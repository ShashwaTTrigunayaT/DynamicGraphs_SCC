// ======================================================================
// scc_cuda_incremental_kernels.cu — GPU kernels for incremental graph
// construction (methods 5, 6, 7, 11)
//
// Contains:
//   - filter_cross_scc_edges_kernel  — filter edges by scc_list mismatch
//   - compute_out_degree_kernel      — count outgoing edges per SCC
//   - find_max_pivot_kernel          — reduction to find max out-degree SCC
//   - build_csr_begin_kernel         — build CSR begin array from sorted edges
//   - build_gpu_condensation_graph   — host wrapper: orchestrates GPU CSR build
//
// Compiled with nvcc.
// ======================================================================

#include "scc_cuda.h"
#include <cub/cub.cuh>
#include <vector>
#include <utility>
#include <cstring>
#include <algorithm>

using namespace std;

// ======================================================================
// filter_cross_scc_edges_kernel
//
// Filters an edge list (d_src, d_dst, num_edges) where
// scc_list[src] != scc_list[dst], producing a compacted cross-SCC edge
// list (d_out_src, d_out_dst) with atomic counter d_out_count.
//
// Warp-ballot compaction (same pattern as trim1's compact build).
// Each warp votes on whether its edge is cross-SCC, does one atomicAdd
// per warp for the output position, then scatters via shuffle.
// ======================================================================
__global__ void filter_cross_scc_edges_kernel(
    const int* d_src, const int* d_dst, int num_edges,
    const int* d_scc_list,
    int* d_out_src, int* d_out_dst, int* d_out_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool cross_scc = false;
    int out_src_val = 0, out_dst_val = 0;

    if (idx < num_edges) {
        int u = d_src[idx];
        int v = d_dst[idx];
        int scc_u = d_scc_list[u];
        int scc_v = d_scc_list[v];
        if (scc_u != scc_v) {
            cross_scc = true;
            out_src_val = scc_u;
            out_dst_val = scc_v;
        }
    }

    // Warp-ballot compact
    unsigned mask = __ballot_sync(0xffffffff, cross_scc);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0)
        warp_base = atomicAdd(d_out_count, warp_count);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    int local_rank = __popc(mask & ((1u << lane) - 1));
    if (cross_scc) {
        d_out_src[warp_base + local_rank] = out_src_val;
        d_out_dst[warp_base + local_rank] = out_dst_val;
    }
}

// ======================================================================
// compute_out_degree_kernel
//
// Counts outgoing edges per SCC node in the compacted cross-SCC edge
// list. Each thread processes one edge, atomicAdd on d_out_degree[src].
// ======================================================================
__global__ void compute_out_degree_kernel(
    const int* d_edges_src, int num_edges,
    int* d_out_degree)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_edges; i += stride) {
        atomicAdd(&d_out_degree[d_edges_src[i]], 1);
    }
}

// ======================================================================
// find_max_pivot_kernel
//
// Reduction kernel to find the SCC node with maximum out-degree.
// Each block scans a chunk of the out_degree array, finding its local
// max (index, value). The final max is aggregated on the host.
// ======================================================================
__global__ void find_max_pivot_kernel(
    const int* d_out_degree, int num_sccs,
    int* d_max_idx, int* d_max_val)
{
    extern __shared__ int s_max_val[];
    int* s_max_idx = &s_max_val[blockDim.x];  // second half of SMEM

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int local_max_val = -1;
    int local_max_idx = 0;

    for (int i = tid; i < num_sccs; i += stride) {
        int val = d_out_degree[i];
        if (val > local_max_val) {
            local_max_val = val;
            local_max_idx = i;
        }
    }

    s_max_val[threadIdx.x] = local_max_val;
    s_max_idx[threadIdx.x] = local_max_idx;
    __syncthreads();

    // Warp-reduce in shared memory
    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz) {
            if (s_max_val[threadIdx.x + sz] > s_max_val[threadIdx.x]) {
                s_max_val[threadIdx.x] = s_max_val[threadIdx.x + sz];
                s_max_idx[threadIdx.x] = s_max_idx[threadIdx.x + sz];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(d_max_val, s_max_val[0]);
        if (s_max_val[0] == *d_max_val) {
            // Best effort: if tie, last writer wins
            *d_max_idx = s_max_idx[0];
        }
    }
}

// ======================================================================
// deinterleave_pairs_kernel
//
// Deinterleaves a pair array [first, second, first, second, ...] into
// two separate arrays d_src and d_dst.
//
// Avoids creating 500MB of temporary host vectors for host-side
// concatenation of orig_edges and insert_edges.
// ======================================================================
__global__ void deinterleave_pairs_kernel(
    const int* d_pairs, int num_pairs,
    int* d_src, int* d_dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_pairs; i += stride) {
        d_src[i] = d_pairs[2 * i];
        d_dst[i] = d_pairs[2 * i + 1];
    }
}

// ======================================================================
// build_csr_begin_kernel
//
// Builds the CSR begin array from a sorted edge list.
// For each edge position i, if i==0 or sorted_src[i]!=sorted_src[i-1],
// writes i to begin[sorted_src[i]].
//
// d_begin MUST be initialized to -1 by the caller (cudaMemset) before
// launching this kernel — avoids cross-block races on initialization.
//
// After this kernel: begin[node] = start position for the node's edges,
// or -1 if the node has no outgoing edges.
//
// Caller must:
//   1. Set begin[num_nodes] = M
//   2. Fill forward: for i = num_nodes-1 down to 0:
//        if begin[i] == -1: begin[i] = begin[i+1]
// ======================================================================
__global__ void build_csr_begin_kernel(
    const int* sorted_src, int num_edges,
    int* d_begin, int num_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Mark start positions where src changes
    // (d_begin already initialized to -1 via cudaMemset before launch)
    for (int e = i; e < num_edges; e += stride) {
        if (e == 0 || sorted_src[e] != sorted_src[e - 1]) {
            d_begin[sorted_src[e]] = e;
        }
    }
}

// ======================================================================
// build_gpu_condensation_graph() — HOST FUNCTION
//
// Orchestrates the GPU-based condensation graph construction for
// method 6. Uploads host data, launches kernels, builds CSR,
// downloads results.
//
// Returns true on success.
// ======================================================================
bool build_gpu_condensation_graph(
    const vector<pair<int,int>>& orig_edges,
    const vector<pair<int,int>>& insert_edges,
    const vector<int>& h_scc_list,
    int num_sccs,
    int& good_init_pivot,
    vector<edge_t>& h_begin,
    vector<node_t>& h_node_idx,
    vector<edge_t>& h_r_begin,
    vector<node_t>& h_r_node_idx,
    int& N, int& M)
{
    fprintf(stderr, "[DBG] GPU build start\n");
    int num_orig = (int)orig_edges.size();
    int num_ins  = (int)insert_edges.size();
    int total_edges = num_orig + num_ins;
    int num_vertices = (int)h_scc_list.size();

    // Skip if no edges at all
    if (total_edges == 0) {
        N = (num_sccs > 0) ? num_sccs : 1;
        M = 0;
        good_init_pivot = 0;
        h_begin.assign(N + 1, 0);
        h_node_idx.clear();
        h_r_begin.assign(N + 1, 0);
        h_r_node_idx.clear();
        return true;
    }

    // ---------------------------------------------------------------
    // 1. Upload host data to GPU
    // ---------------------------------------------------------------
    int* d_all_src = NULL;
    int* d_all_dst = NULL;
    int* d_scc_list = NULL;

    CUDA_CHECK(cudaMalloc(&d_all_src,  total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_dst,  total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scc_list, num_vertices * sizeof(int)));

    // Upload edge data directly as pairs (avoids 500MB of temporary host vectors)
    // We upload to a GPU pair buffer, then deinterleave via a kernel.
    {
        int* d_pairs = NULL;
        CUDA_CHECK(cudaMalloc(&d_pairs, total_edges * sizeof(int) * 2));

        // Upload orig edges as pairs (first, second, first, second, ...)
        CUDA_CHECK(cudaMemcpy(d_pairs, orig_edges.data(),
                               num_orig * sizeof(pair<int,int>),
                               cudaMemcpyHostToDevice));
        // Upload insert edges after orig
        CUDA_CHECK(cudaMemcpy(d_pairs + num_orig * 2, insert_edges.data(),
                               num_ins * sizeof(pair<int,int>),
                               cudaMemcpyHostToDevice));

        // Deinterleave: pairs -> d_all_src and d_all_dst
        int db_blocks = (total_edges + 255) / 256;
        db_blocks = min(db_blocks, 1024);
        deinterleave_pairs_kernel<<<db_blocks, 256>>>(
            d_pairs, total_edges, d_all_src, d_all_dst);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_pairs));
    }
    CUDA_CHECK(cudaMemcpy(d_scc_list, h_scc_list.data(),
                           num_vertices * sizeof(int), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------------
    // 2. Allocate output buffers for filtered edges
    //    (worst case: all edges are cross-SCC)
    // ---------------------------------------------------------------
    int* d_filtered_src = NULL;
    int* d_filtered_dst = NULL;
    int* d_filtered_count = NULL;

    CUDA_CHECK(cudaMalloc(&d_filtered_src,   total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_filtered_dst,   total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_filtered_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_filtered_count, 0, sizeof(int)));

    // ---------------------------------------------------------------
    // 3. Launch filter kernel (warp-ballot compact)
    // ---------------------------------------------------------------
    int block_size = 256;
    int grid_size = (total_edges + block_size - 1) / block_size;
    grid_size = min(grid_size, 1024);

    filter_cross_scc_edges_kernel<<<grid_size, block_size>>>(
        d_all_src, d_all_dst, total_edges,
        d_scc_list,
        d_filtered_src, d_filtered_dst, d_filtered_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_cross;
    CUDA_CHECK(cudaMemcpy(&num_cross, d_filtered_count,
                           sizeof(int), cudaMemcpyDeviceToHost));

    // No cross-SCC edges → single-node condensation graph
    if (num_cross == 0) {
        N = (num_sccs > 0) ? num_sccs : 1;
        M = 0;
        good_init_pivot = 0;
        h_begin.assign(N + 1, 0);
        h_node_idx.clear();
        h_r_begin.assign(N + 1, 0);
        h_r_node_idx.clear();

        CUDA_CHECK(cudaFree(d_all_src));
        CUDA_CHECK(cudaFree(d_all_dst));
        CUDA_CHECK(cudaFree(d_scc_list));
        CUDA_CHECK(cudaFree(d_filtered_src));
        CUDA_CHECK(cudaFree(d_filtered_dst));
        CUDA_CHECK(cudaFree(d_filtered_count));
        return true;
    }

    // ---------------------------------------------------------------
    // 4. Compute out-degree per SCC → find max pivot
    // ---------------------------------------------------------------
    int* d_out_degree = NULL;
    CUDA_CHECK(cudaMalloc(&d_out_degree, num_sccs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_degree, 0, num_sccs * sizeof(int)));

    compute_out_degree_kernel<<<grid_size, block_size>>>(
        d_filtered_src, num_cross, d_out_degree);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Find max pivot on GPU (reduction)
    {
        int* d_max_idx = NULL;
        int* d_max_val = NULL;
        CUDA_CHECK(cudaMalloc(&d_max_idx, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_max_val, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_max_val, -1, sizeof(int)));

        int red_gs = (num_sccs + block_size - 1) / block_size;
        red_gs = min(red_gs, 1024);
        size_t smem_sz = 2 * block_size * sizeof(int);

        find_max_pivot_kernel<<<red_gs, block_size, smem_sz>>>(
            d_out_degree, num_sccs, d_max_idx, d_max_val);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_max_val;
        CUDA_CHECK(cudaMemcpy(&good_init_pivot, d_max_idx,
                               sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_max_val, d_max_val,
                               sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_max_idx));
        CUDA_CHECK(cudaFree(d_max_val));

        if (h_max_val <= 0) good_init_pivot = 0;
    }

    // ---------------------------------------------------------------
    // 5. Build forward CSR from compacted edge list
    //    Using cub::DeviceRadixSort::SortPairs
    // ---------------------------------------------------------------
    int* d_sorted_src = NULL;
    int* d_sorted_dst = NULL;
    CUDA_CHECK(cudaMalloc(&d_sorted_src, num_cross * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_dst, num_cross * sizeof(int)));

    // Determine temporary storage size for cub sort
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_filtered_src, d_sorted_src,
        d_filtered_dst, d_sorted_dst,
        num_cross, 0, sizeof(int) * 8);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sort pairs (src, dst) by src
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_filtered_src, d_sorted_src,
        d_filtered_dst, d_sorted_dst,
        num_cross, 0, sizeof(int) * 8);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_temp_storage));

    // ---------------------------------------------------------------
    // 6. Build forward CSR begin array
    // ---------------------------------------------------------------
    int* d_begin_gpu = NULL;
    int* d_node_idx_gpu = NULL;
    CUDA_CHECK(cudaMalloc(&d_begin_gpu, (num_sccs + 1) * sizeof(edge_t)));
    CUDA_CHECK(cudaMalloc(&d_node_idx_gpu, num_cross * sizeof(node_t)));

    // Initialize begin array to -1, then build from sorted edges
    CUDA_CHECK(cudaMemset(d_begin_gpu, 0xFF, (num_sccs + 1) * sizeof(edge_t)));
    build_csr_begin_kernel<<<grid_size, block_size>>>(
        d_sorted_src, num_cross, d_begin_gpu, num_sccs);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Set sentinel
    CUDA_CHECK(cudaMemcpy(&d_begin_gpu[num_sccs], &num_cross,
                           sizeof(int), cudaMemcpyHostToDevice));

    // Fill zero-degree nodes on host (num_sccs is small)
    h_begin.resize(num_sccs + 1);
    CUDA_CHECK(cudaMemcpy(h_begin.data(), d_begin_gpu,
                           (num_sccs + 1) * sizeof(edge_t),
                           cudaMemcpyDeviceToHost));

    // Fill forward: propagate last edge position to zero-degree nodes
    for (int i = num_sccs - 1; i >= 0; i--) {
        if (h_begin[i] == -1)
            h_begin[i] = h_begin[i + 1];
    }
    // Upload corrected begin array back
    CUDA_CHECK(cudaMemcpy(d_begin_gpu, h_begin.data(),
                           (num_sccs + 1) * sizeof(edge_t),
                           cudaMemcpyHostToDevice));

    // node_idx = sorted_dst (already in correct order)
    CUDA_CHECK(cudaMemcpy(d_node_idx_gpu, d_sorted_dst,
                           num_cross * sizeof(node_t),
                           cudaMemcpyDeviceToDevice));

    // ---------------------------------------------------------------
    // 7. Build reverse CSR — sort by dst
    // ---------------------------------------------------------------
    int* d_sorted_by_dst_src = NULL;
    int* d_sorted_by_dst_dst = NULL;
    CUDA_CHECK(cudaMalloc(&d_sorted_by_dst_src, num_cross * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_by_dst_dst, num_cross * sizeof(int)));

    // Determine temp storage for second sort
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_filtered_dst, d_sorted_by_dst_dst,  // key = dst
        d_filtered_src, d_sorted_by_dst_src,  // value = src
        num_cross, 0, sizeof(int) * 8);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sort pairs (dst, src) by dst
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_filtered_dst, d_sorted_by_dst_dst,
        d_filtered_src, d_sorted_by_dst_src,
        num_cross, 0, sizeof(int) * 8);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_temp_storage));

    // Build reverse begin array
    int* d_r_begin_gpu = NULL;
    int* d_r_node_idx_gpu = NULL;
    CUDA_CHECK(cudaMalloc(&d_r_begin_gpu, (num_sccs + 1) * sizeof(edge_t)));
    CUDA_CHECK(cudaMalloc(&d_r_node_idx_gpu, num_cross * sizeof(node_t)));

    // Initialize reverse begin array to -1, then build from sorted edges
    CUDA_CHECK(cudaMemset(d_r_begin_gpu, 0xFF, (num_sccs + 1) * sizeof(edge_t)));
    build_csr_begin_kernel<<<grid_size, block_size>>>(
        d_sorted_by_dst_dst, num_cross, d_r_begin_gpu, num_sccs);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&d_r_begin_gpu[num_sccs], &num_cross,
                           sizeof(int), cudaMemcpyHostToDevice));

    // Fill zero-degree nodes on host
    h_r_begin.resize(num_sccs + 1);
    CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin_gpu,
                           (num_sccs + 1) * sizeof(edge_t),
                           cudaMemcpyDeviceToHost));
    for (int i = num_sccs - 1; i >= 0; i--) {
        if (h_r_begin[i] == -1)
            h_r_begin[i] = h_r_begin[i + 1];
    }
    CUDA_CHECK(cudaMemcpy(d_r_begin_gpu, h_r_begin.data(),
                           (num_sccs + 1) * sizeof(edge_t),
                           cudaMemcpyHostToDevice));

    // Reverse node_idx = sorted_src (by dst sort, the payload is src)
    CUDA_CHECK(cudaMemcpy(d_r_node_idx_gpu, d_sorted_by_dst_src,
                           num_cross * sizeof(node_t),
                           cudaMemcpyDeviceToDevice));

    // ---------------------------------------------------------------
    // 8. Copy to host output arrays
    // ---------------------------------------------------------------
    h_node_idx.assign(d_sorted_dst, d_sorted_dst + num_cross);
    h_r_node_idx.assign(d_sorted_by_dst_src, d_sorted_by_dst_src + num_cross);

    N = num_sccs;
    M = num_cross;

    // ---------------------------------------------------------------
    // 9. Build GPUGraph directly on GPU
    //
    // NOTE: d_begin_gpu, d_node_idx_gpu, d_r_begin_gpu, d_r_node_idx_gpu
    // are already on device. We create a GPUGraph struct pointing to them.
    // The caller is responsible for freeing these buffers.
    //
    // But we also need d_begin_gpu etc. to survive — they're now owned
    // by the GPUGraph. The caller must call graph_free() when done.
    // ---------------------------------------------------------------
    // Note: graph_upload() in main() will re-upload the host CSR arrays
    // which is redundant but keeps the code path uniform. We could skip
    // it by returning a GPUGraph, but that complicates the flow.

    // Free GPU filter/sort buffers (keep CSR arrays)
    CUDA_CHECK(cudaFree(d_all_src));
    CUDA_CHECK(cudaFree(d_all_dst));
    CUDA_CHECK(cudaFree(d_scc_list));
    CUDA_CHECK(cudaFree(d_filtered_src));
    CUDA_CHECK(cudaFree(d_filtered_dst));
    CUDA_CHECK(cudaFree(d_filtered_count));
    CUDA_CHECK(cudaFree(d_out_degree));
    CUDA_CHECK(cudaFree(d_sorted_src));
    CUDA_CHECK(cudaFree(d_sorted_dst));
    CUDA_CHECK(cudaFree(d_sorted_by_dst_src));
    CUDA_CHECK(cudaFree(d_sorted_by_dst_dst));
    CUDA_CHECK(cudaFree(d_begin_gpu));
    CUDA_CHECK(cudaFree(d_node_idx_gpu));
    CUDA_CHECK(cudaFree(d_r_begin_gpu));
    CUDA_CHECK(cudaFree(d_r_node_idx_gpu));

    return true;
}
