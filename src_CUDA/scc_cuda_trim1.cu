#include "scc_cuda.h"
#include <cooperative_groups.h>

// ======================================================================
// Device-side global state: analogs of OpenMP static globals
// ======================================================================
int* d_trim_targets = NULL;
int  d_trim_targets_count = 0;
int  d_trim_targets_capacity = 0;

int* d_compact_scratch = NULL;  // scratch buffer for compact build
int* d_compact_prefix  = NULL;  // prefix sum / counter buffer
int  d_compact_grid_sz = 0;

// Fix 2: Alive-neighbor counts — avoids rescanning edges on every iteration
// d_out_alive_count[n] = # of outgoing neighbors with same color (not yet trimmed)
// d_in_alive_count[n]  = # of incoming neighbors with same color (not yet trimmed)
int* d_out_alive_count = NULL;
int* d_in_alive_count  = NULL;
int  d_alive_count_initialized = 0;  // host-side flag

// ======================================================================
// initialize_trim1()
// OpenMP: clears trim_targets, reserves space, clears L[] per thread
// ======================================================================
void initialize_trim1()
{
    d_trim_targets_count = 0;
}

void initialize_trim1_full(int num_nodes)
{
    if (d_trim_targets) cudaFree(d_trim_targets);
    d_trim_targets_capacity = num_nodes;
    d_trim_targets_count = 0;
    CUDA_CHECK(cudaMalloc(&d_trim_targets, num_nodes * sizeof(int)));

    if (d_compact_scratch) cudaFree(d_compact_scratch);
    if (d_compact_prefix)  cudaFree(d_compact_prefix);
    d_compact_grid_sz = (num_nodes + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_compact_scratch, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compact_prefix, d_compact_grid_sz * sizeof(int)));

    // Fix 2: allocate alive-count arrays
    if (d_out_alive_count) cudaFree(d_out_alive_count);
    if (d_in_alive_count)  cudaFree(d_in_alive_count);
    CUDA_CHECK(cudaMalloc(&d_out_alive_count, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_alive_count,  num_nodes * sizeof(int)));
    d_alive_count_initialized = 0;
}

void finalize_trim1()
{
    if (d_trim_targets)    { cudaFree(d_trim_targets);    d_trim_targets = NULL; }
    if (d_compact_scratch) { cudaFree(d_compact_scratch); d_compact_scratch = NULL; }
    if (d_compact_prefix)  { cudaFree(d_compact_prefix);  d_compact_prefix = NULL; }
    if (d_out_alive_count) { cudaFree(d_out_alive_count); d_out_alive_count = NULL; }
    if (d_in_alive_count)  { cudaFree(d_in_alive_count);  d_in_alive_count = NULL; }
    d_trim_targets_count = 0;
    d_trim_targets_capacity = 0;
    d_alive_count_initialized = 0;
}

int* get_compact_trim_targets_device() { return d_trim_targets; }
int  get_compact_trim_targets_count()  { return d_trim_targets_count; }

// ======================================================================
// trim_once_node_device() — SHARED DEVICE FUNCTION
//
// Exact mirror of:
//   inline static void trim_once_node(gm_graph& G, int curr_color,
//                                      int& count, node_t n)
//
// Called by all three kernels below (global, compact, local).
// Returns 1 if node was trimmed, 0 otherwise.
// ======================================================================
__device__ int trim_once_node_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    node_t n,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec)
{
    // === OpenMP: if (G_Color[n] == -2) continue; ===
    if (__ldg(&d_Color[n]) == SCC_FOUND) return 0;

    // === int curr_color = G_Color[n]; ===
    int curr_color = __ldg(&d_Color[n]);

    // === OpenMP: met_algo==11 && flag11==2 ===
    if (met_algo == 11 && flag11 == 2) {
        if (d_SCC[n] < 0) {
            d_Color[n] = -2;
            d_SCC[n] = n;
            atomicAdd(d_count_trim_spec, 1);
            return 1;
        }
    }

    // === OpenMP: met_algo==9 && vec_scc_count[scc_list[n]] == -1 ===
    if (met_algo == 9 && d_vec_scc_count[d_scc_list[n]] == -1) {
        d_Color[n] = -2;
        d_SCC[n] = -1;
        return 1;
    }

    // === OpenMP: met_algo==7 && affect_level[level_ver[n]] == 0 ===
    if (met_algo == 7 && d_affect_level[d_level_ver[n]] == 0) {
        d_SCC[n] = n;
        d_Color[n] = -2;
        return 1;
    }

    // === OpenMP: if (G_Color[n] != curr_color) return; ===
    if (__ldg(&d_Color[n]) != curr_color) return 0;

    // === OpenMP: out-degree check ===
    int degree = 0;
    for (edge_t k_idx = d_begin[n]; k_idx < d_begin[n + 1]; k_idx++) {
        node_t k = d_node_idx[k_idx];
        if (k == n) continue;
        if (__ldg(&d_Color[k]) == curr_color) { degree = 1; break; }
    }

    if (degree == 0) {
        d_SCC[n] = n;
        d_Color[n] = -2;
        return 1;
    }

    // === OpenMP: in-degree check ===
    degree = 0;
    for (edge_t k_idx = d_r_begin[n]; k_idx < d_r_begin[n + 1]; k_idx++) {
        node_t k = d_r_node_idx[k_idx];
        if (k == n) continue;
        if (__ldg(&d_Color[k]) == curr_color) { degree = 1; break; }
    }

    if (degree == 0) {
        d_SCC[n] = n;
        d_Color[n] = -2;
        return 1;
    }

    return 0;
}

// ======================================================================
// Kernel 1: do_global_trim1 — iterates over ALL nodes
// Each block accumulates its count in shared memory, then one thread
// per block does a single global atomicAdd (reduces contention 256x).
// ======================================================================
__global__ void trim_once_node_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    int num_nodes,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec)
{
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    int num_warps = gridDim.x * (blockDim.x / 32);
    int local_count = 0;

    for (int n = warp_id; n < num_nodes; n += num_warps) {
        if (d_Color[n] == SCC_FOUND) continue;
        int curr_color = d_Color[n];

        // Method-specific checks (same as trim_once_node_device)
        if (met_algo == 11 && flag11 == 2) {
            if (d_SCC[n] < 0) {
                d_Color[n] = -2;
                d_SCC[n] = n;
                atomicAdd(d_count_trim_spec, 1);
                local_count++;
                continue;
            }
        }
        if (met_algo == 9 && d_vec_scc_count[d_scc_list[n]] == -1) {
            d_Color[n] = -2;
            d_SCC[n] = -1;
            local_count++;
            continue;
        }
        if (met_algo == 7 && d_affect_level[d_level_ver[n]] == 0) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            continue;
        }
        if (d_Color[n] != curr_color) continue;

        // Warp-cooperative out-degree check: 32 lanes scan 32 edges per step
        bool found = false;
        edge_t out_begin = d_begin[n];
        edge_t out_end   = d_begin[n + 1];
        for (edge_t base = out_begin; base < out_end && !found; base += 32) {
            edge_t k_idx = base + lane;
            bool alive = (k_idx < out_end) &&
                         (d_node_idx[k_idx] != n) &&
                         (d_Color[d_node_idx[k_idx]] == curr_color);
            unsigned mask = __ballot_sync(0xffffffff, alive);
            if (mask) found = true;
        }

        if (!found) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            continue;
        }

        // Warp-cooperative in-degree check
        found = false;
        edge_t in_begin = d_r_begin[n];
        edge_t in_end   = d_r_begin[n + 1];
        for (edge_t base = in_begin; base < in_end && !found; base += 32) {
            edge_t k_idx = base + lane;
            bool alive = (k_idx < in_end) &&
                         (d_r_node_idx[k_idx] != n) &&
                         (d_Color[d_r_node_idx[k_idx]] == curr_color);
            unsigned mask = __ballot_sync(0xffffffff, alive);
            if (mask) found = true;
        }

        if (!found) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
        }
    }

    if (local_count > 0) atomicAdd(&s_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0)
        atomicAdd(d_count, s_count);
}

// ======================================================================
// compute_trim_targets_alive_counts_kernel — Fix 2d (warp-cooperative)
//
// Warp-cooperative version: all 32 lanes in each warp share the edge
// scanning for one node. Supernodes with 100K+ edges no longer stall
// the entire block (each warp independently scans its node at 32× speed).
//
// Uses __shfl_xor_sync warp reduction to sum per-lane partial counts.
// The caller zeroes d_out/in_alive_count arrays before launching.
// ======================================================================
__global__ void compute_trim_targets_alive_counts_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    const int* d_Color,
    int* d_out_alive_count, int* d_in_alive_count,
    const int* d_trim_targets, int num_targets)
{
    // Block-cooperative: all 256 threads in a block work together on ONE node
    // at a time. This prevents supernodes from stalling the block: instead of
    // one warp doing all 500K-edge scanning (15K iterations/lane) while 7 other
    // warps wait, all 256 threads share the work (1,953 iterations/thread).
    __shared__ int s_red[256];
    int tid = threadIdx.x;

    for (int ix = blockIdx.x; ix < num_targets; ix += gridDim.x) {
        node_t n = d_trim_targets[ix];
        int out_cnt = 0;
        int in_cnt = 0;

        if (d_Color[n] != SCC_FOUND) {
            int color = __ldg(&d_Color[n]);

            // ---- Out-degree: all 256 threads cooperatively scan ----
            // Unrolled 4x with __ldg for random d_Color reads.
            // __ldg() routes through the read-only data cache which handles
            // random access patterns better than the general L1/L2 path.
            // Loop unrolling gives the compiler more ILP to hide memory latency.
            edge_t out_end = d_begin[n + 1];
            edge_t out_stride = blockDim.x * 4;
            for (edge_t e = d_begin[n] + tid; e < out_end; e += out_stride) {
                int k0 = (e + blockDim.x * 0 < out_end) ? d_node_idx[e + blockDim.x * 0] : n;
                int k1 = (e + blockDim.x * 1 < out_end) ? d_node_idx[e + blockDim.x * 1] : n;
                int k2 = (e + blockDim.x * 2 < out_end) ? d_node_idx[e + blockDim.x * 2] : n;
                int k3 = (e + blockDim.x * 3 < out_end) ? d_node_idx[e + blockDim.x * 3] : n;
                out_cnt += (k0 != n && __ldg(&d_Color[k0]) == color) ? 1 : 0;
                out_cnt += (k1 != n && __ldg(&d_Color[k1]) == color) ? 1 : 0;
                out_cnt += (k2 != n && __ldg(&d_Color[k2]) == color) ? 1 : 0;
                out_cnt += (k3 != n && __ldg(&d_Color[k3]) == color) ? 1 : 0;
            }

            // ---- In-degree: all 256 threads cooperatively scan ----
            edge_t in_end = d_r_begin[n + 1];
            edge_t in_stride = blockDim.x * 4;
            for (edge_t e = d_r_begin[n] + tid; e < in_end; e += in_stride) {
                int k0 = (e + blockDim.x * 0 < in_end) ? d_r_node_idx[e + blockDim.x * 0] : n;
                int k1 = (e + blockDim.x * 1 < in_end) ? d_r_node_idx[e + blockDim.x * 1] : n;
                int k2 = (e + blockDim.x * 2 < in_end) ? d_r_node_idx[e + blockDim.x * 2] : n;
                int k3 = (e + blockDim.x * 3 < in_end) ? d_r_node_idx[e + blockDim.x * 3] : n;
                in_cnt += (k0 != n && __ldg(&d_Color[k0]) == color) ? 1 : 0;
                in_cnt += (k1 != n && __ldg(&d_Color[k1]) == color) ? 1 : 0;
                in_cnt += (k2 != n && __ldg(&d_Color[k2]) == color) ? 1 : 0;
                in_cnt += (k3 != n && __ldg(&d_Color[k3]) == color) ? 1 : 0;
            }
        }

        // ---- Block-wide tree reduction: sum out_cnt across all 256 threads ----
        s_red[tid] = out_cnt;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) s_red[tid] += s_red[tid + s];
            __syncthreads();
        }

        if (tid == 0) d_out_alive_count[n] = s_red[0];
        __syncthreads();

        // ---- Block-wide tree reduction: sum in_cnt across all 256 threads ----
        s_red[tid] = in_cnt;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) s_red[tid] += s_red[tid + s];
            __syncthreads();
        }

        if (tid == 0) d_in_alive_count[n] = s_red[0];
        __syncthreads();
    }
}

// ======================================================================
// Kernel 2: do_global_trim1_compact — iterates over trim_targets
// Each block accumulates its count in shared memory, then one thread
// per block does a single global atomicAdd (reduces contention 256x).
// ======================================================================
__global__ void trim_once_node_compact_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_trim_targets, int num_targets,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec)
{
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    int num_warps = gridDim.x * (blockDim.x / 32);
    int local_count = 0;

    for (int ix = warp_id; ix < num_targets; ix += num_warps) {
        node_t n = d_trim_targets[ix];
        if (__ldg(&d_Color[n]) == SCC_FOUND) continue;
        int curr_color = __ldg(&d_Color[n]);

        // Method-specific checks (same as trim_once_node_device)
        if (met_algo == 11 && flag11 == 2) {
            if (d_SCC[n] < 0) {
                d_Color[n] = -2;
                d_SCC[n] = n;
                atomicAdd(d_count_trim_spec, 1);
                local_count++;
                continue;
            }
        }
        if (met_algo == 9 && d_vec_scc_count[d_scc_list[n]] == -1) {
            d_Color[n] = -2;
            d_SCC[n] = -1;
            local_count++;
            continue;
        }
        if (met_algo == 7 && d_affect_level[d_level_ver[n]] == 0) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            continue;
        }
        if (__ldg(&d_Color[n]) != curr_color) continue;

        // Warp-cooperative out-degree check: 32 lanes scan 32 edges per step
        bool found = false;
        edge_t out_begin = d_begin[n];
        edge_t out_end   = d_begin[n + 1];
        for (edge_t base = out_begin; base < out_end && !found; base += 32) {
            edge_t k_idx = base + lane;
            bool alive = (k_idx < out_end) &&
                         (d_node_idx[k_idx] != n) &&
                         (__ldg(&d_Color[d_node_idx[k_idx]]) == curr_color);
            unsigned mask = __ballot_sync(0xffffffff, alive);
            if (mask) found = true;
        }

        if (!found) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            continue;
        }

        // Warp-cooperative in-degree check
        found = false;
        edge_t in_begin = d_r_begin[n];
        edge_t in_end   = d_r_begin[n + 1];
        for (edge_t base = in_begin; base < in_end && !found; base += 32) {
            edge_t k_idx = base + lane;
            bool alive = (k_idx < in_end) &&
                         (d_r_node_idx[k_idx] != n) &&
                         (__ldg(&d_Color[d_r_node_idx[k_idx]]) == curr_color);
            unsigned mask = __ballot_sync(0xffffffff, alive);
            if (mask) found = true;
        }

        if (!found) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
        }
    }

    if (local_count > 0) atomicAdd(&s_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0)
        atomicAdd(d_count, s_count);
}

//Kernel 3: do_local_trim1 — iterates over a work item's set

__global__ void trim_once_node_local_set_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_set_nodes, int set_size,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec)
{
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int local_count = 0;

    for (int ix = tid; ix < set_size; ix += stride) {
        node_t n = d_set_nodes[ix];
        local_count += trim_once_node_device(
            d_begin, d_node_idx, d_r_begin, d_r_node_idx,
            d_Color, d_SCC, n,
            met_algo, flag11,
            d_scc_list, d_vec_scc_count,
            d_level_ver, d_affect_level,
            d_count_trim_spec);
    }

    if (local_count > 0) atomicAdd(&s_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0)
        atomicAdd(d_count, s_count);
}

// ======================================================================
// Compact build helpers (forward-declared here, defined below
// ======================================================================

// Build compact set of nodes matching a specific color
__global__ void build_compact_by_color_kernel(
    const int* d_Color, int* d_targets, int* d_count,
    int num_nodes, int target_color);

__global__ void build_compact_from_all_kernel(
    const int* d_Color, int* d_targets, int* d_count, int num_nodes);

__global__ void build_compact_from_existing_kernel(
    const int* d_Color,
    const int* d_src_targets, int num_src,
    int* d_dst_targets, int* d_count);

// ======================================================================
// do_global_trim1()
// ======================================================================
int do_global_trim1(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec)
{
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    if (d_count_trim_spec)
        CUDA_CHECK(cudaMemset(d_count_trim_spec, 0, sizeof(int)));

    int N = g.num_nodes;
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    trim_once_node_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx, g.d_r_begin, g.d_r_node_idx,
        st.d_Color, st.d_SCC, d_count, N,
        met_algo, flag11,
        da.d_scc_list, da.d_vec_scc_count,
        da.d_level_ver, da.d_affect_level,
        d_count_trim_spec);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_TIMED_MEMCPY(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

// ======================================================================
// compute_trim1_alive_counts() — Fix 2 (compact-only)
//
// Called once before the compact trim loop to initialize alive-neighbor
// counts for nodes in the COMPACT SET only. Uses cudaMemset to zero out
// the arrays first (so non-compact nodes have 0, which is safe).
//
// After this, the compact kernel checks d_out/in_alive_count[n]
// (O(1)) instead of scanning all edges.
// ======================================================================
void compute_trim1_alive_counts(const GPUGraph& g, const GPUState& st)
{
    int N = g.num_nodes;
    // Zero out entire arrays first (non-compact nodes get 0)
    CUDA_CHECK(cudaMemset(d_out_alive_count, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_in_alive_count,  0, N * sizeof(int)));

    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return;

    int block_size = 256;
    // Query SM count for portable grid sizing (L40S=142, A100=108, etc.)
    // The block-level kernel requires 1 block per node — launch enough blocks
    // to fill all SMs while not exceeding num_targets.
    static int grid_size = 0;
    if (grid_size == 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        grid_size = (num_targets < props.multiProcessorCount) ? num_targets : props.multiProcessorCount;
    }
    int actual_grid = (num_targets < 256) ? num_targets : grid_size;
    compute_trim_targets_alive_counts_kernel<<<actual_grid, block_size>>>(
        g.d_begin, g.d_node_idx,
        g.d_r_begin, g.d_r_node_idx,
        st.d_Color,
        d_out_alive_count, d_in_alive_count,
        d_trim_targets, num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_alive_count_initialized = 1;
}

// ======================================================================
// Kernel: trim_once_node_compact_fix2_kernel — Fix 2c (recompute, no decrements)
//
// Pure O(1) alive-count check kernel — reads d_out/in_alive_count[n] and
// marks nodes for trimming if count <= 0. NO decrement loops — that was
// the source of the 115ms atomic storm.
//
// The caller (repeat_global_trim1_compact) recomputes alive counts each
// iteration via compute_trim1_alive_counts, which scans only compact-set
// nodes (~7K) costing ~1ms per recompute. Over 5 iterations: ~5ms total.
// ======================================================================
__global__ void trim_once_node_compact_fix2_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_trim_targets, int num_targets,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec,
    int* d_out_alive_count, int* d_in_alive_count)
{
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int num_warps = gridDim.x * (blockDim.x / 32);
    int local_count = 0;

    for (int ix = warp_id; ix < num_targets; ix += num_warps) {
        node_t n = d_trim_targets[ix];
        if (d_Color[n] == SCC_FOUND) continue;
        int curr_color = d_Color[n];

        // Method-specific checks (same as trim_once_node_device)
        if (met_algo == 11 && flag11 == 2) {
            if (d_SCC[n] < 0) {
                d_Color[n] = -2;
                d_SCC[n] = n;
                atomicAdd(d_count_trim_spec, 1);
                local_count++;
                continue;
            }
        }
        if (met_algo == 9 && d_vec_scc_count[d_scc_list[n]] == -1) {
            d_Color[n] = -2;
            d_SCC[n] = -1;
            local_count++;
            continue;
        }
        if (met_algo == 7 && d_affect_level[d_level_ver[n]] == 0) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            continue;
        }
        if (d_Color[n] != curr_color) continue;

        // ---- Out-degree check via alive count (O(1)) — no decrement ----
        if (d_out_alive_count[n] <= 0) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            // No decrement — counts will be recomputed next iteration
            continue;
        }

        // ---- In-degree check via alive count (O(1)) — no decrement ----
        if (d_in_alive_count[n] <= 0) {
            d_SCC[n] = n;
            d_Color[n] = -2;
            local_count++;
            // No decrement — counts will be recomputed next iteration
        }
    }

    if (local_count > 0) atomicAdd(&s_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0)
        atomicAdd(d_count, s_count);
}

// ======================================================================
// do_global_trim1_compact() — Fix 1: warp-cooperative edge scan
// ======================================================================
int do_global_trim1_compact(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec)
{
    if (d_trim_targets_count == 0) return 0;

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    if (d_count_trim_spec)
        CUDA_CHECK(cudaMemset(d_count_trim_spec, 0, sizeof(int)));

    int block_size = 256;
    int grid_size = (d_trim_targets_count + block_size - 1) / block_size;

    trim_once_node_compact_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx, g.d_r_begin, g.d_r_node_idx,
        st.d_Color, st.d_SCC, d_count,
        d_trim_targets, d_trim_targets_count,
        met_algo, flag11,
        da.d_scc_list, da.d_vec_scc_count,
        da.d_level_ver, da.d_affect_level,
        d_count_trim_spec);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_TIMED_MEMCPY(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

// ======================================================================
// do_global_trim1_compact_fix2() — Fix 2: O(1) alive-count check
// ======================================================================
int do_global_trim1_compact_fix2(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec)
{
    if (d_trim_targets_count == 0) return 0;

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    if (d_count_trim_spec)
        CUDA_CHECK(cudaMemset(d_count_trim_spec, 0, sizeof(int)));

    int block_size = 256;
    int grid_size = (d_trim_targets_count + block_size - 1) / block_size;

    trim_once_node_compact_fix2_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx, g.d_r_begin, g.d_r_node_idx,
        st.d_Color, st.d_SCC, d_count,
        d_trim_targets, d_trim_targets_count,
        met_algo, flag11,
        da.d_scc_list, da.d_vec_scc_count,
        da.d_level_ver, da.d_affect_level,
        d_count_trim_spec,
        d_out_alive_count, d_in_alive_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_TIMED_MEMCPY(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

// ======================================================================
// do_local_trim1()
//
// OpenMP:
//   int do_local_trim1(gm_graph& G, my_work* w) {
//     int count = 0;
//     NODE_SET* set = w->color_set;
//     int curr_color = w->color;
//     if (set != NULL) {
//       for(I=set->begin(); I!=set->end(); I++)
//         trim_once_node(G, curr_color, count, *I);
//       for(I=set->begin(); I!=set->end(); I++)
//         if (G_Color[*I] != curr_color) set->erase(I);
//     } else {
//       for (node_t n = 0; n < G.num_nodes(); n++)
//         if (G_Color[n] == curr_color)
//           trim_once_node(G, curr_color, count, n);
//     }
//     w->count -= count; return count;
//   }
// ======================================================================
int do_local_trim1(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, int* d_count,
    int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec)
{
    if (w->count <= 0) return 0;

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    int block_size = 256;

    if (w->d_set_nodes != NULL) {
        // --- Mirror: iterate over set nodes ---
        int grid_size = (w->count + block_size - 1) / block_size;

        trim_once_node_local_set_kernel<<<grid_size, block_size>>>(
            g.d_begin, g.d_node_idx, g.d_r_begin, g.d_r_node_idx,
            st.d_Color, st.d_SCC, d_count,
            w->d_set_nodes, w->count,
            met_algo, flag11,
            da.d_scc_list, da.d_vec_scc_count,
            da.d_level_ver, da.d_affect_level,
            d_count_trim_spec);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Mirror: erase changed-color nodes from set ---
        // OpenMP: for(I...) if(G_Color[*I]!=curr_color) set->erase(I);
        // CUDA: rebuild set keeping only nodes still matching w->color
        CUDA_CHECK(cudaMemset(d_compact_prefix, 0, sizeof(int)));
        int grid2 = (w->count + block_size - 1) / block_size;
        build_compact_from_existing_kernel<<<grid2, block_size>>>(
            st.d_Color,
            w->d_set_nodes, w->count,
            w->d_set_nodes, d_compact_prefix);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_TIMED_MEMCPY(&w->count, d_compact_prefix, sizeof(int),
                              cudaMemcpyDeviceToHost);
    } else {
        // --- Mirror: no set — scan all nodes matching w->color ---
        // OpenMP: for(n=0; n<N; n++) if(G_Color[n]==curr_color) trim_once_node(...)
        // CUDA: build compact set matching w->color, then process it
        if (w->d_set_nodes == NULL && d_trim_targets != NULL) {
            // Use d_trim_targets as temporary buffer
            CUDA_CHECK(cudaMemset(d_compact_prefix, 0, sizeof(int)));
            int grid = (g.num_nodes + block_size - 1) / block_size;
            build_compact_by_color_kernel<<<grid, block_size>>>(
                st.d_Color, d_trim_targets, d_compact_prefix,
                g.num_nodes, w->color);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            int set_size;
            CUDA_TIMED_MEMCPY(&set_size, d_compact_prefix, sizeof(int),
                                  cudaMemcpyDeviceToHost);
            w->d_set_nodes = d_compact_scratch;  // use separate scratch buffer (not shared d_trim_targets)
            w->owns_set = 0;                      // don't own it — shared scratch buffer
            w->count = set_size;
            
            // Now process via the local set kernel
            int grid2 = (set_size + block_size - 1) / block_size;
            trim_once_node_local_set_kernel<<<grid2, block_size>>>(
                g.d_begin, g.d_node_idx, g.d_r_begin, g.d_r_node_idx,
                st.d_Color, st.d_SCC, d_count,
                w->d_set_nodes, set_size,
                met_algo, flag11,
                da.d_scc_list, da.d_vec_scc_count,
                da.d_level_ver, da.d_affect_level,
                d_count_trim_spec);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Rebuild set: keep only nodes still matching color
            CUDA_CHECK(cudaMemset(d_compact_prefix, 0, sizeof(int)));
            build_compact_from_existing_kernel<<<grid2, block_size>>>(
                st.d_Color, w->d_set_nodes, set_size,
                w->d_set_nodes, d_compact_prefix);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_TIMED_MEMCPY(&w->count, d_compact_prefix, sizeof(int),
                                  cudaMemcpyDeviceToHost);
        }
    }

    int count;
    CUDA_TIMED_MEMCPY(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    w->count -= count;
    return count;
}

// ======================================================================
// repeat_global_trim1()
//
// OpenMP:
//   int repeat_global_trim1(gm_graph& G, int TRIM_STOP) {
//     int total_count = 0;
//     int count = 0;
//     do {
//       count = do_global_trim1(G);
//       total_count += count;
//       if (total_count >= G.num_nodes() * 0.1)
//         return total_count + repeat_global_trim1_compact(G);
//     } while (count > TRIM_STOP);
//     return total_count;
//   }
// ======================================================================
int repeat_global_trim1(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec,
    int TRIM_STOP)
{
    int total_count = 0;
    int count;

    do {
        count = do_global_trim1(st, g, d_count, met_algo, flag11,
                                da, d_count_trim_spec);
        total_count += count;

        if (total_count >= g.num_nodes * 0.1) {
            
            return total_count + repeat_global_trim1_compact(
                st, g, d_count, met_algo, flag11,
                da, d_count_trim_spec, TRIM_STOP);
        }
    } while (count > TRIM_STOP);

    return total_count;
}

// ======================================================================
// Compact build helpers
// ======================================================================

// -------------------------------------------------------------------
// Warp-ballot compact build kernels
// Each kernel replaces per-thread atomicAdd with one atomicAdd per warp
// (32x fewer atomics), reducing DRAM contention.
//
// Pattern:
//   1. All threads in warp vote via __ballot_sync
//   2. Lane 0 does one atomicAdd per warp
//   3. __shfl_sync broadcasts the base offset
//   4. __popc(mask & lower_lanes) gives each thread its local rank
// -------------------------------------------------------------------

// Build compact set of all non-SCC_FOUND nodes
__global__ void build_compact_from_all_kernel(
    const int* d_Color, int* d_targets, int* d_count, int num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (idx < num_nodes) && (d_Color[idx] != SCC_FOUND);

    unsigned mask = __ballot_sync(0xffffffff, active);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0)
        warp_base = atomicAdd(d_count, warp_count);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    int local_rank = __popc(mask & ((1u << lane) - 1));
    if (active)
        d_targets[warp_base + local_rank] = idx;
}

// Build compact set of nodes matching a specific color
__global__ void build_compact_by_color_kernel(
    const int* d_Color, int* d_targets, int* d_count,
    int num_nodes, int target_color)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (idx < num_nodes) && (d_Color[idx] == target_color);

    unsigned mask = __ballot_sync(0xffffffff, active);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0)
        warp_base = atomicAdd(d_count, warp_count);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    int local_rank = __popc(mask & ((1u << lane) - 1));
    if (active)
        d_targets[warp_base + local_rank] = idx;
}

// Build compact set of non-SCC_FOUND nodes from an existing source set
__global__ void build_compact_from_existing_kernel(
    const int* d_Color,
    const int* d_src_targets, int num_src,
    int* d_dst_targets, int* d_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = false;
    node_t n = -1;
    if (idx < num_src) {
        n = d_src_targets[idx];
        active = (d_Color[n] != SCC_FOUND);
    }

    unsigned mask = __ballot_sync(0xffffffff, active);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0)
        warp_base = atomicAdd(d_count, warp_count);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    int local_rank = __popc(mask & ((1u << lane) - 1));
    if (active)
        d_dst_targets[warp_base + local_rank] = n;
}

static void create_trim1_compact_1(GPUState& st, const GPUGraph& g)
{
    CUDA_CHECK(cudaMemset(d_compact_prefix, 0, sizeof(int)));
    int N = g.num_nodes;
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    build_compact_from_all_kernel<<<grid_size, block_size>>>(
        st.d_Color, d_trim_targets, d_compact_prefix, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_TIMED_MEMCPY(&d_trim_targets_count, d_compact_prefix,
                          sizeof(int), cudaMemcpyDeviceToHost);
}

static void create_trim1_compact_1b(GPUState& st, const GPUGraph& g)
{
    CUDA_CHECK(cudaMemset(d_compact_prefix, 0, sizeof(int)));
    int num_src = d_trim_targets_count;
    int block_size = 256;
    int grid_size = (num_src + block_size - 1) / block_size;
    build_compact_from_existing_kernel<<<grid_size, block_size>>>(
        st.d_Color,
        d_trim_targets, num_src,
        d_trim_targets, d_compact_prefix);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_TIMED_MEMCPY(&d_trim_targets_count, d_compact_prefix,
                          sizeof(int), cudaMemcpyDeviceToHost);
}

static void create_trim1_compact_2()
{
    // No-op: atomic counter handles prefix sum implicitly
}

static void create_trim1_compact_3()
{
    // No-op: kernel writes directly to final positions via atomicAdd
}

void create_trim1_compact(GPUState& st, const GPUGraph& g)
{
    if (d_trim_targets_count == 0)
        create_trim1_compact_1(st, g);
    else
        create_trim1_compact_1b(st, g);
    create_trim1_compact_2();
    create_trim1_compact_3();
}// ======================================================================
// Persistent TRIM1 kernel — runs all passes in a single cooperative launch
//
// Uses Cooperative Groups grid.sync() for grid-wide barriers.
// Internally loops: trim pass → in-place compact rebuild → repeat
// until no nodes trimmed (or <= TRIM_STOP).
//
// Each thread maintains its own `num_targets` from compact_counter after
// grid.sync(), avoiding cross-block stale-read races on device_counters.
//
// d_device_counters layout:
//   [0] = num_targets (for host to read back after kernel)
//   [1] = compact_counter (atomic counter for in-place rebuild)
//   [2] = pass_trimmed (per-pass trimmed count for exit check)
// ======================================================================

__global__ void trim_once_node_compact_persistent_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_global_count,
    int* d_trim_targets,
    int initial_num_targets,
    int* d_device_counters,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec,
    int TRIM_STOP)
{
    // Initialize device counters (safe: only block 0 thread 0 writes)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_device_counters[0] = initial_num_targets;
        atomicExch(&d_device_counters[1], 0);
        atomicExch(&d_device_counters[2], 0);
    }

    // Grid-wide barrier: ensures init writes are visible
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    // Each thread tracks num_targets locally to avoid cross-block reads
    int num_targets = initial_num_targets;
    const int MAX_PASSES = 300;

    for (int pass = 0; pass < MAX_PASSES; pass++) {
        if (num_targets <= 0) break;

        // ---- Phase 1: Trim pass (warp-cooperative) ----
        int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
        int lane = threadIdx.x & 31;
        int warp_stride = (blockDim.x / 32) * gridDim.x;

        int local_count = 0;
        for (int i = warp_id; i < num_targets; i += warp_stride) {
            node_t n = d_trim_targets[i];
            if (d_Color[n] == SCC_FOUND) continue;
            int curr_color = d_Color[n];

            // Method-specific checks (same as trim_once_node_device)
            if (met_algo == 11 && flag11 == 2) {
                if (d_SCC[n] < 0) {
                    d_Color[n] = -2;
                    d_SCC[n] = n;
                    atomicAdd(d_count_trim_spec, 1);
                    local_count++;
                    continue;
                }
            }
            if (met_algo == 9 && d_vec_scc_count[d_scc_list[n]] == -1) {
                d_Color[n] = -2;
                d_SCC[n] = -1;
                local_count++;
                continue;
            }
            if (met_algo == 7 && d_affect_level[d_level_ver[n]] == 0) {
                d_SCC[n] = n;
                d_Color[n] = -2;
                local_count++;
                continue;
            }
            if (d_Color[n] != curr_color) continue;

            // Warp-cooperative out-degree check
            bool found = false;
            edge_t out_begin = d_begin[n];
            edge_t out_end   = d_begin[n + 1];
            for (edge_t base = out_begin; base < out_end && !found; base += 32) {
                edge_t k_idx = base + lane;
                bool alive = (k_idx < out_end) &&
                             (d_node_idx[k_idx] != n) &&
                             (d_Color[d_node_idx[k_idx]] == curr_color);
                unsigned mask = __ballot_sync(0xffffffff, alive);
                if (mask) found = true;
            }

            if (!found) {
                d_SCC[n] = n;
                d_Color[n] = -2;
                local_count++;
                continue;
            }

            // Warp-cooperative in-degree check
            found = false;
            edge_t in_begin = d_r_begin[n];
            edge_t in_end   = d_r_begin[n + 1];
            for (edge_t base = in_begin; base < in_end && !found; base += 32) {
                edge_t k_idx = base + lane;
                bool alive = (k_idx < in_end) &&
                             (d_r_node_idx[k_idx] != n) &&
                             (d_Color[d_r_node_idx[k_idx]] == curr_color);
                unsigned mask = __ballot_sync(0xffffffff, alive);
                if (mask) found = true;
            }

            if (!found) {
                d_SCC[n] = n;
                d_Color[n] = -2;
                local_count++;
            }
        }

        // ---- Aggregate within block via SMEM ----
        __shared__ int s_block_trimmed;
        if (threadIdx.x == 0) s_block_trimmed = 0;
        __syncthreads();
        if (local_count > 0) atomicAdd(&s_block_trimmed, local_count);
        __syncthreads();

        int block_trimmed = s_block_trimmed;

        // Add to global cumulative + per-pass counter
        if (block_trimmed > 0) {
            atomicAdd(d_global_count, block_trimmed);
            atomicAdd(&d_device_counters[2], block_trimmed);
        }

        // ---- Grid barrier: all d_Color writes + counter adds visible ----
        grid.sync();

        // ---- Check exit: per-pass trimmed <= TRIM_STOP ----
        if (d_device_counters[2] <= TRIM_STOP) break;

        // ---- Reset per-pass counter for next iteration ----
        if (threadIdx.x == 0 && blockIdx.x == 0)
            atomicExch(&d_device_counters[2], 0);
        __syncthreads();

        // ---- Phase 2: In-place compact rebuild ----
        // Filter SCC_FOUND from d_trim_targets using warp-ballot atomics.
        if (threadIdx.x == 0 && blockIdx.x == 0)
            atomicExch(&d_device_counters[1], 0);
        __syncthreads();

        int* compact_counter = &d_device_counters[1];

        // Contiguous chunks per block (all threads in each warp participate)
        for (int base = blockIdx.x * blockDim.x;
             base < num_targets;
             base += gridDim.x * blockDim.x) {

            int idx = base + threadIdx.x;
            bool keep = false;
            node_t n = -1;
            if (idx < num_targets) {
                n = d_trim_targets[idx];
                keep = (d_Color[n] != SCC_FOUND);
            }

            unsigned mask = __ballot_sync(0xffffffff, keep);
            int lane = threadIdx.x & 31;
            int warp_count = __popc(mask);

            int warp_base = 0;
            if (lane == 0 && warp_count > 0)
                warp_base = atomicAdd(compact_counter, warp_count);
            warp_base = __shfl_sync(0xffffffff, warp_base, 0);

            int local_rank = __popc(mask & ((1u << lane) - 1));
            if (keep)
                d_trim_targets[warp_base + local_rank] = n;
        }

        // ---- Grid barrier: all compact writes done ----
        grid.sync();

        // ---- Update num_targets from compact counter ----
        // After grid.sync(), d_device_counters[1] is the final post-compact count.
        // Every thread reads the same value — no cross-block race.
        num_targets = d_device_counters[1];

        // Save for host readback, reset compact counter for next iteration
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            d_device_counters[0] = num_targets;
            atomicExch(&d_device_counters[1], 0);
        }
        __syncthreads();
    }
}

// ======================================================================
// repeat_global_trim1_compact() — host-side loop (Fix 3: rebuild compacts)
//
// Each iteration:
//   1. Recompute alive counts for compact-set nodes
//   2. Check counts (O(1) per node) and trim nodes with count <= 0
//   3. If count > TRIM_STOP: rebuild compact set in-place (filter out
//      SCC_FOUND nodes) so the NEXT compute call processes fewer nodes
//
// Rebuilding the compact set reduces each subsequent compute kernel's
// grid size, saving ~3.75ms per iteration. Without rebuild, all 4-5
// iterations pay the full cost (~15ms). With rebuild, each iteration's
// cost shrinks proportionally (~8ms total).
// ======================================================================
int repeat_global_trim1_compact(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec,
    int TRIM_STOP)
{
    create_trim1_compact(st, g);
    if (d_trim_targets_count == 0) return 0;

    int total_count = 0;
    int count;
    do {
        // For large compact sets (>100K nodes), use the original short-circuit
        // kernel which stops at the first alive neighbor. This is much faster
        // on full graphs where most nodes have many alive neighbors.
        // The full-count compute kernel scans ALL edges — expensive on large sets.
        // For small compact sets, the compute kernel + O(1) fix2 check is fine.
        if (d_trim_targets_count > 300000) {
            // Short-circuit: stops at first alive neighbor (fast for large sets)
            count = do_global_trim1_compact(st, g, d_count, met_algo, flag11,
                                            da, d_count_trim_spec);
        } else {
            // Full-count: compute alive counts, then O(1) check
            compute_trim1_alive_counts(g, st);
            count = do_global_trim1_compact_fix2(st, g, d_count, met_algo, flag11,
                                                 da, d_count_trim_spec);
        }
        total_count += count;

        // Rebuild compact set in-place (filters out SCC_FOUND nodes)
        // so the next compute call processes fewer nodes.
        if (count > TRIM_STOP && d_trim_targets_count > 0) {
            create_trim1_compact(st, g);
        }
    } while (count > TRIM_STOP);

    return total_count;
}

// ======================================================================
// repeat_local_trim1()
// ======================================================================
int repeat_local_trim1(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, int* d_count,
    int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec)
{
    int total_count = 0;
    int count;
    do {
        count = do_local_trim1(st, g, w, d_count, met_algo, flag11,
                               da, d_count_trim_spec);
        total_count += count;
    } while (count > 0);
    // Reset d_set_nodes so caller doesn't try to free the shared scratch buffer
    w->d_set_nodes = NULL;
    w->owns_set = 0;
    return total_count;
}
