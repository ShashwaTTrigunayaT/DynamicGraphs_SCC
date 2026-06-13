#include "scc_cuda.h"

// ======================================================================
// Device-side global state: analogs of OpenMP static globals
// ======================================================================
int* d_trim_targets = NULL;
int  d_trim_targets_count = 0;
int  d_trim_targets_capacity = 0;

int* d_compact_scratch = NULL;  // scratch buffer for compact build
int* d_compact_prefix  = NULL;  // prefix sum / counter buffer
int  d_compact_grid_sz = 0;

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
}

void finalize_trim1()
{
    if (d_trim_targets)    { cudaFree(d_trim_targets);    d_trim_targets = NULL; }
    if (d_compact_scratch) { cudaFree(d_compact_scratch); d_compact_scratch = NULL; }
    if (d_compact_prefix)  { cudaFree(d_compact_prefix);  d_compact_prefix = NULL; }
    d_trim_targets_count = 0;
    d_trim_targets_capacity = 0;
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
// ======================================================================
__device__ void trim_once_node_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    node_t n,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec)
{
    // === OpenMP: if (G_Color[n] == -2) continue; ===
    if (d_Color[n] == SCC_FOUND) return;

    // === int curr_color = G_Color[n]; ===
    int curr_color = d_Color[n];

    // === OpenMP: met_algo==11 && flag11==2 ===
    if (met_algo == 11 && flag11 == 2) {
        if (d_SCC[n] < 0) {
            d_Color[n] = -2;
            d_SCC[n] = n;
            atomicAdd(d_count, 1);
            atomicAdd(d_count_trim_spec, 1);
            return;
        }
    }

    // === OpenMP: met_algo==9 && vec_scc_count[scc_list[n]] == -1 ===
    if (met_algo == 9 && d_vec_scc_count[d_scc_list[n]] == -1) {
        d_Color[n] = -2;
        d_SCC[n] = -1;
        atomicAdd(d_count, 1);
        // NO RETURN — falls through to guard below
    }

    // === OpenMP: met_algo==7 && affect_level[level_ver[n]] == 0 ===
    if (met_algo == 7 && d_affect_level[d_level_ver[n]] == 0) {
        d_SCC[n] = n;
        d_Color[n] = -2;
        atomicAdd(d_count, 1);
        return;
    }

    // === OpenMP: if (G_Color[n] != curr_color) return; ===
    if (d_Color[n] != curr_color) return;

    // === OpenMP: out-degree check ===
    int degree = 0;
    for (edge_t k_idx = d_begin[n]; k_idx < d_begin[n + 1]; k_idx++) {
        node_t k = d_node_idx[k_idx];
        if (k == n) continue;
        if (d_Color[k] == curr_color) { degree = 1; break; }
    }

    if (degree == 0) {
        d_SCC[n] = n;
        d_Color[n] = -2;
        atomicAdd(d_count, 1);
        return;
    }

    // === OpenMP: in-degree check ===
    degree = 0;
    for (edge_t k_idx = d_r_begin[n]; k_idx < d_r_begin[n + 1]; k_idx++) {
        node_t k = d_r_node_idx[k_idx];
        if (k == n) continue;
        if (d_Color[k] == curr_color) { degree = 1; break; }
    }

    if (degree == 0) {
        d_SCC[n] = n;
        d_Color[n] = -2;
        atomicAdd(d_count, 1);
        return;
    }
}

// ======================================================================
// Kernel 1: do_global_trim1 — iterates over ALL nodes
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int n = tid; n < num_nodes; n += stride) {
        trim_once_node_device(
            d_begin, d_node_idx, d_r_begin, d_r_node_idx,
            d_Color, d_SCC, d_count, n,
            met_algo, flag11,
            d_scc_list, d_vec_scc_count,
            d_level_ver, d_affect_level,
            d_count_trim_spec);
    }
}

// ======================================================================
// Kernel 2: do_global_trim1_compact — iterates over trim_targets
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ix = tid; ix < num_targets; ix += stride) {
        node_t n = d_trim_targets[ix];
        trim_once_node_device(
            d_begin, d_node_idx, d_r_begin, d_r_node_idx,
            d_Color, d_SCC, d_count, n,
            met_algo, flag11,
            d_scc_list, d_vec_scc_count,
            d_level_ver, d_affect_level,
            d_count_trim_spec);
    }
}

//Kernel 3: do_local_trim1 — iterates over a work item's set--------------------------------------

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ix = tid; ix < set_size; ix += stride) {
        node_t n = d_set_nodes[ix];
        trim_once_node_device(
            d_begin, d_node_idx, d_r_begin, d_r_node_idx,
            d_Color, d_SCC, d_count, n,
            met_algo, flag11,
            d_scc_list, d_vec_scc_count,
            d_level_ver, d_affect_level,
            d_count_trim_spec);
    }
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
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

// ======================================================================
// do_global_trim1_compact()
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
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
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
        CUDA_CHECK(cudaMemcpy(&w->count, d_compact_prefix, sizeof(int),
                               cudaMemcpyDeviceToHost));
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
            CUDA_CHECK(cudaMemcpy(&set_size, d_compact_prefix, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            w->d_set_nodes = d_trim_targets;
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
            CUDA_CHECK(cudaMemcpy(&w->count, d_compact_prefix, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
    }

    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaMemcpy(&d_trim_targets_count, d_compact_prefix,
                          sizeof(int), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaMemcpy(&d_trim_targets_count, d_compact_prefix,
                          sizeof(int), cudaMemcpyDeviceToHost));
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
}

// ======================================================================
// repeat_global_trim1_compact()
// ======================================================================
int repeat_global_trim1_compact(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec,
    int TRIM_STOP)
{
    create_trim1_compact(st, g);
    int total_count = 0;
    int count;
    do {
        count = do_global_trim1_compact(st, g, d_count, met_algo, flag11,
                                        da, d_count_trim_spec);
        total_count += count;
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
    return total_count;
}
