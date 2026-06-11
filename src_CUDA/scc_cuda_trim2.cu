#include "scc_cuda.h"

// ======================================================================
// Device-side global state: mirrors G_nbr and G_maybe_2nd
// ======================================================================
int*  d_G_nbr = NULL;        // [N] — recorded neighbor (Phase 1)
int*  d_G_maybe_2nd = NULL;  // [N] — 0/1 flag: has exactly 1 in OR 1 out

// ======================================================================
// initialize_trim2()
// OpenMP: allocates G_nbr and G_maybe_2nd, initializes to CUDA_NIL_NODE/false
// ======================================================================
void initialize_trim2(int num_nodes)
{
    if (d_G_nbr) cudaFree(d_G_nbr);
    if (d_G_maybe_2nd) cudaFree(d_G_maybe_2nd);

    CUDA_CHECK(cudaMalloc(&d_G_nbr,        num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_G_maybe_2nd,  num_nodes * sizeof(int)));

    // Init: G_nbr[i] = CUDA_NIL_NODE, G_maybe_2nd[i] = 0 (false)
    CUDA_CHECK(cudaMemset(d_G_nbr,        0xFF, num_nodes * sizeof(int)));  // -1 = CUDA_NIL_NODE
    CUDA_CHECK(cudaMemset(d_G_maybe_2nd,  0x00, num_nodes * sizeof(int)));  // false
}

void finalize_trim2()
{
    if (d_G_nbr)        { cudaFree(d_G_nbr);        d_G_nbr = NULL; }
    if (d_G_maybe_2nd)  { cudaFree(d_G_maybe_2nd);  d_G_maybe_2nd = NULL; }
}

// ======================================================================
// trim_2nd_main1() — Phase 1
// OpenMP: classifies each node by its degree pattern.
//   - If both in/out-degree >0 and one is exactly 1, marks as maybe_2nd.
//   - If both are exactly 1 and point to same neighbor, records neighbor.
// ======================================================================
__device__ void trim_2nd_main1_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count, node_t n)
{
    int curr_color = d_Color[n];
    if (curr_color == SCC_FOUND) return;

    // Reset state for this node
    d_G_maybe_2nd[n] = 0;  // false
    d_G_nbr[n] = CUDA_NIL_NODE;

    // ---- Check out-degree ----
    int out_degree = 0;
    node_t last_seen2 = CUDA_NIL_NODE;

    for (edge_t k_idx = d_begin[n]; k_idx < d_begin[n + 1]; k_idx++) {
        node_t k = d_node_idx[k_idx];
        if (k == n) continue;              // self-loop
        if (k == last_seen2) continue;     // duplicate
        if (d_Color[k] != curr_color) continue;
        if (out_degree == 0) {
            last_seen2 = k;
            out_degree = 1;
        } else if (out_degree == 1) {
            out_degree = 2;
            last_seen2 = CUDA_NIL_NODE;
            break;
        }
    }

    // ---- Check in-degree ----
    int in_degree = 0;
    node_t last_seen = CUDA_NIL_NODE;

    for (edge_t k_idx = d_r_begin[n]; k_idx < d_r_begin[n + 1]; k_idx++) {
        node_t k = d_r_node_idx[k_idx];
        if (k == n) continue;
        if (k == last_seen) continue;      // duplicate
        if (d_Color[k] != curr_color) continue;
        if (in_degree == 0) {
            in_degree = 1;
            last_seen = k;
        } else if (in_degree == 1) {
            in_degree = 2;
            last_seen = CUDA_NIL_NODE;
            break;
        }
    }

    // ---- Classify ----
    if ((in_degree == 0) || (out_degree == 0)) {
        // No action needed (trim1 would handle these)
    } else if ((in_degree == 1) && (out_degree == 1)) {
        if (last_seen == last_seen2) {
            d_G_nbr[n] = last_seen;   // single neighbor
            atomicAdd(d_count, 1);
        }
    } else if ((in_degree == 1) || (out_degree == 1)) {
        d_G_maybe_2nd[n] = 1;  // true
    }
}

// ======================================================================
// trim_2nd_main2() — Phase 2
// OpenMP: for nodes where G_nbr is set, checks if it's a 2-node SCC.
// ======================================================================
__device__ void trim_2nd_main2_device(
    int* d_Color, int* d_SCC,
    int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count, node_t n)
{
    int curr_color = d_Color[n];
    if (curr_color == SCC_FOUND) return;
    if (d_G_nbr[n] == CUDA_NIL_NODE) return;

    node_t k = d_G_nbr[n];

    if (d_G_nbr[k] != CUDA_NIL_NODE) {
        // Mutual n->k and k->n confirmed via Phase 1
        d_SCC[n] = (n < k) ? n : k;
        d_Color[n] = SCC_FOUND;
        atomicAdd(d_count, 1);
    } else if (d_G_maybe_2nd[k]) {
        // k has exactly 1 in OR 1 out neighbor -> {n,k} = 2-node SCC
        d_SCC[n] = n;
        d_Color[n] = SCC_FOUND;
        d_SCC[k] = n;
        d_Color[k] = SCC_FOUND;
        atomicAdd(d_count, 2);
    }
}

// ======================================================================
// Kernels: Phase 1 and Phase 2 over compact targets
// ======================================================================
__global__ void trim_2nd_main1_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ix = tid; ix < num_targets; ix += stride) {
        node_t n = d_targets[ix];
        trim_2nd_main1_device(
            d_begin, d_node_idx, d_r_begin, d_r_node_idx,
            d_Color, d_G_nbr, d_G_maybe_2nd,
            d_count, n);
    }
}

__global__ void trim_2nd_main2_kernel(
    int* d_Color, int* d_SCC,
    int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ix = tid; ix < num_targets; ix += stride) {
        node_t n = d_targets[ix];
        trim_2nd_main2_device(
            d_Color, d_SCC,
            d_G_nbr, d_G_maybe_2nd,
            d_count, n);
    }
}

// ======================================================================
// do_global_trim2() — launches Phase 1 then Phase 2
// OpenMP:
//   int do_global_trim2(gm_graph& G) {
//       V = get_compact_trim_targets();
//       Phase 1: trim_2nd_main1 over all V
//       Phase 2: trim_2nd_main2 over all V
//       return count;
//   }
// ======================================================================
int do_global_trim2(GPUState& st, const GPUGraph& g, int* d_count)
{
    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return 0;

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;

    // ---- Phase 1 ----
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    trim_2nd_main1_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx,
        g.d_r_begin, g.d_r_node_idx,
        st.d_Color, d_G_nbr, d_G_maybe_2nd,
        d_count,
        d_trim_targets, num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (count == 0) return 0;

    // ---- Phase 2 ----
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    trim_2nd_main2_kernel<<<grid_size, block_size>>>(
        st.d_Color, st.d_SCC,
        d_G_nbr, d_G_maybe_2nd,
        d_count,
        d_trim_targets, num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

// ======================================================================
// repeat_global_trim2()
// OpenMP:
//   int repeat_global_trim2(gm_graph& G, int exit_count) {
//       do {
//           count_this = do_global_trim2(G);
//           count += count_this;
//           printf("trim2 = %d\n", count_this);
//       } while (count_this > exit_count);
//       return count;
//   }
// ======================================================================
int repeat_global_trim2(GPUState& st, const GPUGraph& g,
    int* d_count, int exit_count)
{
    int total = 0;
    int count_this;

    do {
        count_this = do_global_trim2(st, g, d_count);
        total += count_this;
        printf("trim2 = %d\n", count_this);
    } while (count_this > exit_count);

    return total;
}
