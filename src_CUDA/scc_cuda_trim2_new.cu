#include "scc_cuda.h"

// ======================================================================
// Degree-one bitmap — pre-filter for TRIM2
// Nodes with OUT or IN degree != 1 cannot form a 2-node SCC.
// These bitmaps mark which nodes have exactly 1 neighbor in their color.
// ======================================================================
int8_t* d_out_deg_one = NULL;  // [N] 1 if node has exactly 1 out-neighbor in current color
int8_t* d_in_deg_one  = NULL;  // [N] 1 if node has exactly 1 in-neighbor in current color
int     d_deg_bitmap_allocated = 0;

void initialize_trim2_bitmaps(int num_nodes)
{
    if (d_out_deg_one) cudaFree(d_out_deg_one);
    if (d_in_deg_one)  cudaFree(d_in_deg_one);
    CUDA_CHECK(cudaMalloc(&d_out_deg_one, num_nodes * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_in_deg_one,  num_nodes * sizeof(int8_t)));
    d_deg_bitmap_allocated = num_nodes;
}

void finalize_trim2_bitmaps()
{
    if (d_out_deg_one) { cudaFree(d_out_deg_one); d_out_deg_one = NULL; }
    if (d_in_deg_one)  { cudaFree(d_in_deg_one);  d_in_deg_one = NULL; }
    d_deg_bitmap_allocated = 0;
}

// ======================================================================
// Kernel: compute degree-one bitmap for all target nodes
// For each target, scans outgoing/incoming edges to count neighbors
// in the same color. Stops at 2 (early exit for high-degree nodes).
// Writes to d_out_deg_one[n] and d_in_deg_one[n] (0 or 1).
// ======================================================================
__global__ void compute_deg_one_bitmap_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    const int* d_Color,
    const int* d_targets, int num_targets,
    int8_t* d_out_deg_one, int8_t* d_in_deg_one)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_targets) return;

    node_t n = d_targets[idx];
    int curr_color = d_Color[n];
    if (curr_color == SCC_FOUND) {
        d_out_deg_one[n] = 0;
        d_in_deg_one[n] = 0;
        return;
    }

    // Scan outgoing edges: count same-color neighbors (stop at 2)
    // Must match check_out_degree_is_one_device: skip self-loops and repeated edges
    int out_cnt = 0;
    node_t first_out_nbr = CUDA_NIL_NODE;
    for (edge_t k_idx = d_begin[n]; k_idx < d_begin[n + 1]; k_idx++) {
        node_t k = d_node_idx[k_idx];
        if (k == n) continue;                 // self edge
        if (k == first_out_nbr) continue;     // repeated edge
        if (d_Color[k] == curr_color) {
            if (++out_cnt == 2) break;
            first_out_nbr = k;
        }
    }
    d_out_deg_one[n] = (out_cnt == 1) ? 1 : 0;

    // Scan incoming edges: count same-color neighbors (stop at 2)
    int in_cnt = 0;
    node_t first_in_nbr = CUDA_NIL_NODE;
    for (edge_t k_idx = d_r_begin[n]; k_idx < d_r_begin[n + 1]; k_idx++) {
        node_t k = d_r_node_idx[k_idx];
        if (k == n) continue;                 // self edge
        if (k == first_in_nbr) continue;      // repeated edge
        if (d_Color[k] == curr_color) {
            if (++in_cnt == 2) break;
            first_in_nbr = k;
        }
    }
    d_in_deg_one[n] = (in_cnt == 1) ? 1 : 0;
}

// ======================================================================
// check_out_degree_is_one()
// OpenMP:
//   bool check_out_degree_is_one(gm_graph& G, int curr_color,
//                                  node_t n, node_t& the_nbr)
//   {
//       the_nbr = NIL_NODE;
//       int cnt = 0;
//       for (edge_t k_idx = G.begin[n]; k_idx < G.begin[n+1]; k_idx++) {
//           node_t k = G.node_idx[k_idx];
//           if (k==n) continue;            // self edge
//           if (k==the_nbr) continue;      // repeated edge
//           if (G_Color[k] != curr_color) continue;
//           cnt++;
//           the_nbr = k;
//           if (cnt == 2) return false;
//       }
//       return (cnt == 1);
//   }
// ======================================================================
__device__ bool check_out_degree_is_one_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color, node_t n, int curr_color, node_t* the_nbr)
{
    *the_nbr = CUDA_NIL_NODE;
    int cnt = 0;

    for (edge_t k_idx = d_begin[n]; k_idx < d_begin[n + 1]; k_idx++) {
        node_t k = d_node_idx[k_idx];
        if (k == n) continue;               // self edge
        if (k == *the_nbr) continue;         // repeated edge
        if (d_Color[k] != curr_color) continue;
        cnt++;
        *the_nbr = k;
        if (cnt == 2) return false;
    }
    return (cnt == 1);
}

// ======================================================================
// check_in_degree_is_one()
// OpenMP:
//   bool check_in_degree_is_one(gm_graph& G, int curr_color,
//                                 node_t n, node_t& the_nbr)
//   {
//       the_nbr = NIL_NODE;
//       int cnt = 0;
//       for (edge_t k_idx = G.r_begin[n]; k_idx < G.r_begin[n+1]; k_idx++) {
//           node_t k = G.r_node_idx[k_idx];
//           if (k==n) continue;            // self edge
//           if (k==the_nbr) continue;      // repeated edge
//           if (G_Color[k] != curr_color) continue;
//           cnt++;
//           the_nbr = k;
//           if (cnt == 2) return false;
//       }
//       return (cnt == 1);
//   }
// ======================================================================
__device__ bool check_in_degree_is_one_device(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, node_t n, int curr_color, node_t* the_nbr)
{
    *the_nbr = CUDA_NIL_NODE;
    int cnt = 0;

    for (edge_t k_idx = d_r_begin[n]; k_idx < d_r_begin[n + 1]; k_idx++) {
        node_t k = d_r_node_idx[k_idx];
        if (k == n) continue;               // self edge
        if (k == *the_nbr) continue;         // repeated edge
        if (d_Color[k] != curr_color) continue;
        cnt++;
        *the_nbr = k;
        if (cnt == 2) return false;
    }
    return (cnt == 1);
}

// ======================================================================
// trim_2nd_new_main() — single-pass 2-node SCC detection
//
// OpenMP:
//   void trim_2nd_new_main(gm_graph& G, int curr_color, int& count,
//                           node_t n)
//   {
//       node_t k;
//       if (G_Color[n] != curr_color) return;
//
//       if (check_out_degree_is_one(G, curr_color, n, k)) {
//           if (n < k) {
//               node_t kk;
//               if (check_out_degree_is_one(G, curr_color, k, kk)) {
//                   if (kk == n) {
//                       count += 2;
//                       G_Color[n] = G_Color[k] = -2;
//                       G_SCC[n] = G_SCC[k] = n;
//                       return;
//                   }
//               }
//           }
//       }
//
//       if (check_in_degree_is_one(G, curr_color, n, k)) {
//           if (n < k) {
//               node_t kk;
//               if (check_in_degree_is_one(G, curr_color, k, kk)) {
//                   if (kk == n) {
//                       count += 2;
//                       G_Color[n] = G_Color[k] = -2;
//                       G_SCC[n] = G_SCC[k] = n;
//                       return;
//                   }
//               }
//           }
//       }
//   }
// ======================================================================
__device__ void trim_2nd_new_main_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count, node_t n)
{
    int curr_color = d_Color[n];
    if (curr_color == SCC_FOUND) return;

    node_t k;

    // Check out-degree: n has exactly 1 out-neighbor = k
    if (check_out_degree_is_one_device(d_begin, d_node_idx, d_Color, n, curr_color, &k)) {
        if (n < k) {  // avoid double counting
            node_t kk;
            // Check if k also has exactly 1 out-neighbor = kk
            if (check_out_degree_is_one_device(d_begin, d_node_idx, d_Color, k, curr_color, &kk)) {
                if (kk == n) {  // mutual: n->k and k->n
                    atomicAdd(d_count, 2);
                    d_Color[n] = d_Color[k] = SCC_FOUND;
                    d_SCC[n] = d_SCC[k] = n;
                    return;
                }
            }
        }
    }

    // Check in-degree: n has exactly 1 in-neighbor = k
    if (check_in_degree_is_one_device(d_r_begin, d_r_node_idx, d_Color, n, curr_color, &k)) {
        if (n < k) {  // avoid double counting
            node_t kk;
            // Check if k also has exactly 1 in-neighbor = kk
            if (check_in_degree_is_one_device(d_r_begin, d_r_node_idx, d_Color, k, curr_color, &kk)) {
                if (kk == n) {  // mutual: k->n and n->k
                    atomicAdd(d_count, 2);
                    d_Color[n] = d_Color[k] = SCC_FOUND;
                    d_SCC[n] = d_SCC[k] = n;
                    return;
                }
            }
        }
    }
}

// ======================================================================
// Kernel: trim_2nd_new_main over compact targets (block-contiguous)
// ======================================================================
__global__ void trim_2nd_new_main_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_targets, int num_targets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_targets) return;

    node_t n = d_targets[idx];
    trim_2nd_new_main_device(
        d_begin, d_node_idx, d_r_begin, d_r_node_idx,
        d_Color, d_SCC,
        d_count, n);
}

// ======================================================================
// Kernel: filtered trim2 using degree-one bitmap
// Only processes nodes where out_deg_one[n] || in_deg_one[n]
// Uses the same trim logic but skips the majority of nodes.
// ======================================================================
__global__ void trim_2nd_new_filtered_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_targets, int num_targets,
    const int8_t* d_out_deg_one, const int8_t* d_in_deg_one)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_targets) return;

    node_t n = d_targets[idx];
    // Skip nodes that can't possibly be part of a 2-node SCC
    if (!d_out_deg_one[n] && !d_in_deg_one[n]) return;

    trim_2nd_new_main_device(
        d_begin, d_node_idx, d_r_begin, d_r_node_idx,
        d_Color, d_SCC,
        d_count, n);
}

// ======================================================================
// do_global_trim2_new()
// OpenMP:
//   int do_global_trim2_new(gm_graph& G) {
//       V = get_compact_trim_targets();
//       #pragma omp parallel for
//       for each node in V: trim_2nd_new_main(G, ...)
//       return count;
//   }
// ======================================================================
int do_global_trim2_new(GPUState& st, const GPUGraph& g, int* d_count)
{
    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return 0;    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;

    // Pass 1: compute degree-one bitmap (fast, coalesced edge scan)
    compute_deg_one_bitmap_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx,
        g.d_r_begin, g.d_r_node_idx,
        st.d_Color,
        d_trim_targets, num_targets,
        d_out_deg_one, d_in_deg_one);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 2: run TRIM2 only on nodes where degree=1 (1-5% of targets)
    trim_2nd_new_filtered_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx,
        g.d_r_begin, g.d_r_node_idx,
        st.d_Color, st.d_SCC,
        d_count,
        d_trim_targets, num_targets,
        d_out_deg_one, d_in_deg_one);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    return count;}

// ======================================================================
// repeat_global_trim2_new()
// OpenMP:
//   int repeat_global_trim2_new(gm_graph& G, int exit_count) {
//       do {
//           count_this = do_global_trim2(G);  // NOTE: calls do_global_trim2!
//           count += count_this;
//           printf("trim2 = %d\n", count_this);
//       } while (count_this > exit_count);
//       return count;
//   }
// ======================================================================
int repeat_global_trim2_new(GPUState& st, const GPUGraph& g,
    int* d_count, int exit_count)
{
    int total = 0;
    int count_this;

    do {
        count_this = do_global_trim2(st, g, d_count); // MIRRORS OpenMP: calls do_global_trim2!
        total += count_this;
        printf("trim2 = %d\n", count_this);
    } while (count_this > exit_count);

    return total;
}
