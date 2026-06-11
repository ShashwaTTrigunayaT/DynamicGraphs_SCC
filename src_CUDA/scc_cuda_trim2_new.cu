#include "scc_cuda.h"

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
// Kernel: trim_2nd_new_main over compact targets
// ======================================================================
__global__ void trim_2nd_new_main_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ix = tid; ix < num_targets; ix += stride) {
        node_t n = d_targets[ix];
        trim_2nd_new_main_device(
            d_begin, d_node_idx, d_r_begin, d_r_node_idx,
            d_Color, d_SCC,
            d_count, n);
    }
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
    if (num_targets == 0) return 0;

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;

    trim_2nd_new_main_kernel<<<grid_size, block_size>>>(
        g.d_begin, g.d_node_idx,
        g.d_r_begin, g.d_r_node_idx,
        st.d_Color, st.d_SCC,
        d_count,
        d_trim_targets, num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

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
