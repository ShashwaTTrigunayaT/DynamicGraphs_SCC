#include "scc_cuda.h"

// ======================================================================
// fw_trim_dfs — Forward DFS (sequential in OpenMP)
//
// OpenMP:
//   class fw_trim_dfs : public gm_dfs_template<true, false, true, false>
//   {
//   public:
//       fw_trim_dfs(gm_graph& _G, int32_t _base_color, int32_t _fw_color,
//                   NODE_SET* _base_set)
//       : gm_dfs_template<true,false,true,false>(_G),
//         G(_G), fw_color(_fw_color), base_color(_base_color), base_set(_base_set)
//       { count = 0; fw_set = new NODE_SET(); }
//
//       int get_fw_count() {return count;}
//       NODE_SET* get_fw_set() {
//           if ((fw_set != NULL) && (fw_set->size() != 0)) return fw_set;
//           if (fw_set != NULL) { delete fw_set; fw_set = NULL; }
//           return NULL;
//       }
//
//   protected:
//       void visit_pre(node_t k) {
//           G_Color[k] = fw_color;
//           count++;
//           if (base_set != NULL) base_set->erase(k);
//           if (fw_set != NULL) {
//               if (fw_set->size() >= G_num_nodes * 0.01) {
//                   delete fw_set;
//                   fw_set = NULL;
//               } else {
//                   fw_set->insert(k);
//               }
//           }
//       }
//       bool check_navigator(node_t k9, edge_t) {
//           return (G_Color[k9] == base_color);
//       }
//   };
//
// CUDA: Reuses fw_bfs_level_kernel from scc_cuda_fb_global.cu
//       (GPU uses level-by-level BFS instead of DFS, which is inherently
//        sequential and not GPU-friendly. The computed partitions are
//        identical — only traversal order differs.)
// ======================================================================

// ======================================================================
// bw_trim_dfs — Backward DFS (sequential in OpenMP)
//
// OpenMP:
//   class bw_trim_dfs : public gm_dfs_template<true, false, true, true>
//   {
//   public:
//       bw_trim_dfs(gm_graph& _G, int32_t _base_color, int32_t _fw_color,
//                   int32_t _bw_color, NODE_SET* _base_set, NODE_SET* _fw_set,
//                   node_t _pivot)
//       : gm_dfs_template<true,false,true,true>(_G),
//         G(_G), fw_color(_fw_color), base_color(_base_color),
//         bw_color(_bw_color), base_set(_base_set), fw_set(_fw_set), pivot(_pivot)
//       { count = 0; scc_count = 0; bw_set = new NODE_SET(); }
//
//       int get_bw_count() {return count;}
//       int get_scc_count() {return scc_count;}
//       NODE_SET* get_bw_set() { ... }
//       NODE_SET* get_fw_set() { ... }
//
//   protected:
//       void visit_pre(node_t k) {
//           if (G_Color[k] == fw_color) {     // intersection
//               G_SCC[k] = pivot; G_Color[k] = -2; scc_count++;
//               if (fw_set != NULL) fw_set->erase(k);
//           } else {                           // bw-set
//               G_Color[k] = bw_color; count++;
//               if (base_set != NULL) base_set->erase(k);
//               if (bw_set != NULL) {
//                   if (bw_set->size() >= G_num_nodes * 0.01) {
//                       delete bw_set; bw_set = NULL;
//                   } else { bw_set->insert(k); }
//               }
//           }
//       }
//       bool check_navigator(node_t k10, edge_t) {
//           return (G_Color[k10] == fw_color) || (G_Color[k10] == base_color);
//       }
//   };
//
// CUDA: Reuses bw_bfs_level_kernel from scc_cuda_fb_global.cu
// ======================================================================

// ======================================================================
// do_fw_bw_dfs()
//
// OpenMP:
//   int do_fw_bw_dfs(gm_graph& G, my_work* work,
//                     std::vector<my_work*>& new_works)
//   {
//       if (work->count == 0) return 0;
//
//       NODE_SET* base_set = work->color_set;
//       int base_color = work->color;
//       int base_count = work->count;
//
//       // choose pivot
//       node_t pivot;
//       if (base_set == NULL)
//           pivot = choose_pivot_from_color(G, base_color);
//       else {
//           if (base_set->size() == 0) {
//               delete base_set; base_set = NULL;
//               pivot = choose_pivot_from_color(G, base_color);
//           } else {
//               pivot = *(base_set->begin());
//           }
//       }
//       assert(pivot != NIL_NODE && G_Color[pivot] == base_color);
//
//       if (work->count == 1) {
//           G_Color[pivot] = -2; G_SCC[pivot] = pivot;
//           delete base_set;
//           return 1;
//       }
//
//       int fw_color = get_new_color();
//       int bw_color = get_new_color();
//
//       // FW DFS
//       fw_trim_dfs FW_BFS(G, base_color, fw_color, base_set);
//       FW_BFS.prepare(pivot);
//       FW_BFS.do_dfs();
//       int fw_count = FW_BFS.get_fw_count();
//       NODE_SET* fw_set = FW_BFS.get_fw_set();
//
//       // BW DFS
//       bw_trim_dfs BW_BFS(G, base_color, fw_color, bw_color,
//                          base_set, fw_set, pivot);
//       BW_BFS.prepare(pivot);
//       BW_BFS.do_dfs();
//
//       fw_set = BW_BFS.get_fw_set();
//       NODE_SET* bw_set = BW_BFS.get_bw_set();
//       if (base_set != NULL && base_set->size() == 0) {
//           delete base_set; base_set = NULL;
//       }
//
//       int bw_count = BW_BFS.get_bw_count();
//       int scc_count = BW_BFS.get_scc_count();
//       fw_count = fw_count - scc_count;
//       base_count = base_count - fw_count - bw_count - scc_count;
//
//       // create new work items
//       int depth = work->depth + 1;
//       if (fw_count > 0) {
//           work = new my_work(); work->color = fw_color;
//           work->count = fw_count; work->color_set = fw_set;
//           work->depth = depth; new_works.push_back(work);
//           if (fw_set != NULL) assert(fw_set->size() == fw_count);
//       }
//       if (bw_count > 0) { ... bw_set ... }
//       if (base_count > 0) { ... base_set ... }
//
//       return scc_count;
//   }
// ======================================================================
int do_fw_bw_dfs(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, std::vector<CUDAMyWork*>& new_works)
{
    // OpenMP: if (work->count == 0) return 0;
    if (w->count == 0) return 0;

    // OpenMP: NODE_SET* base_set = work->color_set;
    //         int base_color = work->color;
    //         int base_count = work->count;
    int* d_base_set = w->d_set_nodes;  // CUDA: device compact set
    int base_color = w->color;
    int base_count = w->count;

    int N = g.num_nodes;
    int block_size = 256;
    int grid_size;

    // ---------------------------------------------------------------
    // Choose pivot — EXACT mirror of OpenMP
    // OpenMP:
    //   node_t pivot;
    //   if (base_set == NULL)
    //       pivot = choose_pivot_from_color(G, base_color);
    //   else {
    //       if (base_set->size() == 0) {
    //           delete base_set; base_set = NULL;
    //           pivot = choose_pivot_from_color(G, base_color);
    //       } else {
    //           pivot = *(base_set->begin());
    //       }
    //   }
    //   assert(pivot != NIL_NODE && G_Color[pivot] == base_color);
    //
    // OpenMP choose_pivot_from_color():
    //   std::vector<node_t>& V = get_compact_trim_targets();
    //   if (V.size() == 0) {
    //       for(node_t i=0;i<G.num_nodes(); i++)
    //           if (G_Color[i] == color) return i;
    //   } else {
    //       for(node_t j=0;j<V.size(); j++)
    //           if (G_Color[i] == color) return i;
    //   }
    // ---------------------------------------------------------------
    int h_pivot = -1;
    int PIVOT_NONE = 0x7FFFFFFF;
    int* d_pivot = NULL;
    CUDA_CHECK(cudaMalloc(&d_pivot, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pivot, &PIVOT_NONE, sizeof(int), cudaMemcpyHostToDevice));

    // OpenMP: if (base_set == NULL)
    //            pivot = choose_pivot_from_color(G, base_color);
    //         else
    //            ... (handles empty set separately)
    if (d_base_set == NULL) {
        // OpenMP: pivot = choose_pivot_from_color(G, base_color);
        int num_targets = d_trim_targets_count;
        if (num_targets > 0) {
            // OpenMP: scan compact trim targets
            grid_size = (num_targets + block_size - 1) / block_size;
            find_pivot_by_color_kernel<<<grid_size, block_size>>>(
                st.d_Color, d_trim_targets, num_targets,
                base_color, d_pivot);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // OpenMP: if (V.size() == 0) for(node_t i=0;i<G.num_nodes(); i++) ...
        if (num_targets == 0) {
            grid_size = (N + block_size - 1) / block_size;
            find_pivot_all_nodes_kernel<<<grid_size, block_size>>>(
                st.d_Color, N, base_color, d_pivot);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } else {
        // OpenMP: else { if (base_set->size() == 0) ... else ... }
        if (w->count == 0) {
            // (dead code — w->count > 0 guaranteed by early return above;
            //  mirrors CPU's base_set->size() == 0 branch identically)
            if (w->owns_set && d_base_set != NULL) {
                CUDA_CHECK(cudaFree(d_base_set));
            }
            d_base_set = NULL;
            int num_targets = d_trim_targets_count;
            if (num_targets > 0) {
                grid_size = (num_targets + block_size - 1) / block_size;
                find_pivot_by_color_kernel<<<grid_size, block_size>>>(
                    st.d_Color, d_trim_targets, num_targets,
                    base_color, d_pivot);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            if (num_targets == 0) {
                grid_size = (N + block_size - 1) / block_size;
                find_pivot_all_nodes_kernel<<<grid_size, block_size>>>(
                    st.d_Color, N, base_color, d_pivot);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        } else {
            // OpenMP: pivot = *(base_set->begin());
            grid_size = (w->count + block_size - 1) / block_size;
            find_pivot_in_set_kernel<<<grid_size, block_size>>>(
                st.d_Color, d_base_set, w->count, base_color, d_pivot);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(&h_pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pivot));

    // OpenMP: assert(pivot != gm_graph::NIL_NODE);
    //         assert(G_Color[pivot] == base_color);
    if (h_pivot == PIVOT_NONE || h_pivot == -1) return 0;

    // OpenMP: if (work->count == 1)
    if (w->count == 1) {
        // OpenMP: G_Color[pivot] = -2; G_SCC[pivot] = pivot;
        //         delete base_set; return 1;
        CUDA_CHECK(cudaMemcpy(&st.d_Color[h_pivot], &SCC_FOUND, sizeof(int),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&st.d_SCC[h_pivot], &h_pivot, sizeof(int),
                               cudaMemcpyHostToDevice));
        if (d_base_set != NULL && w->owns_set) {
            CUDA_CHECK(cudaFree(d_base_set));
            d_base_set = NULL;
        }
        return 1;
    }

    // ---------------------------------------------------------------
    // Assign colors — EXACT mirror of OpenMP
    // OpenMP: int fw_color = get_new_color(); int bw_color = get_new_color();
    // ---------------------------------------------------------------
    int fw_color = cuda_get_new_color();
    int bw_color = cuda_get_new_color();

    // ---------------------------------------------------------------
    // Forward traversal from pivot
    // OpenMP:
    //   fw_trim_dfs FW_BFS(G, base_color, fw_color, base_set);
    //   FW_BFS.prepare(pivot);
    //   FW_BFS.do_dfs();
    //   int fw_count = FW_BFS.get_fw_count();
    //   NODE_SET* fw_set = FW_BFS.get_fw_set();
    //
    // CUDA: Uses BFS kernels (same navigator logic, same partition result)
    // ---------------------------------------------------------------
    int queue_size = 1;
    CUDA_CHECK(cudaMemcpy(d_bfs_queue, &h_pivot, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&st.d_Color[h_pivot], &fw_color, sizeof(int),
                           cudaMemcpyHostToDevice));

    int total_fw = 1;  // pivot counted

    while (queue_size > 0) {
        CUDA_CHECK(cudaMemset(d_bfs_next_count, 0, sizeof(int)));

        int grid = (queue_size + block_size - 1) / block_size;
        grid = min(grid, 1024);

        // OpenMP: iterate_neighbor_small(t) + visit_pre(t)
        // CUDA: fw_bfs_level_kernel — same navigator check:
        //       check_navigator: return (G_Color[k9] == base_color)
        fw_bfs_level_kernel<<<grid, block_size>>>(
            g.d_begin, g.d_node_idx,
            st.d_Color,
            d_bfs_queue, queue_size,
            d_bfs_next_queue, d_bfs_next_count,
            fw_color, base_color);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* tmp = d_bfs_queue;
        d_bfs_queue = d_bfs_next_queue;
        d_bfs_next_queue = tmp;

        CUDA_CHECK(cudaMemcpy(&queue_size, d_bfs_next_count, sizeof(int),
                               cudaMemcpyDeviceToHost));
        total_fw += queue_size;
    }

    // OpenMP: int fw_count = FW_BFS.get_fw_count();
    int fw_count = total_fw;

    // ---------------------------------------------------------------
    // Backward traversal from pivot
    // OpenMP:
    //   bw_trim_dfs BW_BFS(G, base_color, fw_color, bw_color,
    //                       base_set, fw_set, pivot);
    //   BW_BFS.prepare(pivot);
    //   BW_BFS.do_dfs();
    //   int bw_count = BW_BFS.get_bw_count();
    //   int scc_count = BW_BFS.get_scc_count();
    //   fw_set = BW_BFS.get_fw_set();
    //   NODE_SET* bw_set = BW_BFS.get_bw_set();
    // ---------------------------------------------------------------
    // Mark pivot itself as SCC
    CUDA_CHECK(cudaMemcpy(&st.d_Color[h_pivot], &SCC_FOUND, sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&st.d_SCC[h_pivot], &h_pivot, sizeof(int),
                           cudaMemcpyHostToDevice));
    int scc_count = 1;

    CUDA_CHECK(cudaMemcpy(d_bfs_queue, &h_pivot, sizeof(int), cudaMemcpyHostToDevice));
    queue_size = 1;
    CUDA_CHECK(cudaMemset(d_bfs_scc_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bfs_bw_count, 0, sizeof(int)));

    while (queue_size > 0) {
        CUDA_CHECK(cudaMemset(d_bfs_next_count, 0, sizeof(int)));

        int grid = (queue_size + block_size - 1) / block_size;
        grid = min(grid, 1024);

        // OpenMP: iterate_neighbor_small(t) + visit_pre(t)
        // CUDA: bw_bfs_level_kernel — same navigator check:
        //       check_navigator: return (color == fw_color) || (color == base_color)
        bw_bfs_level_kernel<<<grid, block_size>>>(
            g.d_r_begin, g.d_r_node_idx,
            st.d_Color, st.d_SCC,
            d_bfs_queue, queue_size,
            d_bfs_next_queue, d_bfs_next_count,
            fw_color, bw_color, base_color, h_pivot,
            d_bfs_scc_count, d_bfs_bw_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* tmp = d_bfs_queue;
        d_bfs_queue = d_bfs_next_queue;
        d_bfs_next_queue = tmp;

        CUDA_CHECK(cudaMemcpy(&queue_size, d_bfs_next_count, sizeof(int),
                               cudaMemcpyDeviceToHost));
    }

    int extra_scc;
    CUDA_CHECK(cudaMemcpy(&extra_scc, d_bfs_scc_count, sizeof(int),
                           cudaMemcpyDeviceToHost));
    scc_count += extra_scc;

    int bw_count;
    CUDA_CHECK(cudaMemcpy(&bw_count, d_bfs_bw_count, sizeof(int),
                           cudaMemcpyDeviceToHost));

    // ---------------------------------------------------------------
    // Compute partition sizes (mirrors OpenMP exactly)
    // OpenMP:
    //   int bw_count = BW_BFS.get_bw_count();
    //   int scc_count = BW_BFS.get_scc_count();
    //   fw_count = fw_count - scc_count;
    //   base_count = base_count - fw_count - bw_count - scc_count;
    // ---------------------------------------------------------------
    fw_count = fw_count - scc_count;
    base_count = base_count - fw_count - bw_count - scc_count;

    // ---------------------------------------------------------------
    // Build compact sets for new partitions (mirrors OpenMP's NODE_SET*)
    //
    // OpenMP: fw_set from FW_BFS.get_fw_set()
    //         bw_set from BW_BFS.get_bw_set()
    //         base_set from BW_BFS (remaining after erase)
    //
    // The CPU discards sets larger than 1% of total nodes:
    //   if (set->size() >= G_num_nodes * 0.01) { delete set; set = NULL; }
    //
    // CUDA: build device compact sets; skip if > 1% of N (set to NULL)
    // ---------------------------------------------------------------
    int* d_fw_set        = NULL;
    int* d_bw_set        = NULL;
    int* d_base_set_new  = NULL;  // 'new' to avoid naming conflict with input d_base_set
    int* d_scatter_fw_pos   = NULL;
    int* d_scatter_bw_pos   = NULL;
    int* d_scatter_base_pos = NULL;

    // Determine which source set to scatter from.
    // OpenMP: if (base_set exists) base_set->erase(k) during traversal
    //         else choose_pivot_from_color scans compact_trim_targets (or all nodes)
    //
    // CUDA: mirror the same fallback:
    //   d_base_set != NULL → scatter from it
    //   d_trim_targets_count > 0 → scatter from d_trim_targets
    //   else → scatter from ALL N nodes
    int* d_src_set = NULL;
    int src_size = 0;
    if (d_base_set != NULL) {
        d_src_set = d_base_set;
        src_size = w->count;
    } else if (d_trim_targets_count > 0) {
        // OpenMP: std::vector<node_t>& V = get_compact_trim_targets(); V.size() > 0
        d_src_set = d_trim_targets;
        src_size = d_trim_targets_count;
    } else {
        // OpenMP: if (V.size() == 0) for(node_t i=0;i<G.num_nodes(); i++) ...
        d_src_set = NULL;  // signal: use all-node kernels
        src_size = N;
    }

    int FW_SET_THRESHOLD   = (int)(N * 0.01);
    int BW_SET_THRESHOLD   = (int)(N * 0.01);
    int BASE_SET_THRESHOLD = (int)(N * 0.01);

    // Allocate buffers for sets that are below the size threshold
    if (fw_count > 0 && fw_count <= FW_SET_THRESHOLD) {
        CUDA_CHECK(cudaMalloc(&d_fw_set, fw_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scatter_fw_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scatter_fw_pos, 0, sizeof(int)));
    }
    if (bw_count > 0 && bw_count <= BW_SET_THRESHOLD) {
        CUDA_CHECK(cudaMalloc(&d_bw_set, bw_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scatter_bw_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scatter_bw_pos, 0, sizeof(int)));
    }
    // OpenMP: base_set is the ORIGINAL work->color_set pointer, NOT freshly allocated.
    // If the input d_base_set was NULL, the output base_set must also be NULL
    // (lazy generation in start_workers_fw_bw_dfs will build it when needed).
    // This mirrors the CPU behavior exactly.
    if (base_count > 0 && base_count <= BASE_SET_THRESHOLD && d_base_set != NULL) {
        CUDA_CHECK(cudaMalloc(&d_base_set_new, base_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scatter_base_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scatter_base_pos, 0, sizeof(int)));
    }

    // Scatter nodes by color into the new compact sets.
    // Use per-color scatter kernels so NULL output buffers (for sets that
    // exceed the threshold) are never dereferenced by the kernel.
    // When d_src_set is NULL, use all-node kernels (mirrors CPU's
    // scan of for(node_t i=0;i<G.num_nodes(); i++) when trim_targets empty).
    if (src_size > 0 && (fw_count > 0 || bw_count > 0 || base_count > 0)) {
        grid_size = (src_size + block_size - 1) / block_size;
        grid_size = min(grid_size, 1024);

        if (d_fw_set != NULL) {
            if (d_src_set != NULL) {
                scatter_single_color_kernel<<<grid_size, block_size>>>(
                    st.d_Color,
                    d_src_set, src_size,
                    fw_color,
                    d_fw_set, d_scatter_fw_pos);
            } else {
                scatter_single_color_all_nodes_kernel<<<grid_size, block_size>>>(
                    st.d_Color, N,
                    fw_color,
                    d_fw_set, d_scatter_fw_pos);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (d_bw_set != NULL) {
            if (d_src_set != NULL) {
                scatter_single_color_kernel<<<grid_size, block_size>>>(
                    st.d_Color,
                    d_src_set, src_size,
                    bw_color,
                    d_bw_set, d_scatter_bw_pos);
            } else {
                scatter_single_color_all_nodes_kernel<<<grid_size, block_size>>>(
                    st.d_Color, N,
                    bw_color,
                    d_bw_set, d_scatter_bw_pos);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (d_base_set_new != NULL) {
            if (d_src_set != NULL) {
                scatter_single_color_kernel<<<grid_size, block_size>>>(
                    st.d_Color,
                    d_src_set, src_size,
                    base_color,
                    d_base_set_new, d_scatter_base_pos);
            } else {
                scatter_single_color_all_nodes_kernel<<<grid_size, block_size>>>(
                    st.d_Color, N,
                    base_color,
                    d_base_set_new, d_scatter_base_pos);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    if (d_scatter_fw_pos)   CUDA_CHECK(cudaFree(d_scatter_fw_pos));
    if (d_scatter_bw_pos)   CUDA_CHECK(cudaFree(d_scatter_bw_pos));
    if (d_scatter_base_pos) CUDA_CHECK(cudaFree(d_scatter_base_pos));

    // ---------------------------------------------------------------
    // Create new work items (mirrors OpenMP exactly)
    //
    // OpenMP:
    //   int depth = work->depth + 1;
    //   if (fw_count > 0) {
    //       work = new my_work();
    //       work->color = fw_color;
    //       work->count = fw_count;
    //       work->color_set = fw_set;    // may be NULL if too large
    //       work->depth = depth;
    //       new_works.push_back(work);
    //       if (fw_set != NULL) assert(fw_set->size() == fw_count);
    //   }
    //   ... (same for bw, base)
    // ---------------------------------------------------------------
    int depth = w->depth + 1;

    if (fw_count > 0) {
        CUDAMyWork* work = new CUDAMyWork();
        work->color       = fw_color;
        work->count       = fw_count;
        work->d_set_nodes = d_fw_set;    // NULL if above threshold (matches OpenMP)
        work->set_capacity = (d_fw_set != NULL) ? fw_count : 0;
        work->depth       = depth;
        work->owns_set    = (d_fw_set != NULL);
        new_works.push_back(work);
    }
    if (bw_count > 0) {
        CUDAMyWork* work = new CUDAMyWork();
        work->color       = bw_color;
        work->count       = bw_count;
        work->d_set_nodes = d_bw_set;    // NULL if above threshold
        work->set_capacity = (d_bw_set != NULL) ? bw_count : 0;
        work->depth       = depth;
        work->owns_set    = (d_bw_set != NULL);
        new_works.push_back(work);
    }
    if (base_count > 0) {
        CUDAMyWork* work = new CUDAMyWork();
        work->color       = base_color;
        work->count       = base_count;
        work->d_set_nodes = d_base_set_new;  // NULL if above threshold
        work->set_capacity = (d_base_set_new != NULL) ? base_count : 0;
        work->depth       = depth;
        work->owns_set    = (d_base_set_new != NULL);
        new_works.push_back(work);
    }

    // OpenMP: if (fw_set != NULL) assert(fw_set->size() == fw_count);
    // CUDA: compact arrays have count = allocated (assertion is implicit via allocation)

    // OpenMP: delete old work item (done by caller in start_workers)
    // CUDA: free old d_set_nodes if we owned it and won't reuse it
    //       (w->d_set_nodes == d_base_set that we already used as source)
    //       We DON'T free d_base_set here — we already scattered from it.
    //       If the old w owned d_base_set, the caller should free w.

    return scc_count;
}

// ======================================================================
// start_workers_fw_bw_dfs()
//
// OpenMP:
//   void start_workers_fw_bw_dfs(gm_graph& G, int N)
//   {
//       #pragma omp parallel
//       {
//           int tid = gm_rt_thread_id();
//           std::vector<my_work*> new_works;
//           std::vector<my_work*> my_works;
//           my_works.reserve(4096);
//
//           while (true) {
//               if (my_works.size() == 0) {
//                   work_q_fetch_N(tid, std::max(N/2,1), my_works);
//               }
//               if (my_works.size() == 0) break;
//
//               while (my_works.size() > 0) {
//                   my_work* w = my_works.back();
//                   my_works.pop_back();
//
//                   // Lazy compact set generation (note: threshold is * 1, not * 0.01)
//                   if ((w->count < G.num_nodes() * 1) && (w->color_set == NULL)) {
//                       w->color_set = generate_compact_set(G, w->color);
//                   }
//
//                   do_fw_bw_dfs(G, w, new_works);
//                   delete w;
//
//                   while ((my_works.size() < N) && (new_works.size() > 0)) {
//                       my_work* w = new_works.back();
//                       new_works.pop_back();
//                       my_works.push_back(w);
//                   }
//
//                   if (new_works.size() > 0) {
//                       work_q_put_all(tid, new_works);
//                       new_works.clear();
//                   }
//               }
//           }
//       }
//   }
//
// CUDA: Sequential host loop (no OpenMP threads). Each iteration:
//   1. Fetch work items from the queue (up to N items)
//   2. For each work item:
//      a. Lazily build compact set if needed (count < N * 1 — always, matching CPU)
//      b. Call do_fw_bw_dfs (which launches GPU kernels)
//      c. Delete old work item
//      d. Keep some new items locally, push rest to queue
// ======================================================================
void start_workers_fw_bw_dfs(GPUState& st, const GPUGraph& g, int N)
{
    // CUDA: Single-threaded host loop (mirrors OpenMP parallel body)
    //       Each "iteration" simulates one thread's work loop.
    std::vector<CUDAMyWork*> new_works;
    std::vector<CUDAMyWork*> my_works;
    my_works.reserve(4096);  // OpenMP: my_works.reserve(4096);

    // OpenMP: while (true) { ... }
    while (true) {
        // OpenMP: if (my_works.size() == 0) work_q_fetch_N(tid, std::max(N/2,1), my_works);
        if (my_works.size() == 0) {
            work_q_fetch_N(0, std::max(N/2,1), my_works);
        }

        // OpenMP: if (my_works.size() == 0) break;
        if (my_works.size() == 0) break;

        // OpenMP: while (my_works.size() > 0) { ... }
        while (my_works.size() > 0) {
            CUDAMyWork* w = my_works.back();
            my_works.pop_back();

            // OpenMP: lazy compact set generation
            //   if ((w->count < G.num_nodes() * 1) && (w->color_set == NULL))
            //       w->color_set = generate_compact_set(G, w->color);
            //
            // Note: CPU uses threshold * 1 (i.e., count < N always true),
            // so this branch ALWAYS fires on CPU. CUDA mirrors this behavior.
            if ((w->count < g.num_nodes * 1) && (w->d_set_nodes == NULL)) {
                // OpenMP: w->color_set = generate_compact_set(G, w->color);
                //         generate_compact_set:
                //           if (get_compact_trim_targets().size() == 0)
                //               for(node_t i=0;i<G.num_nodes(); i++) if (G_Color[i]==color) ...
                //           else
                //               for each node in compact_trim_targets ...
                int* d_new_set = NULL;
                int* d_pos = NULL;
                CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));
                CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_new_set, w->count * sizeof(int)));

                int bs = 256;
                if (d_trim_targets_count > 0) {
                    int gs = (d_trim_targets_count + bs - 1) / bs;
                    scatter_single_color_kernel<<<gs, bs>>>(
                        st.d_Color,
                        d_trim_targets, d_trim_targets_count,
                        w->color,
                        d_new_set, d_pos);
                    CUDA_CHECK(cudaDeviceSynchronize());
                } else {
                    // OpenMP: for(node_t i=0;i<G.num_nodes(); i++) ...
                    int gs = (g.num_nodes + bs - 1) / bs;
                    scatter_single_color_all_nodes_kernel<<<gs, bs>>>(
                        st.d_Color, g.num_nodes,
                        w->color,
                        d_new_set, d_pos);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                CUDA_CHECK(cudaFree(d_pos));
                w->d_set_nodes = d_new_set;
                w->set_capacity = w->count;
                w->owns_set = 1;
            }

            // OpenMP: do_fw_bw_dfs(G, w, new_works);
            do_fw_bw_dfs(st, g, w, new_works);

            // OpenMP: delete w;
            if (w->d_set_nodes != NULL && w->owns_set) {
                CUDA_CHECK(cudaFree(w->d_set_nodes));
            }
            delete w;
            w = NULL;

            // OpenMP: keep some new items locally
            //   while ((my_works.size() < N) && (new_works.size() > 0)) {
            //       my_work* w = new_works.back();
            //       new_works.pop_back();
            //       my_works.push_back(w);
            //   }
            while ((my_works.size() < (size_t)N) && (new_works.size() > 0)) {
                CUDAMyWork* w = new_works.back();
                new_works.pop_back();
                my_works.push_back(w);
            }

            // OpenMP: push rest to global queue
            //   if (new_works.size() > 0) {
            //       work_q_put_all(tid, new_works); new_works.clear();
            //   }
            if (new_works.size() > 0) {
                work_q_put_all(0, new_works);
                new_works.clear();
            }
        }
    }
}
