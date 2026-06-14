#include "scc_cuda.h"
#include "gm.h"

#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <map>

using namespace std;

// Host-side CSR arrays (shared with scc_cuda_fb_seq2.cu for host-side FB processing)
const edge_t* g_h_begin = NULL;
const node_t* g_h_node_idx = NULL;
const edge_t* g_h_r_begin = NULL;
const node_t* g_h_r_node_idx = NULL;
int g_h_N = 0;

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------
    // Parse CLI: same as OpenMP: <graph_file> <num_threads> <method> [-d|-a|-p]
    // ---------------------------------------------------------------
    if (argc < 4) {
        printf("Usage: %s <graph_file> <num_threads> <method> [-d|-a|-p]\n", argv[0]);
        printf("  method 0: Trim1 + FW-BW BFS (Baseline)\n");
        printf("  method 1: Trim1 + Global FW-BW + Trim1 + FW-BW DFS\n");
        printf("  method 2: Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW DFS\n");
        printf("  method 5: Incremental (naive graph)\n");
        printf("  method 6: Incremental (SCC condensation)\n");
        printf("  method 7: Incremental (SCC condensation + BFS levels)\n");
        printf("  method 11: Incremental (SCC condensation + pivot hint)\n");
        printf("  -p: Print SCC list to file\n");
        return 1;
    }

    const char* graph_file = argv[1];
    int num_threads = atoi(argv[2]);
    int met_algo_original = atoi(argv[3]);  // SAVE original before mapping

    // Parse flags
    int detail_time = 0;
    int analyze     = 0;
    int print       = 0;
    if (argc >= 5) {
        if (strncmp(argv[4], "-d", 2) == 0) detail_time = 1;
        else if (strncmp(argv[4], "-a", 2) == 0) analyze = 1;
        else if (strncmp(argv[4], "-p", 2) == 0) print = 1;
    }

    // Map methods 5-7, 11 to 2 for the pipeline (same as OpenMP)
    int met_algo = met_algo_original;
    if ((met_algo >= 5 && met_algo <= 7) || met_algo == 11)
        met_algo = 2;

    int flag11 = 0;
    int good_init_pivot = -1;
    double insert_runtime = 0.0;
    int num_sccs = 0;

    // Host-side dynamic arrays (populated by graph construction, uploaded to GPU)
    vector<int> h_scc_list;
    vector<int> h_level_ver;
    vector<int> h_affect_level;
    vector<int> h_new_edge_nodes;

    // ---------------------------------------------------------------
    // Load graph — 1:1 mirror of OpenMP common_main.h::main()
    // ---------------------------------------------------------------
    struct timeval T1, T2, T6_1, T6_2;
    string fname = graph_file;

    gm_graph G;
    gm_rt_set_num_threads(num_threads);
    gm_rt_initialize();

    gettimeofday(&T1, NULL);
    {
        vector<pair<int,int>> orig_edges;
        vector<pair<int,int>> insert_edges;
        vector<pair<int,int>> scc_edges;

        // ---- Static methods (0-4): load directly ----
        if (met_algo_original == 0 || met_algo_original == 1 ||
            met_algo_original == 2 || met_algo_original == 3 ||
            met_algo_original == 4)
        {
            // OpenMP: int num_vertices = read_file(fname, orig_edges);
            //         for (int i = 0; i < num_vertices; i++) G.add_node();
            //         for all edges: G.add_edge(...);
            int num_vertices = read_file(fname, orig_edges);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            for (size_t i = 0; i < orig_edges.size(); i++)
                G.add_edge(orig_edges[i].first, orig_edges[i].second);
        }

        // ---- Method 5 (Incremental, naive graph) ----
        // OpenMP: read refined_edges.txt + insert_edges, insert_idea1
        if (met_algo_original == 5)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            // OpenMP: insert_idea1(G, orig_edges, insert_edges)
            insert_idea1(G, orig_edges, insert_edges);
        }

        // ---- Method 6 (Incremental, SCC condensation graph) ----
        // OpenMP: read refined_edges.txt + scc_list.txt + insert_edges,
        //         create_scc_edges, build condensation graph
        if (met_algo_original == 6)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            num_sccs = read_file1(
                "/home/tk.temp/par-scc/scc_list.txt", h_scc_list, num_vertices);

            gettimeofday(&T6_1, NULL);
            // OpenMP: create_scc_edges(orig_edges, insert_edges, scc_edges, ...)
            create_scc_edges(orig_edges, insert_edges, scc_edges,
                num_vertices, num_sccs, met_algo_original,
                h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
                insert_runtime);
            gettimeofday(&T6_2, NULL);

            // OpenMP: choose good_init_pivot (SCC with max neighbors)
            {
                vector<vector<int>> scc_adj_list(num_sccs);
                int maxi_neigh = 0;
                good_init_pivot = 0;
                for (size_t i = 0; i < scc_edges.size(); i++) {
                    int ver1 = scc_edges[i].first;
                    int ver2 = scc_edges[i].second;
                    scc_adj_list[ver1].push_back(ver2);
                }
                for (int i = 0; i < num_sccs; i++) {
                    if ((int)scc_adj_list[i].size() > maxi_neigh) {
                        maxi_neigh = (int)scc_adj_list[i].size();
                        good_init_pivot = i;
                    }
                }
            }

            // OpenMP: build condensation graph (nodes = SCCs, edges = cross-SCC edges)
            if (num_sccs > 1) {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            } else {
                G.add_node();
            }
        }

        // ---- Method 11 (Incremental, SCC condensation + pivot hint) ----
        if (met_algo_original == 11)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            num_sccs = read_file1(
                "/home/tk.temp/par-scc/scc_list.txt", h_scc_list, num_vertices);

            gettimeofday(&T6_1, NULL);
            create_scc_edges(orig_edges, insert_edges, scc_edges,
                num_vertices, num_sccs, met_algo_original,
                h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
                insert_runtime);
            gettimeofday(&T6_2, NULL);

            // OpenMP: choose good_init_pivot
            {
                vector<vector<int>> scc_adj_list(num_sccs);
                int maxi_neigh = 0;
                good_init_pivot = 0;
                for (const auto& p : scc_edges) {
                    scc_adj_list[p.first].push_back(p.second);
                }
                for (int i = 0; i < num_sccs; i++) {
                    if ((int)scc_adj_list[i].size() > maxi_neigh) {
                        maxi_neigh = (int)scc_adj_list[i].size();
                        good_init_pivot = i;
                    }
                }
            }

            // OpenMP: build condensation graph
            if (num_sccs > 1) {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            } else {
                G.add_node();
            }
        }

        // ---- Method 7 (Incremental, SCC condensation + BFS levels) ----
        if (met_algo_original == 7)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            num_sccs = read_file1(
                "/home/tk.temp/par-scc/scc_list.txt", h_scc_list, num_vertices);

            // OpenMP: create_scc_edges (includes BFS levels + affect_level for met_algo==7)
            create_scc_edges(orig_edges, insert_edges, scc_edges,
                num_vertices, num_sccs, met_algo_original,
                h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
                insert_runtime);

            // OpenMP: build condensation graph
            if (num_sccs > 1) {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            } else {
                G.add_node();
            }
        }

    }
    gettimeofday(&T2, NULL);
    printf("graph loading time=%lf\n",
           (T2.tv_sec - T1.tv_sec) * 1000 + (T2.tv_usec - T1.tv_usec) * 0.001);

    gettimeofday(&T1, NULL);
    G.make_reverse_edges();
    gettimeofday(&T2, NULL);
    printf("reverse edge creation time=%lf\n",
           (T2.tv_sec - T1.tv_sec) * 1000 + (T2.tv_usec - T1.tv_usec) * 0.001);

    printf("data=%s %d %d\n", graph_file, met_algo, num_threads);

    int N = G.num_nodes();
    int M = G.num_edges();

    // ---------------------------------------------------------------
    // Extract CSR arrays
    // ---------------------------------------------------------------
    vector<edge_t> h_begin(N + 1);
    vector<node_t> h_node_idx(M);
    vector<edge_t> h_r_begin(N + 1);
    vector<node_t> h_r_node_idx(M);
    memcpy(h_begin.data(),     G.begin,      (N + 1) * sizeof(edge_t));
    if (M > 0) {
        memcpy(h_node_idx.data(),  G.node_idx,    M * sizeof(node_t));
        memcpy(h_r_node_idx.data(), G.r_node_idx, M * sizeof(node_t));
    }
    memcpy(h_r_begin.data(),   G.r_begin,     (N + 1) * sizeof(edge_t));

    // ---------------------------------------------------------------
    // Upload to GPU
    // ---------------------------------------------------------------
    GPUGraph gpuG;
    graph_upload(gpuG, h_begin, h_node_idx, h_r_begin, h_r_node_idx, N, M);

    // Set global host CSR arrays (for host-side FB processing)
    g_h_begin     = h_begin.data();
    g_h_node_idx  = h_node_idx.data();
    g_h_r_begin   = h_r_begin.data();
    g_h_r_node_idx = h_r_node_idx.data();
    g_h_N         = N;

    GPUState st;
    state_allocate(st, N);
    state_init(st);

    initialize_trim1_full(N);
    initialize_trim2(N);
    initialize_WCC(N);
    work_q_init(num_threads);

    // Device counter
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    int* d_count_trim_spec = NULL;
    if (met_algo_original == 11) {
        CUDA_CHECK(cudaMalloc(&d_count_trim_spec, sizeof(int)));
    }

    // Dynamic arrays for methods 6, 7, 11
    DynamicArrays da;
    memset(&da, 0, sizeof(da));
    int da_alloc_size = (num_sccs > 0) ? num_sccs : N;
    if (met_algo_original == 11 || met_algo_original == 7 || met_algo_original == 6) {
        dynamic_arrays_allocate(da, N, da_alloc_size);

        // --- Upload host dynamic arrays to GPU (1:1 mirror of CPU globals) ---
        // OpenMP: these are global vectors accessible by trim_once_node
        // CUDA: explicit upload to device DynamicArrays
        if (!h_scc_list.empty())
            dynamic_arrays_upload_scc_list(da, h_scc_list, N);
        if (!h_level_ver.empty())
            dynamic_arrays_upload_level_ver(da, h_level_ver, N);
        if (!h_affect_level.empty())
            dynamic_arrays_upload_affect_level(da, h_affect_level, N + 5);
    }
    // Method 6 is already covered above (included in the allocation condition).

    // ---------------------------------------------------------------
    // Run selected method
    // ---------------------------------------------------------------
    struct timeval R1, R2;
    double runtime_ms = 0.0;
    int trimmed = 0;

    if (met_algo == 0) {
        // ============================================================
        // Method 0 (Baseline): Trim1 + FW-BW (BFS-based)
        // OpenMP: do_baseline()
        // ============================================================
        printf("Running Method 0 (Baseline): Trim1 + FW-BW BFS\n");

        // OpenMP timer starts before the pipeline
        gettimeofday(&R1, NULL);

        // ---------- Phase 1: TRIM1 ----------
        trimmed = repeat_global_trim1(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 0);
        int remaining = N - trimmed;
        printf("[CUDA] Trimmed = %d\n", trimmed);

        if (remaining == 0) {
            printf("[CUDA] No remaining nodes after trim\n");
        } else {
            // ---------- Phase 2: FB (BFS) ----------
            initialize_global_fb(N);

            CUDAMyWork* work = new CUDAMyWork();
            work->color       = COLOR_UNASSIGNED;  // curr_color = -1
            work->count       = remaining;
            work->d_set_nodes = NULL;
            work->set_capacity = 0;
            work->depth       = 0;
            work->owns_set    = 0;
            work_q_put(0, work);

            start_workers_fw_bw(st, gpuG, 1);

            finalize_global_fb();
        }

        gettimeofday(&R2, NULL);
        runtime_ms = (R2.tv_sec - R1.tv_sec) * 1000.0 +
                     (R2.tv_usec - R1.tv_usec) * 0.001;

    } else if (met_algo == 1) {
        // ============================================================
        // Method 1: Trim1 + Global FW-BW + Trim1 + FW-BW DFS
        // OpenMP: do_baseline_global_fb()
        // ============================================================
        printf("Running Method 1: Trim1 + Global FW-BW + Trim1 + FW-BW DFS\n");

        // OpenMP timer starts before the pipeline
        gettimeofday(&R1, NULL);

        // ---------- Phase 1: TRIM1 ----------
        trimmed = repeat_global_trim1(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 0);
        printf("[CUDA] Trimmed = %d\n", trimmed);

        int curr_count = N - trimmed;
        if (curr_count == 0) {
            printf("[CUDA] No remaining nodes after trim\n");
        } else {
            // Ensure d_trim_targets_count is up-to-date for do_global_fw_bw_main
            create_trim1_compact(st, gpuG);

            // ---------- Phase 2: GLOBAL BFS ----------
            initialize_global_fb(N);
            int scc_size = do_global_fw_bw_main(
                st, gpuG,
                COLOR_UNASSIGNED,
                curr_count,
                -1,
                false);
            printf("[CUDA] First SCC size = %d\n", scc_size);

            // ---------- Phase 3: TRIM1 (compact) ----------
            trimmed = repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 0);

            curr_count = d_trim_targets_count;
            if (curr_count > 0) {
                // ---------- Phase 4: FB (DFS) ----------
                create_works_after_bfs_trim(st, gpuG);
                start_workers_fw_bw_dfs(st, gpuG, 1);
            }
            finalize_global_fb();
        }

        gettimeofday(&R2, NULL);
        runtime_ms = (R2.tv_sec - R1.tv_sec) * 1000.0 +
                     (R2.tv_usec - R1.tv_usec) * 0.001;

    } else if (met_algo == 2) {
        // ============================================================
        // Method 2: Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW DFS
        // OpenMP: do_baseline_global_wcc_fb()
        // ============================================================
        printf("Running Method 2: Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW DFS\n");

        // Per-phase timing using gettimeofday (host-side, works on any server)
        struct timeval t_start, t_trim1, t_compact, t_bfs, t_trim12, t_wcc, t_end;
        gettimeofday(&R1, NULL);
        gettimeofday(&t_start, NULL);

        // ---------- Phase 1: TRIM1 ----------
        trimmed = repeat_global_trim1(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 0);
        gettimeofday(&t_trim1, NULL);
        printf("[CUDA] Trimmed = %d\n", trimmed);

        int curr_count = N - trimmed;
        if (curr_count == 0) {
            printf("[CUDA] No remaining nodes after trim\n");
            gettimeofday(&t_compact, NULL);
            gettimeofday(&t_bfs, NULL);
            gettimeofday(&t_trim12, NULL);
            gettimeofday(&t_wcc, NULL);
        } else {
            // Ensure d_trim_targets_count is up-to-date for do_global_fw_bw_main
            create_trim1_compact(st, gpuG);
            gettimeofday(&t_compact, NULL);

            // ---------- Phase 2: GLOBAL BFS ----------
            // OpenMP: do_fw_bw_global_main(G, curr_color, curr_count, false)
            initialize_global_fb(N);
            int scc_size = do_global_fw_bw_main(
                st, gpuG,
                COLOR_UNASSIGNED,   // base_color = curr_color = -1
                curr_count,          // base_count from trim_targets
                -1,                  // good_init_pivot (-1 = not met_algo 6/11)
                false);              // create_work_items = false
            gettimeofday(&t_bfs, NULL);
            printf("[CUDA] First SCC size = %d\n", scc_size);

            // ---------------------------------------------------------------
            // Phase 2.5: Method-11 flag check (1:1 mirror of OpenMP)
            // ---------------------------------------------------------------
            if (met_algo_original == 11)
            {
                flag11 = 2;
                vector<int> check_indices;
                for (size_t i = 0; i < h_new_edge_nodes.size(); i++) {
                    if (h_new_edge_nodes[i] >= 0)
                        check_indices.push_back(h_new_edge_nodes[i]);
                }
                if (!check_indices.empty()) {
                    vector<int> h_scc_check(check_indices.size());
                    for (size_t i = 0; i < check_indices.size(); i++) {
                        CUDA_CHECK(cudaMemcpy(&h_scc_check[i],
                            &st.d_SCC[check_indices[i]],
                            sizeof(int), cudaMemcpyDeviceToHost));
                    }
                    for (size_t i = 0; i < h_scc_check.size(); i++) {
                        if (h_scc_check[i] < 0) {
                            cout << "Helloooo" << endl;
                            flag11 = 1;
                            break;
                        }
                    }
                }
            }

            // ---------- Phase 3: TRIM1/2 (compact) ----------
            trimmed = repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 0);
            int trim_total = do_global_trim2_new(st, gpuG, d_count);
            trim_total += repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 100);
            trimmed += trim_total;
            gettimeofday(&t_trim12, NULL);

            curr_count = d_trim_targets_count;
            if (curr_count > 0) {
                // ---------- Phase 4: WCC ----------
                do_global_wcc(st, gpuG);
                create_work_items_from_wcc(st, gpuG);
                gettimeofday(&t_wcc, NULL);

                // ---------- Phase 5: FB (DFS) — processed on host to avoid kernel launch overhead ----------
                start_workers_fw_bw_dfs_host(st, gpuG, 40);
            } else {
                gettimeofday(&t_wcc, NULL);
            }
            finalize_global_fb();
        }

        gettimeofday(&t_end, NULL);

        // Compute per-phase timings using gettimeofday
        double t1 = (t_trim1.tv_sec - t_start.tv_sec) * 1000.0 +
                    (t_trim1.tv_usec - t_start.tv_usec) * 0.001;
        double t2 = (t_compact.tv_sec - t_trim1.tv_sec) * 1000.0 +
                    (t_compact.tv_usec - t_trim1.tv_usec) * 0.001;
        double t3 = (t_bfs.tv_sec - t_compact.tv_sec) * 1000.0 +
                    (t_bfs.tv_usec - t_compact.tv_usec) * 0.001;
        double t4 = (t_trim12.tv_sec - t_bfs.tv_sec) * 1000.0 +
                    (t_trim12.tv_usec - t_bfs.tv_usec) * 0.001;
        double t5 = (t_wcc.tv_sec - t_trim12.tv_sec) * 1000.0 +
                    (t_wcc.tv_usec - t_trim12.tv_usec) * 0.001;
        double t6 = (t_end.tv_sec - t_wcc.tv_sec) * 1000.0 +
                    (t_end.tv_usec - t_wcc.tv_usec) * 0.001;
        double t_total = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                         (t_end.tv_usec - t_start.tv_usec) * 0.001;

        printf(">>>>CUDA_PROFILE: TRIM1=%.2fms COMPACT_BUILD=%.2fms GLOBAL_BFS=%.2fms TRIM12=%.2fms WCC=%.2fms FB=%.2fms TOTAL=%.2fms\n",
               t1, t2, t3, t4, t5, t6, t_total);
        fflush(stdout);
        fprintf(stderr, "[CUDA_PROFILE_STDERR] TRIM1=%.2f COMPACT=%.2f GLOBAL_BFS=%.2f TRIM12=%.2f WCC=%.2f FB=%.2f TOTAL=%.2f\n",
                t1, t2, t3, t4, t5, t6, t_total);

        // Original timing (for total)
        gettimeofday(&R2, NULL);
        runtime_ms = (R2.tv_sec - R1.tv_sec) * 1000.0 +
                     (R2.tv_usec - R1.tv_usec) * 0.001;

    } else {
        printf("Running CUDA Method %d: (not implemented)\n", met_algo);
        printf("Supported methods: 0 (Baseline), 1 (Global FB + FB DFS), 2 (Full pipeline)\n");
        cudaFree(d_count);
        if (d_count_trim_spec) cudaFree(d_count_trim_spec);
        dynamic_arrays_free(da);
        finalize_WCC();
        finalize_trim2();
        finalize_trim1();
        state_free(st);
        graph_free(gpuG);
        return 0;
    }

    printf("[CUDA]running_time(ms)=%lf\n", runtime_ms + insert_runtime);

    // ---------------------------------------------------------------
    // Post-processing: count SCCs
    // OpenMP (scc_main.cc post_process):
    //   int count = 0;
    //   for(int i=0;i<G.num_nodes(); i++) {
    //       if (G_SCC[i] == i) count++;
    //       else if(G_SCC[i]==-1) trim_9+=1;
    //   }
    //   printf("Total # SCCs = %d\n", count);
    // ---------------------------------------------------------------
    {
        vector<int> h_SCC(N);
        CUDA_CHECK(cudaMemcpy(h_SCC.data(), st.d_SCC, N * sizeof(int),
                               cudaMemcpyDeviceToHost));

        int scc_count = 0;
        // Count SCC sizes for histogram
        std::vector<int> scc_size(N, 0);
        for (int i = 0; i < N; i++) {
            if (h_SCC[i] == i) scc_count++;
            if (h_SCC[i] >= 0) scc_size[h_SCC[i]]++;
        }
        printf("Total # SCCs = %d\n", scc_count);

        if (analyze) {
            std::map<int, int> hist;
            for (int i = 0; i < N; i++) {
                if (h_SCC[i] == i) {
                    int sz = scc_size[i];
                    hist[sz]++;
                }
            }
            for (auto& p : hist) {
                printf("%d => %d\n", p.first, p.second);
            }
            printf("\n");
        }

        if (print) {
            FILE* fp = fopen("scc_output_cuda.txt", "w");
            if (fp) {
                for (int i = 0; i < N; i++)
                    fprintf(fp, "%d %d\n", i, h_SCC[i]);
                fclose(fp);
                printf("SCC list written to scc_output_cuda.txt\n");
            }
        }
    }

    // ---------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------
    // Note: finalize_global_fb() is called inside each method block (0, 1, 2)
    // when initialize_global_fb() was called. Do NOT call it again here —
    // the double-free causes "double free or corruption (out)" on larger
    // datasets where the BFS allocates many pinned-memory buffers.
    cudaFree(d_count);
    if (d_count_trim_spec) cudaFree(d_count_trim_spec);
    dynamic_arrays_free(da);
    finalize_WCC();
    finalize_trim2();
    finalize_trim1();
    state_free(st);
    graph_free(gpuG);

    return 0;
}
