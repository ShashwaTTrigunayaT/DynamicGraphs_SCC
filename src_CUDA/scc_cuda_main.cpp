#include "scc_cuda.h"
#include "gm.h"

#include <string>
#include <vector>
#include <cstring>

using namespace std;

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------
    // Parse CLI: same as OpenMP: <graph_file> <num_threads> <method> [-d|-a|-p]
    // ---------------------------------------------------------------
    if (argc < 4) {
        printf("Usage: %s <graph_file> <num_threads> <method> [-d|-a|-p]\n", argv[0]);
        printf("  method 0: Trim1 + FW-BW BFS (Baseline — exact mirror of OpenMP method 0)\n");
        printf("  method 1: Trim1 + Global FW-BW + Trim1 + FW-BW DFS (exact mirror of OpenMP method 1)\n");
        printf("  method 2: Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW DFS (exact mirror of OpenMP method 2)\n");
        printf("  -p: Print SCC list to file\n");
        return 1;
    }

    const char* graph_file = argv[1];
    int num_threads = atoi(argv[2]);
    int met_algo    = atoi(argv[3]);

    // Map methods 5-11 to 2 (same as OpenMP)
    if (met_algo >= 5 && met_algo <= 11)
        met_algo = 2;

    // Parse flags
    int detail_time = 0;
    int analyze     = 0;
    int print       = 0;
    if (argc >= 5) {
        if (strncmp(argv[4], "-d", 2) == 0) detail_time = 1;
        else if (strncmp(argv[4], "-a", 2) == 0) analyze = 1;
        else if (strncmp(argv[4], "-p", 2) == 0) print = 1;
    }

    int flag11 = 0;

    // ---------------------------------------------------------------
    // Load graph (same as OpenMP: gm_graph)
    // ---------------------------------------------------------------
    struct timeval T1, T2;

    gm_graph G;
    gm_rt_set_num_threads(num_threads);
    gm_rt_initialize();

    gettimeofday(&T1, NULL);
    {
        vector<pair<int,int>> edges_list;
        int max_vertex = 0;
        ifstream inputFile(graph_file);
        string line;
        while (getline(inputFile, line)) {
            vector<string> tokens;
            string token;
            stringstream ss(line);
            while (getline(ss, token, ' '))
                tokens.push_back(token);
            int v1 = stoi(tokens[0]) - 1;
            int v2 = stoi(tokens[1]) - 1;
            edges_list.push_back(make_pair(v1, v2));
            max_vertex = max(max_vertex, max(stoi(tokens[0]), stoi(tokens[1])));
        }
        inputFile.close();

        for (int i = 0; i < max_vertex; i++)
            G.add_node();
        for (size_t i = 0; i < edges_list.size(); i++)
            G.add_edge(edges_list[i].first, edges_list[i].second);
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

    GPUState st;
    state_allocate(st, N);
    state_init(st);

    initialize_trim1_full(N);
    initialize_trim2(N);
    initialize_WCC(N);

    // Device counter
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    int* d_count_trim_spec = NULL;
    if (met_algo == 11) {
        CUDA_CHECK(cudaMalloc(&d_count_trim_spec, sizeof(int)));
    }

    // Dynamic arrays for methods 7, 9, 11
    DynamicArrays da;
    memset(&da, 0, sizeof(da));
    if (met_algo == 9 || met_algo == 11 || met_algo == 7) {
        dynamic_arrays_allocate(da, N, 1);
    }

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

        // Phase 1: Trim1 (mirrors OpenMP: repeat_global_trim1)
        // Note: GPU stride-parallelism trims fewer nodes per iteration than
        // CPU's sequential-within-chunk processing (OpenMP schedule(dynamic,512)),
        // so we use TRIM_STOP=0 to ensure full convergence.
        trimmed = repeat_global_trim1(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 0);
        int remaining = N - trimmed;
        printf("[CUDA] Trimmed = %d\n", trimmed);

        if (remaining == 0) {
            printf("[CUDA] No remaining nodes after trim\n");
        } else {
            // Phase 2: Initialize BFS buffers, create work item, run FW-BW BFS
            // OpenMP:
            //   my_work* work = new my_work();
            //   work->color = get_curr_color();  // -1
            //   work->color_set = NULL;
            //   work->count = G_num_nodes - trimmed;
            //   work->depth = 0;
            //   work_q_put(0, work);
            //   start_workers_fw_bw(G, 1);
            initialize_global_fb(N);

            CUDAMyWork* work = new CUDAMyWork();
            work->color       = COLOR_UNASSIGNED;  // curr_color = -1
            work->count       = remaining;
            work->d_set_nodes = NULL;   // NULL -> lazy gen in start_workers_fw_bw
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

        // Phase 1: Trim1
        trimmed = repeat_global_trim1(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 0);
        printf("[CUDA] Trimmed = %d\n", trimmed);

        int curr_count = N - trimmed;
        if (curr_count == 0) {
            printf("[CUDA] No remaining nodes after trim\n");
        } else {
            // Ensure d_trim_targets_count is up-to-date for do_global_fw_bw_main
            create_trim1_compact(st, gpuG);

            // Phase 2: Global FW-BW (finds one large SCC)
            // OpenMP: do_fw_bw_global_main(G, curr_color, curr_count, false)
            initialize_global_fb(N);
            int scc_size = do_global_fw_bw_main(
                st, gpuG,
                COLOR_UNASSIGNED,   // base_color = curr_color = -1
                curr_count,          // base_count from trim_targets
                -1,                  // good_init_pivot (-1 = not met_algo 6/11)
                false);              // create_work_items = false
            printf("[CUDA] First SCC size = %d\n", scc_size);

            // Phase 3: Re-trim (compact)
            // OpenMP: repeat_global_trim1_compact(G)
            trimmed = repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 0);

            curr_count = d_trim_targets_count;
            if (curr_count > 0) {
                // Phase 4: Create work items from colored partition + FB
                // OpenMP:
                //   create_works_after_bfs_trim(G);
                //   start_workers_fw_bw_dfs(G, 1);
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

        // OpenMP timer starts before the pipeline
        gettimeofday(&R1, NULL);

        // Phase 1: Trim1
        trimmed = repeat_global_trim1(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 0);
        printf("[CUDA] Trimmed = %d\n", trimmed);

        int curr_count = N - trimmed;
        if (curr_count == 0) {
            printf("[CUDA] No remaining nodes after trim\n");
        } else {
            // Ensure d_trim_targets_count is up-to-date for do_global_fw_bw_main
            create_trim1_compact(st, gpuG);

            // Phase 2: Global FW-BW (finds one large SCC)
            // OpenMP: do_fw_bw_global_main(G, curr_color, curr_count, false)
            initialize_global_fb(N);
            int scc_size = do_global_fw_bw_main(
                st, gpuG,
                COLOR_UNASSIGNED,   // base_color = curr_color = -1
                curr_count,          // base_count from trim_targets
                -1,                  // good_init_pivot (-1 = not met_algo 6/11)
                false);              // create_work_items = false
            printf("[CUDA] First SCC size = %d\n", scc_size);

            // Phase 3: Re-trim (compact + trim2)
            // OpenMP:
            //   trimmed = repeat_global_trim1_compact(G);
            //   trim_total = do_global_trim2_new(G);
            //   trim_total += repeat_global_trim1_compact(G, 100);
            trimmed = repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 0);
            int trim_total = do_global_trim2_new(st, gpuG, d_count);
            trim_total += repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 0);
            trimmed += trim_total;

            curr_count = d_trim_targets_count;
            if (curr_count > 0) {
                // Phase 4: WCC + create work items + FW-BW DFS
                // OpenMP:
                //   do_global_wcc(G);
                //   create_work_items_from_wcc(G);
                //   start_workers_fw_bw_dfs(G, 40);
                do_global_wcc(st, gpuG);
                create_work_items_from_wcc(st, gpuG);
                start_workers_fw_bw_dfs(st, gpuG, 40);
            }
            finalize_global_fb();
        }

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

    printf("[CUDA]running_time(ms)=%lf\n", runtime_ms);

    // ---------------------------------------------------------------
    // Post-processing: count SCCs
    // ---------------------------------------------------------------
    {
        vector<int> h_SCC(N);
        CUDA_CHECK(cudaMemcpy(h_SCC.data(), st.d_SCC, N * sizeof(int),
                               cudaMemcpyDeviceToHost));

        int scc_count = 0;
        int trim9_count = 0;
        for (int i = 0; i < N; i++) {
            if (h_SCC[i] == i) scc_count++;
            else if (h_SCC[i] == -1) trim9_count++;
        }
        printf("Total # SCCs = %d\n", scc_count);

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
    cudaFree(d_count);
    if (d_count_trim_spec) cudaFree(d_count_trim_spec);
    dynamic_arrays_free(da);
    if (met_algo >= 1 && met_algo <= 2) finalize_global_fb();
    finalize_WCC();
    finalize_trim2();
    finalize_trim1();
    state_free(st);
    graph_free(gpuG);

    return 0;
}
