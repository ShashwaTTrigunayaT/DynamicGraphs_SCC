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
        printf("  method 0: Trim1 only (CUDA)\n");
        printf("  method 2: Trim1 + Trim2 (CUDA)\n");
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
    if (met_algo == 0) {
        printf("Running CUDA Method 0: Trim1\n");
    } else if (met_algo == 2) {
        printf("Running CUDA Method 2: Trim1 + Trim2\n");
    } else {
        printf("Running CUDA Method %d: (not fully implemented)\n", met_algo);
        cudaFree(d_count);
        if (d_count_trim_spec) cudaFree(d_count_trim_spec);
        dynamic_arrays_free(da);
        finalize_trim2();
        finalize_trim1();
        state_free(st);
        graph_free(gpuG);
        return 0;
    }

    struct timeval R1, R2;
    gettimeofday(&R1, NULL);

    int trimmed = 0;

    // Phase 1: Trim1
    trimmed += repeat_global_trim1(st, gpuG, d_count,
        met_algo, flag11, da, d_count_trim_spec, 100);

    if (met_algo == 2) {
        // Phase 2: Trim2
        trimmed += do_global_trim2_new(st, gpuG, d_count);
        trimmed += repeat_global_trim1_compact(st, gpuG, d_count,
            met_algo, flag11, da, d_count_trim_spec, 100);
    }

    gettimeofday(&R2, NULL);
    double runtime = (R2.tv_sec - R1.tv_sec) * 1000.0 +
                     (R2.tv_usec - R1.tv_usec) * 0.001;

    printf("[CUDA]running_time(ms)=%lf\n", runtime);

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
        if (met_algo == 9)
            printf("Trimmed nodes=%d\n", trim9_count);

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
    finalize_trim2();
    finalize_trim1();
    state_free(st);
    graph_free(gpuG);

    return 0;
}
