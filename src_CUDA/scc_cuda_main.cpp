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
#include <set>
#include <cctype>
#include <omp.h>

using namespace std;

// ---- Algorithm Time tracking globals ----
double g_algo_memcpy_ms = 0;

void algo_memcpy_init() {
    g_algo_memcpy_ms = 0;
}
void algo_memcpy_finalize() {
    // Nothing to free — std::chrono has no handles
}

// ======================================================================
// Fast integer parser: parse next int from string, advance pointer
// Returns -1 on failure (end of buffer)
// ======================================================================
static inline int fast_parse_int(const char*& p, const char* end) {
    while (p < end && !isdigit((unsigned char)*p) && *p != '-') p++;
    if (p >= end) return -1;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    int val = 0;
    while (p < end && isdigit((unsigned char)*p)) {
        val = val * 10 + (*p - '0');
        p++;
    }
    return val * sign;
}

// ======================================================================
// Fast line counter: count newlines in buffer
// ======================================================================
static inline int count_lines(const char* buf, size_t size) {
    int count = 0;
    for (size_t i = 0; i < size; i++) {
        if (buf[i] == '\n') count++;
    }
    return count;
}

// ======================================================================
// read_file() — FAST version (from common_main.h)
//
// Replaces the original getline+stringstream+stoi implementation.
// Uses fread to load entire file, then manual integer parsing.
// Typically 5-10x faster for large graphs (117M edges).
//
// 1:1 functional mirror of main_t::read_file() in common_main.h
// ======================================================================
int read_file(const string& filename, vector<pair<int, int>>& edges_list)
{
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", filename.c_str());
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) { fclose(fp); return 0; }

    char* buf = new char[fsize + 1];
    size_t bytes_read = fread(buf, 1, fsize, fp);
    fclose(fp);
    buf[bytes_read] = '\0';

    int est_lines = count_lines(buf, bytes_read);
    if (est_lines < 0) est_lines = 0;
    edges_list.reserve(est_lines);

    const char* p = buf;
    const char* end = buf + bytes_read;
    int max_vertex = 0;

    while (p < end) {
        int v1 = fast_parse_int(p, end);
        if (v1 < 0) break;
        int v2 = fast_parse_int(p, end);
        if (v2 < 0) break;

        while (p < end && *p != '\n') p++;
        if (p < end) p++;

        edges_list.push_back(make_pair(v1 - 1, v2 - 1));
        if (v1 > max_vertex) max_vertex = v1;
        if (v2 > max_vertex) max_vertex = v2;
    }

    delete[] buf;
    return max_vertex;
}

// ======================================================================
// read_file1() — 1:1 mirror of main_t::read_file1() in common_main.h
// ======================================================================
int read_file1(const string& filename, vector<int>& scc_list_out, int num_vertices)
{
    ifstream inputFile(filename);
    string line;
    int max_vertex = 0;
    scc_list_out.resize(num_vertices);

    while (getline(inputFile, line))
    {
        vector<string> tokens;
        string token;
        stringstream ss(line);
        while (getline(ss, token, ' ')) tokens.push_back(token);
        scc_list_out[stoi(tokens[0])] = stoi(tokens[1]);
        max_vertex = max(max_vertex, stoi(tokens[1]) + 1);
    }

    inputFile.close();
    return max_vertex;
}

// ======================================================================
// BFS() — 1:1 mirror of main_t::BFS() in common_main.h
// ======================================================================
void BFS(vector<vector<int>>& adj_list, vector<int>& level,
         queue<int>& qu, vector<int>& in_degree, int* max_level)
{
    int top;
    while (!qu.empty())
    {
        top = qu.front();
        qu.pop();
        for (int i = 0; i < (int)adj_list[top].size(); i++)
        {
            in_degree[adj_list[top][i]]--;
            if (in_degree[adj_list[top][i]] == 0)
                qu.push(adj_list[top][i]);
            level[adj_list[top][i]] = level[top] + 1;
            *(max_level) = max(*(max_level), level[adj_list[top][i]]);
        }
    }
    cout << "max_level:" << (*(max_level)) << endl;
}

// ======================================================================
// parallel_prefix_sum() — 1:1 mirror of main_t::parallel_prefix_sum()
// ======================================================================
void parallel_prefix_sum(std::vector<int>& a)
{
    int N = (int)a.size();
    if (N == 0) return;

    int num_threads = 0;
#pragma omp parallel
    {
#pragma omp master
        { num_threads = omp_get_num_threads(); }
    }

    std::vector<float> partial_sums(num_threads + 1, 0.0f);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float local_sum = 0.0f;
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) { local_sum += a[i]; a[i] = local_sum; }
        partial_sums[tid + 1] = local_sum;
    }

    for (int i = 1; i <= num_threads; ++i) partial_sums[i] += partial_sums[i - 1];

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();
        a[i] += partial_sums[tid];
    }
}

// ======================================================================
// create_scc_edges() — 1:1 mirror of main_t::create_scc_edges()
//
// Creates cross-SCC edge list from original + insert edges.
// For method 7: also builds BFS levels + affect_level.
// For method 11: tracks new_edge_nodes for pivot hint.
// ======================================================================
void create_scc_edges(vector<pair<int, int>> orig_edges,
                      vector<pair<int, int>> insert_edges,
                      vector<pair<int, int>>& scc_edges,
                      int num_vertices, int num_sccs, int met_algo,
                      vector<int>& scc_list,
                      vector<int>& level_ver,
                      vector<int>& affect_level,
                      vector<int>& new_edge_nodes,
                      double& insert_runtime)
{
    int root_node = 0;
    vector<vector<int>> adj_list(num_sccs);
    level_ver.resize(num_sccs, 0);
    new_edge_nodes.resize(num_sccs, -1);
    affect_level.resize(num_sccs + 5, 0);
    queue<int> qu;
    vector<int> in_degree(num_sccs, 0);
    vector<int> unaffected_levels;
    int max_level = 0;
    scc_edges.resize(orig_edges.size() + insert_edges.size(), {0, 0});
    struct timeval T_insert1, T_insert2;

    // --- Process orig edges (parallel) ---
#pragma omp parallel for
    for (int i = 0; i < (int)orig_edges.size(); i++)
    {
        int ver1 = orig_edges[i].first;
        int ver2 = orig_edges[i].second;
        if (scc_list[ver1] != scc_list[ver2])
        {
            scc_edges[i] = make_pair(scc_list[ver1], scc_list[ver2]);
            if (met_algo == 7)
            {
#pragma omp critical
                {
                    adj_list[scc_list[ver1]].push_back(scc_list[ver2]);
                    in_degree[scc_list[ver2]] += 1;
                }
            }
        }
    }

    // --- Method 7: BFS on condensation DAG ---
    if (met_algo == 7)
    {
        for (int i = 0; i < num_sccs; i++)
        {
            if (in_degree[i] == 0 && adj_list[i].size() != 0)
            {
                qu.push(i);
                level_ver[i] = 0;
            }
        }
        BFS(adj_list, level_ver, qu, in_degree, &max_level);
    }

    // --- Process insert edges (parallel) ---
    gettimeofday(&T_insert1, NULL);

#pragma omp parallel for
    for (int i = 0; i < (int)insert_edges.size(); i++)
    {
        int ver1 = insert_edges[i].first;
        int ver2 = insert_edges[i].second;
        if (scc_list[ver1] != scc_list[ver2])
        {
            scc_edges[orig_edges.size() + i] = make_pair(scc_list[ver1], scc_list[ver2]);
            if (met_algo == 7)
            {
                int scc1 = scc_list[ver1];
                int scc2 = scc_list[ver2];
#pragma omp atomic
                affect_level[min(level_ver[scc1], level_ver[scc2])] += 1;
#pragma omp atomic
                affect_level[max(level_ver[scc1], level_ver[scc2]) + 1] += -1;
            }
        }
        if (met_algo == 11)
        {
            new_edge_nodes[scc_list[ver1]] = 1;
        }
    }

    gettimeofday(&T_insert2, NULL);
    insert_runtime = (T_insert2.tv_sec - T_insert1.tv_sec) * 1000.0 +
                     (T_insert2.tv_usec - T_insert1.tv_usec) * 0.001;

    // --- Method 7: prefix sum on affect_level ---
    if (met_algo == 7)
    {
        parallel_prefix_sum(affect_level);
        for (int i = 0; i < (int)affect_level.size() && i <= max_level; i++)
        {
            if (affect_level[i] == 0)
            {
                unaffected_levels.push_back(i);
            }
        }
        cout << "size of unaffected_levels:" << unaffected_levels.size() << endl;
        for (int i = 0; i < (int)unaffected_levels.size() - 1; i++)
        {
            if (unaffected_levels[i + 1] - unaffected_levels[i] > 1)
                cout << "hey-hey-found" << endl;
        }
    }
}
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
        printf("  method 1: Trim1 + Global FW-BW + Trim1 + FW-BW DFS\n");        printf("  method  2: Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW DFS\n");
        printf("  method 22: Trim1 + Trim1/2 + WCC + FW-BW DFS (skip GLOBAL_BFS)\n");

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

    // GPU graph construction outputs (populated by build_incremental_graph for method 6)
    bool gpu_graph_built = false;
    vector<edge_t> h_gpu_begin;
    vector<node_t> h_gpu_node_idx;
    vector<edge_t> h_gpu_r_begin;
    vector<node_t> h_gpu_r_node_idx;
    int gpu_N = 0, gpu_M = 0;

    // ---------------------------------------------------------------
    // Load graph — 1:1 mirror of OpenMP common_main.h::main()
    // ---------------------------------------------------------------
    struct timeval T1, T2;
    string fname = graph_file;

    gm_graph G;
    gm_rt_set_num_threads(num_threads);
    gm_rt_initialize();

    gettimeofday(&T1, NULL);
    {
        vector<pair<int,int>> orig_edges;

        // ---- Static methods (0-4): load directly ----
        if (met_algo_original == 0 || met_algo_original == 1 ||
            met_algo_original == 2 || met_algo_original == 3 ||
            met_algo_original == 4 || met_algo_original == 12 ||
            met_algo_original == 22)
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

        // ---- Incremental methods (5, 6, 7, 11) — in scc_cuda_incremental_build.cpp ----
        // OpenMP: inline in common_main.h::main()
        if (met_algo_original == 5 || met_algo_original == 6 ||
            met_algo_original == 7 || met_algo_original == 11)
        {
            build_incremental_graph(G, fname, met_algo_original,
                num_sccs, good_init_pivot, insert_runtime,
                h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
                gpu_graph_built,
                h_gpu_begin, h_gpu_node_idx, h_gpu_r_begin, h_gpu_r_node_idx,
                gpu_N, gpu_M);
        }

    }
    gettimeofday(&T2, NULL);
    printf("graph loading time=%lf\n",
           (T2.tv_sec - T1.tv_sec) * 1000 + (T2.tv_usec - T1.tv_usec) * 0.001);

    int N, M;
    vector<edge_t> h_begin;
    vector<node_t> h_node_idx;
    vector<edge_t> h_r_begin;
    vector<node_t> h_r_node_idx;

    if (gpu_graph_built) {
        // GPU path: use pre-built CSR arrays from build_gpu_condensation_graph
        // Skips gm_graph::make_reverse_edges() and CSR extraction entirely.
        N = gpu_N;
        M = gpu_M;
        h_begin = std::move(h_gpu_begin);
        h_node_idx = std::move(h_gpu_node_idx);
        h_r_begin = std::move(h_gpu_r_begin);
        h_r_node_idx = std::move(h_gpu_r_node_idx);
        printf("[CUDA] Condensation graph built on GPU: N=%d, M=%d\n", N, M);
    } else {
        // Existing path via gm_graph
        gettimeofday(&T1, NULL);
        G.make_reverse_edges();
        gettimeofday(&T2, NULL);
        printf("reverse edge creation time=%lf\n",
               (T2.tv_sec - T1.tv_sec) * 1000 + (T2.tv_usec - T1.tv_usec) * 0.001);

        N = G.num_nodes();
        M = G.num_edges();

        // Extract CSR arrays from gm_graph
        h_begin.resize(N + 1);
        h_node_idx.resize(M);
        h_r_begin.resize(N + 1);
        h_r_node_idx.resize(M);
        memcpy(h_begin.data(),     G.begin,      (N + 1) * sizeof(edge_t));
        if (M > 0) {
            memcpy(h_node_idx.data(),  G.node_idx,    M * sizeof(node_t));
            memcpy(h_r_node_idx.data(), G.r_node_idx, M * sizeof(node_t));
        }
        memcpy(h_r_begin.data(),   G.r_begin,     (N + 1) * sizeof(edge_t));
    }

    printf("data=%s %d %d\n", graph_file, met_algo, num_threads);

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

    // Initialize algo timer (tracks H2D/D2H time for ALGO_TIME computation)
    algo_memcpy_init();

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
    double cuda_profile_total_ms = 0.0;
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
                met_algo, flag11, da, d_count_trim_spec, 100);

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

    } else if (met_algo == 2 || met_algo == 22) {
        // ============================================================
        // Method 2: Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW DFS
        // OpenMP: do_baseline_global_wcc_fb()
        // ============================================================
        if (met_algo == 22)
            printf("Running Method 22: Trim1 + Trim1/2 + WCC + FW-BW DFS (skip GLOBAL_BFS)\n");
        else
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
        double fb_algo_time = 0;
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
            int scc_size = 0;
            if (met_algo == 2) {
                initialize_global_fb(N);
                scc_size = do_global_fw_bw_main(
                    st, gpuG,
                    COLOR_UNASSIGNED,   // base_color = curr_color = -1
                    curr_count,          // base_count from trim_targets
                    -1,                  // good_init_pivot (-1 = not met_algo 6/11)
                    false);              // create_work_items = false
            } else {  // method 22: skip GLOBAL_BFS
                // No GLOBAL_BFS — go straight to TRIM1/2 + WCC + GPU FB
                // The SCC will be found during the FB phase.
                printf("[CUDA] Skipping GLOBAL_BFS (method 22)\n");
            }
            gettimeofday(&t_bfs, NULL);
            if (met_algo == 2)
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

        // ---------------------------------------------------------------
        // Timing: record start for each phase (also used for method 22)
        // ---------------------------------------------------------------
        gettimeofday(&t_bfs, NULL);  // GLOBAL_BFS end time (will be same as start if skipped)


            // ---------- Phase 3: TRIM1/2 (compact) — separate passes ----------
            trimmed = repeat_global_trim1_compact(st, gpuG, d_count,
                met_algo, flag11, da, d_count_trim_spec, 100);
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

                // ---------- Phase 5: FB — try GPU batch first, fall back to host ----------
                // GPU batch FB: processes all WCC components in parallel on GPU.
                // For many-SCC graphs (it-2004: 30.5M, wb-edu: 4.3M), this is much faster
                // than the host path which requires D2H/H2D transfers + CPU processing.
                // Fallback to host path if any components were too large for SMEM
                // (run_gpu_fb returns them to the work queue).
                fb_algo_time = run_gpu_fb(st, gpuG, num_threads);
                if (fb_algo_time < 0.0) fb_algo_time = 0.0;  // clamp signal value
                if (!is_work_q_empty_from_seq_context()) {
                    double host_time = start_workers_fw_bw_dfs_host(st, gpuG, num_threads);
                    fb_algo_time += host_time;
                }
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
        double t3;
        if (met_algo == 2) {
            t3 = (t_bfs.tv_sec - t_compact.tv_sec) * 1000.0 +
                 (t_bfs.tv_usec - t_compact.tv_usec) * 0.001;
        } else {
            t3 = 0.0;  // method 22: skipped GLOBAL_BFS
        }
        double t4 = (t_trim12.tv_sec - t_bfs.tv_sec) * 1000.0 +
                    (t_trim12.tv_usec - t_bfs.tv_usec) * 0.001;
        double t5 = (t_wcc.tv_sec - t_trim12.tv_sec) * 1000.0 +
                    (t_wcc.tv_usec - t_trim12.tv_usec) * 0.001;
        // FB time = CPU-only algorithm time (excludes D2H/H2D transfer overhead)
        double t6 = fb_algo_time;
        // Total = sum of per-phase algorithm times (excludes D2H/H2D transfers)
        double t_total = t1 + t2 + t3 + t4 + t5 + t6;

        printf(">>>>CUDA_PROFILE: TRIM1=%.2fms COMPACT_BUILD=%.2fms GLOBAL_BFS=%.2fms TRIM12=%.2fms WCC=%.2fms FB=%.2fms TOTAL=%.2fms\n",
               t1, t2, t3, t4, t5, t6, t_total);
        fflush(stdout);
        fprintf(stderr, "[CUDA_PROFILE_STDERR] TRIM1=%.2f COMPACT=%.2f GLOBAL_BFS=%.2f TRIM12=%.2f WCC=%.2f FB=%.2f TOTAL=%.2f\n",
                t1, t2, t3, t4, t5, t6, t_total);

        // Save CUDA_PROFILE total for ALGO_TIME computation below
        cuda_profile_total_ms = t_total;

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
    // Compute and print Algorithm Time: CUDA_PROFILE_TOTAL - H2D/D2H transfer time
    // (uses CUDA_PROFILE total, not separate gettimeofday, for consistency)
    double algo_time = cuda_profile_total_ms - g_algo_memcpy_ms;
    if (algo_time < 0) algo_time = 0;
    printf(">>>>ALGO_TIME: %.2fms (CUDA_PROFILE_TOTAL %.2fms - H2D/D2H %.2fms)\n",
           algo_time, cuda_profile_total_ms, g_algo_memcpy_ms);
    fprintf(stderr, "[ALGO_TIME] algo=%.2f profile_total=%.2f memcpy=%.2f\n",
            algo_time, cuda_profile_total_ms, g_algo_memcpy_ms);

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
    // Finalize algo timer
    algo_memcpy_finalize();

    fprintf(stderr, "[DEBUG] cleanup: cudaFree(d_count)\n");
    cudaFree(d_count);
    if (d_count_trim_spec) { fprintf(stderr, "[DEBUG] cleanup: cudaFree(d_count_trim_spec)\n"); cudaFree(d_count_trim_spec); }
    fprintf(stderr, "[DEBUG] cleanup: dynamic_arrays_free\n");
    dynamic_arrays_free(da);
    fprintf(stderr, "[DEBUG] cleanup: finalize_fb_gpu\n");
    finalize_fb_gpu();

    fprintf(stderr, "[DEBUG] cleanup: finalize_WCC\n");
    finalize_WCC();
    fprintf(stderr, "[DEBUG] cleanup: finalize_trim2\n");
    finalize_trim2();
    fprintf(stderr, "[DEBUG] cleanup: finalize_trim1\n");
    finalize_trim1();
    fprintf(stderr, "[DEBUG] cleanup: state_free\n");
    state_free(st);
    fprintf(stderr, "[DEBUG] cleanup: graph_free\n");
    graph_free(gpuG);
    fprintf(stderr, "[DEBUG] cleanup: DONE\n");

    // NOTE: gm_graph's destructor crashes on LiveJournal1 (4.8M nodes)
    // due to heap corruption from the 1.7GB of CSR array allocations.
    // Both OpenMP and CUDA binaries have this issue.
    // The SCC result is already printed — skip cleanup and exit.
    // All GPU memory was freed above; OS will reclaim the rest.
    // exit() avoids running destructors (which crash on corrupted heap)
    // while still properly flushing stdio buffers.
    exit(0);
}
