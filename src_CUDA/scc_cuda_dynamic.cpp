// ======================================================================
// scc_cuda_dynamic.cpp
//
// 1:1 mirror of scc_incremental.cc + common_main.h helper functions.
// Host-side graph construction helpers for incremental edge insertion.
//
// Compiled with g++ (host-only, uses gm_graph which is not CUDA-compatible).
// Function names match the CPU originals exactly — no "cuda_" prefix.
// ======================================================================

#include "scc_cuda.h"
#include "gm.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <omp.h>

using namespace std;

// ======================================================================
// insert_idea1() — 1:1 mirror of scc_incremental.cc
// ======================================================================
void insert_idea1(gm_graph &G, vector<pair<int,int> > orig_edges, vector<pair<int,int> > insert_edges)
{
    for(int i=0;i<(int)orig_edges.size();i++)
    {
        G.add_edge(orig_edges[i].first,orig_edges[i].second);
    }
    for(int i=0;i<(int)insert_edges.size();i++)
    {
        G.add_edge(insert_edges[i].first,insert_edges[i].second);
    }
}

// ======================================================================
// insert_idea2() — 1:1 mirror of scc_incremental.cc
// ======================================================================
void insert_idea2(gm_graph &G, vector<pair<int,int> > scc_edges)
{
    for(int i=0;i<(int)scc_edges.size();i++)
    {
        int ver1=scc_edges[i].first;
        int ver2=scc_edges[i].second;
        if(ver1+ver2 !=0)
            G.add_edge(ver1,ver2);
    }
}

// ======================================================================
// Fast integer parser: parse next int from string, advance pointer
// Returns -1 on failure (end of buffer)
// ======================================================================
static inline int fast_parse_int(const char*& p, const char* end) {
    // Skip non-digit characters (spaces, tabs)
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
// read_file() — FAST version
//
// Replaces the original getline+stringstream+stoi implementation.
// Uses fread to load entire file, then manual integer parsing.
// Typically 5-10x faster for large graphs (117M edges).
//
// 1:1 functional mirror of main_t::read_file() in common_main.h
// ======================================================================
int read_file(string filename, vector<pair<int, int>>& edges_list)
{
    // Open file
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", filename.c_str());
        return 0;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) {
        fclose(fp);
        return 0;
    }

    // Allocate buffer and read entire file at once
    char* buf = new char[fsize + 1];
    size_t bytes_read = fread(buf, 1, fsize, fp);
    fclose(fp);
    buf[bytes_read] = '\0';

    // Estimate edge count and pre-allocate (count newlines)
    int est_lines = count_lines(buf, bytes_read);
    if (est_lines < 0) est_lines = 0;
    edges_list.reserve(est_lines);

    // Parse all integers in one pass
    const char* p = buf;
    const char* end = buf + bytes_read;
    int max_vertex = 0;

    while (p < end) {
        int v1 = fast_parse_int(p, end);
        if (v1 < 0) break;
        int v2 = fast_parse_int(p, end);
        if (v2 < 0) break;

        // Skip to next line
        while (p < end && *p != '\n') p++;
        if (p < end) p++;  // skip '\n'

        // Store edge (0-indexed, matching original)
        edges_list.push_back(make_pair(v1 - 1, v2 - 1));
        if (v1 > max_vertex) max_vertex = v1;
        if (v2 > max_vertex) max_vertex = v2;
    }

    delete[] buf;
    return max_vertex;
}

// ======================================================================
// read_file1()
//
// 1:1 mirror of main_t::read_file1() in common_main.h
// ======================================================================
int read_file1(string filename, vector<int>& scc_list_out, int num_vertices)
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
        while (getline(ss, token, ' '))
        {
            tokens.push_back(token);
        }
        scc_list_out[stoi(tokens[0])] = stoi(tokens[1]);
        max_vertex = max(max_vertex, stoi(tokens[1]) + 1);
    }

    inputFile.close();

    return max_vertex;
}

// ======================================================================
// BFS()
//
// 1:1 mirror of main_t::BFS() in common_main.h
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
// parallel_prefix_sum()
//
// 1:1 mirror of main_t::parallel_prefix_sum() in common_main.h
// ======================================================================
void parallel_prefix_sum(std::vector<int>& a)
{
    int N = (int)a.size();
    if (N == 0)
        return;

    int num_threads = 0;
#pragma omp parallel
    {
#pragma omp master
        {
            num_threads = omp_get_num_threads();
        }
    }

    std::vector<float> partial_sums(num_threads + 1, 0.0f);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float local_sum = 0.0f;

#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            local_sum += a[i];
            a[i] = local_sum;
        }
        partial_sums[tid + 1] = local_sum;
    }

    for (int i = 1; i <= num_threads; ++i)
    {
        partial_sums[i] += partial_sums[i - 1];
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
    {
        int tid = omp_get_thread_num();
        a[i] += partial_sums[tid];
    }
}

// ======================================================================
// create_scc_edges()
//
// 1:1 mirror of main_t::create_scc_edges() in common_main.h
// All vector parameters (scc_list, level_ver, affect_level, new_edge_nodes)
// match the CPU globals passed by reference.
// ======================================================================
void create_scc_edges(vector<pair<int, int>> orig_edges,
                      vector<pair<int, int>> insert_edges,
                      vector<pair<int, int>>& scc_edges,
                      int num_vertices, int num_sccs,
                      int met_algo,
                      vector<int>& scc_list,
                      vector<int>& level_ver,
                      vector<int>& affect_level,
                      vector<int>& new_edge_nodes,
                      double& insert_runtime)
{
    unordered_map<string, int> ump;
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
