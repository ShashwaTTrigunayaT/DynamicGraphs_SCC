// ==========================================================================
// scc_trim_batch_node_check.cc
//
// Experiment: For each delete batch, count how many of its unique nodes
// (from source and destination of each edge) are trimmed by TRIM1 on the
// full original graph (no deletions).
//
// Process:
//   1. Read original graph
//   2. Run TRIM1 on full graph → record which nodes get trimmed
//   3. For each batch file:
//      a. Read all edges, collect unique node IDs
//      b. Count how many of those unique nodes were trimmed by TRIM1
//      c. Report
//
// Build:
//   g++ -O3 -I. -I../gm_graph/inc -fopenmp -std=gnu++0x \
//       -o scc_batch_node_check scc_trim_batch_node_check.cc \
//       -L../gm_graph/lib -lgmgraph -lgomp
//
// Usage:
//   ./scc_batch_node_check <refined_edges.txt> <batch1> [batch2 ...]
//
// Example:
//   ./scc_batch_node_check /hdd/thej_par_scc_datasets/soc-Pokec/refined_edges.txt \
//       /hdd/thej_par_scc_datasets/soc-Pokec/0.01_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/soc-Pokec/0.03_delete_edges.txt
// ==========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <omp.h>

#include "gm.h"

// ==========================================================================
// Constants
// ==========================================================================
#define SCC_FOUND        (-2)
#define COLOR_UNASSIGNED (-1)
#define NIL_NODE         (-1)

// ==========================================================================
// Global state (needed by TRIM1 logic)
// ==========================================================================
static int* G_Color = NULL;   // node colors: -1 = unassigned, -2 = SCC found
static int* G_SCC   = NULL;   // SCC root for each node
static int  G_num_nodes = 0;

// ==========================================================================
// Read edge list from file (1-indexed, space-separated)
// Returns: max node ID (+1 = number of vertices)
// ==========================================================================
static int read_edge_file(const std::string& filename,
                          std::vector<std::pair<int,int>>& edges)
{
    int max_vertex = 0;
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename.c_str());
        exit(1);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;

        size_t p1 = line.find_first_of(" \t");
        if (p1 == std::string::npos) continue;
        size_t p2 = line.find_first_of(" \t", p1 + 1);
        if (p2 == std::string::npos) p2 = line.size();

        int v1 = std::stoi(line.substr(0, p1));
        int v2 = std::stoi(line.substr(p1 + 1, p2 - p1 - 1));

        edges.push_back({v1 - 1, v2 - 1});  // 1-indexed → 0-indexed
        max_vertex = std::max(max_vertex, std::max(v1, v2));
    }
    fin.close();
    return max_vertex;
}

// ==========================================================================
// Read a delete batch file: return unique node IDs + edge count
// ==========================================================================
static std::pair<std::unordered_set<int>, int>
read_batch_nodes(const std::string& filename)
{
    std::unordered_set<int> nodes;
    int edge_count = 0;
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename.c_str());
        exit(1);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;

        size_t p1 = line.find_first_of(" \t");
        if (p1 == std::string::npos) continue;
        size_t p2 = line.find_first_of(" \t", p1 + 1);
        if (p2 == std::string::npos) p2 = line.size();

        int v1 = std::stoi(line.substr(0, p1)) - 1;  // 1-indexed → 0-indexed
        int v2 = std::stoi(line.substr(p1 + 1, p2 - p1 - 1)) - 1;

        nodes.insert(v1);
        nodes.insert(v2);
        edge_count++;
    }
    fin.close();
    return {nodes, edge_count};
}

// ==========================================================================
// do_global_trim1 — one pass of TRIM1 over all nodes
// ==========================================================================
static int do_global_trim1(const gm_graph& G)
{
    int count = 0;

    #pragma omp parallel
    {
        int local_count = 0;

        #pragma omp for nowait schedule(dynamic, 512)
        for (int n = 0; n < G_num_nodes; n++) {
            if (G_Color[n] == SCC_FOUND) continue;

            bool has_out = false;
            for (edge_t k_idx = G.begin[n]; k_idx < G.begin[n + 1]; k_idx++) {
                node_t k = G.node_idx[k_idx];
                if (k == n) continue;
                if (G_Color[k] != SCC_FOUND) {
                    has_out = true;
                    break;
                }
            }

            if (!has_out) {
                G_SCC[n]   = n;
                G_Color[n] = SCC_FOUND;
                local_count++;
                continue;
            }

            bool has_in = false;
            for (edge_t k_idx = G.r_begin[n]; k_idx < G.r_begin[n + 1]; k_idx++) {
                node_t k = G.r_node_idx[k_idx];
                if (k == n) continue;
                if (G_Color[k] != SCC_FOUND) {
                    has_in = true;
                    break;
                }
            }

            if (!has_in) {
                G_SCC[n]   = n;
                G_Color[n] = SCC_FOUND;
                local_count++;
            }
        }

        #pragma omp atomic
        count += local_count;
    }

    return count;
}

// ==========================================================================
// repeat_global_trim1 — Iteratively run TRIM1 until convergence
// ==========================================================================
static int repeat_global_trim1(const gm_graph& G, int TRIM_STOP = 100)
{
    int total_count = 0;
    int count = 0;

    do {
        count = do_global_trim1(G);
        total_count += count;
    } while (count > TRIM_STOP);

    return total_count;
}

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <refined_edges.txt> <delete_batch_1> [delete_batch_2 ...]\n", argv[0]);
        printf("\n");
        printf("For each batch, collects unique nodes from its edges and counts\n");
        printf("how many of them were trimmed by TRIM1 on the full original graph.\n");
        return 1;
    }

    const char* graph_file = argv[1];
    int num_batches = argc - 2;
    char** batch_files = argv + 2;

    int num_threads = omp_get_max_threads();

    printf("================================================================================\n");
    printf("  TRIM1 BATCH NODE CHECK EXPERIMENT\n");
    printf("================================================================================\n");
    printf("  Graph file : %s\n", graph_file);
    printf("  Batches    : %d\n", num_batches);
    printf("  OpenMP threads: %d\n", num_threads);
    printf("================================================================================\n\n");

    // ---- Step 1: Read original graph ----
    printf("[1] Reading original graph...\n");
    std::vector<std::pair<int,int>> orig_edges;
    int num_nodes = read_edge_file(graph_file, orig_edges);
    printf("    Nodes: %d, Edges: %zu\n\n", num_nodes, orig_edges.size());

    // ---- Step 2: Build graph and run TRIM1 ----
    printf("[2] Building graph and running TRIM1...\n");
    gm_graph G;
    for (int i = 0; i < num_nodes; i++)
        G.add_node();
    for (const auto& e : orig_edges)
        G.add_edge(e.first, e.second);
    G.make_reverse_edges();

    G_num_nodes = G.num_nodes();
    G_Color = new int[G_num_nodes];
    G_SCC   = new int[G_num_nodes];

    #pragma omp parallel for
    for (int i = 0; i < G_num_nodes; i++) {
        G_Color[i] = COLOR_UNASSIGNED;
        G_SCC[i]   = NIL_NODE;
    }

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    int total_trimmed = repeat_global_trim1(G);
    gettimeofday(&t1, NULL);
    double trim_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_usec - t0.tv_usec) * 0.001;

    printf("    Full graph nodes trimmed: %d (%.2f ms)\n\n", total_trimmed, trim_ms);

    // ---- Step 3: For each batch, check how many of its nodes were trimmed ----
    printf("================================================================================\n");
    printf("  Batch                                     | Edges in  | Nodes in  | Batch nodes\n");
    printf("                                            | batch     | batch     | trimmed\n");
    printf("================================================================================\n");

    for (int i = 0; i < num_batches; i++) {
        std::string full_path = batch_files[i];

        // Extract filename for display
        size_t slash = full_path.rfind('/');
        std::string fname = (slash == std::string::npos)
                            ? full_path
                            : full_path.substr(slash + 1);

        // Read unique nodes and edge count in one pass
        auto [batch_nodes, edge_count] = read_batch_nodes(full_path);

        // Count how many were trimmed
        int trimmed_in_batch = 0;
        for (int n : batch_nodes) {
            if (n < G_num_nodes && G_Color[n] == SCC_FOUND) {
                trimmed_in_batch++;
            }
        }

        printf("  %-42s | %-10d | %-10zu | %-12d\n",
               fname.c_str(), edge_count, batch_nodes.size(), trimmed_in_batch);
    }

    printf("================================================================================\n");
    printf("\nExperiment complete!\n");

    // Cleanup
    delete[] G_Color;
    delete[] G_SCC;

    return 0;
}
