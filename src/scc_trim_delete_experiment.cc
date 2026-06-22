// ==========================================================================
// scc_trim_delete_experiment.cc
//
// Experiment: For each delete batch, measure how many nodes TRIM1 can trim
// after removing those edges from the original graph.
//
// Procedure:
//   1. Read original graph (refined_edges.txt)
//   2. Run TRIM1 on full graph → baseline trimmed count
//   3. For each delete batch file:
//      a. Read the edges to delete
//      b. Build graph = (original_edges - batch_edges)
//      c. Initialize colors
//      d. Run TRIM1
//      e. Print trimmed node count
//
// Build:
//   cd ~/DynamicGraphs_SCC/src
//   g++ -O3 -I. -I../gm_graph/inc -fopenmp -std=gnu++0x \
//       -o scc_trim_experiment scc_trim_delete_experiment.cc \
//       -L../gm_graph/lib -lgmgraph -lgomp
//
// NOTE: On very large graphs (wb-edu, it-2004), gm_graph destructor may
//       trigger "double free or corruption" during cleanup. If so, uncomment
//       the exit(0) call at the end of main(). See DOCUMENTATION.md for details.
//
// Usage:
//   ./scc_trim_experiment <refined_edges.txt> <batch1> [batch2 ...]
//
// Example:
//   ./scc_trim_experiment /hdd/thej_par_scc_datasets/soc-Pokec/refined_edges.txt \
//       /hdd/thej_par_scc_datasets/soc-Pokec/0.01_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/soc-Pokec/0.03_delete_edges.txt
//
// Author: Auto-generated for TRIM1 deletion sensitivity analysis
// ==========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <set>
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

        // Parse two integers (space or tab separated)
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
// Read a delete batch file into a set for O(1) lookup
// ==========================================================================
static std::set<std::pair<int,int>> read_delete_set(const std::string& filename)
{
    std::set<std::pair<int,int>> del_set;
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

        del_set.insert({v1 - 1, v2 - 1});  // 1-indexed → 0-indexed
    }
    fin.close();
    return del_set;
}

// ==========================================================================
// do_global_trim1 — one pass of TRIM1 over all nodes
//
// Identifies nodes with 0 out-degree or 0 in-degree among the remaining
// (non-SCC_FOUND) nodes. Marks them as SCC_FOUND (self-loop SCC).
//
// Returns: number of nodes trimmed in this pass
// ==========================================================================
static int do_global_trim1(const gm_graph& G)
{
    int count = 0;

    #pragma omp parallel
    {
        int local_count = 0;

        #pragma omp for nowait schedule(dynamic, 512)
        for (int n = 0; n < G_num_nodes; n++) {
            if (G_Color[n] == SCC_FOUND) continue;  // already trimmed

            // --- Check out-degree in remaining graph ---
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

            // --- Check in-degree in remaining graph ---
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
//
// Stops when a single pass trims ≤ TRIM_STOP nodes.
// Returns: total number of nodes trimmed
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
// run_single_experiment — Build graph, init colors, run TRIM1
//
// Parameters:
//   orig_edges   — all edges from the original graph
//   num_nodes    — number of vertices
//   del_edges    — set of edges to EXCLUDE (empty = full graph)
//   label        — display label for the output
//
// Returns: number of nodes trimmed by TRIM1
// ==========================================================================
static int run_single_experiment(const std::vector<std::pair<int,int>>& orig_edges,
                                  int num_nodes,
                                  const std::set<std::pair<int,int>>& del_edges,
                                  const std::string& label)
{
    struct timeval t0, t1;

    // ---- Build graph from (orig_edges - del_edges) ----
    gettimeofday(&t0, NULL);

    gm_graph G;
    for (int i = 0; i < num_nodes; i++)
        G.add_node();

    int edge_count = 0;
    for (const auto& e : orig_edges) {
        if (del_edges.find(e) == del_edges.end()) {
            G.add_edge(e.first, e.second);
            edge_count++;
        }
    }

    G.make_reverse_edges();

    gettimeofday(&t1, NULL);
    double build_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                      (t1.tv_usec - t0.tv_usec) * 0.001;

    // ---- Initialize color arrays ----
    G_num_nodes = G.num_nodes();
    G_Color = new int[G_num_nodes];
    G_SCC   = new int[G_num_nodes];

    #pragma omp parallel for
    for (int i = 0; i < G_num_nodes; i++) {
        G_Color[i] = COLOR_UNASSIGNED;  // -1
        G_SCC[i]   = NIL_NODE;          // -1
    }

    // ---- Run TRIM1 ----
    gettimeofday(&t0, NULL);
    int trimmed = repeat_global_trim1(G);
    gettimeofday(&t1, NULL);
    double trim_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_usec - t0.tv_usec) * 0.001;

    // ---- Print result ----
    int remaining = G_num_nodes - trimmed;
    printf("  %-40s | edges=%-10d | trimmed=%-8d | remaining=%-8d | build=%-8.2fms | trim=%-7.2fms\n",
           label.c_str(), edge_count, trimmed, remaining, build_ms, trim_ms);

    // ---- Cleanup ----
    delete[] G_Color;  G_Color = NULL;
    delete[] G_SCC;    G_SCC   = NULL;

    return trimmed;
}

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <refined_edges.txt> <delete_batch_1> [delete_batch_2 ...]\n", argv[0]);
        printf("\n");
        printf("For EACH delete batch file:\n");
        printf("  1. Build graph = (original_edges - batch_edges)\n");
        printf("  2. Run TRIM1 (iterative)\n");
        printf("  3. Count and report trimmed nodes\n");
        printf("\n");
        printf("First run: TRIM1 on the FULL original graph (baseline, no deletions).\n");
        return 1;
    }

    const char* graph_file = argv[1];
    int num_batches = argc - 2;
    char** batch_files = argv + 2;

    // ---- Set OpenMP threads ----
    int num_threads = omp_get_max_threads();
    printf("================================================================================\n");
    printf("  TRIM1 DELETION SENSITIVITY EXPERIMENT\n");
    printf("================================================================================\n");
    printf("  Graph file : %s\n", graph_file);
    printf("  Batches    : %d\n", num_batches);
    printf("  OpenMP threads: %d\n", num_threads);
    printf("================================================================================\n\n");

    // ---- Read original graph edges ----
    printf("[Step 1] Reading original graph...\n");
    std::vector<std::pair<int,int>> orig_edges;
    int num_nodes = read_edge_file(graph_file, orig_edges);
    printf("  Nodes: %d, Edges: %zu\n\n", num_nodes, orig_edges.size());

    // ---- Print table header ----
    printf("================================================================================\n");
    printf("  %-40s | %-11s | %-9s | %-10s | %-9s | %-8s\n",
           "Dataset", "Edges", "Trimmed", "Remaining", "Build", "TRIM1");
    printf("  %-40s | %-11s | %-9s | %-10s | %-9s | %-8s\n",
           "", "", "nodes", "nodes", "time", "time");
    printf("================================================================================\n");

    // ---- Baseline: full graph, no deletions ----
    std::set<std::pair<int,int>> empty_set;
    int baseline_trimmed = run_single_experiment(orig_edges, num_nodes, empty_set,
                                                  "Full graph (no deletions)");

    // ---- For each delete batch ----
    for (int i = 0; i < num_batches; i++) {
        std::string full_path = batch_files[i];

        // Extract just the filename for display
        size_t slash = full_path.rfind('/');
        std::string fname = (slash == std::string::npos)
                            ? full_path
                            : full_path.substr(slash + 1);

        // Read delete edges
        std::set<std::pair<int,int>> del_edges = read_delete_set(full_path);
        printf("  --- Batch %d/%d: %s (deleting %zu edges) ---\n",
               i + 1, num_batches, fname.c_str(), del_edges.size());

        int trimmed = run_single_experiment(orig_edges, num_nodes, del_edges,
                                             "Original - " + fname);
    }

    printf("================================================================================\n");
    printf("\nExperiment complete!\n");
    return 0;
}
