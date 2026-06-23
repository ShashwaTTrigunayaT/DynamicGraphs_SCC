// ==========================================================================
// scc_trim_batch_node_check.cc
//
// Experiment: For each delete batch, count how many of its unique nodes
// (from source and destination of each edge) are trimmed by TRIM1 on the
// MODIFIED graph (original graph with batch edges removed).
//
// This is the updated version: Experiment 1 now runs on the same modified
// graph that Experiment 2 uses, so results can be directly compared.
//
// Process:
//   1. Read original graph edges
//   2. For each batch file:
//      a. Read all edges, collect unique node IDs (B)
//      b. Build graph = (original_edges - batch_edges)
//      c. Run TRIM1 on modified graph → T_mod
//      d. Count how many batch nodes are in T_mod (T_mod ∩ B)
//      e. Report
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
//   ./scc_batch_node_check /hdd/thej_par_scc_datasets/ljournal-2008/refined_edges.txt \
//       /hdd/thej_par_scc_datasets/ljournal-2008/0.01_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/ljournal-2008/0.03_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/ljournal-2008/0.05_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/ljournal-2008/0.07_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/ljournal-2008/0.1_delete_edges.txt \
//       /hdd/thej_par_scc_datasets/ljournal-2008/0.15_delete_edges.txt
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
// Read a delete batch file: return unique node IDs + edges to delete
// ==========================================================================
static std::pair<std::unordered_set<int>, std::vector<std::pair<int,int>>>
read_batch(const std::string& filename)
{
    std::unordered_set<int> nodes;
    std::vector<std::pair<int,int>> del_edges;
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
        del_edges.push_back({v1, v2});
    }
    fin.close();
    return {nodes, del_edges};
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
// Build graph from edges (optionally excluding a set)
// ==========================================================================
static gm_graph build_graph(int num_nodes,
                            const std::vector<std::pair<int,int>>& orig_edges,
                            const std::set<std::pair<int,int>>& exclude = {})
{
    gm_graph G;
    for (int i = 0; i < num_nodes; i++)
        G.add_node();

    for (const auto& e : orig_edges) {
        if (exclude.find(e) == exclude.end()) {
            G.add_edge(e.first, e.second);
        }
    }

    G.make_reverse_edges();
    return G;
}

// ==========================================================================
// Initialize color arrays
// ==========================================================================
static void init_colors()
{
    #pragma omp parallel for
    for (int i = 0; i < G_num_nodes; i++) {
        G_Color[i] = COLOR_UNASSIGNED;
        G_SCC[i]   = NIL_NODE;
    }
}

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <refined_edges.txt> <delete_batch_1> [delete_batch_2 ...]\n", argv[0]);
        printf("\n");
        printf("For each batch, builds graph = (original - deleted edges),\n");
        printf("runs TRIM1 on the MODIFIED graph, and counts how many batch\n");
        printf("nodes (unique nodes from deleted edges) are trimmed.\n");
        return 1;
    }

    const char* graph_file = argv[1];
    int num_batches = argc - 2;
    char** batch_files = argv + 2;

    int num_threads = omp_get_max_threads();

    printf("========================================================================\n");
    printf("  TRIM1 BATCH NODE CHECK (on MODIFIED graph)\n");
    printf("========================================================================\n");
    printf("  Graph file : %s\n", graph_file);
    printf("  Batches    : %d\n", num_batches);
    printf("  OpenMP threads: %d\n", num_threads);
    printf("\n");
    printf("  For each batch, builds graph = (original - deleted edges),\n");
    printf("  runs TRIM1 on the MODIFIED graph, and reports how many of the\n");
    printf("  unique nodes in the deleted edges are trimmed (T_mod ∩ B).\n");
    printf("========================================================================\n\n");

    // ---- Step 1: Read original graph edges ----
    printf("[1] Reading original graph...\n");
    std::vector<std::pair<int,int>> orig_edges;
    int num_nodes = read_edge_file(graph_file, orig_edges);
    printf("    Nodes: %d, Edges: %zu\n\n", num_nodes, orig_edges.size());

    // Allocate global state (reused across batches)
    G_num_nodes = num_nodes;
    G_Color = new int[G_num_nodes];
    G_SCC   = new int[G_num_nodes];

    // ---- Step 2: For each batch, build modified graph and run TRIM1 ----
    printf("========================================================================\n");
    printf("  Batch                                     | Edges in  | Nodes in  | Batch nodes\n");
    printf("                                            | batch     | batch     | trimmed (on\n");
    printf("                                            |           |           | modified graph)\n");
    printf("========================================================================\n");

    for (int i = 0; i < num_batches; i++) {
        std::string full_path = batch_files[i];

        // Extract filename for display
        size_t slash = full_path.rfind('/');
        std::string fname = (slash == std::string::npos)
                            ? full_path
                            : full_path.substr(slash + 1);

        printf("\n  --- Batch %d/%d: %s ---\n", i + 1, num_batches, fname.c_str());

        // Read batch: unique nodes + edges to delete
        auto [batch_nodes, del_edges] = read_batch(full_path);
        int edge_count = (int)del_edges.size();

        // Build set of edges to exclude
        std::set<std::pair<int,int>> exclude_set(del_edges.begin(), del_edges.end());

        // Build graph = (original - deleted edges) and run TRIM1
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        gm_graph G_mod = build_graph(num_nodes, orig_edges, exclude_set);
        init_colors();
        int trimmed_mod = repeat_global_trim1(G_mod);

        gettimeofday(&t1, NULL);
        double time_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                         (t1.tv_usec - t0.tv_usec) * 0.001;

        // Count how many batch nodes are trimmed on the modified graph (T_mod ∩ B)
        int trimmed_in_batch = 0;
        for (int n : batch_nodes) {
            if (n < G_num_nodes && G_Color[n] == SCC_FOUND) {
                trimmed_in_batch++;
            }
        }

        printf("  %-42s | %-10d | %-10zu | %-12d\n",
               fname.c_str(), edge_count, batch_nodes.size(), trimmed_in_batch);
        printf("  -> Total trimmed on modified graph: %d  (%.2f ms)\n", trimmed_mod, time_ms);
    }

    printf("========================================================================\n");
    printf("\nExperiment complete!\n");

    // Cleanup
    delete[] G_Color;
    delete[] G_SCC;

    // On very large graphs (wb-edu, it-2004), gm_graph destructor may cause
    // "double free or corruption". Uncomment exit(0) to skip cleanup.
    // fflush(stdout); fflush(stderr); exit(0);
    return 0;
}
