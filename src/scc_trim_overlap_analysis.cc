// ==========================================================================
// scc_trim_overlap_analysis.cc
//
// Overlap analysis: For each delete batch, compute the intersection between
// the nodes that become newly trimmable AFTER edge deletion (Δ) and the
// unique nodes that appear in the deleted edges (batch nodes, B).
//
// This distinguishes two effects:
//   Direct effect  → Δ ∩ B  : nodes whose last in/out edge was one we deleted
//   Cascade effect → Δ \ B  : nodes that lost connectivity because a
//                              neighbor (upstream/downstream) was affected
//
// IMPORTANT: Δ = T_mod \ T_full (newly trimmed). By definition, Δ has ZERO
// overlap with T_full (nodes already trimmed on full graph). So the answer
// to "how many of Δ are in the batch nodes that were already trimmed?" is
// ALWAYS 0. The meaningful comparison is:
//   - Δ ∩ B  = how many Δ nodes are batch nodes (any trim status)
//   - Δ \ B  = how many Δ nodes are NOT batch nodes (cascade effect)
//
// Process:
//   1. Read original graph
//   2. Run TRIM1 on full graph → record T_full (set of trimmed nodes)
//   3. For each batch file:
//      a. Read delete edges, collect unique batch nodes B
//      b. Build graph = (original_edges - delete_edges)
//      c. Run TRIM1 → record T_mod
//      d. Compute Δ = T_mod \ T_full  (newly trimmed after deletion)
//      e. Report |Δ|, |Δ ∩ B|, |Δ \ B|, and percentages
//
// Build:
//   g++ -O3 -I. -I../gm_graph/inc -fopenmp -std=gnu++0x \
//       -o scc_overlap_analysis scc_trim_overlap_analysis.cc \
//       -L../gm_graph/lib -lgmgraph -lgomp
//
// NOTE: On very large graphs (wb-edu, it-2004), gm_graph destructor may
//       trigger "double free or corruption" during cleanup. If so, uncomment
//       the exit(0) call at the end of main().
//
// Usage:
//   ./scc_overlap_analysis <refined_edges.txt> <batch1> [batch2 ...]
//
// Example:
//   ./scc_overlap_analysis /data/soc-Pokec/refined_edges.txt \
//       /data/soc-Pokec/0.01_delete_edges.txt \
//       /data/soc-Pokec/0.03_delete_edges.txt
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
// Global state
// ==========================================================================
static int* G_Color = NULL;
static int* G_SCC   = NULL;
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
// Read delete edges and unique batch nodes
// ==========================================================================
struct BatchData {
    std::set<std::pair<int,int>> del_edges;  // edges to delete
    std::vector<bool> is_batch_node;          // boolean mask for batch nodes
    std::vector<int>  batch_node_list;        // list of unique batch node IDs
    int edge_count;
};

static BatchData read_batch(const std::string& filename, int num_nodes)
{
    BatchData bd;
    bd.is_batch_node.assign(num_nodes, false);
    bd.edge_count = 0;

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

        int v1 = std::stoi(line.substr(0, p1)) - 1;
        int v2 = std::stoi(line.substr(p1 + 1, p2 - p1 - 1)) - 1;

        bd.del_edges.insert({v1, v2});
        bd.edge_count++;

        if (!bd.is_batch_node[v1]) {
            bd.is_batch_node[v1] = true;
            bd.batch_node_list.push_back(v1);
        }
        if (!bd.is_batch_node[v2]) {
            bd.is_batch_node[v2] = true;
            bd.batch_node_list.push_back(v2);
        }
    }
    fin.close();
    return bd;
}

// ==========================================================================
// TRIM1 — one pass over all nodes (adapted from scc_trim1.cc)
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
// Record trimmed nodes into a boolean mask
// ==========================================================================
static void record_trimmed_set(std::vector<bool>& trimmed)
{
    trimmed.assign(G_num_nodes, false);
    #pragma omp parallel for
    for (int i = 0; i < G_num_nodes; i++) {
        if (G_Color[i] == SCC_FOUND) {
            trimmed[i] = true;
        }
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
        printf("For each batch, computes node-level overlap between:\n");
        printf("  1. Δ = newly trimmed after deletion (T_mod \\ T_full)\n");
        printf("  2. B = unique nodes appearing in deleted edges\n");
        printf("\n");
        printf("This quantifies DIRECT vs CASCADE effect of edge deletions on TRIM1.\n");
        return 1;
    }

    const char* graph_file = argv[1];
    int num_batches = argc - 2;
    char** batch_files = argv + 2;

    int num_threads = omp_get_max_threads();

    printf("================================================================================\n");
    printf("  TRIM1 OVERLAP ANALYSIS (Direct vs Cascade Effect)\n");
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

    // ---- Step 2: Build full graph and run TRIM1 ----
    printf("[2] Building full graph and running TRIM1 (baseline)...\n");
    gm_graph G_full = build_graph(num_nodes, orig_edges);
    G_num_nodes = G_full.num_nodes();

    G_Color = new int[G_num_nodes];
    G_SCC   = new int[G_num_nodes];

    init_colors();
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    int trimmed_full = repeat_global_trim1(G_full);
    gettimeofday(&t1, NULL);
    double trim_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_usec - t0.tv_usec) * 0.001;
    printf("    Full graph nodes trimmed: %d (%.2f ms)\n\n", trimmed_full, trim_ms);

    // Record T_full
    std::vector<bool> T_full;
    record_trimmed_set(T_full);
    printf("    T_full set recorded (%zu nodes)\n\n", T_full.size());

    // ---- Step 3: For each batch, compute overlaps ----
    printf("================================================================================\n");
    printf("  KEY SET RELATIONSHIPS (important for interpreting results):\n");
    printf("\n");
    printf("  Let T_full = nodes trimmed by TRIM1 on the FULL graph\n");
    printf("  Let T_mod  = nodes trimmed by TRIM1 on the MODIFIED graph (after deletion)\n");
    printf("  Let B      = unique nodes appearing in the deleted edges (batch nodes)\n");
    printf("  Let Δ      = T_mod \\ T_full (nodes newly trimmed AFTER deletion)\n");
    printf("\n");
    printf("  By definition: Δ ∩ T_full = ∅ always. So the answer to\n");
    printf("  'how many Δ are batch nodes already trimmed?' is ALWAYS 0.\n");
    printf("\n");
    printf("  The MEANINGFUL questions are:\n");
    printf("    1. Δ ∩ B  = how many newly trimmed nodes are batch nodes? (DIRECT effect)\n");
    printf("    2. Δ \\ B  = how many newly trimmed nodes are NOT batch nodes? (CASCADE)\n");
    printf("    3. Of batch nodes NOT already trimmed, how many got newly trimmed?\n");
    printf("================================================================================\n\n");

    for (int i = 0; i < num_batches; i++) {
        std::string full_path = batch_files[i];

        // Extract filename for display
        size_t slash = full_path.rfind('/');
        std::string fname = (slash == std::string::npos)
                            ? full_path
                            : full_path.substr(slash + 1);

        printf("  --- Batch %d/%d: %s ---\n", i + 1, num_batches, fname.c_str());

        // Read batch data
        BatchData bd = read_batch(full_path, num_nodes);

        // Build graph without deleted edges
        gm_graph G_mod = build_graph(num_nodes, orig_edges, bd.del_edges);

        // Run TRIM1 on modified graph
        init_colors();
        int trimmed_mod = repeat_global_trim1(G_mod);

        // Record T_mod
        std::vector<bool> T_mod;
        record_trimmed_set(T_mod);

        // Compute Δ = T_mod \ T_full
        std::vector<int> delta_nodes;
        int delta_count = 0;
        int delta_and_batch = 0;  // |Δ ∩ B|
        int delta_not_batch = 0;  // |Δ \ B|

        for (int n = 0; n < num_nodes; n++) {
            if (T_mod[n] && !T_full[n]) {
                delta_count++;
                delta_nodes.push_back(n);
                if (bd.is_batch_node[n]) {
                    delta_and_batch++;
                } else {
                    delta_not_batch++;
                }
            }
        }

        // Also compute: batch nodes NOT trimmed on full graph, how many got newly trimmed?
        int batch_not_trimmed_full = 0;
        int batch_newly_trimmed = 0;
        for (int n : bd.batch_node_list) {
            if (!T_full[n]) {
                batch_not_trimmed_full++;
                if (T_mod[n]) {
                    batch_newly_trimmed++;
                }
            }
        }

        // Compute: batch nodes already trimmed on full graph (= Experiment 1)
        int batch_already_trimmed = 0;
        for (int n : bd.batch_node_list) {
            if (T_full[n]) batch_already_trimmed++;
        }

        double direct_pct = (delta_count > 0)
            ? (100.0 * delta_and_batch / delta_count) : 0.0;
        double cascade_pct = (delta_count > 0)
            ? (100.0 * delta_not_batch / delta_count) : 0.0;
        double batch_already_pct = (bd.batch_node_list.size() > 0)
            ? (100.0 * batch_already_trimmed / bd.batch_node_list.size()) : 0.0;
        double batch_newly_pct = (batch_not_trimmed_full > 0)
            ? (100.0 * batch_newly_trimmed / batch_not_trimmed_full) : 0.0;

        printf("\n");
        printf("  Results for %s:\n", fname.c_str());
        printf("  %-30s %10d\n", "Edges deleted:", bd.edge_count);
        printf("  %-30s %10d\n", "Unique batch nodes (B):", (int)bd.batch_node_list.size());
        printf("  %-30s %10d (%.1f%% of B)\n",
               "Batch already trimmed (T_full∩B):",
               batch_already_trimmed,
               batch_already_pct);
        if (batch_not_trimmed_full > 0) {
            printf("  %-30s %10d (%.1f%% of B\\T_full)\n",
                   "Batch newly trimmed (Δ∩B):",
                   batch_newly_trimmed,
                   batch_newly_pct);
        } else {
            printf("  %-30s %10d (all batch nodes already trimmed)\n",
                   "Batch newly trimmed (Δ∩B):", batch_newly_trimmed);
        }
        printf("  %-30s %10d\n", "Total trimmed on full graph:", trimmed_full);
        printf("  %-30s %10d\n", "Total trimmed after deletion:", trimmed_mod);
        printf("  --------------------------------------------\n");
        printf("  %-30s %10d   (Δ = T_mod \\ T_full)\n", "Newly trimmed (Δ):", delta_count);
        printf("  %-30s %10d   (Δ ∩ B: direct effect)\n",
               "  |-- Batch nodes (direct):", delta_and_batch);
        printf("  %-30s %10d   (Δ \\ B: cascade effect)\n",
               "  +-- Non-batch (cascade):", delta_not_batch);
        printf("  %-30s %10.1f%%\n", "Direct % of Δ:", direct_pct);
        printf("  %-30s %10.1f%%\n", "Cascade % of Δ:", cascade_pct);

#if 0   // Enable for detailed node dump
        if (delta_and_batch > 0) {
            printf("\n  Direct-effect nodes (Δ ∩ B):\n  ");
            int count = 0;
            for (int n : delta_nodes) {
                if (bd.is_batch_node[n]) {
                    printf("%d ", n);
                    if (++count % 20 == 0) printf("\n  ");
                }
            }
            printf("\n");
        }
#endif

        printf("\n");
    }

    printf("================================================================================\n");
    printf("\nExperiment complete!\n");

    // Cleanup
    delete[] G_Color;
    delete[] G_SCC;

    // On very large graphs (wb-edu, it-2004), gm_graph destructor may cause
    // "double free or corruption". Uncomment exit(0) to skip cleanup.
    // fflush(stdout); fflush(stderr); exit(0);
    return 0;
}
