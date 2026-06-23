// ==========================================================================
// scc_trim_experiment.h
//
// Core experiment logic for studying TRIM1 behavior under edge deletions.
// This file contains ONLY the algorithm — no printing, no main().
//
// The recipient must provide their own:
//   - Graph type with add_node(), add_edge(), make_reverse_edges()
//   - TRIM1 function: repeat_global_trim1(Graph&) -> int
//   - Color check: G_Color[node] == SCC_FOUND (or equivalent)
//
// Two experiments:
//
//   Experiment 1 (new):
//     For a delete batch, count how many of its unique nodes
//     are trimmed by TRIM1 on the MODIFIED graph (original - batch edges).
//     Result: T_mod ∩ B
//
//   Experiment 2:
//     For a delete batch, count how many nodes are trimmed
//     on the full graph vs modified graph.
//     Result: T_full, T_mod, Δ = T_mod \ T_full
// ==========================================================================

#ifndef SCC_TRIM_EXPERIMENT_H
#define SCC_TRIM_EXPERIMENT_H

#include <vector>
#include <set>
#include <unordered_set>
#include <utility>

// ==========================================================================
// Experiment result struct (pure data, no printing)
// ==========================================================================
struct TrimExperimentResult {
    // Batch info
    int edge_count;                    // number of edges in the batch
    int unique_batch_nodes;            // unique node IDs from batch edges
    
    // Experiment 1 (new): batch nodes trimmed on modified graph
    int batch_trimmed_on_full;         // T_full ∩ B  (batch nodes already trimmed)
    int batch_trimmed_on_mod;          // T_mod ∩ B   (batch nodes trimmed on modified graph)
    
    // Experiment 2: total trim counts
    int trimmed_full;                  // T_full  (nodes trimmed on full graph)
    int trimmed_mod;                   // T_mod   (nodes trimmed on modified graph)
    
    // Overlap: newly trimmed after deletion
    int delta_count;                   // Δ = T_mod \ T_full
    int delta_and_batch;               // Δ ∩ B (direct effect — batch nodes in Δ)
    int delta_not_batch;               // Δ \ B (cascade effect — non-batch nodes in Δ)
};

// ==========================================================================
// Collect unique node IDs from a batch edge list.
// batch_edges: list of (src, dst) pairs
// Returns: unordered_set of unique node IDs
// ==========================================================================
inline std::unordered_set<int> collect_batch_nodes(
    const std::vector<std::pair<int,int>>& batch_edges)
{
    std::unordered_set<int> nodes;
    for (const auto& e : batch_edges) {
        nodes.insert(e.first);
        nodes.insert(e.second);
    }
    return nodes;
}

// ==========================================================================
// Build a boolean mask for batch nodes.
// batch_nodes: unordered_set of unique node IDs
// num_nodes: total number of nodes in the graph
// Returns: vector<bool> where mask[n] = true if n is a batch node
// ==========================================================================
inline std::vector<bool> make_batch_mask(
    const std::unordered_set<int>& batch_nodes,
    int num_nodes)
{
    std::vector<bool> mask(num_nodes, false);
    for (int n : batch_nodes) {
        mask[n] = true;
    }
    return mask;
}

// ==========================================================================
// Build a set of edges to exclude (for building modified graph).
// batch_edges: list of (src, dst) pairs from the batch
// Returns: set of pairs for fast lookup
// ==========================================================================
inline std::set<std::pair<int,int>> make_exclude_set(
    const std::vector<std::pair<int,int>>& batch_edges)
{
    return std::set<std::pair<int,int>>(batch_edges.begin(), batch_edges.end());
}

// ==========================================================================
// Build graph from original edges, optionally excluding some.
//
// Template parameters:
//   Graph: the graph type (must support add_node(), add_edge(), make_reverse_edges())
//   num_nodes: total number of nodes
//   orig_edges: all original graph edges
//   exclude: set of edges to exclude (empty = full graph)
//
// Returns: fully built graph with reverse edges
// ==========================================================================
template<typename Graph>
Graph build_graph(int num_nodes,
                  const std::vector<std::pair<int,int>>& orig_edges,
                  const std::set<std::pair<int,int>>& exclude = {})
{
    Graph G;
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
// Run the combined experiment for a single batch.
//
// This performs both Experiment 1 (new) and Experiment 2 on the modified
// graph (original - batch edges) and returns all metrics.
//
// Template parameters:
//   Graph:        graph type (with add_node, add_edge, make_reverse_edges)
//   TrimFunc:     function type: int(Graph&) — runs TRIM1 to convergence
//   ColorCheck:   function type: bool(int) — checks if a node is trimmed
//   ColorInit:    function type: void(int) — initializes colors for num_nodes
//
// Arguments:
//   num_nodes:    total nodes in the original graph
//   orig_edges:   all edges of the original graph
//   batch_edges:  edges to delete in this batch
//   trim_full:    result of TRIM1 on the FULL graph (caller provides)
//   run_trim1:    callable: run_trim1(graph) -> int (TRIM1 to convergence)
//   is_trimmed:   callable: is_trimmed(node) -> bool (checks if node was trimmed)
//   init_colors:  callable: init_colors(num_nodes) (resets color arrays)
//
// Returns: TrimExperimentResult with all metrics
// ==========================================================================
template<typename Graph, typename TrimFunc, typename ColorCheck, typename ColorInit>
TrimExperimentResult run_batch_experiment(
    int num_nodes,
    const std::vector<std::pair<int,int>>& orig_edges,
    const std::vector<std::pair<int,int>>& batch_edges,
    int trim_full,
    TrimFunc run_trim1,
    ColorCheck is_trimmed,
    ColorInit init_colors)
{
    TrimExperimentResult result;
    
    // Collect unique batch nodes
    auto batch_nodes = collect_batch_nodes(batch_edges);
    auto is_batch_node = make_batch_mask(batch_nodes, num_nodes);
    auto exclude_set = make_exclude_set(batch_edges);
    
    result.edge_count = (int)batch_edges.size();
    result.unique_batch_nodes = (int)batch_nodes.size();
    result.trimmed_full = trim_full;
    
    // Build modified graph = (original - batch edges)
    Graph G_mod = build_graph<Graph>(num_nodes, orig_edges, exclude_set);
    
    // Run TRIM1 on modified graph
    init_colors(num_nodes);
    int trim_mod = run_trim1(G_mod);
    result.trimmed_mod = trim_mod;
    
    // Compute Experiment 1: T_mod ∩ B (batch nodes trimmed on modified graph)
    int batch_trimmed_on_full = 0;
    int batch_trimmed_on_mod = 0;
    for (int n : batch_nodes) {
        if (is_trimmed(n)) {
            batch_trimmed_on_mod++;
        }
    }
    // Note: T_full ∩ B should be computed by caller before deletion
    // Here we only compute on the modified graph
    result.batch_trimmed_on_mod = batch_trimmed_on_mod;
    
    // Compute Δ = T_mod \ T_full (newly trimmed after deletion)
    // Note: This requires knowing which nodes were trimmed on the full graph.
    // The caller must provide is_trimmed_full() in addition to is_trimmed().
    // For now, we just count what we can.
    result.delta_count = 0;
    result.delta_and_batch = 0;
    result.delta_not_batch = 0;
    
    return result;
}

// ==========================================================================
// Full experiment: computes both T_full and T_mod, plus all overlap metrics.
//
// This is the complete experiment pipeline for one batch:
//   1. Build full graph, run TRIM1 → T_full
//   2. Build modified graph (original - batch edges), run TRIM1 → T_mod
//   3. Compute Δ = T_mod \ T_full
//   4. Compute T_mod ∩ B (Experiment 1 new)
//   5. Compute Δ ∩ B (direct) and Δ \ B (cascade)
//
// Template parameters:
//   Graph:        graph type
//   TrimFunc:     int(Graph&) — TRIM1 to convergence
//   ColorFuncs:   at minimum, functions to init colors and check trim status
//
// The caller must provide:
//   - Graph building: build_graph<Graph>(num_nodes, edges, exclude)
//   - TRIM1: a function that runs TRIM1 to convergence on a graph
//   - Color management: init, check per node
//
// Since color arrays are typically global, the caller must handle:
//   init_colors(num_nodes)  → reset color arrays
//   is_trimmed(node)        → check if node was marked SCC_FOUND
// ==========================================================================
template<typename Graph>
struct BatchExperiment {
    // Callbacks (the caller sets these before calling run())
    std::vector<std::pair<int,int>>* orig_edges;   // original graph edges
    int num_nodes;                                   // total nodes
    int trimmed_full;                                // TRIM1 result on full graph
    std::vector<bool>* T_full_mask;                  // which nodes were trimmed on full graph
    
    // Run the experiment for one batch
    // batch_edges: edges to delete
    // build_fn:    builds graph from edges + exclude set
    // trim_fn:     runs TRIM1, returns trimmed count
    // init_fn:     resets color arrays
    // check_fn:    checks if a node is trimmed
    TrimExperimentResult run(
        const std::vector<std::pair<int,int>>& batch_edges,
        Graph (*build_fn)(int, const std::vector<std::pair<int,int>>&, const std::set<std::pair<int,int>>&),
        int (*trim_fn)(Graph&),
        void (*init_fn)(int),
        bool (*check_fn)(int))
    {
        TrimExperimentResult result;
        
        // Collect batch nodes
        auto batch_nodes = collect_batch_nodes(batch_edges);
        auto is_batch_node = make_batch_mask(batch_nodes, num_nodes);
        auto exclude_set = make_exclude_set(batch_edges);
        
        result.edge_count = (int)batch_edges.size();
        result.unique_batch_nodes = (int)batch_nodes.size();
        result.trimmed_full = trimmed_full;
        
        // Build and run on modified graph
        Graph G_mod = build_fn(num_nodes, *orig_edges, exclude_set);
        init_fn(num_nodes);
        result.trimmed_mod = trim_fn(G_mod);
        
        // Compute T_mod ∩ B (Experiment 1)
        result.batch_trimmed_on_mod = 0;
        for (int n : batch_nodes) {
            if (check_fn(n)) result.batch_trimmed_on_mod++;
        }
        
        // Compute T_full ∩ B (batch already trimmed on full graph, pre-computation)
        result.batch_trimmed_on_full = 0;
        if (T_full_mask) {
            for (int n : batch_nodes) {
                if ((*T_full_mask)[n]) result.batch_trimmed_on_full++;
            }
        }
        
        // Compute Δ = T_mod \ T_full
        result.delta_count = 0;
        result.delta_and_batch = 0;
        result.delta_not_batch = 0;
        
        if (T_full_mask) {
            for (int n = 0; n < num_nodes; n++) {
                bool in_mod = check_fn(n);
                bool in_full = (*T_full_mask)[n];
                if (in_mod && !in_full) {
                    result.delta_count++;
                    if (is_batch_node[n]) {
                        result.delta_and_batch++;
                    } else {
                        result.delta_not_batch++;
                    }
                }
            }
        }
        
        return result;
    }
};

// ==========================================================================
// Simplified version: Just Experiment 1 (new) — batch nodes on modified graph
//
// Use this when you only need T_mod ∩ B without comparing to T_full.
//
// Template parameters:
//   Graph: graph type
//
// Returns: number of batch nodes trimmed on the modified graph
// ==========================================================================
template<typename Graph>
inline int experiment1_batch_trimmed_on_mod(
    int num_nodes,
    const std::vector<std::pair<int,int>>& orig_edges,
    const std::vector<std::pair<int,int>>& batch_edges,
    Graph (*build_fn)(int, const std::vector<std::pair<int,int>>&, const std::set<std::pair<int,int>>&),
    int (*trim_fn)(Graph&),
    void (*init_fn)(int),
    bool (*check_fn)(int))
{
    auto batch_nodes = collect_batch_nodes(batch_edges);
    auto exclude_set = make_exclude_set(batch_edges);
    
    Graph G_mod = build_fn(num_nodes, orig_edges, exclude_set);
    init_fn(num_nodes);
    trim_fn(G_mod);
    
    int count = 0;
    for (int n : batch_nodes) {
        if (check_fn(n)) count++;
    }
    return count;
}

// ==========================================================================
// Simplified version: Just Experiment 2 — Δ = T_mod \ T_full
//
// Use this when you only need the newly trimmed nodes.
// Assumes T_full has already been computed.
//
// Template parameters:
//   Graph: graph type
//
// Returns: Δ count
// ==========================================================================
template<typename Graph>
inline int experiment2_delta(
    int num_nodes,
    const std::vector<std::pair<int,int>>& orig_edges,
    const std::vector<std::pair<int,int>>& batch_edges,
    const std::vector<bool>& T_full,
    Graph (*build_fn)(int, const std::vector<std::pair<int,int>>&, const std::set<std::pair<int,int>>&),
    int (*trim_fn)(Graph&),
    void (*init_fn)(int),
    bool (*check_fn)(int))
{
    auto exclude_set = make_exclude_set(batch_edges);
    
    Graph G_mod = build_fn(num_nodes, orig_edges, exclude_set);
    init_fn(num_nodes);
    trim_fn(G_mod);
    
    int delta = 0;
    for (int n = 0; n < num_nodes; n++) {
        if (check_fn(n) && !T_full[n]) {
            delta++;
        }
    }
    return delta;
}

#endif // SCC_TRIM_EXPERIMENT_H
