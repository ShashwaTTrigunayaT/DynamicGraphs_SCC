

#ifndef SCC_TRIM_EXPERIMENT_H
#define SCC_TRIM_EXPERIMENT_H

#include <vector>
#include <set>
#include <unordered_set>
#include <utility>


// Collect unique node IDs from a batch edge list.
// Goes through each edge, takes both src and dst nodes.
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
// Build graph from original edges, optionally excluding some.
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
// EXPERIMENT 1 (new):
// For one batch, count how many of its unique nodes are trimmed by TRIM1
// on the MODIFIED graph (original graph minus batch edges).
//
// Steps:
//   1. Go through each edge of batch → collect unique node IDs
//   2. Build modified graph = (original_edges - batch_edges)
//   3. Run TRIM1 on the modified graph
//   4. Count how many of the unique batch nodes are trimmed
//
// Returns: number of batch nodes trimmed (T_mod ∩ B)
//
// Callbacks needed from the recipient:
//   build_fn(num_nodes, edges, exclude) -> Graph
//   trim_fn(Graph&) -> int           (run TRIM1 to convergence)
//   init_fn(int)                     (reset color arrays)
//   check_fn(int) -> bool            (is node trimmed?)
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
    // Step 1: Go through each edge of batch, collect unique nodes
    std::unordered_set<int> batch_nodes;
    for (const auto& e : batch_edges) {
        batch_nodes.insert(e.first);
        batch_nodes.insert(e.second);
    }

    // Step 2: Build modified graph = (original edges - batch edges)
    std::set<std::pair<int,int>> exclude(batch_edges.begin(), batch_edges.end());
    Graph G_mod = build_fn(num_nodes, orig_edges, exclude);

    // Step 3: Run TRIM1 on the modified graph
    init_fn(num_nodes);
    trim_fn(G_mod);

    // Step 4: Count how many batch nodes are trimmed
    int count = 0;
    for (int n : batch_nodes) {
        if (check_fn(n)) count++;
    }
    return count;
}

#endif // SCC_TRIM_EXPERIMENT_H
