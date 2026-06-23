#ifndef SCC_TRIM_EXPERIMENT_H
#define SCC_TRIM_EXPERIMENT_H

#include <vector>
#include <unordered_set>
#include <utility>
template<typename Graph>
inline int experiment1_batch_trimmed_on_mod(
    Graph& G_mod,
    const std::vector<std::pair<int,int>>& batch_edges,
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

    // Step 2: Run TRIM1 on the modified graph
    init_fn(G_mod.num_nodes());
    trim_fn(G_mod);

    // Step 3: Count how many batch nodes are trimmed
    int count = 0;
    for (int n : batch_nodes) {
        if (check_fn(n)) count++;
    }
    return count;
}

#endif // SCC_TRIM_EXPERIMENT_H
