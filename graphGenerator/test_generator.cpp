#include "graphGenerator.h"

int main()
{
    // Explicit structure LCC graphs
    // 30% LCC (one giant SCC) + 30% singletons + 40% small SCCs
    // => ~50% total SCCs compared to nodes
    printf("\n=== Explicit-structure LCC graphs (30% LCC + 30% singletons) ===\n");

    // 10K nodes: 3K LCC + 3K singletons + 4K small SCCs, 100K edges (quick test)
    generate_lcc_graph_to_file(10000, 30, 100000, -1, 30);

    // 100K nodes: 30K LCC + 30K singletons + 40K small SCCs, 1M edges
    generate_lcc_graph_to_file(100000, 30, 1000000, -1, 30);

    // 500K nodes: 150K LCC + 150K singletons + 200K small SCCs, 5M edges
    generate_lcc_graph_to_file(500000, 30, 5000000, -1, 30);

    // 500K nodes, varying LCC % (10, 40, 50, 60, 90) with same 30% singletons
    printf("\n=== Varying LCC %% at 500K/5M ===\n");
    generate_lcc_graph_to_file(500000, 10, 5000000, -1, 30);  // 10% LCC, 30% singletons
    generate_lcc_graph_to_file(500000, 40, 5000000, -1, 30);  // 40% LCC, 30% singletons
    generate_lcc_graph_to_file(500000, 50, 5000000, -1, 30);  // 50% LCC, 30% singletons
    generate_lcc_graph_to_file(500000, 60, 5000000, -1, 30);  // 60% LCC, 30% singletons
    generate_lcc_graph_to_file(500000, 90, 5000000, -1, 30);  // 90% LCC, remaining 10% split: singletons + small SCCs

    // 1M nodes: 300K LCC + 300K singletons + 400K small SCCs, 10M edges
    generate_lcc_graph_to_file(1000000, 30, 10000000, -1, 30);

    printf("\nDone!\n");
    return 0;
}
