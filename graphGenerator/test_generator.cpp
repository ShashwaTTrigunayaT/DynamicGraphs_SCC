#include "graphGenerator.h"

int main()
{
    // Generate old-style LCC graphs (existing behavior)
    generate_lcc_graph_to_file(3000000,30,98000000);
    generate_lcc_graph_to_file(3000000,40,98000000);
    generate_lcc_graph_to_file(3000000,50,98000000);
    generate_lcc_graph_to_file(3000000,70,98000000);
    
    // New: explicit structure LCC graphs
    // 30% LCC (one giant SCC) + 30% singletons + 40% small SCCs
    // => ~50% total SCCs compared to nodes
    printf("\n=== New explicit-structure LCC graphs ===\n");

    // 10K nodes: 3K LCC + 3K singletons + 4K small SCCs, 100K edges (quick test)
    generate_lcc_graph_to_file(10000, 30, 100000, -1, 30);

    // 100K nodes: 30K LCC + 30K singletons + 40K small SCCs, 1M edges
    generate_lcc_graph_to_file(100000, 30, 1000000, -1, 30);

    // 500K nodes: 150K LCC + 150K singletons + 200K small SCCs, 5M edges
    generate_lcc_graph_to_file(500000, 30, 5000000, -1, 30);

    // 1M nodes: 300K LCC + 300K singletons + 400K small SCCs, 10M edges
    generate_lcc_graph_to_file(1000000, 30, 10000000, -1, 30);

    printf("\nDone!\n");
    return 0;
}
