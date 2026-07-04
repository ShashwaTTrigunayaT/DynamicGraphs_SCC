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

    // 100K nodes: 30K LCC + 30K singletons + 40K small SCCs
    generate_lcc_graph_to_file(100000, 30, 9800000, -1, 30);

    // 1M nodes: 300K LCC + 300K singletons + 400K small SCCs
    generate_lcc_graph_to_file(1000000, 30, 98000000, -1, 30);

    printf("\nDone!\n");
    return 0;
}
