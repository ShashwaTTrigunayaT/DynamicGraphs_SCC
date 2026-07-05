#include "graphGenerator.h"

int main()
{
    
    

    printf("\n=== SCC-Ratio graphs (1M nodes, 10M edges) ===\n");
    generate_scc_ratio_graph_to_file(1000000, 0.30, 10000000);  // 30% SCC (time-based seed)
    generate_scc_ratio_graph_to_file(1000000, 0.50, 10000000);  // 50% SCC
    generate_scc_ratio_graph_to_file(1000000, 0.70, 10000000);  // 70% SCC

    printf("\n=== SCC-Ratio graphs (5M nodes, 50M edges) ===\n");
    generate_scc_ratio_graph_to_file(5000000, 0.30, 50000000);  // 30% SCC
    generate_scc_ratio_graph_to_file(5000000, 0.50, 50000000);  // 50% SCC
    generate_scc_ratio_graph_to_file(5000000, 0.70, 50000000);  // 70% SCC

    printf("\n=== SCC-Ratio graphs (10M nodes, 100M edges) ===\n");
    generate_scc_ratio_graph_to_file(10000000, 0.30, 100000000);  // 30% SCC
    generate_scc_ratio_graph_to_file(10000000, 0.50, 100000000);  // 50% SCC
    generate_scc_ratio_graph_to_file(10000000, 0.70, 100000000);  // 70% SCC

    printf("\nDone!\n");
    return 0;
}
