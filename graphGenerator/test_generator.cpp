#include "graphGenerator.h"

int main()
{
    printf("\n=== SCC-Ratio graphs (100K nodes, 1M edges) ===\n");
    generate_scc_ratio_graph_to_file(100000, 0.30, 1000000);
    generate_scc_ratio_graph_to_file(100000, 0.50, 1000000);
    generate_scc_ratio_graph_to_file(100000, 0.70, 1000000);

    printf("\n=== SCC-Ratio graphs (500K nodes, 5M edges) ===\n");
    generate_scc_ratio_graph_to_file(500000, 0.30, 5000000);
    generate_scc_ratio_graph_to_file(500000, 0.50, 5000000);
    generate_scc_ratio_graph_to_file(500000, 0.70, 5000000);

    printf("\n=== SCC-Ratio graphs (1M nodes, 10M edges) ===\n");
    generate_scc_ratio_graph_to_file(1000000, 0.30, 10000000);
    generate_scc_ratio_graph_to_file(1000000, 0.50, 10000000);
    generate_scc_ratio_graph_to_file(1000000, 0.70, 10000000);

    printf("\n=== SCC-Ratio graphs (5M nodes, 50M edges) ===\n");
    generate_scc_ratio_graph_to_file(5000000, 0.30, 50000000);
    generate_scc_ratio_graph_to_file(5000000, 0.50, 50000000);
    generate_scc_ratio_graph_to_file(5000000, 0.70, 50000000);

    printf("\nDone!\n");
    return 0;
}
