#include "graphGenerator.h"

int main()
{
    // LCC graphs with ~50% SCC condensation ratio
    // Structure: ring (LCC%) + 35% singletons + remaining small SCCs (2-5 nodes)
    printf("\n=== LCC graphs (100K nodes, 1M edges, ~50%% SCCs) ===\n");
    generate_lcc_graph_to_file(100000, 30, 1000000, -1, 35);  // 30% LCC, 35% singletons
    generate_lcc_graph_to_file(100000, 50, 1000000, -1, 35);  // 50% LCC, 35% singletons
    generate_lcc_graph_to_file(100000, 60, 1000000, -1, 35);  // 60% LCC, 35% singletons
    generate_lcc_graph_to_file(100000, 70, 1000000, -1, 35);  // 70% LCC, 35% singletons

    printf("\n=== LCC graphs (500K nodes, 5M edges, ~50%% SCCs) ===\n");
    generate_lcc_graph_to_file(500000, 30, 5000000, -1, 35);  // 30% LCC, 35% singletons
    generate_lcc_graph_to_file(500000, 50, 5000000, -1, 35);  // 50% LCC, 35% singletons
    generate_lcc_graph_to_file(500000, 60, 5000000, -1, 35);  // 60% LCC, 35% singletons
    generate_lcc_graph_to_file(500000, 70, 5000000, -1, 35);  // 70% LCC, 35% singletons

    // Diameter graphs (chain structure, sparse edges between layers)
    printf("\n=== Diameter graphs (100K nodes, varying diameter) ===\n");
    generate_diameter_graph_to_file(30, 100000, 10, 10000);   // dia=30, 10K SCCs
    generate_diameter_graph_to_file(40, 100000, 10, 10000);   // dia=40, 10K SCCs
    generate_diameter_graph_to_file(70, 100000, 10, 10000);   // dia=70, 10K SCCs
    generate_diameter_graph_to_file(100, 100000, 10, 10000);  // dia=100, 10K SCCs

    printf("\n=== Diameter graphs (500K nodes, varying diameter) ===\n");
    generate_diameter_graph_to_file(30, 500000, 10, 50000);   // dia=30, 50K SCCs
    generate_diameter_graph_to_file(40, 500000, 10, 50000);   // dia=40, 50K SCCs
    generate_diameter_graph_to_file(70, 500000, 10, 50000);   // dia=70, 50K SCCs
    generate_diameter_graph_to_file(100, 500000, 10, 50000);  // dia=100, 50K SCCs

    printf("\nDone!\n");
    return 0;
}
