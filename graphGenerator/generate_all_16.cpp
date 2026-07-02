#include "graphGenerator.h"

int main() {
    // ========================================
    // Diameter graphs (layered DAG-style SCCs)
    // ========================================
    // format: generate_diameter_graph_to_file(layers, nodes, k, sccs)

    // 1M node diameter graphs
    generate_diameter_graph_to_file(20,   1000000, 10, 100000);
    generate_diameter_graph_to_file(40,   1000000, 10, 100000);
    generate_diameter_graph_to_file(60,   1000000, 10, 100000);
    generate_diameter_graph_to_file(100,  1000000, 10, 100000);

    // 3M node diameter graphs
    generate_diameter_graph_to_file(20,   3000000, 10, 300000);
    generate_diameter_graph_to_file(40,   3000000, 10, 300000);
    generate_diameter_graph_to_file(60,   3000000, 10, 300000);
    generate_diameter_graph_to_file(100,  3000000, 10, 300000);

    // ========================================
    // LCC graphs (giant SCC + satellite SCCs)
    // ========================================
    // format: generate_lcc_graph_to_file(nodes, lcc_pct, edges)

    // 1M node LCC graphs (10M edges)
    generate_lcc_graph_to_file(1000000, 30, 10000000);
    generate_lcc_graph_to_file(1000000, 40, 10000000);
    generate_lcc_graph_to_file(1000000, 50, 10000000);
    generate_lcc_graph_to_file(1000000, 70, 10000000);

    // 3M node LCC graphs (98M edges)
    generate_lcc_graph_to_file(3000000, 30, 98000000);
    generate_lcc_graph_to_file(3000000, 40, 98000000);
    generate_lcc_graph_to_file(3000000, 50, 98000000);
    generate_lcc_graph_to_file(3000000, 70, 98000000);

    printf("\nAll 16 synthetic graphs generated successfully!\n");
    return 0;
}
