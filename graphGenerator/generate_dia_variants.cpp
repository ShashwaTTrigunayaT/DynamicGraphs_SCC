#include "graphGenerator.h"

int main() {
    // 100K nodes, varying diameters, k=10, SCCs=10% of nodes
    generate_diameter_graph_to_file(20, 100000, 10, 10000);
    generate_diameter_graph_to_file(30, 100000, 10, 10000);
    generate_diameter_graph_to_file(70, 100000, 10, 10000);
    generate_diameter_graph_to_file(100, 100000, 10, 10000);
    printf("\nAll 4 graphs generated!\n");
    return 0;
}
