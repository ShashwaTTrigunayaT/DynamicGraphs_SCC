#include "graphGenerator.h"

int main() {
    // Generate diameter graphs: dia=20, k=10, SCCs=10% of nodes
    generate_diameter_graph_to_file(20, 200000, 10, 20000);
    generate_diameter_graph_to_file(20, 300000, 10, 30000);
    generate_diameter_graph_to_file(20, 400000, 10, 40000);
    generate_diameter_graph_to_file(20, 500000, 10, 50000);
    printf("\nAll 4 graphs generated!\n");
    return 0;
}
