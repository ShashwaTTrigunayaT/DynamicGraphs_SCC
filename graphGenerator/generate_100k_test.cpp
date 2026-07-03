#include "graphGenerator.h"

int main() {
    // Generate diameter graph: dia=20, nodes=100000, k=10, total_sccs=10000
    // This produces: diameter_20_100000_10000.txt in the current directory
    generate_diameter_graph_to_file(20, 100000, 10, 10000);
    printf("\nDone! Generated diameter_20_100000_10000.txt\n");
    return 0;
}
