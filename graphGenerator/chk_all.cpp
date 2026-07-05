#include "graphGenerator.h"
int main(int argc, char** argv) {
    std::vector<std::pair<int,int>> edges;
    int nodes = 0;
    read_graph_file(argv[1], edges, nodes);
    auto scc = compute_scc(edges, nodes);
    std::string out_name = std::string(argv[2]);
    std::ofstream out(out_name);
    for (int i = 0; i < (int)scc.size(); i++)
        out << i << " " << scc[i] << "\n";
    int max_scc = *max_element(scc.begin(), scc.end()) + 1;
    printf("  SCCs: %d / %d (%.1f%%)\n", max_scc, nodes, 100.0*max_scc/nodes);
    return 0;
}
