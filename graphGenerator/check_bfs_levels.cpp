#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <graph_file>\n", argv[0]);
        return 1;
    }

    // Read graph
    printf("Reading %s ...\n", argv[1]);
    std::ifstream in(argv[1]);
    if (!in.is_open()) { printf("Error opening file\n"); return 1; }

    std::vector<std::pair<int,int>> edges;
    int num_nodes = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        int u, v;
        if (sscanf(line.c_str(), "%d %d", &u, &v) == 2) {
            u--; v--;
            edges.push_back({u, v});
            if (u + 1 > num_nodes) num_nodes = u + 1;
            if (v + 1 > num_nodes) num_nodes = v + 1;
        }
    }
    in.close();
    printf("Nodes: %d, Edges: %zu\n", num_nodes, edges.size());

    // Build adjacency (outgoing) and reverse adjacency (incoming)
    std::vector<std::vector<int>> adj(num_nodes);
    std::vector<std::vector<int>> radj(num_nodes);
    std::vector<int> out_degree(num_nodes, 0);
    std::vector<int> in_degree(num_nodes, 0);

    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
        radj[e.second].push_back(e.first);
        out_degree[e.first]++;
        in_degree[e.second]++;
    }

    // Find node with highest total degree as BFS source
    int source = 0, max_deg = 0;
    for (int i = 0; i < num_nodes; i++) {
        int deg = out_degree[i] + in_degree[i];
        if (deg > max_deg) { max_deg = deg; source = i; }
    }
    printf("Source node: %d (out-degree: %d, in-degree: %d, total: %d)\n",
           source, out_degree[source], in_degree[source], max_deg);

    // ====== 1. DIRECTED BFS (outgoing edges only) ======
    printf("\n========== DIRECTED BFS (outgoing edges only) ==========\n");
    {
        std::vector<int> dist(num_nodes, -1);
        std::queue<int> q;
        dist[source] = 0;
        q.push(source);

        int max_dist = 0, visited = 0;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            visited++;
            for (int v : adj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    if (dist[v] > max_dist) max_dist = dist[v];
                    q.push(v);
                }
            }
        }

        printf("Visited nodes: %d / %d (%.1f%%)\n", visited, num_nodes, 100.0 * visited / num_nodes);
        printf("Max BFS distance (levels-1): %d\n", max_dist);
        printf("Total BFS levels: %d\n", max_dist + 1);

        std::vector<int> npl(max_dist + 1, 0);
        for (int d : dist) if (d >= 0) npl[d]++;
        printf("\nNodes per BFS level:\n");
        for (int l = 0; l <= max_dist; l++)
            printf("  Level %3d: %d nodes\n", l, npl[l]);
    }

    // ====== 2. UNDIRECTED BFS (follow both directions) ======
    printf("\n========== UNDIRECTED BFS (outgoing + incoming) ==========\n");
    {
        std::vector<int> dist(num_nodes, -1);
        std::queue<int> q;
        dist[source] = 0;
        q.push(source);

        int max_dist = 0, visited = 0;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            visited++;
            // Follow outgoing edges
            for (int v : adj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    if (dist[v] > max_dist) max_dist = dist[v];
                    q.push(v);
                }
            }
            // Follow incoming edges (reverse direction)
            for (int v : radj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    if (dist[v] > max_dist) max_dist = dist[v];
                    q.push(v);
                }
            }
        }

        printf("Visited nodes: %d / %d (%.1f%%)\n", visited, num_nodes, 100.0 * visited / num_nodes);
        printf("Max BFS distance (levels-1): %d\n", max_dist);
        printf("Total BFS levels: %d\n", max_dist + 1);

        std::vector<int> npl(max_dist + 1, 0);
        for (int d : dist) if (d >= 0) npl[d]++;
        printf("\nNodes per BFS level:\n");
        for (int l = 0; l <= max_dist; l++)
            printf("  Level %3d: %d nodes\n", l, npl[l]);
    }

    return 0;
}
