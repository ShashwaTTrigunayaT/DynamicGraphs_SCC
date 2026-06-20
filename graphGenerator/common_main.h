#ifndef COMMON_MAIN_H
#define COMMON_MAIN_H

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "gm.h"
#include <pthread.h>
// #include <execution>
// #include <algorithm>
#include <bits/stdc++.h>

using namespace std;

extern vector<int> scc_list;
extern vector<int> vec_scc_count;
extern vector<int> new_edge_nodes;
extern int met_algo;
extern int unaffect_sccs;
extern vector<int> level_ver;
extern vector<int> affect_level;
extern int good_init_pivot;
extern int maxi_neigh_del;
extern vector<pair<int, int>> scc_num_nodes;
extern int affect_sccs;
extern float count_ver;
extern float count_scc;
extern float insert_runtime;

class main_t
{
public:
    gm_graph G2;

protected:
    gm_graph G;
    int num_threads;
    bool is_all_thread_mode() { return num_threads == -1; }

public:
    main_t()
    {
        time_to_exclude = 0;
        num_threads = 0;
    }

    void pin_CPU()
    {
#pragma omp parallel
        {
            pthread_t thread;
            thread = pthread_self();
            cpu_set_t CPU;
            CPU_ZERO(&CPU);
            CPU_SET(omp_get_thread_num(), &CPU);
            pthread_setaffinity_np(thread, sizeof(CPU), &CPU);
        }
    }

    #include <iostream>
#include <vector>
#include <omp.h>

// Simple Directed Graph Structure
struct Graph {
    int num_vertices;
    // Top-down requires outgoing edges
    std::vector<std::vector<int>> out_edges;
    // Bottom-up requires incoming edges (crucial for directed graphs!)
    std::vector<std::vector<int>> in_edges;

    Graph(int n) : num_vertices(n), out_edges(n), in_edges(n) {}

    void add_edge(int u, int v) {
        out_edges[u].push_back(v);
        in_edges[v].push_back(u);
    }
};

// Top-Down (Push) Step: Frontier vertices push updates to unvisited neighbors
int top_down_step(const Graph& g, const std::vector<uint8_t>& frontier, std::vector<uint8_t>& next_frontier, std::vector<int>& distance, int curr_dist) {
    int awake_count = 0;

    #pragma omp parallel for reduction(+:awake_count) schedule(dynamic, 64)
    for (int u = 0; u < g.num_vertices; ++u) {
        if (!frontier[u]) continue;

        for (int v : g.out_edges[u]) {
            // Check if unvisited
            if (distance[v] == -1) {
                // Safely attempt to claim the node using atomic compare-and-swap
                int expected = -1;
                if (__atomic_compare_exchange_n(&distance[v], &expected, curr_dist + 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    next_frontier[v] = 1;
                    awake_count++;
                }
            }
        }
    }
    return awake_count;
}

// Bottom-Up (Pull) Step: Unvisited vertices pull status from active parents in the frontier
int bottom_up_step(const Graph& g, const std::vector<uint8_t>& frontier, std::vector<uint8_t>& next_frontier, std::vector<int>& distance, int curr_dist) {
    int awake_count = 0;

    #pragma omp parallel for reduction(+:awake_count) schedule(dynamic, 64)
    for (int v = 0; v < g.num_vertices; ++v) {
        if (distance[v] != -1) continue; // Already visited

        for (int u : g.in_edges[v]) {
            if (frontier[u]) { 
                // Parent is in the frontier! Claim immediately and stop checking neighbors.
                distance[v] = curr_dist + 1;
                next_frontier[v] = 1;
                awake_count++;
                break; 
            }
        }
    }
    return awake_count;
}

// Direction-Optimizing BFS Orchestrator
std::vector<int> do_bfs(const Graph& g, int source) {
    std::vector<int> distance(g.num_vertices, -1);
    std::vector<uint8_t> frontier(g.num_vertices, 0);
    std::vector<uint8_t> next_frontier(g.num_vertices, 0);

    distance[source] = 0;
    frontier[source] = 1;
    
    int frontier_size = 1;
    int curr_dist = 0;

    // Simple heuristic threshold based on total graph size
    const int alpha = g.num_vertices / 20; 

    while (frontier_size > 0) {
        int next_frontier_size = 0;

        // Dynamic Heuristic Selection
        if (frontier_size > alpha) {
            // Pull phase reduces edge examinations when the active pool is dense
            next_frontier_size = bottom_up_step(g, frontier, next_frontier, distance, curr_dist);
        } else {
            // Push phase scales better when exploring small components
            next_frontier_size = top_down_step(g, frontier, next_frontier, distance, curr_dist);
        }

        // Cycle Frontiers
        frontier = next_frontier;
        std::fill(next_frontier.begin(), next_frontier.end(), 0);
        
        frontier_size = next_frontier_size;
        curr_dist++;
    }

    return distance;
}

    // int main() {
    //     // Construct a simple sample graph with 6 vertices
    //     Graph g;
    //     g.num_vertices = 6;
    //     g.adj.resize(6);

    //     // Adding sample edges (Undirected)
    //     auto add_edge = [&](int u, int v) {
    //         g.adj[u].push_back(v);
    //         g.adj[v].push_back(u);
    //     };

    //     add_edge(0, 1); add_edge(0, 2);
    //     add_edge(1, 3); add_edge(1, 4);
    //     add_edge(2, 4);
    //     add_edge(3, 5); add_edge(4, 5);

    //     // Run DO-BFS from Source node 0
    //     do_bfs(g, 0);

    //     return 0;
    // }



    // struct Edge {
    //     int u, v;
    // };

    // // Find root of component (with path compression)
    // int find_root(int i, std::vector<int>& parent) {
    //     int root = i;
    //     while (root != parent[root])
    //         root = parent[root];
        
    //     // Path compression
    //     int curr = i;
    //     while (curr != root) {
    //         int next = parent[curr];
    //         parent[curr] = root;
    //         curr = next;
    //     }
    //     return root;
    // }

    // // Union components
    // void union_sets(int i, int j, std::vector<int>& parent, std::vector<int>& rank) {
    //     int root_i = find_root(i, parent);
    //     int root_j = find_root(j, parent);
    //     if (root_i != root_j) {
    //         if (rank[root_i] < rank[root_j]) {
    //             parent[root_i] = root_j;
    //         } else if (rank[root_i] > rank[root_j]) {
    //             parent[root_j] = root_i;
    //         } else {
    //             parent[root_j] = root_i;
    //             rank[root_i]++;
    //         }
    //     }
    // }

    // void boruvkas_mst(int V, const std::vector<Edge>& edges) {
    //     std::vector<int> parent(V);
    //     std::vector<int> rank(V, 0);

    //     // Each vertex forms its own component initially
    //     for (int i = 0; i < V; ++i)
    //         parent[i] = i;

    //     int num_components = V;
    //     std::vector<Edge> mst_edges;

    //     while (num_components > 1) {
    //         // Store the cheapest edge for each component. 
    //         // Since graph is unweighted, we can pick any valid outgoing edge.
    //         std::vector<int> cheapest(V, -1);

    //         #pragma omp parallel
    //         {
    //             std::vector<int> local_cheapest(V, -1);
                
    //             #pragma omp for schedule(dynamic)
    //             for (size_t i = 0; i < edges.size(); ++i) {
    //                 int u = edges[i].u;
    //                 int v = edges[i].v;

    //                 int root_u = find_root(u, parent);
    //                 int root_v = find_root(v, parent);

    //                 // If vertices belong to different components, it's a valid cut
    //                 if (root_u != root_v) {
    //                     if (local_cheapest[root_u] == -1) local_cheapest[root_u] = i;
    //                     if (local_cheapest[root_v] == -1) local_cheapest[root_v] = i;
    //                 }
    //             }

    //             // Merge local cheapest edges into the global array safely
    //             #pragma omp critical
    //             {
    //                 for (int i = 0; i < V; ++i) {
    //                     if (local_cheapest[i] != -1) {
    //                         if (cheapest[i] == -1) {
    //                             cheapest[i] = local_cheapest[i];
    //                         }
    //                     }
    //                 }
    //             }
    //         }

    //         bool components_merged = false;

    //         // Add edges to MST and perform unions
    //         for (int i = 0; i < V; ++i) {
    //             if (cheapest[i] != -1) {
    //                 int edge_idx = cheapest[i];
    //                 int u = edges[edge_idx].u;
    //                 int v = edges[edge_idx].v;

    //                 int root_u = find_root(u, parent);
    //                 int root_v = find_root(v, parent);

    //                 if (root_u != root_v) {
    //                     mst_edges.push_back(edges[edge_idx]);
    //                     union_sets(root_u, root_v, parent, rank);
    //                     num_components--;
    //                     components_merged = true;
    //                 }
    //             }
    //         }

    //         // If no more components can be merged, break to avoid infinite loop
    //         if (!components_merged) {
    //             break; 
    //         }
    //     }

    //     // // Output the resulting edges
    //     // std::cout << "Edges in the Boruvka's MST:\n";
    //     // for (const auto& edge : mst_edges) {
    //     //     std::cout << edge.u << " -> " << edge.v << "\n";
    //     // }
    // }

    int read_file(string filename, vector<pair<int, int>> &edges_list)
    {
        int max_vertex = 0;
        ifstream inputFile(filename);
        string line;

        while (getline(inputFile, line))
        {
            if (line.empty() || line[0] == '%') continue;

            vector<string> tokens;
            string token;
            stringstream ss(line);
            while (getline(ss, token, ' '))
            {
                tokens.push_back(token);
            }
            edges_list.push_back(make_pair(stoi(tokens[0]) - 1, stoi(tokens[1]) - 1));
            max_vertex = max(max_vertex, max(stoi(tokens[0]), stoi(tokens[1])));
        }

        inputFile.close();

        return max_vertex;
    }

    void BFS(vector<vector<int>> &adj_list, vector<int> &level, queue<int> &qu, vector<int> &in_degree, int *max_level)
    {
        vector<int> visited(adj_list.size(), 0);
        int top;
        // vector<vector<pair<int,int> > > topo_edges;
        while (!qu.empty())
        {
            top = qu.front();
            qu.pop();
            for (int i = 0; i < adj_list[top].size(); i++)
            {
                in_degree[adj_list[top][i]]--;
                if (in_degree[adj_list[top][i]] == 0)
                    qu.push(adj_list[top][i]);
                level[adj_list[top][i]] = level[top] + 1;
                // cout<<level[adj_list[top][i]];
                *(max_level) = max(*(max_level), level[adj_list[top][i]]);
                // topo_edges[level[top]].push_back(make_pair(top,adj_list[top][i]));
                /*Store the number of nodes also*/
            }
            // qu.pop();
        }
        cout << "max_level:" << (*(max_level)) << endl;
    }

    void parallel_prefix_sum(std::vector<int> &a)
    {
        int N = a.size();
        if (N == 0)
            return;

        int num_threads = 0;
#pragma omp parallel
        {
#pragma omp master
            {
                num_threads = omp_get_num_threads();
            }
        }

        std::vector<float> partial_sums(num_threads + 1, 0.0f);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float local_sum = 0.0f;

#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i)
            {
                local_sum += a[i];
                a[i] = local_sum;
            }
            partial_sums[tid + 1] = local_sum;
        }

        for (int i = 1; i <= num_threads; ++i)
        {
            partial_sums[i] += partial_sums[i - 1];
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            int tid = omp_get_thread_num();
            a[i] += partial_sums[tid];
        }
    }

    typedef std::pair<int, int> Pair;

    void parallelMergeSort(std::vector<Pair> &vec, int left, int right)
    {
        if (left < right)
        {
            // Threshold for switching to serial sort (e.g., 1000)
            if (right - left < 1000)
            {
                std::sort(vec.begin() + left, vec.begin() + right + 1);
                return;
            }

            int mid = left + (right - left) / 2;

#pragma omp task shared(vec)
            parallelMergeSort(vec, left, mid);

#pragma omp task shared(vec)
            parallelMergeSort(vec, mid + 1, right);

#pragma omp taskwait
            std::inplace_merge(vec.begin() + left, vec.begin() + mid + 1, vec.begin() + right + 1);
        }
    }

    gm_graph create_synthetic_graph_insert(int num_nodes, int num_insert)
    {
        gm_graph topo_graph_insert;
        for(int i=0;i<num_nodes;i++)
        {
            topo_graph_insert.add_node();
        }
        int count1=1;
        for(int i=0;i<=6;i++)
        {
            topo_graph_insert.add_edge(i,count1);
            count1+=1;
            topo_graph_insert.add_edge(i,count1);
        }

        for(int i=7;i<(num_nodes-8);i++)
        {
            topo_graph_insert.add_edge(i,i+8);
        }
    }
    // (0) ==> (2^0)
    // (1,2) ==> (2^1)
    // (3,4),(5,6) ==> (2^2)
    // (7,8),(9,10),(11,12),(13,14) ==> (2^3)

    void create_scc_edges(vector<pair<int, int>> orig_edges, vector<pair<int, int>> insert_edges, vector<pair<int, int>> &scc_edges, int num_vertices, int num_sccs)
    {
        unordered_map<string, int> ump;
        int root_node = 0;
        vector<vector<int>> adj_list(num_sccs);
        level_ver.resize(num_sccs, 0);
        new_edge_nodes.resize(num_sccs, -1);
        affect_level.resize(num_sccs + 5, 0);
        queue<int> qu;
        vector<int> in_degree(num_sccs, 0);
        vector<int> unaffected_levels;
        int max_level = 0;
        scc_edges.resize(orig_edges.size() + insert_edges.size(), {0, 0});
        struct timeval T_insert1, T_insert2;

#pragma omp parallel for
        for (int i = 0; i < orig_edges.size(); i++)
        {
            int ver1 = orig_edges[i].first;
            int ver2 = orig_edges[i].second;
            if (scc_list[ver1] != scc_list[ver2])
            {
                scc_edges[i] = make_pair(scc_list[ver1], scc_list[ver2]);
                if (met_algo == 7)
                {
                    adj_list[scc_list[ver1]].push_back(scc_list[ver2]);
                    in_degree[scc_list[ver2]] += 1;
                }
            }
        }
        if (met_algo == 7)
        {
            for (int i = 0; i < num_sccs; i++)
            {
                if (in_degree[i] == 0 && adj_list[i].size() != 0)
                {
                    qu.push(i);
                    level_ver[i] = 0;
                }
            }
            BFS(adj_list, level_ver, qu, in_degree, &max_level);
        }

        gettimeofday(&T_insert1, NULL);
#pragma omp parallel for
        for (int i = 0; i < insert_edges.size(); i++)
        {
            int ver1 = insert_edges[i].first;
            int ver2 = insert_edges[i].second;
            if (scc_list[ver1] != scc_list[ver2])
            {
                scc_edges[orig_edges.size() + i] = make_pair(scc_list[ver1], scc_list[ver2]);
                if (met_algo == 7)
                {
                    int scc1 = scc_list[ver1];
                    int scc2 = scc_list[ver2];
                    affect_level[min(level_ver[scc1], level_ver[scc2])] += 1;
                    affect_level[max(level_ver[scc1], level_ver[scc2]) + 1] += -1;
                }
            }
            if (met_algo == 11)
            {
                new_edge_nodes[scc_list[ver1]] = 1;
            }
        }
        gettimeofday(&T_insert2, NULL);
        insert_runtime = (T_insert2.tv_sec - T_insert1.tv_sec) * 1000 + (T_insert2.tv_usec - T_insert1.tv_usec) * 0.001;
        if (met_algo == 7)
        {
            parallel_prefix_sum(affect_level);
            for (int i = 0; i < affect_level.size() && i <= max_level; i++)
            {
                if (affect_level[i] == 0)
                {
                    unaffected_levels.push_back(i);
                }
            }
            cout << "size of unaffected_levels:" << unaffected_levels.size() << endl;
            for (int i = 0; i < (unaffected_levels.size()) - 1; i++)
            {
                if (unaffected_levels[i + 1] - unaffected_levels[i] > 1)
                    cout << "hey-hey-found" << endl;
            }
        }
    }

    int read_file1(string filename, vector<int> &scc_list, int num_vertices)
    {
        ifstream inputFile(filename);
        string line;
        int max_vertex = 0;
        scc_list.resize(num_vertices);
        int root_node = -1;
        vector<int> root_node_pres(num_vertices, -1);

        while (getline(inputFile, line))
        {
            vector<string> tokens;
            string token;
            stringstream ss(line);
            while (getline(ss, token, ' '))
            {
                tokens.push_back(token);
            }
            scc_list[stoi(tokens[0])] = stoi(tokens[1]);
            max_vertex = max(max_vertex, stoi(tokens[1]) + 1);
        }

        inputFile.close();

        return max_vertex;
    }

    void read_file2(string filename, vector<pair<int, int>> orig_edges, vector<int> scc_list, vector<vector<pair<int, int>>> &rem_edges, vector<int> &vec_scc_count, int num_sccs)
    {
        ifstream inputFile(filename);
        string line;
        unordered_map<string, int> ump;
        vec_scc_count.resize(scc_list.size(), -1);
        affect_sccs = 0;
        rem_edges.resize(num_sccs);

        while (getline(inputFile, line))
        {
            vector<string> tokens;
            string token;
            stringstream ss(line);
            while (getline(ss, token, ' '))
            {
                tokens.push_back(token);
            }
            int ver1 = stoi(tokens[0]) - 1;
            int ver2 = stoi(tokens[1]) - 1;
            string str1 = to_string(ver1) + "#" + to_string(ver2);
            ump[str1] = 1;
            if (scc_list[ver1] == scc_list[ver2])
            {
                if (vec_scc_count[scc_list[ver1]] == -1)
                {
                    affect_sccs++;
                }
                vec_scc_count[scc_list[ver1]] = 1;
            }
        }

        inputFile.close();

        for (int i = 0; i < orig_edges.size(); i++)
        {
            int ver1 = orig_edges[i].first;
            int ver2 = orig_edges[i].second;
            string str1 = to_string(ver1) + "#" + to_string(ver2);
            if (ump.find(str1) == ump.end())
            {
                rem_edges[scc_list[ver1]].push_back(make_pair(ver1, ver2));
            }
        }

        if (met_algo == 9)
        {
            unaffect_sccs = num_sccs - affect_sccs;
            cout << "affect_sccs:" << affect_sccs << " unaffect_sccs:" << unaffect_sccs << endl;
        }
    }

    virtual void main(int argc, char **argv)
    {
        bool b;
        if (argc < 3)
        {

            printf("%s <graph_name> <num_threads> ", argv[0]);
            print_arg_info();
            printf("\n");

            exit(EXIT_FAILURE);
        }

        int new_argc = argc - 3;
        char **new_argv = &(argv[3]);
        b = check_args(new_argc, new_argv);
        if (!b)
        {
            printf("error procesing argument\n");
            printf("%s <graph_name> <num_threads> ", argv[0]);
            print_arg_info();
            printf("\n");
            exit(EXIT_FAILURE);
        }

        int num = atoi(argv[2]);
        num_threads = num;
        gm_rt_set_num_threads(num); // gm_runtime.h
        if (num == -1)
        {
            printf("exploration mode\n", num);
        }
        else
        {
            printf("running with %d threads\n", num);
        }

        //--------------------------------------------
        // Load graph and creating reverse edges
        //--------------------------------------------
        struct timeval T1, T2, T6_1, T6_2;
        string fname = argv[1];
        vector<pair<int, int>> orig_edges;
        vector<pair<int, int>> insert_edges;
        // vector<pair<int, int>> scc_edges;
        vector<pair<int, int>> scc_edges;
        vector<vector<pair<int, int>>> rem_edges;
        // gettimeofday(&T1, NULL);
        // b = G.load_binary(fname);
        int met = atoi(new_argv[0]);
        met_algo = met;
        if (met == 5)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            gettimeofday(&T1, NULL);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            insert_idea1(G, orig_edges, insert_edges);
            vector<vector<int>> adj_list(num_vertices);
            for (int i = 0; i < orig_edges.size(); i++)
            {
                int ver1 = orig_edges[i].first;
                int ver2 = orig_edges[i].second;
                adj_list[ver1].push_back(ver2);
            }
            for (int i = 0; i < insert_edges.size(); i++)
            {
                int ver1 = insert_edges[i].first;
                int ver2 = insert_edges[i].second;
                adj_list[ver1].push_back(ver2);
            }
            int maxi_neigh = 0;
            for (int i = 0; i < adj_list.size(); i++)
            {
                if (adj_list[i].size() > (maxi_neigh))
                {
                    maxi_neigh = (int)adj_list[i].size();
                    // good_init_pivot=i;
                }
            }
            // cout<<"maxi_neigh:"<<maxi_neigh<<endl;
        }
        if (met == 6)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            int num_sccs = read_file1("/home/tk.temp/par-scc/scc_list.txt", scc_list, num_vertices);
            gettimeofday(&T6_1, NULL);
            create_scc_edges(orig_edges, insert_edges, scc_edges, num_vertices, num_sccs);
            gettimeofday(&T6_2, NULL);
            count_ver = num_vertices;
            count_scc = num_sccs;
            vector<vector<int>> scc_adj_list(num_sccs);
            for (int i = 0; i < scc_edges.size(); i++)
            {
                int ver1 = scc_edges[i].first;
                int ver2 = scc_edges[i].second;
                scc_adj_list[ver1].push_back(ver2);
            }
            // for (const auto& p : scc_edges) {
            //     int ver1=p.first;
            //     int ver2=p.second;
            //     scc_adj_list[ver1].push_back(ver2);
            // }
            int maxi_neigh = 0;
            for (int i = 0; i < scc_adj_list.size(); i++)
            {
                if (scc_adj_list[i].size() > (maxi_neigh))
                {
                    maxi_neigh = (int)scc_adj_list[i].size();
                    good_init_pivot = i;
                }
            }
            if (num_sccs > 1)
            {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            }
            else
            {
                G.add_node();
            }
            // G.add_node();
        }
        if (met == 11)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            int num_sccs = read_file1("/home/tk.temp/par-scc/scc_list.txt", scc_list, num_vertices);
            gettimeofday(&T6_1, NULL);
            create_scc_edges(orig_edges, insert_edges, scc_edges, num_vertices, num_sccs);
            gettimeofday(&T6_2, NULL);
            // cout<<"num_sccs:"<<num_sccs<<endl;
            vector<vector<int>> scc_adj_list(num_sccs);
            for (const auto &p : scc_edges)
            {
                int ver1 = p.first;
                int ver2 = p.second;
                scc_adj_list[ver1].push_back(ver2);
            }
            int maxi_neigh = 0;
            for (int i = 0; i < scc_adj_list.size(); i++)
            {
                if (scc_adj_list[i].size() > (maxi_neigh))
                {
                    maxi_neigh = (int)scc_adj_list[i].size();
                    good_init_pivot = i;
                }
            }
            // cout<<"maxi_neigh:"<<maxi_neigh<<endl;
            // cout<<"good_init_pivot:"<<good_init_pivot<<endl;
            // gettimeofday(&T6_2, NULL); // int num_vertices = 4;
            // vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}};

            // vector<Edge> mst = boruvkaMST(num_vertices, edges);

            // cout << "Edges in the MST:" << endl;
            // for (const auto& edge : mst) {
            //     cout << edge.u << " - " << edge.v << endl;
            // }


            gettimeofday(&T1, NULL);
            if (num_sccs > 1)
            {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            }
            else
            {
                G.add_node();
            }
        }
        if (met == 7)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            read_file(fname, insert_edges);
            int num_sccs = read_file1("/home/tk.temp/par-scc/scc_list.txt", scc_list, num_vertices);
            create_scc_edges(orig_edges, insert_edges, scc_edges, num_vertices, num_sccs);
            gettimeofday(&T1, NULL);
            if (num_sccs > 1)
            {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            }
            else
            {
                G.add_node();
            }
        }
        if (met == 8)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            int num_sccs = read_file1("/home/tk.temp/par-scc/scc_list.txt", scc_list, num_vertices);
            read_file2(fname, orig_edges, scc_list, rem_edges, vec_scc_count, num_sccs);
            Graph g1(num_vertices);
            for(int i=0;i<rem_edges.size();i++)
            {
                // cout<<rem_edges[0].size()<<endl;
                for(int j=0;j<rem_edges[i].size();j++)
                {
                    int ver1=rem_edges[i][j].first;
                    int ver2=rem_edges[i][j].second;
                    g1.add_edge(ver1,ver2);
                }
            }
            gettimeofday(&T1, NULL);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            /*Remove every (m/kn)th edge*/
            int count11=0;
            int total_edges=0;
            int max_edges=0;
            int maxi_index=0;
            for(int i=0;i<rem_edges.size();i++)
            {
                total_edges+=rem_edges[i].size();
                if(rem_edges[i].size()>max_edges)
                {
                    max_edges=max(max_edges,(int)rem_edges[i].size());
                    maxi_index=i;
                }
            }
            cout<<"Total edges:"<<total_edges<<endl;
            cout<<"Max edges:"<<max_edges<<endl;
            cout<<"Maxi index:"<<maxi_index<<endl;
            delete_idea1(G, rem_edges);
        }
        if (met == 9)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            int num_sccs = read_file1("/home/tk.temp/par-scc/scc_list.txt", scc_list, num_vertices);
            read_file2(fname, orig_edges, scc_list, rem_edges, vec_scc_count, num_sccs);
            gettimeofday(&T1, NULL);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            if (num_sccs < num_vertices)
            {
                delete_idea1(G, rem_edges);
            }
        }
        if (met == 10)
        {
            size_t lastPos = fname.rfind('/');
            string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
            int num_vertices = read_file(orig_fname, orig_edges);
            int num_sccs = read_file1("/home/tk.temp/par-scc/scc_list.txt", scc_list, num_vertices);
            read_file2(fname, orig_edges, scc_list, rem_edges, vec_scc_count, num_sccs);
            gettimeofday(&T1, NULL);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            if (num_sccs < num_vertices)
            {
                delete_idea1(G, rem_edges);
            }
            gettimeofday(&T1, NULL);
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            if (num_sccs < num_vertices)
            {
                delete_idea1(G, rem_edges);
            }
        }
        if (met == 2)
        {
            // cout<<"Num_Threads:"<<omp_get_max_threads()<<" "<<omp_get_num_procs()<<endl;
            int num_vertices = read_file(fname, orig_edges);
            gettimeofday(&T1, NULL);
            // num_vertices=10;
            for (int i = 0; i < num_vertices; i++)
                G.add_node();
            for (int i = 0; i < orig_edges.size(); i++)
            {
                G.add_edge(orig_edges[i].first, orig_edges[i].second);
            }
        }
        if (!b)
        {
            printf("error reading graph\n");
            exit(EXIT_FAILURE);
        }

        gettimeofday(&T2, NULL);
        printf("graph loading time=%lf\n",
               (T2.tv_sec - T1.tv_sec) * 1000 +
                   (T2.tv_usec - T1.tv_usec) * 0.001);

        gettimeofday(&T1, NULL);
        G.make_reverse_edges();
        gettimeofday(&T2, NULL);
        printf("reverse edge creation time=%lf\n",
               (T2.tv_sec - T1.tv_sec) * 1000 +
                   (T2.tv_usec - T1.tv_usec) * 0.001);

        cout << "data=" << fname << " " << met << " " << num << endl;

        //------------------------------------------------
        // Any extra preperation Step (provided by the user)
        //------------------------------------------------
        // if(met_algo!=10)
        // {
        if (num == -1)
        {
            int max = 32;
            for (int i = 1; i <= max; i = i * 2)
            {
                gm_rt_set_num_threads(i); // gm_runtime.h
                do_main_steps();
            }
        }
        else
        {
            // gm_rt_set_num_threads(num); // gm_runtime.h
            do_main_steps();
        }
        // }
    }

    void do_main_steps()
    {
        struct timeval T1, T2;
        printf("\n");
        pin_CPU();

        bool b = prepare();
        if (!b)
        {
            printf("Error prepare data\n");
            exit(EXIT_FAILURE);
        }

        gettimeofday(&T1, NULL);
        b = run();
        gettimeofday(&T2, NULL);
        printf("[%d]running_time(ms)=%lf\n",
               gm_rt_get_num_threads(),
               (T2.tv_sec - T1.tv_sec) * 1000 +
                   (T2.tv_usec - T1.tv_usec) * 0.001 - time_to_exclude + insert_runtime);
        fflush(stdout);
        if (!b)
        {
            printf("Error runing algortihm\n");
            exit(EXIT_FAILURE);
        }

        b = post_process();
        if (!b)
        {
            printf("Error post processing\n");
            exit(EXIT_FAILURE);
        }

        //----------------------------------------------
        // Clean up routine
        //----------------------------------------------
        b = cleanup();
        if (!b)
            exit(EXIT_FAILURE);
    }

    virtual bool check_answer() { return true; }
    virtual bool run() = 0;
    virtual bool prepare() { return true; }
    virtual bool post_process() { return true; }
    virtual bool cleanup() { return true; }
    // check remaining arguments
    virtual bool check_args(int argc, char **argv) { return true; }
    virtual void print_arg_info() {}

protected:
    gm_graph &get_graph() { return G; }
    void add_time_to_exlude(double ms) { time_to_exclude += ms; }
    double time_to_exclude;
};

#endif