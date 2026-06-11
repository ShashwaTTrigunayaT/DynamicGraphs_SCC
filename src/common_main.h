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

    int read_file(string filename, vector<pair<int, int>> &edges_list)
    {
        int max_vertex = 0;
        ifstream inputFile(filename);
        string line;

        while (getline(inputFile, line))
        {
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
        // struct timeval T6_1_new, T6_2_new;
        // gettimeofday(&T6_1_new, NULL);
        // if(met_algo==6)
        // {
        //     cout<<"hello"<<endl;
        //     #pragma omp parallel for
        //     for(int i=0;i<orig_edges.size();i++)
        //     {
        //         int ver1=orig_edges[i].first;
        //         int ver2=orig_edges[i].second;
        //         // if(scc_list[ver1]!=scc_list[ver2])
        //         // {
        //         //     vec_pair[i]=make_pair(scc_list[ver1],scc_list[ver2]);
        //         // }
        //         // else
        //         //     vec_pair[i]=make_pair(INT_MAX,INT_MAX);
        //         vec_pair[i]=make_pair(scc_list[ver1],scc_list[ver2]);
        //     }
        //     #pragma omp parallel for
        //     for(int i=orig_edges.size();i<vec_pair.size();i++)
        //     {
        //         int ver1=insert_edges[i].first;
        //         int ver2=insert_edges[i].second;
        //         // if(scc_list[ver1]!=scc_list[ver2])
        //         // {
        //         //     vec_pair[i]=make_pair(scc_list[ver1],scc_list[ver2]);
        //         // }
        //         // else
        //         //     vec_pair[i]=make_pair(INT_MAX,INT_MAX);
        //         vec_pair[i]=make_pair(scc_list[ver1],scc_list[ver2]);
        //     }
        //     // parallel_sort_pairs(vec_pair);
        //     #pragma omp parallel
        //     {
        //         #pragma omp single
        //         parallelMergeSort(vec_pair, 0, vec_pair.size() - 1);
        //     }

        //     std::vector<bool> is_unique(vec_pair.size(), false);
        //     if (!vec_pair.empty()) is_unique[0] = true;

        //     #pragma omp parallel for
        //     for (size_t i = 1; i < vec_pair.size(); ++i) {
        //         if (vec_pair[i] != vec_pair[i-1]) {
        //             is_unique[i] = true;
        //         }
        //     }

        // }
        // gettimeofday(&T6_2_new, NULL);
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

    void create_synthetic_graph_insert(int num_sccs, int num_new_edges, vector<pair<int, int>> levels_affect)
    {
        gm_graph g_syn_scc;
        for (int i = 0; i < num_sccs; i++)
        {
            g_syn_scc.add_node();
        }
        for (int i = 0; i < 10; i++)
        {
            g_syn_scc.add_edge(0, i + 1);
        }
        for (int i = 11; i < num_new_edges; i++)
        {
            g_syn_scc.add_edge(i, i - 10);
        }

        for (int i = 0; i < levels_affect.size(); i++)
        {
            int level1 = levels_affect[i].first;
            int level2 = levels_affect[i].second;
            /*Pick three-four random vertices and then add edges between them*/
        }
    }

    void create_synthetic_graph_del(int num_nodes)
    {
        int num_high_scc_nodes = (int)(0.4 * num_nodes);
        int num_small_scc_nodes = (int)(0.1 * num_nodes);
        // gm_graph G_del_orig;
        for (int i = 0; i < num_high_scc_nodes; i++)
        {
            G.add_node();
        }
        for (int i = 0; i < num_high_scc_nodes - 1; i++)
        {
            G.add_edge(i, i + 1);
        }
        // G.add_edge(num_high_scc_nodes-10,10);

        int last_vertex = num_high_scc_nodes;

        // for(int j=0;j<6;j++)
        // {
        //     for(int i=0;i<num_small_scc_nodes;i++)
        //     {
        //         G.add_node();
        //     }
        //     for(int i=last_vertex;i<last_vertex+num_small_scc_nodes-1;i++)
        //     {
        //         G.add_edge(i,i+1);
        //     }
        //     G.add_edge(last_vertex+num_small_scc_nodes-1,0);
        //     last_vertex=last_vertex+num_small_scc_nodes;
        // }
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
            // gettimeofday(&T6_2, NULL);
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
            int count12=0;
            cout<<"num removed edges:"<<(4*num_vertices)<<endl;
            cout<<"num removed edges index:"<<(total_edges/(4*num_vertices))<<endl;
            int k=4;
            // for(int i=0;i<1;i++)
            // {
                if(rem_edges[5].size()>k)
                {
                    for(int j=0;j<rem_edges[5].size();j++)
                    {
                        count11++;
                        if((count11%(total_edges/(k*num_vertices)))!=0)
                            rem_edges[5][j]=make_pair(-1,-1);
                        // int ver1=rem_edges[i][j].first;
                        // int ver2=rem_edges[i][j].second;
                        // if(scc_list[ver1]!=scc_list[ver2])
                        // {
                        //     count12++;
                        // }
                    }
                }
            // }
            // cout<<count12<<endl;
            /*Remove kn random edges*/
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