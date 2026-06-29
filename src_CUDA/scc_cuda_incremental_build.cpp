// ======================================================================
// scc_cuda_incremental_build.cpp — Incremental graph construction
// (methods 5, 6, 7, 11)
//
// 1:1 CUDA mirror of common_main.h's incremental graph construction
// blocks. This logic was extracted from scc_cuda_main.cpp to keep the
// main file focused on the pipeline.
//
// OpenMP: inline in main_t::main() inside common_main.h
//   - method 5:  load refined_edges.txt + insert_edges → insert_idea1()
//   - method 6:  + scc_list.txt → create_scc_edges() → good_init_pivot → insert_idea2()
//   - method 11: same as 6 + new_edge_nodes for pivot hint
//   - method 7:  same as 6 + BFS levels + affect_level tracking
//
// Compiled with g++ (gm_graph is not CUDA-compatible).
// ======================================================================

#include "scc_cuda.h"
#include "gm.h"

#include <string>
#include <vector>
#include <utility>
#include <sys/time.h>
#include <iostream>

using namespace std;

// ======================================================================
// insert_idea1() — 1:1 mirror of scc_incremental.cc
//
// Builds a graph with ALL edges (original + insert batch).
// Used by method 5 (naive incremental recomputation).
// ======================================================================
void insert_idea1(gm_graph &G,
                  vector<pair<int,int> > orig_edges,
                  vector<pair<int,int> > insert_edges)
{
    for(int i=0;i<(int)orig_edges.size();i++)
    {
        G.add_edge(orig_edges[i].first,orig_edges[i].second);
    }
    for(int i=0;i<(int)insert_edges.size();i++)
    {
        G.add_edge(insert_edges[i].first,insert_edges[i].second);
    }
}

// ======================================================================
// insert_idea2() — 1:1 mirror of scc_incremental.cc
//
// Builds an SCC condensation graph from cross-SCC edges only.
// Used by methods 6, 7, 11 (incremental on condensation graph).
// ======================================================================
void insert_idea2(gm_graph &G,
                  vector<pair<int,int> > scc_edges)
{
    for(int i=0;i<(int)scc_edges.size();i++)
    {
        int ver1=scc_edges[i].first;
        int ver2=scc_edges[i].second;
        if(ver1+ver2 !=0)
            G.add_edge(ver1,ver2);
    }
}

// ======================================================================
// build_incremental_graph() — 1:1 mirror of common_main.h incremental
// graph construction for methods 5, 6, 7, 11.
//
// Populates the gm_graph G with the appropriate graph structure
// (naive or condensation) for the requested incremental method.
//
// Parameters:
//   G                    — gm_graph to build (modified in-place)
//   fname                — path to insert_edges file
//   met_algo_original    — which incremental method (5, 6, 7, or 11)
//   num_sccs             — [out] number of SCCs in the condensation graph
//   good_init_pivot      — [out] best pivot SCC (SCC with most cross-SCC neighbors)
//   insert_runtime       — [out] time spent in insert edge processing (ms)
//   h_scc_list           — [out] per-vertex SCC label array
//   h_level_ver          — [out] per-SCC BFS level (method 7 only)
//   h_affect_level       — [out] per-level affected count (method 7 only)
//   h_new_edge_nodes     — [out] nodes with new edges (method 11 only)
//   gpu_graph_built      — [out] true if GPU condensation graph was built
//   h_gpu_begin          — [out] CSR begin array (GPU path)
//   h_gpu_node_idx       — [out] CSR node_idx array (GPU path)
//   h_gpu_r_begin        — [out] CSR reverse begin array (GPU path)
//   h_gpu_r_node_idx     — [out] CSR reverse node_idx array (GPU path)
//   gpu_N                — [out] num_nodes for GPU-constructed graph
//   gpu_M                — [out] num_edges for GPU-constructed graph
// ======================================================================
void build_incremental_graph(
    gm_graph& G,
    const string& fname,
    int met_algo_original,
    int& num_sccs,
    int& good_init_pivot,
    double& insert_runtime,
    vector<int>& h_scc_list,
    vector<int>& h_level_ver,
    vector<int>& h_affect_level,
    vector<int>& h_new_edge_nodes,
    bool& gpu_graph_built,
    vector<edge_t>& h_gpu_begin,
    vector<node_t>& h_gpu_node_idx,
    vector<edge_t>& h_gpu_r_begin,
    vector<node_t>& h_gpu_r_node_idx,
    int& gpu_N, int& gpu_M)
{
    vector<pair<int,int>> orig_edges;
    vector<pair<int,int>> insert_edges;
    vector<pair<int,int>> scc_edges;

    gpu_graph_built = false;

    // ---- Method 5 (Incremental, naive graph) ----
    // OpenMP: read refined_edges.txt + insert_edges, insert_idea1
    // Stays on CPU — intentionally the naive full-rebuild baseline.
    if (met_algo_original == 5)
    {
        size_t lastPos = fname.rfind('/');
        string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
        int num_vertices = read_file(orig_fname, orig_edges);
        read_file(fname, insert_edges);
        for (int i = 0; i < num_vertices; i++)
            G.add_node();
        // OpenMP: insert_idea1(G, orig_edges, insert_edges)
        insert_idea1(G, orig_edges, insert_edges);
    }

    // ---- Method 6 (Incremental, SCC condensation graph) ----
    // GPU path: filter edges on GPU, build CSR directly, skip gm_graph.
    // Replaces: create_scc_edges (CPU) + insert_idea2 (gm_graph).
    if (met_algo_original == 6)
    {
        size_t lastPos = fname.rfind('/');
        string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
        int num_vertices = read_file(orig_fname, orig_edges);
        read_file(fname, insert_edges);
        num_sccs = read_file1(
            "/home/tk.temp/par-scc/scc_list.txt", h_scc_list, num_vertices);

        // GPU: filter cross-SCC edges, build CSR, find pivot
        bool gpu_ok = build_gpu_condensation_graph(
            orig_edges, insert_edges, h_scc_list, num_sccs,
            good_init_pivot,
            h_gpu_begin, h_gpu_node_idx, h_gpu_r_begin, h_gpu_r_node_idx,
            gpu_N, gpu_M);

        if (gpu_ok && gpu_M > 0) {
            gpu_graph_built = true;
            insert_runtime = 0.0;
        } else if (gpu_ok && gpu_M == 0) {
            // No cross-SCC edges: single SCC, trivial case
            gpu_graph_built = true;  // CSR is still valid (all zero begin)
            insert_runtime = 0.0;
        } else {
            // Fallback: CPU path (shouldn't happen)
            fprintf(stderr, "[WARN] GPU condensation graph builder failed, falling back to CPU\n");
            create_scc_edges(orig_edges, insert_edges, scc_edges,
                num_vertices, num_sccs, met_algo_original,
                h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
                insert_runtime);
            {
                vector<vector<int>> scc_adj_list(num_sccs);
                int maxi_neigh = 0;
                good_init_pivot = 0;
                for (size_t i = 0; i < scc_edges.size(); i++) {
                    int ver1 = scc_edges[i].first;
                    int ver2 = scc_edges[i].second;
                    scc_adj_list[ver1].push_back(ver2);
                }
                for (int i = 0; i < num_sccs; i++) {
                    if ((int)scc_adj_list[i].size() > maxi_neigh) {
                        maxi_neigh = (int)scc_adj_list[i].size();
                        good_init_pivot = i;
                    }
                }
            }
            if (num_sccs > 1) {
                for (int i = 0; i < num_sccs; i++)
                    G.add_node();
                insert_idea2(G, scc_edges);
            } else {
                G.add_node();
            }
        }
    }

    // ---- Method 11 (Incremental, SCC condensation + pivot hint) ----
    if (met_algo_original == 11)
    {
        size_t lastPos = fname.rfind('/');
        string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
        int num_vertices = read_file(orig_fname, orig_edges);
        read_file(fname, insert_edges);
        num_sccs = read_file1(
            "/home/tk.temp/par-scc/scc_list.txt", h_scc_list, num_vertices);

        create_scc_edges(orig_edges, insert_edges, scc_edges,
            num_vertices, num_sccs, met_algo_original,
            h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
            insert_runtime);

        // OpenMP: choose good_init_pivot
        {
            vector<vector<int>> scc_adj_list(num_sccs);
            int maxi_neigh = 0;
            good_init_pivot = 0;
            for (const auto& p : scc_edges) {
                scc_adj_list[p.first].push_back(p.second);
            }
            for (int i = 0; i < num_sccs; i++) {
                if ((int)scc_adj_list[i].size() > maxi_neigh) {
                    maxi_neigh = (int)scc_adj_list[i].size();
                    good_init_pivot = i;
                }
            }
        }

        // OpenMP: build condensation graph
        if (num_sccs > 1) {
            for (int i = 0; i < num_sccs; i++)
                G.add_node();
            insert_idea2(G, scc_edges);
        } else {
            G.add_node();
        }
    }

    // ---- Method 7 (Incremental, SCC condensation + BFS levels) ----
    if (met_algo_original == 7)
    {
        size_t lastPos = fname.rfind('/');
        string orig_fname = fname.substr(0, lastPos) + "/refined_edges.txt";
        int num_vertices = read_file(orig_fname, orig_edges);
        read_file(fname, insert_edges);
        num_sccs = read_file1(
            "/home/tk.temp/par-scc/scc_list.txt", h_scc_list, num_vertices);

        // OpenMP: create_scc_edges (includes BFS levels + affect_level for met_algo==7)
        create_scc_edges(orig_edges, insert_edges, scc_edges,
            num_vertices, num_sccs, met_algo_original,
            h_scc_list, h_level_ver, h_affect_level, h_new_edge_nodes,
            insert_runtime);

        // OpenMP: build condensation graph
        if (num_sccs > 1) {
            for (int i = 0; i < num_sccs; i++)
                G.add_node();
            insert_idea2(G, scc_edges);
        } else {
            G.add_node();
        }
    }
}
