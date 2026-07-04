/*
Quick and simple graph generator for testing graph algorithms.
For Diameter graphs call :  generate_diameter_graph_to_file(45, 1000000, 10, 100000);
Diameter: 1M nodes, dia=45, total_sccs=100000

For LCC graphs Call :  generate_lcc_graph_to_file(1000000, 70, 10000000);
LCC: 1M nodes, 70% LCC, 10M edges
*/



#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include <vector>
#include <utility>
#include <string>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <stack>
#include <random>
#include <cstdio>
#include "gm.h"





inline std::vector<int> distribute_nodes(int total_nodes, int num_layers) {
    if (num_layers < 2) num_layers = 2;
    if (total_nodes < num_layers) total_nodes = num_layers;
    std::vector<int> sizes(num_layers);
    int base = total_nodes / num_layers;
    int rem  = total_nodes % num_layers;
    for (int i = 0; i < num_layers; i++)
        sizes[i] = base + (i < rem ? 1 : 0);
    return sizes;
}


inline long long base_edges_variable(const std::vector<int>& sizes) {
    if (sizes.empty()) return 0;
    long long total = 0;
    for (int sz : sizes) total += sz;
    for (size_t i = 0; i < sizes.size() - 1; i++)
        total += (long long)sizes[i] * sizes[i+1];
    return total;
}


inline long long base_edges_sparse(const std::vector<int>& sizes, int k) {
    if (sizes.empty()) return 0;
    long long total = 0;
    for (int sz : sizes) total += sz;          
    for (size_t i = 0; i < sizes.size() - 1; i++)
        total += (long long)k * sizes[i];      
    return total;
}


inline long long max_back_attempts_variable(const std::vector<int>& sizes) {
    if (sizes.size() < 2) return 0;
    long long total = 0;
    for (size_t i = 1; i < sizes.size(); i++)
        total += sizes[i];
    return total;
}




inline std::vector<int> distribute_sccs_randomly(int total_sccs, const std::vector<int>& layer_sizes) {
    int num_layers = (int)layer_sizes.size();
    std::vector<int> sub_sccs(num_layers, 0);
    if (num_layers == 0) return sub_sccs;
    
    
    int reserved = std::min(num_layers, total_sccs);
    for (int i = 0; i < reserved; i++) sub_sccs[i] = 1;
    int remaining = total_sccs - reserved;
    if (remaining <= 0) return sub_sccs;
    
    
    int total_nodes = 0;
    for (int sz : layer_sizes) total_nodes += sz;
    if (total_nodes == 0) total_nodes = num_layers;
    
    
    int allocated = 0;
    for (int i = 0; i < num_layers; i++) {
        int extra = (int)((long long)layer_sizes[i] * remaining / total_nodes);
        sub_sccs[i] += extra;
        allocated += extra;
    }
    
    
    int leftover = remaining - allocated;
    for (int i = 0; i < leftover; i++) {
        int r = std::rand() % total_nodes;
        int cumulative = 0;
        for (int j = 0; j < num_layers; j++) {
            cumulative += layer_sizes[j];
            if (r < cumulative) {
                sub_sccs[j]++;
                break;
            }
        }
    }
    
    return sub_sccs;
}



inline std::vector<std::pair<int, int>> generate_chain_edges(
    const std::vector<int>& layer_sizes,
    float back_edge_prob = 0.0f,
    int seed = -1)
{
    int num_layers = (int)layer_sizes.size();
    if (num_layers < 2) return {};
    std::vector<int> starts(num_layers);
    int total_nodes = 0;
    for (int i = 0; i < num_layers; i++) {
        starts[i] = total_nodes;
        total_nodes += layer_sizes[i];
    }
    
    size_t est_intra = (size_t)total_nodes;
    size_t est_inter = 0;
    for (int i = 0; i < num_layers - 1; i++)
        est_inter += (size_t)layer_sizes[i] * layer_sizes[i+1];
    std::vector<std::pair<int, int>> edges;
    edges.reserve(est_intra + est_inter);
    for (int l = 0; l < num_layers; l++) {
        int start = starts[l], sz = layer_sizes[l];
        for (int i = 0; i < sz; i++)
            edges.push_back({start + i, start + (i + 1) % sz});
    }
    for (int l = 0; l < num_layers - 1; l++) {
        int s0 = starts[l], s1 = starts[l+1];
        int sz0 = layer_sizes[l], sz1 = layer_sizes[l+1];
        for (int i = 0; i < sz0; i++)
            for (int j = 0; j < sz1; j++)
                edges.push_back({s0 + i, s1 + j});
    }
    if (back_edge_prob > 0.0f && num_layers > 1) {
        for (int l = num_layers - 1; l >= 1; l--) {
            int sz = layer_sizes[l];
            for (int attempt = 0; attempt < sz; attempt++) {
                if ((float)std::rand() / (float)RAND_MAX < back_edge_prob) {
                    int target = std::rand() % l;
                    int u = starts[l] + (std::rand() % layer_sizes[l]);
                    int v = starts[target] + (std::rand() % layer_sizes[target]);
                    edges.push_back({u, v});
                }
            }
        }
    }
    return edges;
}


inline std::vector<std::pair<int, int>> generate_chain_edges(
    int num_layers, int scc_size,
    float back_edge_prob = 0.0f,
    int seed = -1)
{
    if (num_layers < 2) num_layers = 2;
    if (scc_size < 1)   scc_size = 1;
    return generate_chain_edges(std::vector<int>(num_layers, scc_size), back_edge_prob, seed);
}




inline void stream_chain_edges(std::ofstream& out,
    const std::vector<int>& sizes,
    const std::vector<int>& starts,
    float back_edge_prob,
    int seed)
{
    int num_layers = (int)sizes.size();
    if (num_layers < 2) return;
    if (back_edge_prob > 0.0f) {
        if (seed == -1) seed = (int)std::time(nullptr);
        std::srand((unsigned int)seed);
    }
    for (int l = 0; l < num_layers; l++) {
        int start = starts[l], sz = sizes[l];
        for (int i = 0; i < sz; i++)
            out << (start + i + 1) << " " << (start + (i + 1) % sz + 1) << "\n";
    }
    for (int l = 0; l < num_layers - 1; l++) {
        int s0 = starts[l], s1 = starts[l+1];
        int sz0 = sizes[l], sz1 = sizes[l+1];
        for (int i = 0; i < sz0; i++)
            for (int j = 0; j < sz1; j++)
                out << (s0 + i + 1) << " " << (s1 + j + 1) << "\n";
    }
    if (back_edge_prob > 0.0f && num_layers > 1) {
        for (int l = num_layers - 1; l >= 1; l--) {
            int sz = sizes[l];
            for (int attempt = 0; attempt < sz; attempt++) {
                if ((float)std::rand() / (float)RAND_MAX < back_edge_prob) {
                    int target = std::rand() % l;
                    int u = starts[l] + (std::rand() % sizes[l]);
                    int v = starts[target] + (std::rand() % sizes[target]);
                    out << (u + 1) << " " << (v + 1) << "\n";
                }
            }
        }
    }
}



inline void stream_chain_edges_sparse(std::ofstream& out,
    const std::vector<int>& sizes,
    const std::vector<int>& starts,
    int k, int seed, const std::vector<int>& sub_sccs_per_layer)
{
    int num_layers = (int)sizes.size();
    if (num_layers < 2) return;
    std::srand((unsigned int)seed);

    
    
    for (int l = 0; l < num_layers; l++) {
        int start = starts[l], sz = sizes[l];
        int sub_count = sub_sccs_per_layer[l];
        if (sub_count < 1) sub_count = 1;
        
        
        
        std::vector<int> scc_sizes(sub_count, 1);
        int remaining = sz - sub_count;
        if (remaining < 0) {
            
            for (int i = sz; i < sub_count; i++) scc_sizes[i] = 0;
            remaining = 0;
        }
        for (int i = 0; i < remaining; i++)
            scc_sizes[std::rand() % sub_count]++;
        
        int pos = start;
        for (int s = 0; s < sub_count; s++) {
            int s_sz = scc_sizes[s];
            if (s_sz > 1) {
                for (int i = 0; i < s_sz; i++)
                    out << (pos + i + 1) << " " << (pos + (i + 1) % s_sz + 1) << "\n";
            }
            pos += s_sz;
        }
    }

    
    for (int l = 0; l < num_layers - 1; l++) {
        int s0 = starts[l], s1 = starts[l+1];
        int sz0 = sizes[l], sz1 = sizes[l+1];
        for (int i = 0; i < sz0; i++)
            for (int c = 0; c < k; c++)
                out << (s0 + i + 1) << " " << (s1 + (std::rand() % sz1) + 1) << "\n";
    }
}





inline gm_graph build_graph_from_edges(
    const std::vector<std::pair<int, int>>& edge_list,
    int num_vertices = -1)
{
    if (num_vertices <= 0) {
        num_vertices = 0;
        for (const auto& e : edge_list) {
            if (e.first  + 1 > num_vertices) num_vertices = e.first  + 1;
            if (e.second + 1 > num_vertices) num_vertices = e.second + 1;
        }
    }
    gm_graph g;
    for (int i = 0; i < num_vertices; i++) g.add_node();
    for (const auto& e : edge_list) g.add_edge(e.first, e.second);
    return g;
}





inline bool write_edge_list_to_file(
    const std::vector<std::pair<int, int>>& edge_list,
    const std::string& filename,
    const std::vector<std::string>& metadata = {})
{
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    for (const auto& line : metadata) out << "% " << line << "\n";
    for (const auto& e : edge_list)
        out << (e.first + 1) << " " << (e.second + 1) << "\n";
    out.close();
    int max_v = 0;
    for (const auto& e : edge_list) {
        if (e.first  > max_v) max_v = e.first;
        if (e.second > max_v) max_v = e.second;
    }
    std::cout << "Written " << filename
              << " (" << edge_list.size() << " edges, "
              << (max_v + 1) << " nodes)\n";
    return true;
}





inline long long diameter_base_edges(int layers, int scc_sz) {
    return (long long)layers * scc_sz + (long long)(layers - 1) * scc_sz * scc_sz;
}
inline float compute_back_prob(long long target_extra, int layers, int scc_sz) {
    if (layers < 2 || scc_sz < 1) return 0.0f;
    long long attempts = (long long)(layers - 1) * scc_sz;
    if (attempts < 1 || target_extra <= 0) return 0.0f;
    float p = (float)target_extra / (float)attempts;
    return (p > 1.0f) ? 1.0f : p;
}





inline bool stream_diameter_to_file(
    const std::vector<int>& sizes, const std::string& filename,
    float back_edge_prob, int seed, long long estimated_edges)
{
    int num_layers = (int)sizes.size();
    std::vector<int> starts(num_layers);
    int total_nodes = 0;
    for (int i = 0; i < num_layers; i++) {
        starts[i] = total_nodes; total_nodes += sizes[i];
    }
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "% graph_type: diameter\n% nodes: " << total_nodes
        << "\n% edges: " << estimated_edges
        << "\n% diameter: " << (num_layers - 1)
        << "\n% layers: " << num_layers << "\n";
    stream_chain_edges(out, sizes, starts, back_edge_prob, seed);
    out.close();
    std::cout << "Written " << filename
              << " (~" << estimated_edges << " edges, "
              << total_nodes << " nodes)\n";
    return true;
}

inline bool stream_diameter_uniform_to_file(
    int target_diameter, int scc_size,
    float back_edge_prob, int seed,
    long long estimated_edges, const std::string& filename)
{
    return stream_diameter_to_file(
        std::vector<int>(target_diameter, scc_size),
        filename, back_edge_prob, seed, estimated_edges);
}






inline bool stream_diameter_sparse_to_file(
    const std::vector<int>& sizes, const std::string& filename,
    int k, int seed, long long estimated_edges, const std::vector<int>& sub_sccs_per_layer)
{
    int num_layers = (int)sizes.size();
    std::vector<int> starts(num_layers);
    int total_nodes = 0;
    for (int i = 0; i < num_layers; i++) {
        starts[i] = total_nodes; total_nodes += sizes[i];
    }
    int tot_sccs = 0;
    for (int s : sub_sccs_per_layer) tot_sccs += s;
    
    
    int min_sccs = sub_sccs_per_layer[0], max_sccs = sub_sccs_per_layer[0];
    for (int s : sub_sccs_per_layer) {
        if (s < min_sccs) min_sccs = s;
        if (s > max_sccs) max_sccs = s;
    }
    
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "% graph_type: diameter\n% nodes: " << total_nodes
        << "\n% edges: " << estimated_edges
        << "\n% diameter: " << (num_layers - 1)
        << "\n% layers: " << num_layers
        << "\n% total_sccs: " << tot_sccs
        << "\n% sccs_per_layer: " << min_sccs << "-" << max_sccs << " (random, proportional to layer size)"
        << "\n% interconnect: sparse k=" << k << "\n";
    stream_chain_edges_sparse(out, sizes, starts, k, seed, sub_sccs_per_layer);
    out.close();
    std::cout << "Written " << filename
              << " (" << estimated_edges << " edges, "
              << total_nodes << " nodes, " << tot_sccs << " SCCs, k=" << k << ")\n";
    return true;
}





inline bool generate_diameter_graph_to_file(
    const std::string& filename,
    int target_diameter, int scc_size,
    float back_edge_prob = 0.0f, int seed = -1,
    int num_nodes = -1, int num_edges = -1)
{
    if (num_nodes > 0) {
        if (num_nodes % target_diameter == 0) {
            int s = num_nodes / target_diameter;
            if (s < 1) s = 1;
            scc_size = s;
            long long base = diameter_base_edges(target_diameter, scc_size);
            if (num_edges > 0 && (long long)num_edges > base) {
                back_edge_prob = compute_back_prob(num_edges - base, target_diameter, scc_size);
            } else {
                num_edges = (int)base;
            }
            return stream_diameter_uniform_to_file(
                target_diameter, scc_size, back_edge_prob, seed,
                num_edges, filename);
        } else {
            auto sizes = distribute_nodes(num_nodes, target_diameter);
            long long base = base_edges_variable(sizes);
            if (num_edges > 0 && (long long)num_edges > base) {
                long long attempts = max_back_attempts_variable(sizes);
                float p = (attempts > 0) ? (float)(num_edges - base) / (float)attempts : 0.0f;
                if (p > 1.0f) p = 1.0f;
                back_edge_prob = p;
                return stream_diameter_to_file(sizes, filename, back_edge_prob, seed, num_edges);
            } else {
                return stream_diameter_to_file(sizes, filename, 0.0f, seed, base);
            }
        }
    }
    
    auto edges = generate_chain_edges(target_diameter, scc_size, back_edge_prob, seed);
    int nnodes = target_diameter * scc_size;
    int nedges = (int)edges.size();
    std::vector<std::string> meta = {
        "graph_type: diameter", "nodes: " + std::to_string(nnodes),
        "edges: " + std::to_string(nedges),
        "diameter: " + std::to_string(target_diameter - 1),
        "layers: " + std::to_string(target_diameter),
        "scc_size: " + std::to_string(scc_size),
        "back_edge_prob: " + std::to_string(back_edge_prob)
    };
    return write_edge_list_to_file(edges, filename, meta);
}








inline bool generate_diameter_graph_to_file(int target_diameter, int num_nodes, int k = 10, int total_sccs = 1)
{
    if (target_diameter < 2) target_diameter = 2;
    if (num_nodes < target_diameter) num_nodes = target_diameter;
    if (total_sccs < 1) total_sccs = 1;
    int seed = (int)std::time(nullptr);
    std::srand((unsigned int)seed);
    auto sizes = distribute_nodes(num_nodes, target_diameter);
    auto sub_sccs_per_layer = distribute_sccs_randomly(total_sccs, sizes);
    int actual_sccs = 0;
    for (int s : sub_sccs_per_layer) actual_sccs += s;
    std::cout << "Distributed " << actual_sccs << " SCCs across " << target_diameter
              << " layers (random, proportional to layer size).\n";
    long long edges = base_edges_sparse(sizes, k);
    std::string name = "diameter_" + std::to_string(target_diameter) + "_" + std::to_string(num_nodes) + "_" + std::to_string(actual_sccs) + ".txt";
    return stream_diameter_sparse_to_file(sizes, name, k, seed, edges, sub_sccs_per_layer);
}



inline auto generate_diameter_graph(
    int target_diameter, int scc_size,
    float back_edge_prob = 0.0f, int num_nodes = -1,
    int num_edges = -1, int seed = -1)
    -> std::pair<gm_graph, std::vector<std::pair<int, int>>>
{
    if (num_nodes > 0) {
        if (num_nodes % target_diameter == 0) {
            int s = num_nodes / target_diameter;
            if (s < 1) s = 1;
            scc_size = s;
            long long base = diameter_base_edges(target_diameter, scc_size);
            if (num_edges > 0 && (long long)num_edges > base)
                back_edge_prob = compute_back_prob(num_edges - base, target_diameter, scc_size);
            else back_edge_prob = 0.0f;
        } else {
            auto sizes = distribute_nodes(num_nodes, target_diameter);
            long long base = base_edges_variable(sizes);
            if (num_edges > 0 && (long long)num_edges > base) {
                long long attempts = max_back_attempts_variable(sizes);
                float p = (attempts > 0) ? (float)(num_edges - base) / (float)attempts : 0.0f;
                back_edge_prob = (p > 1.0f) ? 1.0f : p;
            } else back_edge_prob = 0.0f;
            auto edges = generate_chain_edges(sizes, back_edge_prob, seed);
            int total_nodes = 0; for (int sz : sizes) total_nodes += sz;
            gm_graph g = build_graph_from_edges(edges, total_nodes);
            return {g, edges};
        }
    }
    auto edges = generate_chain_edges(target_diameter, scc_size, back_edge_prob, seed);
    int total_nodes = target_diameter * scc_size;
    gm_graph g = build_graph_from_edges(edges, total_nodes);
    return {g, edges};
}







inline bool stream_lcc_to_file(
    int num_nodes, int lcc_size, long long num_edges,
    int seed, const std::string& filename,
    int explicit_singletons = -1)
{
    if (lcc_size < 1) lcc_size = 1;
    if (lcc_size > num_nodes) lcc_size = num_nodes;
    int satellite_count = num_nodes - lcc_size;
    if (explicit_singletons > satellite_count) explicit_singletons = satellite_count;
    long long base_edges = num_nodes;
    long long extra = num_edges - base_edges;
    if (extra < 0) extra = 0;

    
    long long max_sat_to_giant = (long long)satellite_count * lcc_size;
    long long sat_to_giant = (extra < max_sat_to_giant) ? extra : max_sat_to_giant;
    long long intra_giant = extra - sat_to_giant;

    
    long long per_sat = (satellite_count > 0) ? sat_to_giant / satellite_count : 0;
    int rem_sat = (satellite_count > 0) ? (int)(sat_to_giant % satellite_count) : 0;

    std::srand((unsigned int)seed);

    
    std::vector<long long> node_budget(satellite_count, per_sat);
    for (int i = 0; i < rem_sat; i++) node_budget[i]++;
    
    
    std::vector<int> sat_scc_sizes;
    if (explicit_singletons >= 0) {
        // New: explicit_singletons singletons + remaining in small SCCs (size 2-5)
        int small_scc_nodes = satellite_count - explicit_singletons;
        if (small_scc_nodes < 0) small_scc_nodes = 0;
        int remaining_nodes = small_scc_nodes;
        while (remaining_nodes > 0) {
            int max_sz = std::min(remaining_nodes, 5);
            int min_sz = 2;
            int sz = (remaining_nodes <= max_sz) ? remaining_nodes : (std::rand() % (max_sz - min_sz + 1)) + min_sz;
            sat_scc_sizes.push_back(sz);
            remaining_nodes -= sz;
        }
        // Add singletons as size-1
        for (int i = 0; i < explicit_singletons; i++) {
            sat_scc_sizes.push_back(1);
        }
        // Shuffle so structure isn't clumped
        std::random_shuffle(sat_scc_sizes.begin(), sat_scc_sizes.end());
    } else {
        // Old behavior: random SCCs of size 1-200
        int remaining_nodes = satellite_count;
        while (remaining_nodes > 0) {
            int max_sz = std::min(remaining_nodes, std::max(1, 200));
            int sz = (std::rand() % max_sz) + 1;
            sat_scc_sizes.push_back(sz);
            remaining_nodes -= sz;
        }
    }
    int num_sat_sccs = (int)sat_scc_sizes.size();
    
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    int num_singletons = 0;
    int num_small_sccs = 0;
    if (explicit_singletons >= 0) {
        for (int sz : sat_scc_sizes) {
            if (sz == 1) num_singletons++;
            else num_small_sccs++;
        }
    }

    out << "% graph_type: lcc\n% nodes: " << num_nodes
        << "\n% edges: " << num_edges
        << "\n% lcc_size: " << lcc_size
        << "\n% satellite_nodes: " << satellite_count
        << "\n% satellite_sccs: " << num_sat_sccs;
    if (explicit_singletons >= 0) {
        out << "\n% singletons: " << num_singletons
            << "\n% small_sccs: " << num_small_sccs
            << "\n% small_scc_nodes: " << (satellite_count - num_singletons);
    }
    out << "\n% sat_to_giant_edges: " << sat_to_giant
        << "\n% intra_giant_edges: " << intra_giant << "\n";

    
    for (int i = 0; i < lcc_size; i++) {
        int nxt = (i + 1) % lcc_size;
        out << (i + 1) << " " << (nxt + 1) << "\n";
    }

    
    int sat_node_idx = 0;
    for (int s = 0; s < num_sat_sccs; s++) {
        int s_sz = sat_scc_sizes[s];
        int base_node = lcc_size + sat_node_idx;
        
        
        if (s_sz > 1) {
            for (int i = 0; i < s_sz; i++) {
                int nxt = (i + 1) % s_sz;
                out << (base_node + i + 1) << " " << (base_node + nxt + 1) << "\n";
            }
        } else {
            out << (base_node + 1) << " " << (base_node + 1) << "\n";
        }
        
        
        for (int n = 0; n < s_sz; n++) {
            long long edges_for_this = node_budget[sat_node_idx + n];
            for (long long e = 0; e < edges_for_this; e++) {
                int target = std::rand() % lcc_size;
                out << (base_node + n + 1) << " " << (target + 1) << "\n";
            }
        }
        
        sat_node_idx += s_sz;
    }

    
    for (long long e = 0; e < intra_giant; e++) {
        int u = std::rand() % lcc_size;
        int v = std::rand() % lcc_size;
        if (u == v) v = (v + 1) % lcc_size;  
        out << (u + 1) << " " << (v + 1) << "\n";
    }

    out.close();
    std::cout << "Written " << filename
              << " (" << num_edges << " edges, "
              << num_nodes << " nodes, LCC=" << lcc_size
              << ", " << num_sat_sccs << " satellite SCCs";
    if (explicit_singletons >= 0) {
        std::cout << ", " << num_singletons << " singletons, "
                  << num_small_sccs << " small SCCs";
    }
    std::cout << ")\n";
    return true;
}



inline bool generate_lcc_graph_to_file(
    int num_nodes, int lcc_percent, long long num_edges, int seed = -1,
    int singleton_percent = 0)
{
    if (num_nodes < 2) num_nodes = 2;
    if (lcc_percent < 1) lcc_percent = 1;
    if (lcc_percent > 99) lcc_percent = 99;
    if (singleton_percent < 0) singleton_percent = 0;
    if (lcc_percent + singleton_percent > 99) singleton_percent = 99 - lcc_percent;
    int lcc_size = num_nodes * lcc_percent / 100;
    if (lcc_size < 1) lcc_size = 1;
    if (lcc_size > num_nodes - 1) lcc_size = num_nodes - 1;
    int singleton_count = num_nodes * singleton_percent / 100;
    if (singleton_percent > 0 && singleton_count < 1) singleton_count = 1;
    long long min_edges = num_nodes;
    if (num_edges < min_edges) num_edges = min_edges;
    if (seed == -1) seed = (int)std::time(nullptr);

    
    std::string name = "lcc_" + std::to_string(lcc_percent) + "pct_"
                       + std::to_string(num_nodes) + "_"
                       + std::to_string(num_edges) + ".txt";
    if (singleton_percent > 0) {
        name = "lcc_" + std::to_string(lcc_percent) + "pct_"
               + std::to_string(singleton_percent) + "single_"
               + std::to_string(num_nodes) + "_"
               + std::to_string(num_edges) + ".txt";
    }
    int es = (singleton_percent > 0) ? singleton_count : -1;
    bool ok = stream_lcc_to_file(num_nodes, lcc_size, num_edges, seed, name, es);
    return ok;
}

// ================================================================
// SCC-Aware Insert Batch Generation (Split-Existing-Edges Approach)
//
// Takes a FULL graph + its SCC decomposition, splits edges into
// a base graph + batches. Batches are SCC-aware: each cumulative
// target %% controls how many SCCs are "affected" by the cross-SCC
// edges in that batch.
//
// Definition: an SCC is "affected" when it gets >=1 new outgoing
// cross-SCC edge (matching met_algo==11 logic: new_edge_nodes[...]=1).
//
// Python base: like generate_edge_batches(G, ratio=0.7, batches=10)
// but instead of random equal-sized batches, groups edges by SCC
// membership to hit target affected-SCC percentages.
//
// Algorithm:
//   1. Shuffle and split edges: initial_ratio → base, rest → candidate
//   2. Among candidates, separate cross-SCC edges (source_SCC != target_SCC)
//      from intra-SCC edges (same SCC) — O(1) via reverse-lookup
//   3. Sort SCCs by size descending
//   4. Build incremental batches: each batch uses cross-SCC edges
//      from the next block of SCCs to hit the cumulative target %%
//   5. Intra-SCC edges distributed proportionally by batch size
//   6. Result: base + all batches = original graph (all edges accounted for)
//
// Output files (1-indexed, matching project convention):
//   {prefix}_base.txt           (base graph edges)
//   {prefix}_scc10insert.txt    (edges affecting ~10%% SCCs cumulatively)
//   {prefix}_scc20insert.txt    (edges affecting ~20%% SCCs cumulatively)
//   ...
// ================================================================

inline bool generate_scc_aware_insert_batches(
    const std::vector<std::pair<int, int>>& all_edges,
    const std::vector<int>& scc_list,
    int num_sccs,
    const std::string& output_prefix = "insert_batch",
    const std::vector<float>& target_pcts = {0.10f, 0.20f, 0.40f, 0.60f},
    float initial_ratio = 0.70f,
    int seed = -1)
{
    if (all_edges.empty() || scc_list.empty() || num_sccs < 2) {
        std::cerr << "generate_scc_aware_insert_batches: insufficient data.\n";
        return false;
    }

    int num_nodes = (int)scc_list.size();
    if (seed == -1) seed = (int)std::time(nullptr);

    int total_edges = (int)all_edges.size();
    if (initial_ratio < 0.0f || initial_ratio > 1.0f) {
        std::cerr << "Warning: initial_ratio " << initial_ratio
                  << " out of [0,1], clamping to 0.7\n";
        initial_ratio = 0.70f;
    }
    int base_count = (int)(initial_ratio * total_edges);
    if (base_count < 1)      base_count = 1;
    if (base_count > total_edges) base_count = total_edges;

    // -------------------------------------------------------
    // 1. Shuffle and split edges: base + remaining
    //    Use std::shuffle with mt19937 (random_shuffle is deprecated)
    // -------------------------------------------------------
    std::mt19937 rng((unsigned int)seed);

    std::vector<int> order(total_edges);
    for (int i = 0; i < total_edges; i++) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<std::pair<int, int>> base_edges(base_count);
    std::vector<std::pair<int, int>> rem_edges(total_edges - base_count);
    for (int i = 0; i < base_count; i++)  base_edges[i] = all_edges[order[i]];
    for (int i = base_count; i < total_edges; i++) rem_edges[i - base_count] = all_edges[order[i]];

    // -------------------------------------------------------
    // 2. Build SCC node groups and reverse-lookup
    // -------------------------------------------------------
    // Group nodes by SCC
    std::vector<std::vector<int>> scc_nodes(num_sccs);
    for (int n = 0; n < num_nodes; n++) {
        int sid = scc_list[n];
        if (sid >= 0 && sid < num_sccs) scc_nodes[sid].push_back(n);
    }

    // Collect non-empty SCCs sorted by size descending
    struct SccInfo { int id; int size; };
    std::vector<SccInfo> active;
    for (int s = 0; s < num_sccs; s++) {
        if (!scc_nodes[s].empty()) active.push_back({s, (int)scc_nodes[s].size()});
    }
    int total_active = (int)active.size();
    if (total_active < 2) {
        std::cerr << "generate_scc_aware_insert_batches: need >=2 non-empty SCCs.\n";
        return false;
    }
    std::sort(active.begin(), active.end(),
        [](const SccInfo& a, const SccInfo& b) { return a.size > b.size; });

    // Reverse-lookup: SCC id → active[] index (O(1) per edge)
    std::vector<int> scc_to_active(num_sccs, -1);
    for (int i = 0; i < total_active; i++) scc_to_active[active[i].id] = i;

    // -------------------------------------------------------
    // 3. Classify remaining edges: cross-SCC vs intra-SCC (O(N))
    // -------------------------------------------------------
    std::vector<std::vector<std::pair<int, int>>> per_scc_xedges(total_active);
    std::vector<std::pair<int, int>> intra_edges;

    for (const auto& e : rem_edges) {
        int s1 = scc_list[e.first];
        int s2 = scc_list[e.second];
        if (s1 != s2 && s1 >= 0 && s1 < num_sccs && s2 >= 0 && s2 < num_sccs) {
            int idx = scc_to_active[s1];
            if (idx >= 0) per_scc_xedges[idx].push_back(e);
        } else {
            intra_edges.push_back(e);
        }
    }

    // -------------------------------------------------------
    // 4. Build batches, distribute intra edges proportionally
    // -------------------------------------------------------
    int cross_total = (int)(rem_edges.size() - intra_edges.size());
    int intra_total = (int)intra_edges.size();

    std::cout << "\nGenerating " << target_pcts.size()
              << " insert batches (initial_ratio=" << initial_ratio << ")\n"
              << "  Base: " << base_edges.size() << " edges\n"
              << "  Remaining: " << rem_edges.size() << " edges\n"
              << "    Cross-SCC: " << cross_total << "\n"
              << "    Intra-SCC: " << intra_total << "\n";

    // Write the base graph
    {
        std::string base_file = output_prefix + "_base.txt";
        std::vector<std::string> meta = {
            "base graph (" + std::to_string(100.0f * initial_ratio) + "% of full graph)",
            "edges: " + std::to_string(base_edges.size()),
            "nodes: " + std::to_string(num_nodes)
        };
        write_edge_list_to_file(base_edges, base_file, meta);
    }

    int prev_boundary = 0;
    size_t intra_consumed = 0;

    for (size_t t = 0; t < target_pcts.size(); t++) {
        int target_count = (int)(target_pcts[t] * total_active + 0.5f);
        if (target_count < 1) target_count = 1;
        if (target_count > total_active) target_count = total_active;
        int range_size = target_count - prev_boundary;
        if (range_size < 1) range_size = 1;

        std::vector<std::pair<int, int>> batch_edges;
        std::unordered_set<int> affected_scc_set;
        int cross_in_batch = 0;

        // Add cross-SCC edges from SCCs in [prev_boundary, target_count)
        for (int i = prev_boundary; i < target_count && i < total_active; i++) {
            for (const auto& e : per_scc_xedges[i]) {
                batch_edges.push_back(e);
                cross_in_batch++;
            }
            if (!per_scc_xedges[i].empty()) {
                affected_scc_set.insert(active[i].id);
            }
        }

        if (affected_scc_set.empty() && cross_total > 0) {
            std::cerr << "  Warning: batch " << (t+1)
                      << " (" << (int)(target_pcts[t]*100) << "%) has 0 cross-SCC edges.\n";
        }

        // Distribute intra-SCC edges proportionally by this batch's
        // fraction of total SCC range (range_size / total_active)
        float frac = (float)range_size / (float)total_active;
        int intra_target = (int)(frac * intra_total + 0.5f);
        int added = 0;
        while (added < intra_target && intra_consumed < intra_edges.size()) {
            batch_edges.push_back(intra_edges[intra_consumed]);
            intra_consumed++;
            added++;
        }

        // Write batch to file
        int pct_int = (int)(target_pcts[t] * 100 + 0.5f);
        std::string filename = output_prefix + "_scc" + std::to_string(pct_int) + "insert.txt";
        {
            std::ofstream out(filename);
            if (!out.is_open()) {
                std::cerr << "Error: cannot write " << filename << "\n";
                return false;
            }
            out << "% insert_batch: cumulative " << pct_int << "% SCCs affected\n";
            out << "% source_sccs_affected: " << (int)affected_scc_set.size() << "\n";
            out << "% cross_scc_edges: " << cross_in_batch << "\n";
            out << "% intra_scc_edges: " << added << "\n";
            out << "% total_edges: " << (int)batch_edges.size() << "\n";
            for (const auto& e : batch_edges) {
                out << (e.first + 1) << " " << (e.second + 1) << "\n";
            }
        }

        std::cout << "  " << filename << ": " << batch_edges.size()
                  << " edges (" << cross_in_batch << " cross, " << added << " intra), "
                  << affected_scc_set.size() << " affected SCCs"
                  << " (cumulative " << pct_int << "%)\n";

        prev_boundary = target_count;
    }

    // Any remaining intra edges → put in last batch
    if (intra_consumed < intra_edges.size()) {
        int pct_int = (int)(target_pcts.back() * 100 + 0.5f);
        std::string filename = output_prefix + "_scc" + std::to_string(pct_int) + "insert.txt";
        {
            std::ofstream out(filename, std::ios::app);
            if (out.is_open()) {
                size_t remaining = intra_edges.size() - intra_consumed;
                for (size_t i = intra_consumed; i < intra_edges.size(); i++) {
                    out << (intra_edges[i].first + 1) << " " << (intra_edges[i].second + 1) << "\n";
                }
                std::cout << "  (appended " << remaining << " remaining intra edges to "
                          << filename << ")\n";
            }
        }
    }

    std::cout << "  Base + all batches = " << total_edges << " edges (original graph restored)\n";

    return true;
}

// ================================================================
// Kosaraju SCC Computation (Iterative — handles 1M+ nodes safely)
// ================================================================

inline std::vector<int> compute_scc(
    const std::vector<std::pair<int, int>>& edges,
    int num_nodes)
{
    if (num_nodes <= 0) return {};

    // Build adjacency lists
    std::vector<std::vector<int>> adj(num_nodes), radj(num_nodes);
    for (const auto& e : edges) {
        int u = e.first, v = e.second;
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes) {
            adj[u].push_back(v);
            radj[v].push_back(u);
        }
    }

    // --- Pass 1: iterative DFS to get finish order ---
    // Each stack entry: (node, iterator_index, state)
    // state = 0: first visit (push children)
    // state = 1: returning (record finish order)
    struct Frame { int u; size_t idx; int state; };

    std::vector<bool> visited(num_nodes, false);
    std::vector<int> finish_order;
    finish_order.reserve(num_nodes);

    for (int start = 0; start < num_nodes; start++) {
        if (visited[start]) continue;

        std::stack<Frame> stk;
        stk.push({start, 0, 0});
        visited[start] = true;

        while (!stk.empty()) {
            Frame& f = stk.top();

            if (f.state == 0) {
                // Push unvisited children
                if (f.idx < adj[f.u].size()) {
                    int v = adj[f.u][f.idx];
                    f.idx++;
                    if (!visited[v]) {
                        visited[v] = true;
                        stk.push({v, 0, 0});
                    }
                } else {
                    f.state = 1;  // all children processed
                }
            } else {
                finish_order.push_back(f.u);
                stk.pop();
            }
        }
    }

    // --- Pass 2: iterative DFS on reverse graph in finish order ---
    std::vector<int> scc(num_nodes, -1);
    int scc_count = 0;

    for (int i = num_nodes - 1; i >= 0; i--) {
        int start = finish_order[i];
        if (scc[start] != -1) continue;

        std::stack<int> stk;
        stk.push(start);
        scc[start] = scc_count;

        while (!stk.empty()) {
            int u = stk.top();
            stk.pop();
            for (int v : radj[u]) {
                if (scc[v] == -1) {
                    scc[v] = scc_count;
                    stk.push(v);
                }
            }
        }
        scc_count++;
    }

    std::cout << "computed " << scc_count << " SCCs from "
              << num_nodes << " nodes, " << edges.size() << " edges\n";
    return scc;
}


// ================================================================
// File-reader helper: loads edge list, returns edges + num_nodes
// ================================================================

inline bool read_graph_file(
    const std::string& filename,
    std::vector<std::pair<int, int>>& out_edges,
    int& out_num_nodes)
{
    out_edges.clear();
    out_num_nodes = 0;

    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        int u, v;
        if (sscanf(line.c_str(), "%d %d", &u, &v) == 2) {
            u--; v--;  // 1-indexed → 0-indexed
            out_edges.push_back({u, v});
            if (u + 1 > out_num_nodes) out_num_nodes = u + 1;
            if (v + 1 > out_num_nodes) out_num_nodes = v + 1;
        }
    }
    return true;
}


// ================================================================
// File-based overload — NO scc_list needed!
// Reads graph file, computes SCCs internally, then generates batches.
// ================================================================

inline bool generate_scc_aware_insert_batches(
    const std::string& graph_file,
    const std::string& output_prefix = "insert_batch",
    const std::vector<float>& target_pcts = {0.10f, 0.20f, 0.40f, 0.60f},
    float initial_ratio = 0.70f,
    int seed = -1)
{
    // 1. Read graph from file
    std::vector<std::pair<int, int>> edges;
    int num_nodes = 0;
    if (!read_graph_file(graph_file, edges, num_nodes)) return false;

    if (edges.empty() || num_nodes < 2) {
        std::cerr << "Error: graph has insufficient data.\n";
        return false;
    }

    // 2. Compute SCCs internally
    std::cout << "Reading " << graph_file << " (" << edges.size()
              << " edges, " << num_nodes << " nodes)\n";
    std::cout << "Computing SCC decomposition...\n";

    std::vector<int> scc = compute_scc(edges, num_nodes);

    // Count SCCs
    int num_sccs = 0;
    for (int s : scc) if (s + 1 > num_sccs) num_sccs = s + 1;

    if (num_sccs < 2) {
        std::cerr << "Error: graph has only " << num_sccs << " SCC (<2).\n";
        return false;
    }

    // 3. Call the in-memory version
    return generate_scc_aware_insert_batches(
        edges, scc, num_sccs, output_prefix, target_pcts, initial_ratio, seed);
}

#endif 
