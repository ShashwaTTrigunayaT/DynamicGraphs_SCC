#ifndef SCC_CUDA_H
#define SCC_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <vector>
#include <cuda_runtime.h>

typedef int32_t node_t;
typedef int32_t edge_t;
#define NIL_NODE ((node_t)-1)

#define SCC_FOUND        (-2)
#define COLOR_UNASSIGNED (-1)

// GPU Graph (CSR format — mirrors gm_graph on device)
struct GPUGraph {
    edge_t* d_begin;         // [N+1] G.begin
    node_t* d_node_idx;      // [M]   G.node_idx
    edge_t* d_r_begin;       // [N+1] G.r_begin
    node_t* d_r_node_idx;    // [M]   G.r_node_idx
    int num_nodes;
    int num_edges;
};

// GPU SCC State (mirrors G_Color, G_SCC arrays)
struct GPUState {
    int* d_Color;            // [N] G_Color
    int* d_SCC;              // [N] G_SCC
    int num_nodes;
};

// Dynamic Arrays for methods 7, 9, 11
struct DynamicArrays {
    int* d_scc_list;         // [N]
    int* d_vec_scc_count;    // [num_sccs]
    int* d_level_ver;        // [N]
    int* d_affect_level;     // [max_level+1]
};

// CUDAMyWork — mirror of my_work from my_work_queue.h
struct CUDAMyWork {
    int color;               // color of the base-set
    int count;               // count of the set
    int* d_set_nodes;        // device: compact list of nodes in this set
    int set_capacity;        // allocated capacity
    int depth;               // recursion depth
};

// CUDA Error Macro
#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(err)); exit(1);                    \
    }                                                                 \
} while(0)

// ---- graph.cu ----
void graph_upload(GPUGraph& gpu,
    const std::vector<edge_t>& h_begin,
    const std::vector<node_t>& h_node_idx,
    const std::vector<edge_t>& h_r_begin,
    const std::vector<node_t>& h_r_node_idx,
    int N, int M);
void graph_free(GPUGraph& gpu);

// ---- state.cu ----
void state_allocate(GPUState& st, int N);
void state_init(GPUState& st);
void state_free(GPUState& st);

// ---- dynamic_arrays.cu ----
void dynamic_arrays_allocate(DynamicArrays& da, int N, int num_sccs);
void dynamic_arrays_free(DynamicArrays& da);

// ---- scc_cuda_trim1.cu (mirrors scc_trim1.cc) ----

// Device-side global state
extern int* d_trim_targets;
extern int  d_trim_targets_count;
extern int  d_trim_targets_capacity;

// initialize / finalize (mirrors scc_trim1.cc)
void initialize_trim1();                      // exact mirror of OpenMP
void initialize_trim1_full(int num_nodes);    // CUDA: full allocation (GPU memory setup)
void finalize_trim1();                        // CUDA: cleanup (no OpenMP equivalent)

// get_compact_trim_targets() — returns device pointer + count (replaces vector reference)
int* get_compact_trim_targets_device();
int  get_compact_trim_targets_count();

// Core device function — exact mirror of trim_once_node()
__device__ void trim_once_node_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    node_t n,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec);

// Kernel: do_global_trim1 — iterates over ALL nodes
__global__ void trim_once_node_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    int num_nodes,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec);

// Kernel: do_global_trim1_compact — iterates over trim_targets
__global__ void trim_once_node_compact_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_trim_targets, int num_targets,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec);

// Kernel: do_local_trim1 — iterates over a work item's set
__global__ void trim_once_node_local_set_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_set_nodes, int set_size,
    int base_color,
    int met_algo, int flag11,
    const int* d_scc_list, const int* d_vec_scc_count,
    const int* d_level_ver, const int* d_affect_level,
    int* d_count_trim_spec);

// Host functions — exact mirrors of OpenMP functions
int do_global_trim1(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec);

int do_global_trim1_compact(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec);

int do_local_trim1(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, int* d_count,
    int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec);

int repeat_global_trim1(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec,
    int TRIM_STOP);

int repeat_global_trim1_compact(GPUState& st, const GPUGraph& g,
    int* d_count, int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec,
    int TRIM_STOP);

int repeat_local_trim1(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, int* d_count,
    int met_algo, int flag11,
    const DynamicArrays& da, int* d_count_trim_spec);

// Compact build helpers (mirrors create_trim1_compact)
void create_trim1_compact(GPUState& st, const GPUGraph& g);

// ---- scc_cuda_trim2.cu (mirrors scc_trim2.cc) ----

// Device-side global state
extern int* d_G_nbr;
extern int* d_G_maybe_2nd;

// initialize / finalize
void initialize_trim2(int num_nodes);
void finalize_trim2();

// Phase 1: classify nodes by degree pattern
__device__ void trim_2nd_main1_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count, node_t n);

// Phase 2: check mutual pointing -> 2-node SCC
__device__ void trim_2nd_main2_device(
    int* d_Color, int* d_SCC,
    int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count, node_t n);

// Kernels
__global__ void trim_2nd_main1_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count,
    const int* d_targets, int num_targets);

__global__ void trim_2nd_main2_kernel(
    int* d_Color, int* d_SCC,
    int* d_G_nbr, int* d_G_maybe_2nd,
    int* d_count,
    const int* d_targets, int num_targets);

// Host functions
int do_global_trim2(GPUState& st, const GPUGraph& g, int* d_count);
int repeat_global_trim2(GPUState& st, const GPUGraph& g,
    int* d_count, int exit_count);

// ---- scc_cuda_trim2_new.cu (mirrors scc_trim2_new.cc) ----

// Helper: check if node has exactly 1 unique out-neighbor in curr_color
__device__ bool check_out_degree_is_one_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color, node_t n, int curr_color, node_t* the_nbr);

// Helper: check if node has exactly 1 unique in-neighbor in curr_color
__device__ bool check_in_degree_is_one_device(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, node_t n, int curr_color, node_t* the_nbr);

// Single-pass 2-node SCC detection
__device__ void trim_2nd_new_main_device(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count, node_t n);

// Kernel
__global__ void trim_2nd_new_main_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int* d_count,
    const int* d_targets, int num_targets);

// Host functions
int do_global_trim2_new(GPUState& st, const GPUGraph& g, int* d_count);
int repeat_global_trim2_new(GPUState& st, const GPUGraph& g,
    int* d_count, int exit_count);

#endif
