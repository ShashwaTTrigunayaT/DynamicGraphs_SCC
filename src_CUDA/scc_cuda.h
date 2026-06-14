#ifndef SCC_CUDA_H
#define SCC_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <string>
#include <queue>
#include <vector>
#include <cuda_runtime.h>

// Forward declaration of gm_graph (used as reference parameter in dynamic helpers)
// Full definition is in gm.h, included by the .cpp files that instantiate these
class gm_graph;

typedef int32_t node_t;
typedef int32_t edge_t;
#define CUDA_NIL_NODE ((node_t)-1)

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
    int owns_set;            // 1 if this work item owns d_set_nodes (must free)
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

// ---- scc_cuda_fb_global.cu (mirrors scc_fb_global.cc) ----

// BFS queue buffers shared with fb_seq.cu / fb_seq2.cu
extern int* d_bfs_queue;
extern int* d_bfs_next_queue;
extern int* d_bfs_next_count;
extern int* d_bfs_scc_count;
extern int* d_bfs_bw_count;

// Pinned host memory + stream for async BFS level loop (pinned = faster D2H)
extern int* h_pinned_next_count;
extern int* h_pinned_scc_count;
extern int* h_pinned_bw_count;
extern cudaStream_t bfs_stream;

// initialize / finalize
void initialize_global_fb(int num_nodes);
void finalize_global_fb();

// BFS kernels (defined in scc_cuda_fb_global.cu) — extern for separate compilation
// OpenMP: fw_trim_global / bw_trim_global (BFS template classes)
extern __global__ void fw_bfs_level_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int base_color);

extern __global__ void bw_bfs_level_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int bw_color, int base_color, node_t pivot,
    int* d_scc_count, int* d_bw_count);

// Host function — exact mirror of do_fw_bw_global_main()
// Parameters match CPU: (curr_color, count) + good_init_pivot (for met_algo==6/11)
int do_global_fw_bw_main(GPUState& st, const GPUGraph& g,
    int base_color, int base_count, int good_init_pivot,
    bool create_work_items = false);

// Host function — exact mirror of create_works_after_bfs_trim()
// Creates work items from the colored partitions after re-trimming
void create_works_after_bfs_trim(GPUState& st, const GPUGraph& g);

// ---- scc_cuda_weak.cu (mirrors scc_weak.cc) ----

// initialize / finalize
void initialize_WCC(int num_nodes);
void finalize_WCC();

// Host functions — exact mirrors of OpenMP functions
void do_global_wcc(GPUState& st, const GPUGraph& g);
void create_work_items_from_wcc(GPUState& st, const GPUGraph& g);
int* get_WCC();

// ---- scc_cuda_work_queue.cu (mirrors my_work_queue.cc) ----

// initialize
void work_q_init(int num_threads);

// queue operations
int  work_q_size();
bool is_work_q_empty_from_seq_context();
void work_q_print_max_depth();

// put / fetch
void        work_q_put(int thread_id, CUDAMyWork* w);
void        work_q_put_all(int thread_id, std::vector<CUDAMyWork*>& works);
CUDAMyWork* work_q_fetch(int id);
void        work_q_fetch_N(int id, int N, std::vector<CUDAMyWork*>& works);

// scatter kernels for building compact sets
__global__ void scatter_by_color_kernel(
    const int* d_Color,
    const int* d_targets, int num_targets,
    int fw_color, int bw_color, int base_color,
    int* fw_out, int* fw_pos,
    int* bw_out, int* bw_pos,
    int* base_out, int* base_pos);

__global__ void scatter_single_color_kernel(
    const int* d_Color,
    const int* d_targets, int num_targets,
    int target_color,
    int* d_out, int* d_pos);

__global__ void scatter_by_root_kernel(
    const int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets,
    const int* d_root_offsets,
    int* d_root_buf);

__global__ void count_by_wcc_root_kernel(
    const int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets,
    int* d_root_counts);

__global__ void wcc_insert_members_kernel(
    const int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets,
    int* d_root_pos,
    int** d_wcc_sets_dev);

// ---- scc_cuda_color.cu (shared color allocator — mirrors scc_color.cc) ----
// OpenMP: static int _the_color; int get_new_color() { const int CHUNK=1024; ... }
extern int _cuda_the_color;
int  cuda_get_new_color();

// Kernels for all-node scan (used when d_trim_targets_count == 0)
// OpenMP: for(node_t i=0;i<G.num_nodes(); i++) if (G_Color[i]==color) ...
__global__ void find_pivot_all_nodes_kernel(
    const int* d_Color, int num_nodes,
    int base_color, int* d_pivot);

__global__ void scatter_single_color_all_nodes_kernel(
    const int* d_Color, int num_nodes,
    int target_color,
    int* d_out, int* d_pos);

// ---- scc_cuda_dynamic.cpp (mirrors common_main.h helpers) ----
// Host-side helpers for dynamic graph construction (compiled with g++)

int  read_file(const std::string& filename,
               std::vector<std::pair<int,int>>& edges_list);

int  read_file1(const std::string& filename,
                std::vector<int>& scc_list_out, int num_vertices);

void create_scc_edges(
    std::vector<std::pair<int,int>> orig_edges,
    std::vector<std::pair<int,int>> insert_edges,
    std::vector<std::pair<int,int>>& scc_edges,
    int num_vertices, int num_sccs, int met_algo,
    std::vector<int>& scc_list,
    std::vector<int>& level_ver,
    std::vector<int>& affect_level,
    std::vector<int>& new_edge_nodes,
    double& insert_runtime);

void BFS(std::vector<std::vector<int>>& adj_list,
         std::vector<int>& level,
         std::queue<int>& qu,
         std::vector<int>& in_degree, int* max_level);

void parallel_prefix_sum(std::vector<int>& a);

// ---- scc_cuda_dynamic.cpp (mirrors scc_incremental.cc) ----
void insert_idea1(gm_graph& G,
                  std::vector<std::pair<int,int>> orig_edges,
                  std::vector<std::pair<int,int>> insert_edges);
void insert_idea2(gm_graph& G,
                  std::vector<std::pair<int,int>> scc_edges);

// DynamicArrays upload helpers — upload host data to existing device buffers
void dynamic_arrays_upload_scc_list(DynamicArrays& da,
    const std::vector<int>& h_scc_list, int N);
void dynamic_arrays_upload_vec_scc_count(DynamicArrays& da,
    const std::vector<int>& h_vec_scc_count, int num_sccs);
void dynamic_arrays_upload_level_ver(DynamicArrays& da,
    const std::vector<int>& h_level_ver, int N);
void dynamic_arrays_upload_affect_level(DynamicArrays& da,
    const std::vector<int>& h_affect_level, int size);

// ---- scc_cuda_fb_seq.cu (mirrors scc_fb_seq.cc) ----

// Pivot-finding kernels (defined in scc_cuda_fb_seq.cu, also used by fb_seq2.cu)
extern __global__ void find_pivot_in_set_kernel(
    const int* d_Color, const int* d_set, int set_size,
    int target_color, int* d_pivot);

extern __global__ void find_pivot_by_color_kernel(
    const int* d_Color,
    const int* d_targets, int num_targets,
    int base_color, int* d_pivot);

// Host functions — exact mirrors of OpenMP functions
// Per-subgraph FW-BW (BFS-based): consumes CUDAMyWork items from the work queue
int do_fw_bw_single_thread(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, std::vector<CUDAMyWork*>& new_works);

void start_workers_fw_bw(GPUState& st, const GPUGraph& g, int N);

// ---- scc_cuda_fb_seq2.cu (mirrors scc_fb_seq2.cc) ----

// Host functions — exact mirrors of OpenMP functions
// Per-subgraph FW-BW (DFS-based): consumes CUDAMyWork items from the work queue
int do_fw_bw_dfs(GPUState& st, const GPUGraph& g,
    CUDAMyWork* w, std::vector<CUDAMyWork*>& new_works);

void start_workers_fw_bw_dfs(GPUState& st, const GPUGraph& g, int N);

#endif
