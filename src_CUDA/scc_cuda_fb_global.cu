#include "scc_cuda.h"

// ======================================================================
// Device-side global state: analogs of OpenMP static globals
// ======================================================================

// OpenMP:
//   class thread_local_t {
//   public:
//       int val0;
//       int val1;
//       int padding[16];
//   };
//   thread_local_t  thread_data[MAX_THREADS];
//   static int init_fw_color;
//   static int init_bw_color;
//   static int init_base_color;
//   extern int pivot_fix;
//   extern int good_init_pivot;
//   extern int maxi_neigh_del;
//   extern float count_ver;
//   extern float count_scc;
// CUDA mirrors: good_init_pivot for met_algo==6/11 pivot override
// (now passed as parameter to do_global_fw_bw_main, not static)

// CUDA: BFS queue buffers (replace OpenMP's BFS template infrastructure)
// Non-static so they can be shared with fb_seq.cu / fb_seq2.cu
int* d_bfs_queue       = NULL;  // [N] current frontier
int* d_bfs_next_queue   = NULL;  // [N] next frontier  
int* d_bfs_next_count   = NULL;  // [1] atomic counter for next level size
int* d_bfs_scc_count    = NULL;  // [1] atomic counter for SCC nodes found
int* d_bfs_bw_count     = NULL;  // [1] atomic counter for bw-colored nodes

// Visited bitmap for BFS node claiming (separate from d_Color to avoid L2 thrashing)
// Bitmask: 1 bit per node, size = ceil(N/32) uint32_t ≈ 200KB for 1.6M nodes
// Node n corresponds to bit d_bfs_visited_bits[n/32] & (1 << (n%32))
uint32_t* d_bfs_visited_bits = NULL;
int d_bfs_visited_words = 0;

// Persistent scratch buffers to avoid cudaMalloc/cudaFree in hot BFS path
int* d_pivot_scratch    = NULL;  // [1] temp for pivot selection
int* d_remain_scratch   = NULL;  // [1] temp for remaining count

// ---- Block-local persistent kernel state ----
// The persistent kernel replaces the host-level loop for high-diameter graphs.
// It keeps the BFS resident on a single block, looping levels via __syncthreads()
// instead of launching a new kernel per level.
// Spills to global if the frontier exceeds MAX_SMEM_FRONTIER.

// Maximum frontier size that fits in shared memory (matched to gpu_fb_batch_kernel)
// For deep-narrow graphs (wiki-Talk: 1-5 nodes/level), this is always sufficient.
// If the frontier grows beyond this, the kernel spills to global and the host
// takes over (host loop handles broad frontiers efficiently).
#define MAX_SMEM_FRONTIER 2048

// Instrumentation: log frontier sizes per level for tuning MAX_SMEM_FRONTIER.
// Set to 1 to collect data on Pokec/ljournal-2008 frontier distributions.
// Collected data is printed to stderr on finalize_global_fb().
#define ENABLE_FRONTIER_LOG 0

#if ENABLE_FRONTIER_LOG
// Ring buffer of frontier sizes, one per level (max 10000 levels)
static int* d_frontier_log  = NULL;
static int* d_frontier_pos  = NULL;  // [1] atomic position in log
static int* h_frontier_log  = NULL;  // host side after download
static int  g_frontier_log_cap = 10000;
#endif

// Spill flag for persistent kernel:
//   -1 = BFS completed without spill (all done)
//    0 = kernel not yet launched
//    1 = spill occurred, host must continue from d_bfs_next_queue
static int* d_spill_flag     = NULL;

// Pinned host memory + stream for async BFS level loop
// Pinned memory enables faster D2H transfers (avoids staging buffer)
cudaStream_t bfs_stream        = NULL;
int*         h_pinned_next_count = NULL;  // pinned: next frontier size
int*         h_pinned_scc_count  = NULL;  // pinned: SCC count
int*         h_pinned_bw_count   = NULL;  // pinned: BW count

// Host-side persistent kernel control
int g_persistent_kernel_attempts = 0;  // stats counter

static int init_fw_color;
static int init_bw_color;
static int init_base_color;

// ======================================================================
// Shared color allocator — EXACT mirror of scc_color.cc get_new_color()
//
// OpenMP:
//   static int _the_color;
//   int get_new_color() {
//       const int CHUNK=1024;
//       int tid = gm_rt_thread_id();
//       int used         = the_colors[tid*16+0];
//       int max_assigned = the_colors[tid*16+1];
//       if (used == max_assigned) {
//           max_assigned = the_colors[tid*16+1] =
//               __sync_add_and_fetch(&_the_color, CHUNK);
//           used = the_colors[tid*16+0] = max_assigned - CHUNK + 1;
//       } else {
//           used = ++the_colors[tid*16+0];
//       }
//       return used;
//   }
//
// CUDA: single-threaded equivalent (tid=0, stride-1 cache)
// ======================================================================
int _cuda_the_color = -1;

static int _cuda_color_used = -1;
static int _cuda_color_max_assigned = -1;

int cuda_get_new_color()
{
    const int CHUNK = 1024;
    if (_cuda_color_used == _cuda_color_max_assigned) {
        _cuda_color_max_assigned = __sync_add_and_fetch(&_cuda_the_color, CHUNK);
        _cuda_color_used = _cuda_color_max_assigned - CHUNK + 1;
    } else {
        _cuda_color_used++;
    }
    return _cuda_color_used;
}

// ======================================================================
// initialize_global_fb()
// OpenMP:
//   void initialize_global_fb() 
//   {
//       for(int i=0;i<gm_rt_get_num_threads(); i++)
//       {
//           thread_data[i].val0 = 0; thread_data[i].val1= 0;
//       }
//   }
// ======================================================================
void initialize_global_fb(int num_nodes)
{
    if (d_bfs_queue)      cudaFree(d_bfs_queue);
    if (d_bfs_next_queue) cudaFree(d_bfs_next_queue);
    if (d_bfs_next_count) cudaFree(d_bfs_next_count);
    if (d_bfs_scc_count)  cudaFree(d_bfs_scc_count);
    if (d_bfs_bw_count)   cudaFree(d_bfs_bw_count);

    CUDA_CHECK(cudaMalloc(&d_bfs_queue,       num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bfs_next_queue,  num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bfs_next_count,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bfs_scc_count,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bfs_bw_count,    sizeof(int)));

    // Allocate visited bitmap: 1 bit per node, sized as uint32_t words
    d_bfs_visited_words = (num_nodes + 31) / 32;
    if (d_bfs_visited_bits) cudaFree(d_bfs_visited_bits);
    CUDA_CHECK(cudaMalloc(&d_bfs_visited_bits, d_bfs_visited_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_bfs_visited_bits, 0, d_bfs_visited_words * sizeof(uint32_t)));

    // Allocate persistent scratch buffers (allocated once, reused across multiple
    // do_global_fw_bw_main calls, avoiding per-call cudaMalloc/cudaFree overhead)
    if (!d_pivot_scratch)  CUDA_CHECK(cudaMalloc(&d_pivot_scratch,  sizeof(int)));
    if (!d_remain_scratch) CUDA_CHECK(cudaMalloc(&d_remain_scratch, sizeof(int)));

    // Allocate spill flag for persistent kernel (0 = no spill, -1 = done, 1 = spill)
    if (!d_spill_flag) CUDA_CHECK(cudaMalloc(&d_spill_flag, sizeof(int)));

    // Allocate pinned host memory for async D2H copies (faster D2H, no staging)
    if (!h_pinned_next_count) CUDA_CHECK(cudaMallocHost(&h_pinned_next_count, sizeof(int)));
    if (!h_pinned_scc_count)  CUDA_CHECK(cudaMallocHost(&h_pinned_scc_count,  sizeof(int)));
    if (!h_pinned_bw_count)   CUDA_CHECK(cudaMallocHost(&h_pinned_bw_count,  sizeof(int)));

    // Create stream for async kernel launches + memcpy
    if (!bfs_stream) CUDA_CHECK(cudaStreamCreate(&bfs_stream));

#if ENABLE_FRONTIER_LOG
    // Allocate frontier logging buffers
    if (!d_frontier_log) CUDA_CHECK(cudaMalloc(&d_frontier_log, g_frontier_log_cap * sizeof(int)));
    if (!d_frontier_pos) CUDA_CHECK(cudaMalloc(&d_frontier_pos, sizeof(int)));
    if (!h_frontier_log) h_frontier_log = new int[g_frontier_log_cap];
    CUDA_CHECK(cudaMemset(d_frontier_pos, 0, sizeof(int)));
#endif

    // Reset stats
    g_persistent_kernel_attempts = 0;

    init_fw_color = 0;
    init_bw_color = 0;
    init_base_color = 0;

    // OpenMP: initialize_color() sets _the_color = -1 and per-thread caches to -1
    _cuda_the_color = -1;
    _cuda_color_used = -1;
    _cuda_color_max_assigned = -1;
}

void finalize_global_fb()
{
    if (d_bfs_queue)      { cudaFree(d_bfs_queue);      d_bfs_queue = NULL; }
    if (d_bfs_next_queue) { cudaFree(d_bfs_next_queue); d_bfs_next_queue = NULL; }
    if (d_bfs_next_count) { cudaFree(d_bfs_next_count); d_bfs_next_count = NULL; }
    if (d_bfs_scc_count)  { cudaFree(d_bfs_scc_count);  d_bfs_scc_count = NULL; }
    if (d_bfs_bw_count)   { cudaFree(d_bfs_bw_count);   d_bfs_bw_count = NULL; }

    // Free visited bitmap (200KB)
    if (d_bfs_visited_bits) { cudaFree(d_bfs_visited_bits); d_bfs_visited_bits = NULL; }
    d_bfs_visited_words = 0;

    // Free persistent scratch buffers
    if (d_pivot_scratch)  { cudaFree(d_pivot_scratch);  d_pivot_scratch = NULL; }
    if (d_remain_scratch) { cudaFree(d_remain_scratch); d_remain_scratch = NULL; }

    // Free spill flag
    if (d_spill_flag)     { cudaFree(d_spill_flag);     d_spill_flag = NULL; }

    // Free pinned memory and destroy stream
    if (h_pinned_next_count) { cudaFreeHost(h_pinned_next_count); h_pinned_next_count = NULL; }
    if (h_pinned_scc_count)  { cudaFreeHost(h_pinned_scc_count); h_pinned_scc_count = NULL; }
    if (h_pinned_bw_count)   { cudaFreeHost(h_pinned_bw_count);  h_pinned_bw_count = NULL; }
    if (bfs_stream)          { cudaStreamDestroy(bfs_stream);     bfs_stream = NULL; }

#if ENABLE_FRONTIER_LOG
    // Print frontier size log and clean up
    if (h_frontier_log) {
        int count;
        CUDA_CHECK(cudaMemcpy(&count, d_frontier_pos, sizeof(int), cudaMemcpyDeviceToHost));
        if (count > g_frontier_log_cap) count = g_frontier_log_cap;
        CUDA_CHECK(cudaMemcpy(h_frontier_log, d_frontier_log,
                               count * sizeof(int), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[FRONTIER_LOG] %d levels:\n", count);
        for (int i = 0; i < count && i < 1000; i++) {
            fprintf(stderr, "%d%s", h_frontier_log[i],
                    (i + 1) % 50 == 0 ? "\n" : " ");
        }
        if (count > 1000) fprintf(stderr, "... (%d more)\n", count - 1000);
        fprintf(stderr, "\n");
        delete[] h_frontier_log; h_frontier_log = NULL;
    }
    if (d_frontier_log) { cudaFree(d_frontier_log); d_frontier_log = NULL; }
    if (d_frontier_pos) { cudaFree(d_frontier_pos); d_frontier_pos = NULL; }
#endif
}

// ======================================================================
// fw_trim_global — Forward BFS (single level kernel)
//
// OpenMP:
//   class fw_trim_global : public gm_bfs_template
//       <short, true, true, false, false>
//   {
//   public:
//       fw_trim_global(gm_graph& _G, int32_t _base_color, int32_t _fw_color)
//       : gm_bfs_template<short, true, true, false, false>(_G),
//         G(_G), fw_color(_fw_color), base_color(_base_color) { count = 0; }
//
//       int get_fw_count() {return count;}
//
//   protected:
//       virtual void visit_fw(node_t k) 
//       {
//           G_Color[k] = fw_color ;
//           thread_data[gm_rt_thread_id()].val0 ++;
//       }
//       virtual void do_end_of_level_fw() {
//           for(int i=0;i<gm_rt_get_num_threads();i++) {
//               count += thread_data[i].val0; 
//               thread_data[i].val0 = 0;
//           }
//       }
//       virtual void visit_rv(node_t k9) {}
//       virtual bool check_navigator(node_t k9, edge_t k9_idx) 
//       {
//           return (G_Color[k9] == base_color);
//       }
//   };
// ======================================================================

// Device function: check_navigator for forward BFS
// OpenMP: return (G_Color[k9] == base_color);
__device__ bool fw_check_navigator_device(int* d_Color, node_t k9, int base_color)
{
    return (d_Color[k9] == base_color);
}

// ======================================================================
// Kernel: one level of forward BFS
//
// Optimization: Visited bitmap (atomicOr) + per-thread staging.
// Instead of atomicCAS on d_Color (which invalidates L2 cache lines
// needed by navigator reads), claim nodes via atomicOr on a separate
// visited bitmap (200KB, stays L2-resident). On claim success, write
// d_Color with a simple store (no CAS).
// Also: per-thread staging of claimed nodes reduces atomicAdd by 32x.
// ======================================================================
__global__ void fw_bfs_level_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int base_color,
    uint32_t* d_visited_bits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Per-thread staging buffer (STAGE_SIZE=4 fits in registers, no local memory spill)
    const int STAGE_SIZE = 4;
    int staged[STAGE_SIZE];
    int staged_cnt = 0;

    // Helper: flush local buffer to global queue (single atomicAdd per flush)
#define FW_FLUSH() do {                                                 \
    if (staged_cnt > 0) {                                               \
        int base = atomicAdd(d_next_count, staged_cnt);                 \
        for (int _j = 0; _j < staged_cnt; _j++)                         \
            d_next_queue[base + _j] = staged[_j];                       \
        staged_cnt = 0;                                                  \
    }                                                                    \
} while(0)

    for (int i = tid; i < queue_size; i += stride) {
        node_t t = d_queue[i];
        for (edge_t nx = d_begin[t]; nx < d_begin[t + 1]; nx++) {
            node_t k = d_node_idx[nx];
            // Navigate: check if node has base_color (d_Color stays in L2 because
            // we don't CAS it — only write with simple store after claiming)
            if (fw_check_navigator_device(d_Color, k, base_color)) {
                // Claim via visited bitmap: atomicOr on 200KB bitmask (L2-resident)
                int word = k >> 5;  // k / 32
                uint32_t bit = 1u << (k & 31);
                uint32_t old = atomicOr(&d_visited_bits[word], bit);
                if ((old & bit) == 0) {
                    // Claimed! Write color with simple store
                    d_Color[k] = fw_color;
                    staged[staged_cnt++] = k;
                    if (staged_cnt == STAGE_SIZE) {
                        FW_FLUSH();
                    }
                }
            }
        }
    }

    // Flush remaining
    FW_FLUSH();
#undef FW_FLUSH
}

// ======================================================================
// bw_trim_global — Backward BFS (single level kernel)
//
// OpenMP:
//   class bw_trim_global : public gm_bfs_template
//       <short, true, true, true, false>
//   {
//   public:
//       bw_trim_global(gm_graph& _G, int32_t _base_color, int32_t _fw_color,
//                       int32_t _bw_color, node_t _pivot)
//       : gm_bfs_template<short, true, true, true, false>(_G),
//         G(_G), fw_color(_fw_color), base_color(_base_color),
//         bw_color(_bw_color), pivot(_pivot) 
//       { count = 0; scc_count = 0; }
//
//       int get_bw_count() {return count;}
//       int get_scc_count() {return scc_count;}
//
//   protected:
//       virtual void visit_fw(node_t k) 
//       {
//           if (G_Color[k] == fw_color)     // intersection
//           {
//               G_SCC[k] = pivot ;
//               G_Color[k] = -2;
//               thread_data[gm_rt_thread_id()].val1 ++;
//           }
//           else {                          // bw-set
//               G_Color[k] = bw_color;
//               thread_data[gm_rt_thread_id()].val0 ++;
//           }
//       }
//       virtual void do_end_of_level_fw() {
//           for(int i=0;i<gm_rt_get_num_threads();i++) {
//               count += thread_data[i].val0; 
//               scc_count += thread_data[i].val1; 
//               thread_data[i].val0 = 0;
//               thread_data[i].val1 = 0;
//           }
//       }
//       virtual void visit_rv(node_t k10) {}
//       virtual bool check_navigator(node_t k10, edge_t k10_idx) 
//       {
//           int color = G_Color[k10];
//           return (color == fw_color) || (color == base_color) ;
//       }
//   };
// ======================================================================

// Device function: check_navigator for backward BFS
// OpenMP: return (color == fw_color) || (color == base_color);
__device__ bool bw_check_navigator_device(int* d_Color, node_t k10,
    int fw_color, int base_color)
{
    int color = d_Color[k10];
    return (color == fw_color) || (color == base_color);
}

// ======================================================================
// Kernel: one level of backward BFS
//
// Fix: Eliminate TOCTOU race by reading d_Color[k] ONCE before both
//      the navigator check and the visit_fw logic. The single read
//      eliminates the window where another thread could change the
//      color between the navigator check and the if-else dispatch.
//
// Optimization: Visited bitmap (atomicOr) + per-thread staging.
// Same approach as FW BFS: claim via atomicOr on 200KB visited bitmap,
// then write d_Color with simple stores (no CAS).
// ======================================================================
__global__ void bw_bfs_level_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int bw_color, int base_color, node_t pivot,
    int* d_scc_count, int* d_bw_count,
    uint32_t* d_visited_bits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Per-thread staging buffer (STAGE_SIZE=4 fits in registers)
    const int STAGE_SIZE = 4;
    int staged[STAGE_SIZE];
    int staged_cnt = 0;
    int local_scc = 0;
    int local_bw = 0;

    // Helper: flush local buffer to global queue (single atomicAdd per flush)
#define BW_FLUSH() do {                                                 \
    if (staged_cnt > 0) {                                               \
        int base = atomicAdd(d_next_count, staged_cnt);                 \
        for (int _j = 0; _j < staged_cnt; _j++)                         \
            d_next_queue[base + _j] = staged[_j];                       \
        staged_cnt = 0;                                                  \
    }                                                                    \
} while(0)

    for (int i = tid; i < queue_size; i += stride) {
        node_t t = d_queue[i];
        for (edge_t nx = d_r_begin[t]; nx < d_r_begin[t + 1]; nx++) {
            node_t k = d_r_node_idx[nx];
            // Single read of d_Color[k] for TOCTOU safety
            int k_color = d_Color[k];

            // Navigate: check if node is fw_color (intersection) or base_color (bw-set)
            if (k_color == fw_color || k_color == base_color) {
                // Claim via visited bitmap
                int word = k >> 5;
                uint32_t bit = 1u << (k & 31);
                uint32_t old = atomicOr(&d_visited_bits[word], bit);
                if ((old & bit) == 0) {
                    if (k_color == fw_color) {
                        // Intersection: mark as SCC
                        d_Color[k] = SCC_FOUND;
                        d_SCC[k] = pivot;
                        local_scc++;
                    } else {
                        // BW-set
                        d_Color[k] = bw_color;
                        local_bw++;
                    }
                    staged[staged_cnt++] = k;
                    if (staged_cnt == STAGE_SIZE) {
                        BW_FLUSH();
                    }
                }
            }
        }
    }

    // Flush remaining queue entries
    BW_FLUSH();
#undef BW_FLUSH

    // Flush local SCC / BW counters
    if (local_scc > 0) atomicAdd(d_scc_count, local_scc);
    if (local_bw > 0) atomicAdd(d_bw_count, local_bw);
}

// ======================================================================
// pick_pivot_kernel()
// OpenMP: pivot = choose_pivot_from_color(G, base_color);
// Parallel: each thread checks one target, uses atomicMin for race-safe write
// ======================================================================
__global__ void pick_pivot_kernel(
    const int* d_Color, int* d_pivot,
    const int* d_targets, int num_targets,
    int base_color)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t = d_targets[i];
        if (d_Color[t] == base_color) {
            atomicMin(d_pivot, (int)t);
        }
    }
}

// ======================================================================
// count_remaining_kernel()
// OpenMP: count of nodes with base_color (trivial early exit)
// ======================================================================
__global__ void count_remaining_kernel(
    const int* d_Color, int* d_count,
    const int* d_targets, int num_targets,
    int base_color)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local = 0;
    for (int i = tid; i < num_targets; i += stride) {
        if (d_Color[d_targets[i]] == base_color)
            local++;
    }
    if (local > 0)
        atomicAdd(d_count, local);
}

// ======================================================================
// count_by_colors_kernel()
// OpenMP count by color in compact set (for create_works_after_bfs_trim)
// ======================================================================
__global__ void count_by_colors_kernel(
    const int* d_Color,
    const int* d_targets, int num_targets,
    int fw_color, int bw_color, int base_color,
    int* d_fw_count, int* d_bw_count, int* d_base_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_fw = 0, local_bw = 0, local_base = 0;
    for (int i = tid; i < num_targets; i += stride) {
        node_t t = d_targets[i];
        int c = d_Color[t];
        if (c == fw_color)       local_fw++;
        else if (c == bw_color)  local_bw++;
        else if (c == base_color) local_base++;
    }
    if (local_fw > 0)   atomicAdd(d_fw_count, local_fw);
    if (local_bw > 0)   atomicAdd(d_bw_count, local_bw);
    if (local_base > 0) atomicAdd(d_base_count, local_base);
}

// ======================================================================
// persistent_fw_bfs_kernel
//
// Block-local persistent FW BFS that loops levels via __syncthreads()
// instead of re-launching a kernel per level. Designed for deep-narrow
// BFS frontiers (1-5 nodes/level) where kernel launch overhead dominates.
//
// SMEM layout per block:
//   [0..max_f-1]        : s_queue (int)     — current frontier
//   [max_f..2*max_f-1]  : s_next (int)      — next frontier
//   [2*max_f]           : s_qsize (int)     — current frontier size
//   [2*max_f+1]         : s_ncount (int)    — next frontier count
//   [2*max_f+2]         : s_spilled (int)   — 1 if overflow occurred
//
// Layout bytes = 2*max_f*4 + 3*4 = 8*max_f + 12
// For MAX=2048: 16384 + 12 = 16396 bytes < 48KB ✓
//
// Spill behavior:
//   If the next frontier exceeds MAX_SMEM_FRONTIER nodes, the kernel
//   writes all overflow nodes to global d_bfs_next_queue and sets
//   d_spill_flag = 1. The host can then continue the BFS from the
//   spilled frontier using the existing per-level kernel loop.
//
// Output: d_bfs_next_count contains the number of nodes in the spilled
//         frontier (0 if BFS completed). d_spill_flag = -1 (done) or 1 (spill).
// ======================================================================
__global__ void persistent_fw_bfs_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    int pivot, int base_color, int fw_color,
    int* d_bfs_queue,          // [N] global queue buffer (for spill)
    int* d_bfs_next_queue,     // [N] global next queue buffer (for spill)
    int* d_bfs_next_count,     // [1] global next count (for spill)
    uint32_t* d_visited_bits,  // global visited bitmap
    int* d_spill_flag,         // [1] output: -1 done, 1 spilled
    int* d_level_counter)      // [1] output: number of levels processed (for stats)
{
    // Only one block needed — sequential BFS on a single component
    if (blockIdx.x > 0) return;

    extern __shared__ int smem[];
    int* s_queue   = &smem[0];
    int* s_next    = &smem[MAX_SMEM_FRONTIER];
    volatile int& s_qsize   = smem[2 * MAX_SMEM_FRONTIER];
    volatile int& s_ncount  = smem[2 * MAX_SMEM_FRONTIER + 1];
    volatile int& s_spilled = smem[2 * MAX_SMEM_FRONTIER + 2];

    int tid = threadIdx.x;
    int stride = blockDim.x;
    const int STAGE_SIZE = 4;

    // ---- Initialize: pivot is the first frontier ----
    if (tid == 0) {
        s_queue[0] = pivot;
        s_qsize = 1;
        s_spilled = 0;
        *d_bfs_next_count = 0;  // reset spill count
    }
    __syncthreads();

    int total_levels = 0;
    int local_levels = 0;

    int qsize = s_qsize;  // declared outside while for scope in report section

    // ---- Main loop: one BFS level per iteration ----
    // Safety cap at 100000 levels to prevent hangs on corrupted state.
    while (s_qsize > 0 && s_spilled == 0 && local_levels < 100000) {
        qsize = s_qsize;
        if (tid == 0) s_ncount = 0;
        __syncthreads();

#if ENABLE_FRONTIER_LOG
        // Log frontier size for this level
        if (tid == 0) {
            int pos = atomicAdd(d_level_counter, 1);
            if (pos < 10000) d_bfs_next_queue[pos] = qsize;  // reuse buffer for log
        }
#endif

        // Per-thread staging buffer for claimed nodes
        int staged[STAGE_SIZE];
        int staged_cnt = 0;

#define PERSISTENT_FLUSH(to_queue, count_var) do {                         \
    if (staged_cnt > 0) {                                                  \
        int base = atomicAdd((int*)&count_var, staged_cnt);                \
        for (int _j = 0; _j < staged_cnt; _j++) {                          \
            if (base + _j < MAX_SMEM_FRONTIER) {                           \
                to_queue[base + _j] = staged[_j];                          \
            } else {                                                        \
                /* Spill to global */                                       \
                int gpos = atomicAdd(d_bfs_next_count, 1);                  \
                d_bfs_next_queue[gpos] = staged[_j];                        \
                if (tid == 0) s_spilled = 1;                                 \
            }                                                               \
        }                                                                    \
        staged_cnt = 0;                                                      \
    }                                                                        \
} while(0)

        for (int fi = tid; fi < qsize; fi += stride) {
            node_t t = s_queue[fi];
            for (edge_t nx = d_begin[t]; nx < d_begin[t + 1]; nx++) {
                node_t k = d_node_idx[nx];
                // Navigate: check if k has base_color
                if (d_Color[k] == base_color) {
                    // Claim via visited bitmap (atomicOr on 200KB L2-resident bitmap)
                    int word = k >> 5;
                    uint32_t bit = 1u << (k & 31);
                    uint32_t old = atomicOr(&d_visited_bits[word], bit);
                    if ((old & bit) == 0) {
                        // Claimed! Write color with simple store
                        d_Color[k] = fw_color;
                        staged[staged_cnt++] = k;
                        if (staged_cnt == STAGE_SIZE) {
                            // Check spill flag early to avoid unnecessary work
                            if (s_spilled) break;
                            PERSISTENT_FLUSH(s_next, s_ncount);
                        }
                    }
                }
            }
            if (s_spilled) break;
        }

        // Flush remaining staging buffer
        if (!s_spilled || staged_cnt > 0) {
            PERSISTENT_FLUSH(s_next, s_ncount);
        }
#undef PERSISTENT_FLUSH

        __syncthreads();

        if (s_spilled) {
            // *** BUG FIX: Flush SMEM nodes from s_next to global before breaking ***
            // When PERSISTENT_FLUSH overflowed, some nodes were written to s_next
            // (shared memory) by other threads before the spill flag was set.
            // Those nodes are lost when this kernel exits unless we copy them to
            // global d_bfs_next_queue here, appended after the already-spilled
            // overflow nodes. Without this, the host's spill recovery only sees
            // the overflow nodes — nodes only reachable through the s_next nodes
            // would never be visited, causing undercounted frontiers.
            int ncnt = s_ncount;
            if (ncnt > MAX_SMEM_FRONTIER) ncnt = MAX_SMEM_FRONTIER;
            for (int i = tid; i < ncnt; i += stride) {
                int gpos = atomicAdd(d_bfs_next_count, 1);
                d_bfs_next_queue[gpos] = s_next[i];
            }
            __syncthreads();
            break;
        }

        // Swap frontiers: copy s_next → s_queue
        int ncnt = s_ncount;
        if (ncnt > MAX_SMEM_FRONTIER) ncnt = MAX_SMEM_FRONTIER;
        for (int i = tid; i < ncnt; i += stride) {
            s_queue[i] = s_next[i];
        }
        if (tid == 0) s_qsize = ncnt;
        __syncthreads();

        local_levels++;
        total_levels++;
    }        // ---- Report results ----
    // Note: use s_qsize, not qsize. qsize holds the last iteration's frontier
    // size (always > 0 if work was done), while s_qsize is set to ncnt at the
    // end of each iteration and is 0 when BFS genuinely completed.
    if (tid == 0) {
        if (s_qsize == 0 && s_spilled == 0) {
            // BFS completed successfully
            *d_spill_flag = -1;  // done
        } else if (s_spilled) {
            // Spill occurred — host continues from d_bfs_next_queue
            *d_spill_flag = 1;
        } else {
            // Safety exit (hit 100K levels) — shouldn't happen
            *d_spill_flag = -1;
        }
        *d_level_counter = total_levels;
    }
}

// ======================================================================
// persistent_bw_bfs_kernel
//
// Block-local persistent BW BFS. Same pattern as persistent_fw_bfs_kernel
// but processes reverse edges. For each visited node:
//   - If d_Color[k] == fw_color → intersection: mark as SCC, set d_SCC[k] = pivot
//   - If d_Color[k] == base_color → BW set: mark as bw_color
//
// SMEM layout: same as FW version.
//
// Output: d_bfs_next_count contains spilled frontier size (0 if done).
//         d_bfs_scc_count / d_bfs_bw_count contain final counts.
//         d_spill_flag = -1 (done) or 1 (spill).
// ======================================================================
__global__ void persistent_bw_bfs_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    int pivot, int base_color, int fw_color, int bw_color,
    int* d_bfs_queue,          // [N] global queue buffer (for spill)
    int* d_bfs_next_queue,     // [N] global next queue buffer (for spill)
    int* d_bfs_next_count,     // [1] global next count (for spill)
    int* d_bfs_scc_count,      // [1] total SCC count
    int* d_bfs_bw_count,       // [1] total bw count
    uint32_t* d_visited_bits,  // global visited bitmap
    int* d_spill_flag,         // [1] output: -1 done, 1 spilled
    int* d_level_counter)      // [1] output: number of levels processed (for stats)
{
    // Only one block needed
    if (blockIdx.x > 0) return;

    extern __shared__ int smem[];
    int* s_queue   = &smem[0];
    int* s_next    = &smem[MAX_SMEM_FRONTIER];
    volatile int& s_qsize   = smem[2 * MAX_SMEM_FRONTIER];
    volatile int& s_ncount  = smem[2 * MAX_SMEM_FRONTIER + 1];
    volatile int& s_spilled = smem[2 * MAX_SMEM_FRONTIER + 2];

    int tid = threadIdx.x;
    int stride = blockDim.x;
    const int STAGE_SIZE = 4;

    // ---- Initialize: pivot is the first frontier ----
    // Pivot is always at the intersection, so mark it as SCC immediately.
    if (tid == 0) {
        d_Color[pivot] = SCC_FOUND;
        d_SCC[pivot] = pivot;
        s_queue[0] = pivot;
        s_qsize = 1;
        s_spilled = 0;
        *d_bfs_next_count = 0;
        *d_bfs_scc_count = 1;  // count pivot
        *d_bfs_bw_count = 0;
    }
    __syncthreads();

    int local_levels = 0;

    // ---- Main loop ----
    while (s_qsize > 0 && s_spilled == 0 && local_levels < 100000) {
        int qsize = s_qsize;
        if (tid == 0) s_ncount = 0;
        __syncthreads();

#if ENABLE_FRONTIER_LOG
        if (tid == 0) {
            // Use d_bfs_queue as temp log buffer (safe here — not used during kernel)
            int pos = atomicAdd(d_level_counter, 1);
            if (pos < 10000) d_bfs_queue[pos] = qsize;
        }
#endif

        int staged[STAGE_SIZE];
        int staged_cnt = 0;
        int local_scc = 0;
        int local_bw = 0;

#define PERSISTENT_BW_FLUSH(to_queue, count_var) do {                      \
    if (staged_cnt > 0) {                                                  \
        int base = atomicAdd((int*)&count_var, staged_cnt);                \
        for (int _j = 0; _j < staged_cnt; _j++) {                          \
            if (base + _j < MAX_SMEM_FRONTIER) {                           \
                to_queue[base + _j] = staged[_j];                          \
            } else {                                                        \
                int gpos = atomicAdd(d_bfs_next_count, 1);                  \
                d_bfs_next_queue[gpos] = staged[_j];                        \
                if (tid == 0) s_spilled = 1;                                 \
            }                                                               \
        }                                                                    \
        staged_cnt = 0;                                                      \
    }                                                                        \
} while(0)

        for (int fi = tid; fi < qsize; fi += stride) {
            node_t t = s_queue[fi];
            for (edge_t nx = d_r_begin[t]; nx < d_r_begin[t + 1]; nx++) {
                node_t k = d_r_node_idx[nx];
                // Single read of d_Color[k] for TOCTOU safety
                int k_color = d_Color[k];

                // Navigate: check if k is in fw_set (intersection) or base_set
                if (k_color == fw_color || k_color == base_color) {
                    // Claim via visited bitmap
                    int word = k >> 5;
                    uint32_t bit = 1u << (k & 31);
                    uint32_t old = atomicOr(&d_visited_bits[word], bit);
                    if ((old & bit) == 0) {
                        if (k_color == fw_color) {
                            // Intersection: mark as SCC
                            d_Color[k] = SCC_FOUND;
                            d_SCC[k] = pivot;
                            local_scc++;
                        } else {
                            // BW-set
                            d_Color[k] = bw_color;
                            local_bw++;
                        }
                        staged[staged_cnt++] = k;
                        if (staged_cnt == STAGE_SIZE) {
                            if (s_spilled) break;
                            PERSISTENT_BW_FLUSH(s_next, s_ncount);
                        }
                    }
                }
            }
            if (s_spilled) break;
        }

        if (!s_spilled || staged_cnt > 0) {
            PERSISTENT_BW_FLUSH(s_next, s_ncount);
        }
#undef PERSISTENT_BW_FLUSH

        // Flush local SCC / BW counters BEFORE the break check
        // This ensures spill-case nodes are counted even when s_spilled is set
        // by PERSISTENT_BW_FLUSH (overflow nodes are already in d_bfs_next_queue
        // and correctly counted here — no double counting).
        if (local_scc > 0) atomicAdd(d_bfs_scc_count, local_scc);
        if (local_bw > 0) atomicAdd(d_bfs_bw_count, local_bw);

        __syncthreads();

        if (s_spilled) {
            // *** BUG FIX: Flush SMEM nodes from s_next to global before breaking ***
            int ncnt = s_ncount;
            if (ncnt > MAX_SMEM_FRONTIER) ncnt = MAX_SMEM_FRONTIER;
            for (int i = tid; i < ncnt; i += stride) {
                int gpos = atomicAdd(d_bfs_next_count, 1);
                d_bfs_next_queue[gpos] = s_next[i];
            }
            __syncthreads();
            break;
        }

        // Swap frontiers
        int ncnt = s_ncount;
        if (ncnt > MAX_SMEM_FRONTIER) ncnt = MAX_SMEM_FRONTIER;
        for (int i = tid; i < ncnt; i += stride) {
            s_queue[i] = s_next[i];
        }
        if (tid == 0) s_qsize = ncnt;
        __syncthreads();

        local_levels++;
    }

    // ---- Report results ----
    if (tid == 0) {
        if (!s_spilled && s_qsize == 0) {
            *d_spill_flag = -1;  // done
        } else if (s_spilled) {
            *d_spill_flag = 1;
        } else {
            *d_spill_flag = -1;
        }
        *d_level_counter = local_levels;
    }
}

// ======================================================================
// run_persistent_bfs_fw() — host helper
//
// Launches the persistent FW BFS kernel for a given pivot, base_color,
// fw_color. If the kernel completes without spill (d_spill_flag == -1),
// returns the total number of FW nodes reachable.
// If spill occurs (d_spill_flag == 1), returns the count of nodes in
// the spilled frontier so the caller can continue from there.
// ======================================================================
int run_persistent_bfs_fw(GPUState& st, const GPUGraph& g,
    int pivot, int base_color, int fw_color)
{
    g_persistent_kernel_attempts++;

    // ---- Set initial pivot state ----
    CUDA_CHECK(cudaMemcpy(d_bfs_queue, &pivot, sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&st.d_Color[pivot], &fw_color, sizeof(int),
                           cudaMemcpyHostToDevice));

    // ---- Reset spill flag ----
    int spill_init = 0;
    CUDA_CHECK(cudaMemcpy(d_spill_flag, &spill_init, sizeof(int),
                           cudaMemcpyHostToDevice));

    // ---- SMEM size ----
    int smem_size = 2 * MAX_SMEM_FRONTIER * sizeof(int) + 3 * sizeof(int);
    int block_size = 256;

    // ---- Launch persistent kernel ----
    persistent_fw_bfs_kernel<<<1, block_size, smem_size>>>(
        g.d_begin, g.d_node_idx,
        st.d_Color,
        pivot, base_color, fw_color,
        d_bfs_queue, d_bfs_next_queue, d_bfs_next_count,
        d_bfs_visited_bits,
        d_spill_flag,
        d_pivot_scratch);  // reuse d_pivot_scratch as level counter
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Check result ----
    int h_spill_flag;
    CUDA_CHECK(cudaMemcpy(&h_spill_flag, d_spill_flag, sizeof(int),
                           cudaMemcpyDeviceToHost));

    if (h_spill_flag == -1) {
        // BFS completed — count visited nodes
        // Count number of nodes marked with fw_color via a kernel
        int* d_fw_count = d_remain_scratch;
        CUDA_CHECK(cudaMemset(d_fw_count, 0, sizeof(int)));

        // Use compact scan to count fw_color nodes from trim_targets
        if (d_trim_targets_count > 0) {
            int bs = 256;
            int gs = (d_trim_targets_count + bs - 1) / bs;
            count_remaining_kernel<<<gs, bs>>>(
                st.d_Color, d_fw_count,
                d_trim_targets, d_trim_targets_count, fw_color);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int fw_count;
        CUDA_CHECK(cudaMemcpy(&fw_count, d_fw_count, sizeof(int),
                               cudaMemcpyDeviceToHost));
        return fw_count;
    }

    // ---- Spill occurred: host must continue ----
    // d_bfs_next_count contains the number of nodes in the spilled frontier.
    int spill_count;
    CUDA_CHECK(cudaMemcpy(&spill_count, d_bfs_next_count, sizeof(int),
                           cudaMemcpyDeviceToHost));
    // The nodes the persistent kernel visited before spilling are already colored
    // fw_color. The host loop will process nodes from the spilled frontier only.
    // Return negative spill count so the caller can continue the host loop.
    // The caller should run count_remaining_kernel after the host loop finishes
    // to get the accurate total fw_count (includes all SMEM-level nodes + host-loop nodes).
    return -spill_count;  // negative = spill occurred, magnitude = spill count
}

// ======================================================================
// run_persistent_bfs_bw() — host helper for BW BFS
//
// Same pattern as run_persistent_bfs_fw but for backward BFS.
// Returns: positive = scc_count (BFS completed without spill)
//          negative = -(spill_count) if spill occurred
// ======================================================================
int run_persistent_bfs_bw(GPUState& st, const GPUGraph& g,
    int pivot, int base_color, int fw_color, int bw_color,
    int& out_bw_count, int& out_scc_count)
{
    // ---- Mark pivot as SCC ----
    int scc_val = SCC_FOUND;
    CUDA_CHECK(cudaMemcpy(d_bfs_queue, &pivot, sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&st.d_Color[pivot], &scc_val, sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&st.d_SCC[pivot], &pivot, sizeof(int),
                           cudaMemcpyHostToDevice));

    // ---- Initialize SCC and BW counters ----
    CUDA_CHECK(cudaMemset(d_bfs_scc_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bfs_bw_count, 0, sizeof(int)));

    // ---- Reset spill flag ----
    int spill_init = 0;
    CUDA_CHECK(cudaMemcpy(d_spill_flag, &spill_init, sizeof(int),
                           cudaMemcpyHostToDevice));

    // ---- SMEM size ----
    int smem_size = 2 * MAX_SMEM_FRONTIER * sizeof(int) + 3 * sizeof(int);
    int block_size = 256;

    // ---- Launch persistent BW kernel ----
    persistent_bw_bfs_kernel<<<1, block_size, smem_size>>>(
        g.d_r_begin, g.d_r_node_idx,
        st.d_Color, st.d_SCC,
        pivot, base_color, fw_color, bw_color,
        d_bfs_queue, d_bfs_next_queue, d_bfs_next_count,
        d_bfs_scc_count, d_bfs_bw_count,
        d_bfs_visited_bits,
        d_spill_flag,
        d_pivot_scratch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Read results ----
    int h_scc_count, h_bw_count;
    CUDA_CHECK(cudaMemcpy(&h_scc_count, d_bfs_scc_count, sizeof(int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bw_count, d_bfs_bw_count, sizeof(int),
                           cudaMemcpyDeviceToHost));
    out_scc_count = h_scc_count;
    out_bw_count = h_bw_count;

    int h_spill_flag;
    CUDA_CHECK(cudaMemcpy(&h_spill_flag, d_spill_flag, sizeof(int),
                           cudaMemcpyDeviceToHost));

    if (h_spill_flag == -1) {
        // BFS completed
        return out_scc_count;
    }

    // ---- Spill occurred ----
    int spill_count;
    CUDA_CHECK(cudaMemcpy(&spill_count, d_bfs_next_count, sizeof(int),
                           cudaMemcpyDeviceToHost));
    return -spill_count;
}

// ======================================================================
// do_global_fw_bw_main()
// Mirrors OpenMP:
//   int do_fw_bw_global_main(gm_graph& G, int curr_color, int count,
//                              bool create_work_items)
//
// CUDA parameters match the CPU convention:
//   base_color = curr_color  (the color of the subgraph to process)
//   base_count = count       (number of nodes in this subgraph)
//   good_init_pivot          (from common_main.h for met_algo==6/11)
// ======================================================================
int do_global_fw_bw_main(GPUState& st, const GPUGraph& g,
    int base_color, int base_count, int good_init_pivot,
    bool create_work_items)
{
    // OpenMP: base_color = curr_color; base_count = count;
    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return 0;

    // --- Compute grid/block sizes BEFORE pivot selection ---
    // (fix: grid_size and block_size were used in pick_pivot_kernel before definition)
    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;

    // ---------------------------------------------------------------
    // Pick pivot — EXACT mirror of OpenMP
    // OpenMP:
    //   node_t pivot;
    //   if((met_algo==6 || met_algo==11) && G_Color[good_init_pivot]!=-2)
    //       pivot=good_init_pivot;
    //   else
    //       pivot = choose_pivot_from_color(G,base_color);
    //   assert(pivot != gm_graph::NIL_NODE);
    //   assert(G_Color[pivot] == base_color);
    //
    // CUDA: Uses persistent d_pivot_scratch buffer (allocated once in
    //       initialize_global_fb) to avoid per-call cudaMalloc/cudaFree.
    // ---------------------------------------------------------------
    int h_pivot = -1;
    int PIVOT_NONE = 0x7FFFFFFF;
    CUDA_CHECK(cudaMemcpy(d_pivot_scratch, &PIVOT_NONE, sizeof(int), cudaMemcpyHostToDevice));

    // OpenMP: if((met_algo==6 || met_algo==11) && G_Color[good_init_pivot]!=-2)
    //            pivot=good_init_pivot;
    //         else
    //            pivot = choose_pivot_from_color(G,base_color);
    //
    // CUDA: Caller passes good_init_pivot = -1 for methods != 6/11,
    //       or the actual pivot node for methods 6/11.
    //       CPU condition: G_Color[pivot] != -2 (not SCC_FOUND)
    if (good_init_pivot >= 0) {
        // CPU: check G_Color[good_init_pivot] != -2
        int h_color;
        CUDA_CHECK(cudaMemcpy(&h_color, &st.d_Color[good_init_pivot],
                               sizeof(int), cudaMemcpyDeviceToHost));
        if (h_color != SCC_FOUND) {                       // CPU: != -2
            h_pivot = good_init_pivot;
            CUDA_CHECK(cudaMemcpy(d_pivot_scratch, &h_pivot, sizeof(int),
                                   cudaMemcpyHostToDevice));
        }
    }

    if (h_pivot == -1) {
        // CPU: pivot = choose_pivot_from_color(G,base_color);
        pick_pivot_kernel<<<grid_size, block_size>>>(
            st.d_Color, d_pivot_scratch, d_trim_targets, num_targets, base_color);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_pivot, d_pivot_scratch, sizeof(int), cudaMemcpyDeviceToHost));
    }

    // CPU: assert(pivot != gm_graph::NIL_NODE);
    if (h_pivot == 0x7FFFFFFF || h_pivot == -1) return 0;
    // CPU: assert(G_Color[pivot] == base_color); — guaranteed by kernel

    // ---------------------------------------------------------------
    // Count remaining base_color nodes
    // OpenMP: if (count == 1) { G_Color[pivot] = -2; G_SCC[pivot] = pivot; return 1; }
    //
    // CUDA: Uses persistent d_remain_scratch buffer (no per-call malloc/free).
    // ---------------------------------------------------------------

    CUDA_CHECK(cudaMemset(d_remain_scratch, 0, sizeof(int)));

    count_remaining_kernel<<<grid_size, block_size>>>(
        st.d_Color, d_remain_scratch, d_trim_targets, num_targets, base_color);
    CUDA_CHECK(cudaDeviceSynchronize());

    int remain_count;
    CUDA_CHECK(cudaMemcpy(&remain_count, d_remain_scratch, sizeof(int), cudaMemcpyDeviceToHost));

    if (remain_count <= 1) {
        if (remain_count == 1) {
            // OpenMP: G_Color[pivot] = -2; G_SCC[pivot] = pivot;
            { int _scc_val = SCC_FOUND; CUDA_CHECK(cudaMemcpy(&st.d_Color[h_pivot], &_scc_val, sizeof(int),
                                   cudaMemcpyHostToDevice)); }
            CUDA_CHECK(cudaMemcpy(&st.d_SCC[h_pivot], &h_pivot, sizeof(int),
                                   cudaMemcpyHostToDevice));
        }
        return remain_count;
    }

    // ---------------------------------------------------------------
    // Assign colors — EXACT mirror of OpenMP
    // OpenMP: int fw_color = get_new_color(); int bw_color = get_new_color();
    // ---------------------------------------------------------------
    int fw_color = cuda_get_new_color();
    int bw_color = cuda_get_new_color();

    // ---------------------------------------------------------------
    // Forward BFS (with persistent kernel optimization for high-diameter)
    // OpenMP:
    //   fw_trim_global FW_BFS(G, base_color, fw_color);
    //   FW_BFS.prepare(pivot, gm_rt_get_num_threads());
    //   FW_BFS.do_bfs_forward();
    //   int fw_count = FW_BFS.get_fw_count();
    //
    // Strategy:
    //   1. Try block-local persistent kernel (keeps BFS resident, loops
    //      levels via __syncthreads(), avoids per-level launch overhead)
    //   2. If persistent kernel spills (frontier exceeds MAX_SMEM_FRONTIER),
    //      fall back to existing per-level host loop from the spilled frontier
    // ---------------------------------------------------------------
    int total_fw = 1;  // pivot counted
    int fw_count;

    // Always try persistent kernel first — it's a single block launch (~5-15μs)
    // even if it spills immediately. On deep-narrow graphs (wiki-Talk),
    // it eliminates ~100+ kernel launches.
    int fw_result = run_persistent_bfs_fw(st, g, h_pivot, base_color, fw_color);

    if (fw_result >= 0) {
        // Persistent kernel completed the entire FW BFS without spill
        fw_count = fw_result;
    } else {
        // Spill occurred: host loop picks up from spilled frontier
        int spill_count = -fw_result;
        int queue_size = spill_count;
        total_fw += spill_count;

        // Swap buffers so d_bfs_queue holds the spilled frontier
        // The persistent kernel wrote to d_bfs_next_queue; copy to d_bfs_queue
        if (spill_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_bfs_queue, d_bfs_next_queue,
                                       spill_count * sizeof(int),
                                       cudaMemcpyDeviceToDevice, bfs_stream));
            CUDA_CHECK(cudaStreamSynchronize(bfs_stream));
        }

        // Continue BFS with existing per-level kernel loop
        while (queue_size > 0) {
            CUDA_CHECK(cudaMemsetAsync(d_bfs_next_count, 0, sizeof(int), bfs_stream));

            int grid = (queue_size + block_size - 1) / block_size;

            fw_bfs_level_kernel<<<grid, block_size, 0, bfs_stream>>>(
                g.d_begin, g.d_node_idx,
                st.d_Color,
                d_bfs_queue, queue_size,
                d_bfs_next_queue, d_bfs_next_count,
                fw_color, base_color,
                d_bfs_visited_bits);

            CUDA_CHECK(cudaMemcpyAsync(h_pinned_next_count, d_bfs_next_count,
                                        sizeof(int), cudaMemcpyDeviceToHost, bfs_stream));
            CUDA_CHECK(cudaStreamSynchronize(bfs_stream));

            int* tmp = d_bfs_queue;
            d_bfs_queue = d_bfs_next_queue;
            d_bfs_next_queue = tmp;

            queue_size = *h_pinned_next_count;
            total_fw += queue_size;
        }

        // Accurate count: count all fw_color nodes (includes SMEM levels + host loop)
        // The persistent kernel's SMEM-level visits are already in d_Color as fw_color,
        // so count_remaining_kernel gives the true total.
        CUDA_CHECK(cudaMemset(d_remain_scratch, 0, sizeof(int)));
        count_remaining_kernel<<<grid_size, block_size>>>(
            st.d_Color, d_remain_scratch,
            d_trim_targets, d_trim_targets_count, fw_color);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&fw_count, d_remain_scratch, sizeof(int),
                               cudaMemcpyDeviceToHost));
    }

    printf("[PERSISTENT_KERNEL] FW BFS: used=%s fw_count=%d\n",
           fw_result >= 0 ? "persistent" : "spill+host", fw_count);

    // ---------------------------------------------------------------
    // Reset visited bitmap between FW and BW BFS
    // FW BFS set bits for all FW-reachable nodes; BW BFS needs a clean bitmap
    // for claiming SCC/bw-set nodes (tiny 200KB memset, ~0.01ms)
    // ---------------------------------------------------------------
    CUDA_CHECK(cudaMemsetAsync(d_bfs_visited_bits, 0,
                                d_bfs_visited_words * sizeof(uint32_t),
                                bfs_stream));
    CUDA_CHECK(cudaStreamSynchronize(bfs_stream));

    // ---------------------------------------------------------------
    // Backward BFS (with persistent kernel optimization)
    // OpenMP:
    //   bw_trim_global BW_BFS(G, base_color, fw_color, bw_color, pivot);
    //   BW_BFS.prepare(pivot, gm_rt_get_num_threads());
    //   BW_BFS.do_bfs_forward();
    //
    //   int bw_count = BW_BFS.get_bw_count();
    //   int scc_count = BW_BFS.get_scc_count();
    //
    //   fw_count = fw_count - scc_count;
    //   base_count = base_count - fw_count - bw_count - scc_count;
    // ---------------------------------------------------------------
    int out_bw_count = 0;
    int out_scc_count = 0;

    int bw_result = run_persistent_bfs_bw(st, g, h_pivot, base_color, fw_color, bw_color,
                                           out_bw_count, out_scc_count);

    if (bw_result >= 0) {
        // Persistent kernel completed entire BW BFS without spill
    } else {
        // Spill occurred: host loop continues from spilled frontier
        int spill_count = -bw_result;
        int queue_size = spill_count;

        // Copy spilled frontier to d_bfs_queue
        if (spill_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_bfs_queue, d_bfs_next_queue,
                                       spill_count * sizeof(int),
                                       cudaMemcpyDeviceToDevice, bfs_stream));
            CUDA_CHECK(cudaStreamSynchronize(bfs_stream));
        }

        while (queue_size > 0) {
            CUDA_CHECK(cudaMemsetAsync(d_bfs_next_count, 0, sizeof(int), bfs_stream));

            int grid = (queue_size + block_size - 1) / block_size;

            bw_bfs_level_kernel<<<grid, block_size, 0, bfs_stream>>>(
                g.d_r_begin, g.d_r_node_idx,
                st.d_Color, st.d_SCC,
                d_bfs_queue, queue_size,
                d_bfs_next_queue, d_bfs_next_count,
                fw_color, bw_color, base_color, h_pivot,
                d_bfs_scc_count, d_bfs_bw_count,
                d_bfs_visited_bits);

            CUDA_CHECK(cudaMemcpyAsync(h_pinned_next_count, d_bfs_next_count,
                                        sizeof(int), cudaMemcpyDeviceToHost, bfs_stream));
            CUDA_CHECK(cudaStreamSynchronize(bfs_stream));

            int* tmp = d_bfs_queue;
            d_bfs_queue = d_bfs_next_queue;
            d_bfs_next_queue = tmp;

            queue_size = *h_pinned_next_count;
        }

        // Read final SCC / BW counts from spill path
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_scc_count, d_bfs_scc_count, sizeof(int),
                                    cudaMemcpyDeviceToHost, bfs_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_bw_count, d_bfs_bw_count, sizeof(int),
                                    cudaMemcpyDeviceToHost, bfs_stream));
        CUDA_CHECK(cudaStreamSynchronize(bfs_stream));
        out_scc_count += *h_pinned_scc_count;
        out_bw_count += *h_pinned_bw_count;
    }

    int scc_count = out_scc_count;
    int bw_count = out_bw_count;

    printf("[PERSISTENT_KERNEL] BW BFS: used=%s scc=%d bw=%d\n",
           bw_result >= 0 ? "persistent" : "spill+host", scc_count, bw_count);

    // OpenMP: compute counts for each partition
    //   int bw_count = BW_BFS.get_bw_count();
    fw_count = fw_count - scc_count;
    base_count = base_count - fw_count - bw_count - scc_count;

    // OpenMP: init_fw_color = fw_color; init_bw_color = bw_color;
    //         init_base_color = base_color;
    init_fw_color = fw_color;
    init_bw_color = bw_color;
    init_base_color = base_color;

    // OpenMP: if (!create_work_items) return scc_count;
    if (!create_work_items) {
        printf("[CUDA Global BFS] pivot=%d fw_set=%d scc=%d base_remain=%d\n",
               h_pivot, fw_count, scc_count, base_count);
        return scc_count;
    }

    // OpenMP: create work items for fw, bw, base partitions
    //         (pushes them to the work queue with color_set = NULL)
    // CUDA: creates CUDAMyWork items with proper device-side compact
    //       node sets (d_set_nodes) so the consumer can process them
    //       without rebuilding sets.
    create_works_after_bfs_trim(st, g);

    return scc_count;
}

// ======================================================================
// create_works_after_bfs_trim()
// OpenMP:
//   void create_works_after_bfs_trim(gm_graph& G)
//   {
//       int fw_color = init_fw_color;
//       int bw_color = init_bw_color;
//       int base_color = init_base_color;
//       ...
//       #pragma omp parallel for
//       for each node in V: count by color
//       ...
//       create work items for fw_count, bw_count, base_count
//   }
// ======================================================================
// OpenMP:
//   static my_work* base_work_item = NULL;
static CUDAMyWork* base_work_item_cuda = NULL;

void create_works_after_bfs_trim(GPUState& st, const GPUGraph& g)
{
    // OpenMP: int fw_color = init_fw_color; int bw_color = init_bw_color;
    //         int base_color = init_base_color;
    int fw_color = init_fw_color;
    int bw_color = init_bw_color;
    int base_color = init_base_color;

    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return;

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;
    grid_size = min(grid_size, 1024);

    // ---------------------------------------------------------------
    // Phase 1: Count nodes per color (mirrors CPU's omp parallel for)
    // OpenMP:
    //   int fw_count = 0; int bw_count = 0; int base_count = 0;
    //   #pragma omp parallel for
    //   for each node in V: count by color
    // ---------------------------------------------------------------
    int* d_fw_count   = NULL;
    int* d_bw_count   = NULL;
    int* d_base_count = NULL;
    CUDA_CHECK(cudaMalloc(&d_fw_count,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bw_count,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_base_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fw_count,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bw_count,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_base_count, 0, sizeof(int)));

    count_by_colors_kernel<<<grid_size, block_size>>>(
        st.d_Color, d_trim_targets, num_targets,
        fw_color, bw_color, base_color,
        d_fw_count, d_bw_count, d_base_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_fw_count, h_bw_count, h_base_count;
    CUDA_CHECK(cudaMemcpy(&h_fw_count,   d_fw_count,   sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bw_count,   d_bw_count,   sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_base_count, d_base_count, sizeof(int), cudaMemcpyDeviceToHost));

    // ---------------------------------------------------------------
    // Phase 2: Allocate device buffers, scatter nodes by color
    // CUDA-specific enhancement: builds compact device arrays for each
    // partition so consumers can process them without rebuilding.
    // ---------------------------------------------------------------
    int* d_fw_set   = NULL;
    int* d_bw_set   = NULL;
    int* d_base_set = NULL;
    int* d_scatter_fw_pos   = NULL;  // [1] atomic position for FW scatter
    int* d_scatter_bw_pos   = NULL;  // [1] atomic position for BW scatter
    int* d_scatter_base_pos = NULL;  // [1] atomic position for BASE scatter

    if (h_fw_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_fw_set, h_fw_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scatter_fw_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scatter_fw_pos, 0, sizeof(int)));
    }
    if (h_bw_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_bw_set, h_bw_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scatter_bw_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scatter_bw_pos, 0, sizeof(int)));
    }
    if (h_base_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_base_set, h_base_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scatter_base_pos, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scatter_base_pos, 0, sizeof(int)));
    }

    // Scatter: for each node in trim_targets, write to the appropriate
    // per-color device buffer based on its current color.
    // (Uses scatter_by_color_kernel from scc_cuda_work_queue.cu)
    if (h_fw_count > 0 || h_bw_count > 0 || h_base_count > 0) {
        scatter_by_color_kernel<<<grid_size, block_size>>>(
            st.d_Color,
            d_trim_targets, num_targets,
            fw_color, bw_color, base_color,
            d_fw_set,   d_scatter_fw_pos,
            d_bw_set,   d_scatter_bw_pos,
            d_base_set, d_scatter_base_pos);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_fw_count));
    CUDA_CHECK(cudaFree(d_bw_count));
    CUDA_CHECK(cudaFree(d_base_count));

    // ---------------------------------------------------------------
    // Phase 3: Create CUDAMyWork items and push to work queue
    // OpenMP:
    //   int depth = 1;
    //   my_work* work;
    //   if (fw_count > 0) {
    //       work = new my_work();
    //       work->color = fw_color;
    //       work->count = fw_count;
    //       work->color_set = NULL;    // <-- CPU sets NULL, consumer rebuilds
    //       work->depth = depth;
    //       work_q_put(gm_rt_thread_id(), work);
    //   }
    //   ... (same for bw, base)
    //
    // CUDA enhancement: work items carry pre-built d_set_nodes so GPU
    // consumers can process them directly without rebuilding.
    // ---------------------------------------------------------------
    int depth = 1;

    if (h_fw_count > 0) {
        CUDAMyWork* w = new CUDAMyWork();
        w->color       = fw_color;
        w->count       = h_fw_count;
        w->d_set_nodes = d_fw_set;    // device compact set (pre-built)
        w->set_capacity = h_fw_count;
        w->depth       = depth;
        w->owns_set    = 1;              // CUDA: owns d_fw_set buffer
        work_q_put(0, w);
    }
    if (h_bw_count > 0) {
        CUDAMyWork* w = new CUDAMyWork();
        w->color       = bw_color;
        w->count       = h_bw_count;
        w->d_set_nodes = d_bw_set;    // device compact set (pre-built)
        w->set_capacity = h_bw_count;
        w->depth       = depth;
        w->owns_set    = 1;              // CUDA: owns d_bw_set buffer
        work_q_put(0, w);
    }
    if (h_base_count > 0) {
        CUDAMyWork* w = new CUDAMyWork();
        w->color       = base_color;
        w->count       = h_base_count;
        w->d_set_nodes = d_base_set;  // device compact set (pre-built)
        w->set_capacity = h_base_count;
        w->depth       = depth;
        w->owns_set    = 1;              // CUDA: owns d_base_set buffer
        work_q_put(0, w);
        // OpenMP: base_work_item = work;
        base_work_item_cuda = w;
    }

    if (d_scatter_fw_pos)   CUDA_CHECK(cudaFree(d_scatter_fw_pos));
    if (d_scatter_bw_pos)   CUDA_CHECK(cudaFree(d_scatter_bw_pos));
    if (d_scatter_base_pos) CUDA_CHECK(cudaFree(d_scatter_base_pos));
}
