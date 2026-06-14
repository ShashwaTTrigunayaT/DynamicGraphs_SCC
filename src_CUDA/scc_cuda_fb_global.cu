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

// Persistent scratch buffers to avoid cudaMalloc/cudaFree in hot BFS path
int* d_pivot_scratch    = NULL;  // [1] temp for pivot selection
int* d_remain_scratch   = NULL;  // [1] temp for remaining count

// Pinned host memory + stream for async BFS level loop
// Pinned memory enables faster D2H transfers (avoids staging buffer)
cudaStream_t bfs_stream        = NULL;
int*         h_pinned_next_count = NULL;  // pinned: next frontier size
int*         h_pinned_scc_count  = NULL;  // pinned: SCC count
int*         h_pinned_bw_count   = NULL;  // pinned: BW count

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

    // Allocate persistent scratch buffers (allocated once, reused across multiple
    // do_global_fw_bw_main calls, avoiding per-call cudaMalloc/cudaFree overhead)
    if (!d_pivot_scratch)  CUDA_CHECK(cudaMalloc(&d_pivot_scratch,  sizeof(int)));
    if (!d_remain_scratch) CUDA_CHECK(cudaMalloc(&d_remain_scratch, sizeof(int)));

    // Allocate pinned host memory for async D2H copies (faster D2H, no staging)
    if (!h_pinned_next_count) CUDA_CHECK(cudaMallocHost(&h_pinned_next_count, sizeof(int)));
    if (!h_pinned_scc_count)  CUDA_CHECK(cudaMallocHost(&h_pinned_scc_count,  sizeof(int)));
    if (!h_pinned_bw_count)   CUDA_CHECK(cudaMallocHost(&h_pinned_bw_count,  sizeof(int)));

    // Create stream for async kernel launches + memcpy
    if (!bfs_stream) CUDA_CHECK(cudaStreamCreate(&bfs_stream));

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

    // Free persistent scratch buffers
    if (d_pivot_scratch)  { cudaFree(d_pivot_scratch);  d_pivot_scratch = NULL; }
    if (d_remain_scratch) { cudaFree(d_remain_scratch); d_remain_scratch = NULL; }

    // Free pinned memory and destroy stream
    if (h_pinned_next_count) { cudaFreeHost(h_pinned_next_count); h_pinned_next_count = NULL; }
    if (h_pinned_scc_count)  { cudaFreeHost(h_pinned_scc_count); h_pinned_scc_count = NULL; }
    if (h_pinned_bw_count)   { cudaFreeHost(h_pinned_bw_count);  h_pinned_bw_count = NULL; }
    if (bfs_stream)          { cudaStreamDestroy(bfs_stream);     bfs_stream = NULL; }
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
// Optimization: Per-thread local staging of claimed nodes.
// Instead of per-node atomicAdd to d_next_count (1 atomic per claimed
// node), collect up to STAGE_SIZE claimed nodes locally, then flush
// with a single atomicAdd per thread. Reduces atomic contention by
// ~STAGE_SIZE x.
// ======================================================================
__global__ void fw_bfs_level_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int base_color)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Per-thread staging buffer for claimed nodes
    const int STAGE_SIZE = 32;
    int staged[STAGE_SIZE];
    int staged_cnt = 0;

    // Helper: flush local buffer to global queue (single atomicAdd)
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
            if (fw_check_navigator_device(d_Color, k, base_color)) {
                int old = atomicCAS(&d_Color[k], base_color, fw_color);
                if (old == base_color) {
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
// Optimization: Per-thread local staging for d_next_count + per-thread
// local accumulators for d_scc_count and d_bw_count. Reduces global
// atomic contention by ~STAGE_SIZE x.
// ======================================================================
__global__ void bw_bfs_level_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int bw_color, int base_color, node_t pivot,
    int* d_scc_count, int* d_bw_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Per-thread staging buffer for claimed nodes + local counters
    const int STAGE_SIZE = 32;
    int staged[STAGE_SIZE];
    int staged_cnt = 0;
    int local_scc = 0;
    int local_bw = 0;

    // Helper: flush local buffer to global queue (single atomicAdd)
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
            int k_color = d_Color[k];
            if (k_color == fw_color) {
                int old = atomicCAS(&d_Color[k], fw_color, SCC_FOUND);
                if (old == fw_color) {
                    d_SCC[k] = pivot;
                    local_scc++;
                    staged[staged_cnt++] = k;
                    if (staged_cnt == STAGE_SIZE) {
                        BW_FLUSH();
                    }
                }
            }
            else if (k_color == base_color) {
                int old = atomicCAS(&d_Color[k], base_color, bw_color);
                if (old == base_color) {
                    local_bw++;
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

    // Flush local SCC / BW counters (one atomicAdd per thread per counter)
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
    int N = g.num_nodes;
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
    // Forward BFS
    // OpenMP:
    //   fw_trim_global FW_BFS(G, base_color, fw_color);
    //   FW_BFS.prepare(pivot, gm_rt_get_num_threads());
    //   FW_BFS.do_bfs_forward();
    //   int fw_count = FW_BFS.get_fw_count();
    // ---------------------------------------------------------------
    int queue_size = 1;
    CUDA_CHECK(cudaMemcpyAsync(d_bfs_queue, &h_pivot, sizeof(int),
                                cudaMemcpyHostToDevice, bfs_stream));
    CUDA_CHECK(cudaMemcpyAsync(&st.d_Color[h_pivot], &fw_color, sizeof(int),
                                cudaMemcpyHostToDevice, bfs_stream));
    CUDA_CHECK(cudaStreamSynchronize(bfs_stream));

    int total_fw = 1;  // pivot counted

    while (queue_size > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_bfs_next_count, 0, sizeof(int), bfs_stream));

        int grid = (queue_size + block_size - 1) / block_size;
        grid = min(grid, 1024);

        fw_bfs_level_kernel<<<grid, block_size, 0, bfs_stream>>>(
            g.d_begin, g.d_node_idx,
            st.d_Color,
            d_bfs_queue, queue_size,
            d_bfs_next_queue, d_bfs_next_count,
            fw_color, base_color);

        // Async D2H copy — starts as soon as kernel completes on stream
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_next_count, d_bfs_next_count,
                                    sizeof(int), cudaMemcpyDeviceToHost, bfs_stream));

        // Single sync point instead of DeviceSynchronize + blocking Memcpy
        CUDA_CHECK(cudaStreamSynchronize(bfs_stream));

        int* tmp = d_bfs_queue;
        d_bfs_queue = d_bfs_next_queue;
        d_bfs_next_queue = tmp;

        queue_size = *h_pinned_next_count;
        total_fw += queue_size;
    }

    // OpenMP: int fw_count = FW_BFS.get_fw_count();
    int fw_count = total_fw;

    // ---------------------------------------------------------------
    // Backward BFS
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
    // Mark pivot itself as SCC (always in intersection)
    { int _scc_val = SCC_FOUND; CUDA_CHECK(cudaMemcpyAsync(&st.d_Color[h_pivot], &_scc_val, sizeof(int),
                                   cudaMemcpyHostToDevice, bfs_stream)); }
    CUDA_CHECK(cudaMemcpyAsync(&st.d_SCC[h_pivot], &h_pivot, sizeof(int),
                                cudaMemcpyHostToDevice, bfs_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bfs_queue, &h_pivot, sizeof(int),
                                cudaMemcpyHostToDevice, bfs_stream));
    CUDA_CHECK(cudaMemsetAsync(d_bfs_scc_count, 0, sizeof(int), bfs_stream));
    CUDA_CHECK(cudaMemsetAsync(d_bfs_bw_count, 0, sizeof(int), bfs_stream));
    CUDA_CHECK(cudaStreamSynchronize(bfs_stream));
    queue_size = 1;
    int scc_count = 1;

    while (queue_size > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_bfs_next_count, 0, sizeof(int), bfs_stream));

        int grid = (queue_size + block_size - 1) / block_size;
        grid = min(grid, 1024);

        bw_bfs_level_kernel<<<grid, block_size, 0, bfs_stream>>>(
            g.d_r_begin, g.d_r_node_idx,
            st.d_Color, st.d_SCC,
            d_bfs_queue, queue_size,
            d_bfs_next_queue, d_bfs_next_count,
            fw_color, bw_color, base_color, h_pivot,
            d_bfs_scc_count, d_bfs_bw_count);

        // Async D2H — starts after kernel on stream
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_next_count, d_bfs_next_count,
                                    sizeof(int), cudaMemcpyDeviceToHost, bfs_stream));

        CUDA_CHECK(cudaStreamSynchronize(bfs_stream));

        int* tmp = d_bfs_queue;
        d_bfs_queue = d_bfs_next_queue;
        d_bfs_next_queue = tmp;

        queue_size = *h_pinned_next_count;
    }

    // Read final SCC / BW counts
    int h_scc;
    int h_bw;
    CUDA_CHECK(cudaMemcpy(&h_scc, d_bfs_scc_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bw, d_bfs_bw_count, sizeof(int), cudaMemcpyDeviceToHost));
    scc_count += h_scc;
    int bw_count = h_bw;

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
