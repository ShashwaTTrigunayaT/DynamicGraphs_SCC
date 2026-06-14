#include "scc_cuda.h"


static int            g_max_th = 1;
static bool           g_all_finished = false;
static bool           g_queue_empty = true;
static std::vector<CUDAMyWork*> g_work_queue;
static int            g_max_depth = 0;
static int*           g_work_sheet = NULL;  // [max_th * 16] busy flags with stride-16 padding
static int            g_num_threads = 1;

#ifdef _WIN32
#include <windows.h>
#define CUDA_PAUSE()  YieldProcessor()
#define CUDA_SLEEP_US(us) Sleep(((us) + 999) / 1000)  // round up to at least 1ms
#else
#include <unistd.h>
#define CUDA_PAUSE()  asm volatile ("pause" ::: "memory")
#define CUDA_SLEEP_US(us) usleep(us)
#endif

// OpenMP: static gm_spinlock_t q_lock = 0;
// CUDA: use a simple atomic flag
static int g_q_lock = 0;

static void lock_acquire() {
    while (__sync_lock_test_and_set(&g_q_lock, 1)) {
        CUDA_PAUSE();
    }
}

static void lock_release() {
    __sync_lock_release(&g_q_lock);
}


void work_q_init(int num_threads)
{
    g_q_lock = 0;
    g_num_threads = num_threads;
    g_max_th = num_threads;
    g_all_finished = false;
    g_queue_empty = true;
    g_max_depth = 0;
    g_work_queue.clear();

    // OpenMP: for(int i=0;i<num_threads;i++) work_sheet[i*8] = 0;
    if (g_work_sheet) delete[] g_work_sheet;
    g_work_sheet = new int[num_threads * 16];
    for (int i = 0; i < num_threads; i++)
        g_work_sheet[i * 16] = 0;
}

// ======================================================================
// work_q_size()
// OpenMP:
//   int work_q_size() {return the_q.size();}
// ======================================================================
int work_q_size()
{
    lock_acquire();
    int sz = (int)g_work_queue.size();
    lock_release();
    return sz;
}

// ======================================================================
// is_work_q_empty_from_seq_context()
// OpenMP:
//   bool is_work_q_empty_from_seq_context() {return work_queue_empty;}
// ======================================================================
bool is_work_q_empty_from_seq_context()
{
    return g_queue_empty;
}

// ======================================================================
// check_if_all_finished() — static helper
// OpenMP:
//   static bool check_if_all_finished() {
//       bool b = true;
//       if (!work_queue_empty) return false;
//       gm_spinlock_acquire(&q_lock);
//       if ((int)the_q.size() > 0) { b = false; work_queue_empty = false; }
//       else {
//           for(int i=0;i<max_th;i++)
//               if (work_sheet[i*8] == 1) { b = false; break; }
//       }
//       gm_spinlock_release(&q_lock);
//       if (b) all_finished = b;
//       return b;
//   }
// ======================================================================
static bool check_if_all_finished()
{
    bool b = true;
    if (!g_queue_empty) return false;

    lock_acquire();
    if ((int)g_work_queue.size() > 0) {
        b = false;
        g_queue_empty = false;
    } else {
        for (int i = 0; i < g_max_th; i++) {
            // OpenMP: work_sheet[i*8] == 1
            if (g_work_sheet[i * 16] == 1) {
                b = false;
                break;
            }
        }
    }
    lock_release();

    if (b) g_all_finished = b;
    return b;
}

// ======================================================================
// my_sleep() — static helper
// OpenMP:
//   static void my_sleep(int& sleep_cnt) {
//       if (sleep_cnt < 50000) for(int i=0;i<800;i++) asm volatile ("pause"...);
//       else if (sleep_cnt < 80000) usleep(1);
//       else if (sleep_cnt < 100000) usleep(10);
//       sleep_cnt++;
//   }
// ======================================================================
static void my_sleep(int& sleep_cnt)
{
    if (sleep_cnt < 50000) {
        for (int i = 0; i < 800; i++) {
            CUDA_PAUSE();
        }
    } else if (sleep_cnt < 80000) {
        CUDA_SLEEP_US(1);
    } else if (sleep_cnt < 100000) {
        CUDA_SLEEP_US(10);
    }
    sleep_cnt++;
}

// ======================================================================
// work_q_fetch()
// OpenMP:
//   my_work* work_q_fetch(int id) {
//       int sleep_cnt = 1;
//       work_sheet[id*8] = 0;  // set idle
//       while (true) {
//           if (all_finished) return NULL;
//           if (work_queue_empty) {
//               if (id == 0) check_if_all_finished();
//               my_sleep(sleep_cnt);
//               continue;
//           }
//           gm_spinlock_acquire(&q_lock);
//           if ((int)the_q.size() == 0) {
//               work_queue_empty = true;
//               gm_spinlock_release(&q_lock);
//               continue;
//           }
//           sleep_cnt = 1;
//           my_work* ret = the_q.back();
//           the_q.pop_back();
//           work_sheet[id*8] = 1;
//           if (the_q.size() == 0) work_queue_empty = true;
//           gm_spinlock_release(&q_lock);
//           return ret;
//       }
//   }
// ======================================================================
CUDAMyWork* work_q_fetch(int id)
{
    int sleep_cnt = 1;

    // OpenMP: work_sheet[id*8] = 0;
    g_work_sheet[id * 16] = 0;

    while (true) {
        if (g_all_finished) return NULL;

        if (g_queue_empty) {
            if (id == 0) {  // master thread
                bool b = check_if_all_finished();
                if (b) continue;
            }
            my_sleep(sleep_cnt);
            continue;
        }

        lock_acquire();
        if ((int)g_work_queue.size() == 0) {
            g_queue_empty = true;
            lock_release();
            continue;
        }

        sleep_cnt = 1;
        CUDAMyWork* ret = g_work_queue.back();
        g_work_queue.pop_back();
        // OpenMP: work_sheet[id*8] = 1;
        g_work_sheet[id * 16] = 1;

        if (g_work_queue.size() == 0)
            g_queue_empty = true;

        lock_release();
        return ret;
    }
}

// ======================================================================
// work_q_fetch_N()
// OpenMP:
//   void work_q_fetch_N(int id, int N, std::vector<my_work*>& works) {
//       (same structure as fetch but grabs up to N items)
//   }
// ======================================================================
void work_q_fetch_N(int id, int N, std::vector<CUDAMyWork*>& works)
{
    int sleep_cnt = 1;

    // OpenMP: work_sheet[id*8] = 0;
    g_work_sheet[id * 16] = 0;

    while (true) {
        if (g_all_finished) return;

        if (g_queue_empty) {
            if (id == 0) {
                bool b = check_if_all_finished();
                if (b) continue;
            }
            my_sleep(sleep_cnt);
            continue;
        }

        lock_acquire();
        if ((int)g_work_queue.size() == 0) {
            g_queue_empty = true;
            lock_release();
            continue;
        }

        sleep_cnt = 1;
        int max = std::min(N, (int)g_work_queue.size());
        for (int i = 0; i < max; i++) {
            CUDAMyWork* ret = g_work_queue.back();
            g_work_queue.pop_back();
            works.push_back(ret);
        }
        // OpenMP: work_sheet[id*8] = 1;
        g_work_sheet[id * 16] = 1;

        if (g_work_queue.size() == 0)
            g_queue_empty = true;

        lock_release();
        return;
    }
}

// ======================================================================
// work_q_put()
// OpenMP:
//   void work_q_put(int thread_id, my_work* w) {
//       gm_spinlock_acquire(&q_lock);
//       the_q.push_back(w);
//       work_queue_empty = false;
//       int depth = the_q.size();
//       if (depth > max_depth) max_depth = depth;
//       gm_spinlock_release(&q_lock);
//   }
// ======================================================================
void work_q_put(int thread_id, CUDAMyWork* w)
{
    lock_acquire();
    g_work_queue.push_back(w);
    g_queue_empty = false;
    int depth = (int)g_work_queue.size();
    if (depth > g_max_depth) g_max_depth = depth;
    lock_release();
}

// ======================================================================
// work_q_put_all()
// OpenMP:
//   void work_q_put_all(int thread_id, std::vector<my_work*>& work) {
//       gm_spinlock_acquire(&q_lock);
//       for each w in work: the_q.push_back(w);
//       work_queue_empty = false;
//       int depth = the_q.size();
//       if (depth > max_depth) max_depth = depth;
//       gm_spinlock_release(&q_lock);
//   }
// ======================================================================
void work_q_put_all(int thread_id, std::vector<CUDAMyWork*>& works)
{
    lock_acquire();
    for (size_t i = 0; i < works.size(); i++) {
        g_work_queue.push_back(works[i]);
        g_queue_empty = false;
    }
    int depth = (int)g_work_queue.size();
    if (depth > g_max_depth) g_max_depth = depth;
    lock_release();
}

// ======================================================================
// work_q_print_max_depth()
// OpenMP:
//   void work_q_print_max_depth() {printf("max depth = %d\n", max_depth);}
// ======================================================================
void work_q_print_max_depth()
{
    printf("max depth = %d\n", g_max_depth);
}

// ======================================================================
// scatter_by_color_kernel() — scatter targets into per-color compact arrays
// Uses warp ballot (3 colors × 1 atomic per warp = 3 atomics per warp
// instead of 3 per thread).
// Used by create_works_after_bfs_trim() to build device node lists
// ======================================================================
__global__ void scatter_by_color_kernel(
    const int* d_Color,
    const int* d_targets, int num_targets,
    int fw_color, int bw_color, int base_color,
    int* fw_out, int* fw_pos,
    int* bw_out, int* bw_pos,
    int* base_out, int* base_pos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // NOTE: Do NOT early-return — all threads in warp must participate in
    // __ballot_sync below to avoid undefined behavior (warp divergence).
    // Guard data access with conditional instead:
    node_t t = 0;
    bool is_fw = false, is_bw = false, is_base = false;
    if (i < num_targets) {
        t = d_targets[i];
        int c = d_Color[t];
        is_fw   = (c == fw_color);
        is_bw   = (c == bw_color);
        is_base = (c == base_color);
    }

    int lane = threadIdx.x & 31;

    // --- FW scatter ---
    unsigned mask_fw = __ballot_sync(0xffffffff, is_fw);
    int fw_warp_cnt = __popc(mask_fw);
    int fw_base = 0;
    if (lane == 0 && fw_warp_cnt > 0)
        fw_base = atomicAdd(fw_pos, fw_warp_cnt);
    fw_base = __shfl_sync(0xffffffff, fw_base, 0);
    if (is_fw)
        fw_out[fw_base + __popc(mask_fw & ((1u << lane) - 1))] = t;

    // --- BW scatter ---
    unsigned mask_bw = __ballot_sync(0xffffffff, is_bw);
    int bw_warp_cnt = __popc(mask_bw);
    int bw_base = 0;
    if (lane == 0 && bw_warp_cnt > 0)
        bw_base = atomicAdd(bw_pos, bw_warp_cnt);
    bw_base = __shfl_sync(0xffffffff, bw_base, 0);
    if (is_bw)
        bw_out[bw_base + __popc(mask_bw & ((1u << lane) - 1))] = t;

    // --- BASE scatter ---
    unsigned mask_base = __ballot_sync(0xffffffff, is_base);
    int base_warp_cnt = __popc(mask_base);
    int base_base = 0;
    if (lane == 0 && base_warp_cnt > 0)
        base_base = atomicAdd(base_pos, base_warp_cnt);
    base_base = __shfl_sync(0xffffffff, base_base, 0);
    if (is_base)
        base_out[base_base + __popc(mask_base & ((1u << lane) - 1))] = t;
}

// ======================================================================
// scatter_by_root_kernel() — scatter targets into per-WCC-root arrays
// Used by create_work_items_from_wcc()
// ======================================================================
__global__ void scatter_by_root_kernel(
    const int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets,
    int* d_root_offsets,  // [N] prefix sum offsets (mutated as atomic positions)
    int* d_root_buf)            // flat buffer for all root sets
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t = d_targets[i];
        if (d_Color[t] == SCC_FOUND) continue;

        int root = d_WCC[t] & 0x1FFFFFFF;  // GET_WCC_ROOT_MASKED
        int pos = atomicAdd(&d_root_offsets[root], 1);  // use offset storage as position counter
        d_root_buf[pos] = t;
    }
}

// ======================================================================
// scatter_single_color_kernel() — scatter targets of one color into output
// Warp ballot: 1 atomic per warp instead of 1 per thread.
// ======================================================================
__global__ void scatter_single_color_kernel(
    const int* d_Color,
    const int* d_targets, int num_targets,
    int target_color,
    int* d_out, int* d_pos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // NOTE: Do NOT early-return — all threads in warp must participate in
    // __ballot_sync below to avoid undefined behavior (warp divergence).
    // Guard data access with conditional instead:
    bool active = false;
    node_t t = 0;
    if (i < num_targets) {
        t = d_targets[i];
        active = (d_Color[t] == target_color);
    }

    unsigned mask = __ballot_sync(0xffffffff, active);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0)
        warp_base = atomicAdd(d_pos, warp_count);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    if (active)
        d_out[warp_base + __popc(mask & ((1u << lane) - 1))] = t;
}

// ======================================================================
// find_pivot_all_nodes_kernel() — find first node with base_color by
// scanning ALL N nodes (used when d_trim_targets_count == 0)
//
// OpenMP: for(node_t i=0;i<G.num_nodes(); i++)
//             if (G_Color[i] == color) return i;
// ======================================================================
__global__ void find_pivot_all_nodes_kernel(
    const int* d_Color, int num_nodes,
    int base_color, int* d_pivot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    if (d_Color[tid] == base_color) {
        atomicMin(d_pivot, tid);
    }
}

// ======================================================================
// scatter_single_color_all_nodes_kernel() — scatter nodes of a single
// color by scanning ALL N nodes (used when d_trim_targets_count == 0)
//
// OpenMP: for(node_t i=0;i<G.num_nodes(); i++)
//             if (G_Color[i] == color) S->insert(i);
// ======================================================================
__global__ void scatter_single_color_all_nodes_kernel(
    const int* d_Color, int num_nodes,
    int target_color,
    int* d_out, int* d_pos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // NOTE: Do NOT early-return — all threads in warp must participate in
    // __ballot_sync below to avoid undefined behavior (warp divergence).
    // Guard data access with conditional instead:
    bool active = false;
    if (i < num_nodes) {
        active = (d_Color[i] == target_color);
    }

    unsigned mask = __ballot_sync(0xffffffff, active);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0)
        warp_base = atomicAdd(d_pos, warp_count);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    if (active)
        d_out[warp_base + __popc(mask & ((1u << lane) - 1))] = i;
}

// ======================================================================
// count_by_wcc_root_kernel() — count members per WCC root
// ======================================================================
__global__ void count_by_wcc_root_kernel(
    const int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets,
    int* d_root_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t = d_targets[i];
        if (d_Color[t] == SCC_FOUND) continue;

        int root = d_WCC[t] & 0x1FFFFFFF;  // GET_WCC_ROOT_MASKED
        atomicAdd(&d_root_counts[root], 1);
    }
}
