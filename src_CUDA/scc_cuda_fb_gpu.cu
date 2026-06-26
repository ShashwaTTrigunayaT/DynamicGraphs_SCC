#include "scc_cuda.h"

// ======================================================================
// Batch GPU FB Kernel
//
// Replaces start_workers_fw_bw_dfs_host() for many-component graphs
// (it-2004, wb-edu). Processes all components in parallel on GPU.
//
// Strategy: Iterative level-by-level processing on GPU.
//   Each level: one kernel launch with one block per component.
//   Each block: loads component nodes into shared memory,
//   runs FW+BW BFS, marks SCCs, writes sub-component data.
//
// The scatter phase uses pre-allocated device buffers to avoid
// per-component cudaMalloc overhead.
// ======================================================================

#define GPU_FB_MAX_SMEM_NODES 2048
#define GPU_FB_HASH_SIZE     4096   // power of 2, 2× load factor for 2048 nodes

static int* d_fb_color_counter = NULL;

// ---- Bulk scatter buffer (reused across iterations, avoids per-comp cudaMemcpy) ----
static int* d_bulk_buf   = NULL;  // [d_bulk_cap]
static int  d_bulk_cap   = 0;
static int* d_bulk_off   = NULL;  // [max_comps] prefix sum offsets (reused)
static int  d_bulk_off_cap = 0;

// ======================================================================
// gpu_fb_batch_kernel
//
// Shared memory layout per block:
//   [0..comp_size-1]           : smem_nodes (int)     — node IDs
//   [MAX..MAX+comp_size-1]     : smem_frontier (int)  — BFS frontier (positions)
//   [2*MAX..2*MAX+comp_size-1] : smem_next (int)      — next frontier (positions)
//   [3*MAX..3*MAX+comp_size-1] : smem_fw_flag (char)  — 1 if FW-marked
//   [3*MAX+MAX/4..]            : smem_bw_flag (char)  — 1 if BW-marked
//   [3*MAX+MAX/2..]            : smem_fsize (int)     — frontier size [0]
//   [3*MAX+MAX/2+1]            : smem_pivot (int)     — pivot position [0]
//   [3*MAX+MAX/2+2]            : smem_fw_color (int)  — fw color [0]
//   [3*MAX+MAX/2+3]            : smem_bw_color (int)  — bw color [0]
//   [3*MAX+MAX/2+4]            : smem_next_count (int)— next count [0]
//   [3*MAX+MAX/2+5..]          : warp_sums (int)      — warp reduction sums (max 32 warps × 3)
//
// Layout bytes = MAX*3*4 + MAX*2*1 + (5+32*3)*4 = MAX*14 + 404
// For MAX=2048: 28672 + 404 = 29076 bytes < 48KB ✓
// ======================================================================
__global__ void gpu_fb_batch_kernel(
    const edge_t* __restrict__ d_begin,
    const node_t* __restrict__ d_node_idx,
    const edge_t* __restrict__ d_r_begin,
    const node_t* __restrict__ d_r_node_idx,
    int* d_Color,
    int* d_SCC,

    const int* __restrict__ d_in_nodes,
    const int* __restrict__ d_in_comp_start,
    const int* __restrict__ d_in_comp_size,
    const int* __restrict__ d_in_comp_color,
    int num_comps,

    int* d_out_comp_color,
    int* d_out_comp_size,
    int* d_num_out,
    int* d_color_counter
)
{
    int comp_id = blockIdx.x;
    if (comp_id >= num_comps) return;

    int comp_start = d_in_comp_start[comp_id];
    int comp_size  = d_in_comp_size[comp_id];
    int base_color = d_in_comp_color[comp_id];

    if (comp_size == 0 || comp_size > GPU_FB_MAX_SMEM_NODES) return;

    extern __shared__ char smem_char[];
    int*  smem_nodes    = (int*)smem_char;
    int*  smem_frontier = smem_nodes + GPU_FB_MAX_SMEM_NODES;
    int*  smem_next     = smem_frontier + GPU_FB_MAX_SMEM_NODES;
    char* smem_fw_flag  = (char*)(smem_next + GPU_FB_MAX_SMEM_NODES);
    char* smem_bw_flag  = smem_fw_flag + GPU_FB_MAX_SMEM_NODES;
    int*  smem_shared   = (int*)(smem_bw_flag + GPU_FB_MAX_SMEM_NODES);
    int*  smem_hash     = smem_shared + (5 + 32 * 3);  // after scalars + warp_sums
    volatile int& smem_fsize     = smem_shared[0];
    volatile int& smem_pivot_pos = smem_shared[1];
    volatile int& smem_fw_color  = smem_shared[2];
    volatile int& smem_bw_color  = smem_shared[3];
    volatile int& smem_ncount    = smem_shared[4];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // ---- Load component nodes into shared memory ----
    for (int i = tid; i < comp_size; i += stride) {
        smem_nodes[i]   = d_in_nodes[comp_start + i];
        smem_fw_flag[i] = 0;
        smem_bw_flag[i] = 0;
    }
    if (tid == 0) { smem_fsize = 0; smem_pivot_pos = -1; smem_ncount = 0; }
    __syncthreads();

    // ---- Build hash table: node_id → position ----
    for (int i = tid; i < GPU_FB_HASH_SIZE; i += stride) {
        smem_hash[i] = -1;
    }
    __syncthreads();

    for (int i = tid; i < comp_size; i += stride) {
        int key = smem_nodes[i];
        // Fibonacci hash: multiply by golden ratio, take top bits
        unsigned h = (unsigned)key * 2654435761u;
        int slot = (int)(h >> 20) & (GPU_FB_HASH_SIZE - 1);
        // Linear probing with atomicCAS for concurrent threads
        while (atomicCAS(&smem_hash[slot], -1, i) != -1) {
            slot = (slot + 1) & (GPU_FB_HASH_SIZE - 1);
        }
    }
    __syncthreads();

    // ---- Handle single-node component ----
    if (comp_size == 1) {
        if (tid == 0) {
            int n = smem_nodes[0];
            d_Color[n] = SCC_FOUND;
            d_SCC[n]   = n;
        }
        return;
    }

    // ---- Find pivot ----
    if (tid == 0) {
        for (int i = 0; i < comp_size; i++) {
            if (d_Color[smem_nodes[i]] == base_color) {
                smem_pivot_pos = i;
                break;
            }
        }
    }
    __syncthreads();

    if (smem_pivot_pos < 0) return;  // nothing to process

    // ---- Allocate fw_color, bw_color ----
    if (tid == 0) {
        smem_fw_color = atomicAdd(d_color_counter, 1024) + 1;
        smem_bw_color = smem_fw_color + 1;
        // Mark pivot with fw_color
        d_Color[smem_nodes[smem_pivot_pos]] = smem_fw_color;
    }
    __syncthreads();

    int fw_color = smem_fw_color;
    int bw_color = smem_bw_color;
    int pivot_pos = smem_pivot_pos;

    // ================================================================
    // FW BFS from pivot
    // ================================================================
    if (tid == 0) {
        smem_frontier[0] = pivot_pos;
        smem_fw_flag[pivot_pos] = 1;
        smem_fsize = 1;
    }
    __syncthreads();

    for (int iter = 0; iter < 10000 && smem_fsize > 0; iter++) {
        if (tid == 0) smem_ncount = 0;
        __syncthreads();

        int fsize = smem_fsize;
        for (int fi = tid; fi < fsize; fi += stride) {
            int pos = smem_frontier[fi];
            int n = smem_nodes[pos];

            for (edge_t e = d_begin[n]; e < d_begin[n + 1]; e++) {
                node_t k = d_node_idx[e];
                int old = atomicCAS(&d_Color[k], base_color, fw_color);
                if (old == base_color) {
                    // Find position in shared memory
                    // Hash lookup: O(1) instead of O(comp_size) linear search
                    unsigned hk = (unsigned)k * 2654435761u;
                    int slot = (int)(hk >> 20) & (GPU_FB_HASH_SIZE - 1);
                    while (1) {
                        int pos = smem_hash[slot];
                        if (pos < 0) break;  // not found (shouldn't happen for valid k)
                        if (smem_nodes[pos] == k) {
                            smem_fw_flag[pos] = 1;
                            int np = atomicAdd((int*)&smem_ncount, 1);
                            if (np < GPU_FB_MAX_SMEM_NODES) smem_next[np] = pos;
                            break;
                        }
                        slot = (slot + 1) & (GPU_FB_HASH_SIZE - 1);
                    }
                }
            }
        }
        __syncthreads();

        // Swap frontiers (cap at GPU_FB_MAX_SMEM_NODES to prevent OOB SMEM reads)
        // NOTE: must use a stride loop, NOT smem_frontier[tid] = smem_next[tid].
        // With blockDim.x=256 but ncnt up to 2048, a single tid assignment only
        // copies 256 entries — the other 1792 slots retain stale garbage from the
        // previous BFS level, causing OOB SMEM reads in the next iteration.
        int ncnt = smem_ncount;
        if (ncnt > GPU_FB_MAX_SMEM_NODES) ncnt = GPU_FB_MAX_SMEM_NODES;
        for (int i = tid; i < ncnt; i += stride) {
            smem_frontier[i] = smem_next[i];
        }
        if (tid == 0) smem_fsize = ncnt;
        __syncthreads();
    }

    // ================================================================
    // Mark pivot as SCC, prepare BW BFS
    // ================================================================
    if (tid == 0) {
        int pivot_node = smem_nodes[pivot_pos];
        d_Color[pivot_node] = SCC_FOUND;
        d_SCC[pivot_node]   = pivot_node;
        smem_frontier[0] = pivot_pos;
        smem_bw_flag[pivot_pos] = 1;
        smem_fsize = 1;
        smem_ncount = 0;
    }
    __syncthreads();

    // ================================================================
    // BW BFS from pivot
    // ================================================================
    for (int iter = 0; iter < 10000 && smem_fsize > 0; iter++) {
        if (tid == 0) smem_ncount = 0;
        __syncthreads();

        int fsize = smem_fsize;
        for (int fi = tid; fi < fsize; fi += stride) {
            int pos = smem_frontier[fi];
            int n = smem_nodes[pos];

            for (edge_t e = d_r_begin[n]; e < d_r_begin[n + 1]; e++) {
                node_t k = d_r_node_idx[e];

                // TOCTOU-safe: two separate atomicCAS calls instead of read-then-CAS
                // First try: claim fw_color node as SCC_FOUND
                int old = atomicCAS(&d_Color[k], fw_color, SCC_FOUND);
                if (old == fw_color) {
                    d_SCC[k] = smem_nodes[pivot_pos];
                    // Hash lookup: O(1) instead of O(comp_size) linear search
                    unsigned hk = (unsigned)k * 2654435761u;
                    int slot = (int)(hk >> 20) & (GPU_FB_HASH_SIZE - 1);
                    while (1) {
                        int pos = smem_hash[slot];
                        if (pos < 0) break;
                        if (smem_nodes[pos] == k) {
                            smem_bw_flag[pos] = 1;
                            int np = atomicAdd((int*)&smem_ncount, 1);
                            if (np < GPU_FB_MAX_SMEM_NODES) smem_next[np] = pos;
                            break;
                        }
                        slot = (slot + 1) & (GPU_FB_HASH_SIZE - 1);
                    }
                } else {
                    // Second try: claim base_color node as bw_color
                    old = atomicCAS(&d_Color[k], base_color, bw_color);
                    if (old == base_color) {
                        // Hash lookup: O(1) instead of O(comp_size) linear search
                        unsigned hk = (unsigned)k * 2654435761u;
                        int slot = (int)(hk >> 20) & (GPU_FB_HASH_SIZE - 1);
                        while (1) {
                            int pos = smem_hash[slot];
                            if (pos < 0) break;
                            if (smem_nodes[pos] == k) {
                                smem_bw_flag[pos] = 1;
                                int np = atomicAdd((int*)&smem_ncount, 1);
                                if (np < GPU_FB_MAX_SMEM_NODES) smem_next[np] = pos;
                                break;
                            }
                            slot = (slot + 1) & (GPU_FB_HASH_SIZE - 1);
                        }
                    }
                    // else: already claimed by another thread, or SCC_FOUND — skip
                }
            }
        }
        __syncthreads();

        int ncnt = smem_ncount;
        if (ncnt > GPU_FB_MAX_SMEM_NODES) ncnt = GPU_FB_MAX_SMEM_NODES;
        for (int i = tid; i < ncnt; i += stride) {
            smem_frontier[i] = smem_next[i];
        }
        if (tid == 0) smem_fsize = ncnt;
        __syncthreads();
    }

    // ================================================================
    // Count sub-partitions by d_Color value
    // ================================================================
    int local_fw = 0, local_bw = 0, local_base = 0;
    for (int i = tid; i < comp_size; i += stride) {
        int c = d_Color[smem_nodes[i]];
        if (c == fw_color)        local_fw++;
        else if (c == bw_color)   local_bw++;
        else if (c == base_color) local_base++;
    }

    // Warp-level reduction
    #define SUM_REDUCE(v) do { \
        v += __shfl_xor_sync(0xffffffff, v, 16); \
        v += __shfl_xor_sync(0xffffffff, v, 8);  \
        v += __shfl_xor_sync(0xffffffff, v, 4);  \
        v += __shfl_xor_sync(0xffffffff, v, 2);  \
        v += __shfl_xor_sync(0xffffffff, v, 1);  \
    } while(0)

    SUM_REDUCE(local_fw);
    SUM_REDUCE(local_bw);
    SUM_REDUCE(local_base);

    // warp_sums is in dynamic SMEM, after smem_shared (5 ints)
    int* warp_sums = smem_shared + 5;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane == 0 && warp_id < 32) {
        warp_sums[warp_id * 3 + 0] = local_fw;
        warp_sums[warp_id * 3 + 1] = local_bw;
        warp_sums[warp_id * 3 + 2] = local_base;
    }
    __syncthreads();

    if (warp_id == 0 && lane < num_warps) {
        local_fw   = warp_sums[lane * 3 + 0];
        local_bw   = warp_sums[lane * 3 + 1];
        local_base = warp_sums[lane * 3 + 2];
    } else if (warp_id == 0) {
        // Zero out unused lanes (8..31) so the final SUM_REDUCE is correct
        local_fw   = 0;
        local_bw   = 0;
        local_base = 0;
    }
    if (warp_id == 0) {
        SUM_REDUCE(local_fw);
        SUM_REDUCE(local_bw);
        SUM_REDUCE(local_base);
    }

    // ================================================================
    // Write sub-component descriptors to output (exact slot count)
    // ================================================================
    // We don't write the actual node IDs here — the host will scatter them
    // using a single pre-allocated buffer, avoiding per-component cudaMalloc.
    if (tid == 0) {
        // Count how many non-zero sub-components we actually have
        int slots = (local_fw > 0 ? 1 : 0)
                  + (local_bw > 0 ? 1 : 0)
                  + (local_base > 0 ? 1 : 0);
        if (slots > 0) {
            int idx = atomicAdd(d_num_out, slots);
            int i = idx;
            if (local_fw   > 0) { d_out_comp_size[i] = local_fw;   d_out_comp_color[i] = fw_color;   i++; }
            if (local_bw   > 0) { d_out_comp_size[i] = local_bw;   d_out_comp_color[i] = bw_color;   i++; }
            if (local_base > 0) { d_out_comp_size[i] = local_base; d_out_comp_color[i] = base_color; i++; }
        }
    }
}

// ======================================================================
// bulk_scatter_single_color_kernel — one block per sub-component
// Scatters matching nodes into pre-allocated range in d_bulk_buf.
// Uses warp ballot: 1 atomic per warp per iteration.
// All blocks write to the same bulk buffer at different offsets.
// ======================================================================
__global__ void bulk_scatter_single_color_kernel(
    const int* d_Color,
    const int* d_in_nodes, int num_src,
    const int* d_out_sizes,
    const int* d_out_colors,
    int num_out,
    const int* d_bulk_offsets,  // [num_out] prefix sum of sizes
    int* d_bulk_buf)            // [total_nodes] flat output buffer
{
    int oi = blockIdx.x;
    if (oi >= num_out) return;

    int sz = d_out_sizes[oi];
    int color = d_out_colors[oi];
    if (sz <= 0) return;

    int base = d_bulk_offsets[oi];

    // Per-block atomic position counter in shared memory
    __shared__ volatile int s_pos;
    if (threadIdx.x == 0) s_pos = base;
    __syncthreads();

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < num_src; i += stride) {
        bool match = (d_Color[d_in_nodes[i]] == color);

        unsigned mask = __ballot_sync(0xffffffff, match);
        int lane = threadIdx.x & 31;
        int warp_count = __popc(mask);

        int pos = 0;
        if (lane == 0 && warp_count > 0)
            pos = atomicAdd((int*)&s_pos, warp_count);
        pos = __shfl_sync(0xffffffff, pos, 0);

        if (match)
            d_bulk_buf[pos + __popc(mask & ((1u << lane) - 1))] = d_in_nodes[i];
    }
}

// ======================================================================
// run_gpu_fb() — Host driver
//
// Iteratively processes all components on GPU.
// Returns: FB processing time in ms
// ======================================================================
double run_gpu_fb(GPUState& st, const GPUGraph& g, int num_threads)
{
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    int block_size = 256;

    // ---- Initialize device color counter ----
    if (!d_fb_color_counter) {
        CUDA_CHECK(cudaMalloc(&d_fb_color_counter, sizeof(int)));
    }
    CUDA_TIMED_MEMCPY(d_fb_color_counter, &_cuda_the_color, sizeof(int),
                           cudaMemcpyHostToDevice);

    // ---- Drain WCC work queue ----
    std::vector<CUDAMyWork*> all_works;
    work_q_fetch_N(0, 999999, all_works);

    if (all_works.empty()) {
        gettimeofday(&t1, NULL);
        return 0.0;
    }

    // ---- Determine if we can use WCC big buffer directly (avoid D2H) ----
    // All WCC work items' d_set_nodes are slices of d_wcc_big_buffer.
    // If available, compute offsets via pointer arithmetic and keep data on GPU.
    int* wcc_buf = get_wcc_big_buffer();
    int wcc_buf_size = get_wcc_big_buffer_size();
    bool can_use_wcc_buffer = (wcc_buf != NULL && wcc_buf_size > 0);

    // ---- Pack initial components ----
    struct CompData {
        std::vector<int> starts;   // offsets into GPU node buffer
        std::vector<int> sizes;
        std::vector<int> colors;
    };
    struct NodeVec {
        std::vector<int> data;
        int* data_ptr() { return data.data(); }
        const int* data_ptr() const { return data.data(); }
        bool empty() const { return data.empty(); }
        size_t size() const { return data.size(); }
    };

    CompData cur;
    NodeVec cur_host_nodes;  // only populated for fallback path (no WCC buffer)
    const int* d_gpu_nodes = NULL;  // pointer to current GPU node buffer
    std::vector<CUDAMyWork*> large_works;  // for fallback

    if (can_use_wcc_buffer) {
        // Fast path: use WCC big buffer directly, no D2H needed
        for (CUDAMyWork* w : all_works) {
            if (w->count == 0) {
                delete w;
                continue;
            }
            if (w->count > GPU_FB_MAX_SMEM_NODES) {
                large_works.push_back(w);
                continue;
            }
            // Pointer difference is already in int elements (not bytes)
            // because both d_set_nodes and wcc_buf are int*.
            // e.g., if d_set_nodes = &wcc_buf[10], difference = 10.
            cur.starts.push_back((int)(w->d_set_nodes - wcc_buf));
            cur.sizes.push_back(w->count);
            cur.colors.push_back(w->color);
            delete w;
        }
        d_gpu_nodes = wcc_buf;
    } else {
        // Fallback: D2H+upload path
        for (CUDAMyWork* w : all_works) {
            if (w->count == 0) {
                if (w->owns_set && w->d_set_nodes) cudaFree(w->d_set_nodes);
                delete w;
                continue;
            }
            if (w->count > GPU_FB_MAX_SMEM_NODES) {
                large_works.push_back(w);
                continue;
            }
            cur.starts.push_back((int)cur_host_nodes.data.size());
            cur.sizes.push_back(w->count);
            cur.colors.push_back(w->color);
            if (w->d_set_nodes) {
                std::vector<int> tmp(w->count);
                CUDA_TIMED_MEMCPY(tmp.data(), w->d_set_nodes,
                                       w->count * sizeof(int), cudaMemcpyDeviceToHost);
                cur_host_nodes.data.insert(cur_host_nodes.data.end(),
                                           tmp.begin(), tmp.end());
                if (w->owns_set) cudaFree(w->d_set_nodes);
            }
            delete w;
        }
    }

    // Return large components to work queue for fallback
    if (!large_works.empty()) {
        printf("[GPU_FB] %zu large components (>%d nodes) deferred to fallback\n",
               large_works.size(), GPU_FB_MAX_SMEM_NODES);
        work_q_put_all(0, large_works);
        large_works.clear();
    }

    if (cur.sizes.empty()) {
        printf("[GPU_FB] No components fit in SMEM limit (%d) — all deferred to fallback\n",
               GPU_FB_MAX_SMEM_NODES);
        gettimeofday(&t1, NULL);
        return -1.0;  // signal: did no work, use fallback
    }

    printf("[GPU_FB] Initial: %zu comps, %s\n",
           cur.sizes.size(),
           can_use_wcc_buffer ? "using WCC buffer (no D2H)" : "uploaded from host");

    // ---- Allocate persistent device buffers ----
    // Metadata buffers (starts, sizes, colors) — uploaded each iteration
    // d_in_nodes is NOT needed — data lives in d_gpu_nodes (wcc_buf or d_bulk_buf)
    int max_dev_comps = (int)cur.sizes.size() * 4;

    int* d_in_nodes       = NULL;  // only allocated for fallback path (no WCC buffer)
    int* d_in_comp_start  = NULL;
    int* d_in_comp_size   = NULL;
    int* d_in_comp_color  = NULL;
    int* d_out_sizes      = NULL;
    int* d_out_colors     = NULL;
    int* d_num_out        = NULL;

    int max_dev_nodes = 0;
    if (!can_use_wcc_buffer) {
        // Fallback path needs d_in_nodes for host upload
        max_dev_nodes = (int)cur_host_nodes.size() * 4;
    }

    auto alloc_grow = [&]() {
        auto mall = [](auto*& p, int sz) {
            if (p) cudaFree(p);
            if (sz > 0) cudaMalloc(&p, sz * sizeof(int));
            else p = NULL;
        };
        if (!can_use_wcc_buffer) {
            mall(d_in_nodes, max_dev_nodes);
        }
        mall(d_in_comp_start,  max_dev_comps);
        mall(d_in_comp_size,   max_dev_comps);
        mall(d_in_comp_color,  max_dev_comps);
        mall(d_out_sizes,      max_dev_comps * 3);
        mall(d_out_colors,     max_dev_comps * 3);
        if (!d_num_out) cudaMalloc(&d_num_out, sizeof(int));
    };
    alloc_grow();

    // For the fallback path, allocate the GPU node buffer and set d_gpu_nodes
    if (!can_use_wcc_buffer && d_in_nodes) {
        d_gpu_nodes = d_in_nodes;
    }

    // ---- SMEM size ----
    int smem_shared_ints = 5 + 32 * 3;
    int smem_size = GPU_FB_MAX_SMEM_NODES * (3 * sizeof(int) + 2 * sizeof(char))
                  + smem_shared_ints * sizeof(int)
                  + GPU_FB_HASH_SIZE * sizeof(int);

    // ---- Iterative processing ----
    int total_levels = 0;
    double kernel_ms = 0, scatter_ms = 0;
    bool first_level = true;

    // cur_host_nodes is only needed for the first iteration of the fallback path
    // total_src_nodes = sum of all component sizes (needed by scatter kernel)
    int total_src_nodes = 0;
    for (int sz : cur.sizes) total_src_nodes += sz;
    if (!can_use_wcc_buffer) {
        total_src_nodes = (int)cur_host_nodes.size();
    }

    while (!cur.sizes.empty() && total_levels < 1000) {
        total_levels++;
        int cur_comps = (int)cur.sizes.size();

        // ---- Grow metadata buffers if needed ----
        int need_comps = cur_comps * 4;
        if (need_comps > max_dev_comps) {
            max_dev_comps = max(max_dev_comps * 2, need_comps);
            alloc_grow();
        }

        struct timeval ts, te;

        // ---- Upload metadata to device ----
        // d_in_nodes is NOT uploaded — data stays on GPU
        // For the fallback path: upload host nodes on the first iteration only
        if (!can_use_wcc_buffer && first_level) {
            // First level fallback: need to upload host buffer to GPU
            CUDA_TIMED_MEMCPY((void*)d_gpu_nodes, cur_host_nodes.data_ptr(),
                                   cur_host_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);
        }
        CUDA_TIMED_MEMCPY(d_in_comp_start, cur.starts.data(),
                               cur_comps * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_TIMED_MEMCPY(d_in_comp_size, cur.sizes.data(),
                               cur_comps * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_TIMED_MEMCPY(d_in_comp_color, cur.colors.data(),
                               cur_comps * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaMemset(d_num_out, 0, sizeof(int)));

        // ---- Launch batch kernel ----
        gettimeofday(&ts, NULL);

        int max_grid = 65535;
        for (int ch = 0; ch < cur_comps; ch += max_grid) {
            int n = min(max_grid, cur_comps - ch);
            int* starts = d_in_comp_start + ch;

            gpu_fb_batch_kernel<<<n, block_size, smem_size>>>(
                g.d_begin, g.d_node_idx,
                g.d_r_begin, g.d_r_node_idx,
                st.d_Color, st.d_SCC,
                d_gpu_nodes, starts,
                d_in_comp_size + ch,
                d_in_comp_color + ch,
                n,
                d_out_colors, d_out_sizes, d_num_out, d_fb_color_counter
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        gettimeofday(&te, NULL);
        kernel_ms += (te.tv_sec - ts.tv_sec) * 1000.0 + (te.tv_usec - ts.tv_usec) * 0.001;

        // ---- Read back results ----
        int h_num_out = 0;
        CUDA_TIMED_MEMCPY(&h_num_out, d_num_out, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_num_out == 0) break;  // all done

        // ---- Build next batch via bulk scatter ----
        std::vector<int> h_out_sizes(h_num_out);
        std::vector<int> h_out_colors(h_num_out);
        CUDA_TIMED_MEMCPY(h_out_sizes.data(), d_out_sizes,
                               h_num_out * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_TIMED_MEMCPY(h_out_colors.data(), d_out_colors,
                               h_num_out * sizeof(int), cudaMemcpyDeviceToHost);

        gettimeofday(&ts, NULL);

        // Compute prefix sums on host (offsets into d_bulk_buf)
        std::vector<int> h_bulk_offsets(h_num_out + 1, 0);
        int total_sz = 0;
        for (int oi = 0; oi < h_num_out; oi++) {
            h_bulk_offsets[oi] = total_sz;
            total_sz += h_out_sizes[oi];
        }
        h_bulk_offsets[h_num_out] = total_sz;

        // Allocate/grow bulk buffer if needed
        if (d_bulk_cap < total_sz) {
            if (d_bulk_buf) cudaFree(d_bulk_buf);
            d_bulk_cap = max(total_sz * 2, 1024);
            cudaMalloc(&d_bulk_buf, d_bulk_cap * sizeof(int));
        }
        // Allocate/grow offsets buffer if needed
        if (d_bulk_off_cap < h_num_out + 1) {
            if (d_bulk_off) cudaFree(d_bulk_off);
            d_bulk_off_cap = max(h_num_out + 1, 1024);
            cudaMalloc(&d_bulk_off, d_bulk_off_cap * sizeof(int));
        }

        // Upload offsets to device
        CUDA_TIMED_MEMCPY(d_bulk_off, h_bulk_offsets.data(),
                               (h_num_out + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Launch bulk scatter: one block per sub-component, chunked for max grid
        int scatter_max_grid = 65535;
        for (int ch = 0; ch < h_num_out; ch += scatter_max_grid) {
            int n = min(scatter_max_grid, h_num_out - ch);
            bulk_scatter_single_color_kernel<<<n, block_size>>>(
                st.d_Color, d_gpu_nodes, total_src_nodes,
                d_out_sizes + ch, d_out_colors + ch, n,
                d_bulk_off + ch,
                d_bulk_buf
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- GPU-only: keep scatter results on device, no D2H ----
        // Build next iteration metadata from prefix sums (already on host)
        CompData next;
        for (int oi = 0; oi < h_num_out; oi++) {
            int sz = h_out_sizes[oi];
            int color = h_out_colors[oi];
            if (sz <= 0) continue;
            // starts = offsets into d_bulk_buf = the prefix sums we just computed
            next.starts.push_back(h_bulk_offsets[oi]);
            next.sizes.push_back(sz);
            next.colors.push_back(color);
        }

        gettimeofday(&te, NULL);
        scatter_ms += (te.tv_sec - ts.tv_sec) * 1000.0 + (te.tv_usec - ts.tv_usec) * 0.001;

        // ---- Swap: next iteration reads from d_bulk_buf ----
        d_gpu_nodes = d_bulk_buf;
        total_src_nodes = total_sz;
        cur = std::move(next);
        first_level = false;

        if (total_levels % 10 == 0)
            printf("[GPU_FB] Lvl %d: %zu comps, %d nodes\n",
                   total_levels, cur.sizes.size(), total_sz);
    }

    // ---- Sync color counter ----
    CUDA_CHECK(cudaMemcpy(&_cuda_the_color, d_fb_color_counter,
                           sizeof(int), cudaMemcpyDeviceToHost));

    // ---- Cleanup device buffers ----
    auto sf = [](auto*& p) { if (p) { cudaFree(p); p = NULL; } };
    sf(d_in_nodes); sf(d_in_comp_start); sf(d_in_comp_size); sf(d_in_comp_color);
    sf(d_out_sizes); sf(d_out_colors); sf(d_num_out);

    printf("[GPU_FB] Done: %d levels, kernel=%.2fms scatter=%.2fms\n",
           total_levels, kernel_ms, scatter_ms);

    gettimeofday(&t1, NULL);
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) * 0.001;
}

// ======================================================================
// finalize_fb_gpu() — Free persistent buffers
// ======================================================================
void finalize_fb_gpu()
{
    if (d_fb_color_counter) { cudaFree(d_fb_color_counter); d_fb_color_counter = NULL; }
    if (d_bulk_buf)         { cudaFree(d_bulk_buf);         d_bulk_buf = NULL; }
    if (d_bulk_off)         { cudaFree(d_bulk_off);         d_bulk_off = NULL; }
    d_bulk_cap = 0; d_bulk_off_cap = 0;
}
