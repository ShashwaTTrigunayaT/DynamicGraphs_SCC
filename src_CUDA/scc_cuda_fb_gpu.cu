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

static int* d_fb_color_counter = NULL;

// ---- Pre-allocated scatter buffers (reused across iterations) ----
static int* d_scatter_buf   = NULL;  // [max_comp_nodes]
static int* d_scatter_pos   = NULL;  // [1]
static int  d_scatter_cap   = 0;

// ======================================================================
// Ensure scatter buffers are large enough
// ======================================================================
static void ensure_scatter_buf(int max_comp_nodes)
{
    if (d_scatter_cap < max_comp_nodes) {
        if (d_scatter_buf) cudaFree(d_scatter_buf);
        if (d_scatter_pos) cudaFree(d_scatter_pos);
        d_scatter_cap = max_comp_nodes * 2;  // generous
        cudaMalloc(&d_scatter_buf, d_scatter_cap * sizeof(int));
        cudaMalloc(&d_scatter_pos, sizeof(int));
    }
}

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
//
// Total: MAX*3*4 + MAX/4 + MAX/4 + 5*4 ≈ MAX*12 + MAX/2 + 20
// For MAX=2048: 24576 + 1024 + 20 = 25620 bytes < 48KB ✓
// Actually with char arrays: MAX*3*4 + MAX*1 + MAX*1 + 5*4 = MAX*14 + 20
// For MAX=2048: 28672 + 20 = 28692 bytes < 48KB ✓
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
                    for (int j = 0; j < comp_size; j++) {
                        if (smem_nodes[j] == k) {
                            smem_fw_flag[j] = 1;
                            int np = atomicAdd((int*)&smem_ncount, 1);
                            if (np < GPU_FB_MAX_SMEM_NODES) smem_next[np] = j;
                            break;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Swap frontiers
        int ncnt = smem_ncount;
        if (ncnt > 0 && tid < ncnt) smem_frontier[tid] = smem_next[tid];
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
                int k_color = d_Color[k];

                if (k_color == fw_color || k_color == base_color) {
                    int target = (k_color == fw_color) ? SCC_FOUND : bw_color;
                    int old = atomicCAS(&d_Color[k], k_color, target);
                    if (old == k_color) {
                        if (target == SCC_FOUND) {
                            d_SCC[k] = smem_nodes[pivot_pos];
                        }
                        for (int j = 0; j < comp_size; j++) {
                            if (smem_nodes[j] == k && !smem_bw_flag[j]) {
                                smem_bw_flag[j] = 1;
                                int np = atomicAdd((int*)&smem_ncount, 1);
                                if (np < GPU_FB_MAX_SMEM_NODES) smem_next[np] = j;
                                break;
                            }
                        }
                    } else if (old == target) {
                        // Already claimed, but might not be in bw_flag
                        for (int j = 0; j < comp_size; j++) {
                            if (smem_nodes[j] == k && !smem_bw_flag[j]) {
                                smem_bw_flag[j] = 1;
                                int np = atomicAdd((int*)&smem_ncount, 1);
                                if (np < GPU_FB_MAX_SMEM_NODES) smem_next[np] = j;
                                break;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();

        int ncnt = smem_ncount;
        if (ncnt > 0 && tid < ncnt) smem_frontier[tid] = smem_next[tid];
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

    volatile __shared__ int warp_sums[96];  // 3 * 32 warps
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
    }
    if (warp_id == 0) {
        SUM_REDUCE(local_fw);
        SUM_REDUCE(local_bw);
        SUM_REDUCE(local_base);
    }

    // ================================================================
    // Write sub-component descriptors to output
    // ================================================================
    // We don't write the actual node IDs here — the host will scatter them
    // using a single pre-allocated buffer, avoiding per-component cudaMalloc.
    if (tid == 0) {
        // We need to write (size, color) triplets to output.
        // But d_out_comp_color and d_out_comp_size are indexed by output component.
        // We'll write them sequentially using atomic add.
        if (local_fw > 0) {
            int idx = atomicAdd(d_num_out, 3);  // reserve 3 slots
            d_out_comp_size[idx]     = local_fw;
            d_out_comp_color[idx]    = fw_color;
            d_out_comp_size[idx + 1] = local_bw;
            d_out_comp_color[idx + 1] = bw_color;
            d_out_comp_size[idx + 2] = local_base;
            d_out_comp_color[idx + 2] = base_color;
        } else if (local_bw > 0) {
            int idx = atomicAdd(d_num_out, 2);  // reserve 2 slots
            d_out_comp_size[idx]     = local_bw;
            d_out_comp_color[idx]    = bw_color;
            d_out_comp_size[idx + 1] = local_base;
            d_out_comp_color[idx + 1] = base_color;
        } else if (local_base > 0) {
            int idx = atomicAdd(d_num_out, 1);
            d_out_comp_size[idx] = local_base;
            d_out_comp_color[idx] = base_color;
        }
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

    int N = g.num_nodes;
    int block_size = 256;

    // ---- Initialize device color counter ----
    if (!d_fb_color_counter) {
        CUDA_CHECK(cudaMalloc(&d_fb_color_counter, sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpy(d_fb_color_counter, &_cuda_the_color, sizeof(int),
                           cudaMemcpyHostToDevice));
    ensure_scatter_buf(GPU_FB_MAX_SMEM_NODES);

    // ---- Drain WCC work queue ----
    std::vector<CUDAMyWork*> all_works;
    work_q_fetch_N(0, 999999, all_works);

    if (all_works.empty()) {
        gettimeofday(&t1, NULL);
        return 0.0;
    }

    // ---- Pack initial components ----
    // Large components (> GPU_FB_MAX_SMEM_NODES) are returned to the work queue
    // for the host fallback path (start_workers_fw_bw_dfs_host).
    struct CompData {
        std::vector<int> nodes;
        std::vector<int> starts;   // prefix sum into nodes
        std::vector<int> sizes;
        std::vector<int> colors;
    };

    CompData cur;
    std::vector<CUDAMyWork*> large_works;  // for fallback

    for (CUDAMyWork* w : all_works) {
        if (w->count == 0) {
            if (w->owns_set && w->d_set_nodes) cudaFree(w->d_set_nodes);
            delete w;
            continue;
        }
        if (w->count > GPU_FB_MAX_SMEM_NODES) {
            // Too large for SMEM — keep for fallback
            large_works.push_back(w);
            continue;
        }

        cur.starts.push_back((int)cur.nodes.size());
        cur.sizes.push_back(w->count);
        cur.colors.push_back(w->color);

        if (w->d_set_nodes) {
            std::vector<int> tmp(w->count);
            CUDA_CHECK(cudaMemcpy(tmp.data(), w->d_set_nodes,
                                   w->count * sizeof(int), cudaMemcpyDeviceToHost));
            cur.nodes.insert(cur.nodes.end(), tmp.begin(), tmp.end());
            if (w->owns_set) cudaFree(w->d_set_nodes);
        }
        delete w;
    }

    // Return large components to work queue for fallback
    if (!large_works.empty()) {
        printf("[GPU_FB] %zu large components (>%d nodes) deferred to fallback\n",
               large_works.size(), GPU_FB_MAX_SMEM_NODES);
        work_q_put_all(0, large_works);
        large_works.clear();
    }

    if (cur.nodes.empty()) {
        printf("[GPU_FB] No components fit in SMEM limit (%d) — all deferred to fallback\n",
               GPU_FB_MAX_SMEM_NODES);
        gettimeofday(&t1, NULL);
        return -1.0;  // signal: did no work, use fallback
    }

    printf("[GPU_FB] Initial: %zu comps, %zu nodes\n", cur.sizes.size(), cur.nodes.size());

    // ---- Allocate persistent device buffers ----
    // We'll grow these as needed
    int max_dev_comps = (int)cur.sizes.size() * 4;
    int max_dev_nodes = (int)cur.nodes.size() * 4;

    int* d_in_nodes       = NULL;
    int* d_in_comp_start  = NULL;
    int* d_in_comp_size   = NULL;
    int* d_in_comp_color  = NULL;
    int* d_out_sizes      = NULL;
    int* d_out_colors     = NULL;
    int* d_num_out        = NULL;

    auto alloc_grow = [&]() {
        auto mall = [](auto*& p, int sz) {
            if (p) cudaFree(p);
            if (sz > 0) cudaMalloc(&p, sz * sizeof(int));
            else p = NULL;
        };
        mall(d_in_nodes,       max_dev_nodes);
        mall(d_in_comp_start,  max_dev_comps);
        mall(d_in_comp_size,   max_dev_comps);
        mall(d_in_comp_color,  max_dev_comps);
        mall(d_out_sizes,      max_dev_comps * 3);  // up to 3 sub-comps per input
        mall(d_out_colors,     max_dev_comps * 3);
        if (!d_num_out) cudaMalloc(&d_num_out, sizeof(int));
    };
    alloc_grow();

    // ---- SMEM size ----
    int smem_size = GPU_FB_MAX_SMEM_NODES * (3 * sizeof(int))  // nodes, frontier, next
                  + GPU_FB_MAX_SMEM_NODES * 2                   // fw_flag, bw_flag (char)
                  + 5 * sizeof(int);                            // shared scalars
    // = 2048*(12+2)+20 = 28692 bytes < 48KB ✓

    // ---- Iterative processing ----
    int total_levels = 0;
    double kernel_ms = 0, scatter_ms = 0;

    while (!cur.nodes.empty() && total_levels < 1000) {
        total_levels++;
        int cur_comps = (int)cur.sizes.size();

        // ---- Grow buffers if needed ----
        int need_comps = cur_comps * 4;
        int need_nodes = (int)cur.nodes.size();
        if (need_comps > max_dev_comps || need_nodes > max_dev_nodes) {
            max_dev_comps = max(max_dev_comps * 2, need_comps);
            max_dev_nodes = max(max_dev_nodes * 2, need_nodes * 2);
            alloc_grow();
        }

        struct timeval ts, te;

        // ---- Upload to device ----
        CUDA_CHECK(cudaMemcpy(d_in_nodes, cur.nodes.data(),
                               cur.nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_comp_start, cur.starts.data(),
                               cur_comps * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_comp_size, cur.sizes.data(),
                               cur_comps * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_comp_color, cur.colors.data(),
                               cur_comps * sizeof(int), cudaMemcpyHostToDevice));
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
                d_in_nodes, starts,
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
        CUDA_CHECK(cudaMemcpy(&h_num_out, d_num_out, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_num_out == 0) break;  // all done

        // ---- Build next batch via per-component scatter (using pre-allocated buf) ----
        std::vector<int> h_out_sizes(h_num_out);
        std::vector<int> h_out_colors(h_num_out);
        CUDA_CHECK(cudaMemcpy(h_out_sizes.data(), d_out_sizes,
                               h_num_out * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_colors.data(), d_out_colors,
                               h_num_out * sizeof(int), cudaMemcpyDeviceToHost));

        gettimeofday(&ts, NULL);

        CompData next;
        int num_src = (int)cur.nodes.size();
        int gs2 = min((num_src + block_size - 1) / block_size, 1024);

        for (int oi = 0; oi < h_num_out; oi++) {
            int sz = h_out_sizes[oi];
            int color = h_out_colors[oi];
            if (sz <= 0) continue;

            CUDA_CHECK(cudaMemset(d_scatter_pos, 0, sizeof(int)));

            scatter_single_color_kernel<<<gs2, block_size>>>(
                st.d_Color, d_in_nodes, num_src, color,
                d_scatter_buf, d_scatter_pos);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<int> tmp(sz);
            CUDA_CHECK(cudaMemcpy(tmp.data(), d_scatter_buf,
                                   sz * sizeof(int), cudaMemcpyDeviceToHost));

            next.starts.push_back((int)next.nodes.size());
            next.nodes.insert(next.nodes.end(), tmp.begin(), tmp.end());
            next.sizes.push_back(sz);
            next.colors.push_back(color);
        }

        gettimeofday(&te, NULL);
        scatter_ms += (te.tv_sec - ts.tv_sec) * 1000.0 + (te.tv_usec - ts.tv_usec) * 0.001;

        // ---- Swap ----
        cur = std::move(next);

        if (total_levels % 10 == 0)
            printf("[GPU_FB] Lvl %d: %zu comps, %zu nodes\n",
                   total_levels, cur.sizes.size(), cur.nodes.size());
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
    if (d_scatter_buf)      { cudaFree(d_scatter_buf);      d_scatter_buf = NULL; }
    if (d_scatter_pos)      { cudaFree(d_scatter_pos);      d_scatter_pos = NULL; }
    d_scatter_cap = 0;
}
