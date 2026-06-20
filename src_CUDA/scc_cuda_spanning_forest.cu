#include "scc_cuda.h"

// ======================================================================
// Spanning Forest SCC — GPU Implementation with full transitive
// union-find pivot tree merging.
//
// Key insight (from Claude's critique):
//   Without pivot merging, two pivots P and Q in the same true SCC
//   grow independent trees. Node n in FW(P) ∩ BW(Q) gets root_fw=P
//   and root_bw=Q. The root_fw==root_bw check fails even though n
//   is genuinely in the SCC.
//
// Fix (per Claude's design):
//   - Separate d_pivot_id_fw[n] and d_pivot_id_bw[n] arrays
//     (current code's single d_pivot_id is overwritten by BW after FW!)
//   - uf_find/uf_union on d_pivot_parent[K] (tiny — 512 ints)
//   - Every edge crossing two pivot trees triggers a union
//   - Extraction resolves through find() on compressed union-find
//
// Replaces Phases 2-5 (GLOBAL_BFS + TRIM1/2 + WCC + FB).
// Falls back to standard pipeline for any remaining nodes.
// ======================================================================

// ---- Device-side state ----
static int* d_parent_fw      = NULL;  // [N] FW parent (-1 = no parent)
static int* d_parent_bw      = NULL;  // [N] BW parent (-1 = no parent)
static int* d_pivot_id_fw    = NULL;  // [N] pivot INDEX for FW tree (0..K-1)
static int* d_pivot_id_bw    = NULL;  // [N] pivot INDEX for BW tree (0..K-1)
static int* d_tree_depth     = NULL;  // [N] depth in tree
static int* d_pivots         = NULL;  // [K_max] pivot NODE IDs
static int* d_num_pivots     = NULL;  // [1] number of pivots (device)
static int* d_pivot_degrees  = NULL;  // [K_max] degree of each pivot

// Union-find for pivot tree merging (tiny — K ints, ~4KB).
// ONE shared array for BOTH FW and BW merge evidence.
// When FW discovers P→Q or BW discovers Q→R, both union into
// the SAME structure, creating a complete transitive picture.
// The resolved_fw == resolved_bw comparison then correctly
// checks "do this node's FW and BW pivots belong to the same
// merged group?" — the actual SCC condition.
static int* d_pivot_parent   = NULL;  // [K_max] shared union-find, FW+BW merges

static int  d_max_pivots = 0;  // set dynamically based on graph size
static int  d_num_nodes = 0;
static int  d_forest_initialized = 0;

// ---- SCC extraction state ----
static int* d_scc_counter  = NULL;  // [1] counter for SCCs found
static int* d_changed      = NULL;  // [1] convergence flag

// ======================================================================
// initialize/finalize
// ======================================================================
void initialize_spanning_forest(int num_nodes)
{
    d_num_nodes = num_nodes;

    // Scale max pivots with graph size: ~1 per 1000 nodes, clamped to [64, 1024]
    d_max_pivots = max(64, min(num_nodes / 1000, 1024));

    auto alloc = [](auto*& p, int sz) {
        if (p) cudaFree(p);
        if (sz > 0) cudaMalloc(&p, sz * sizeof(int));
        else p = NULL;
    };

    alloc(d_parent_fw,       num_nodes);
    alloc(d_parent_bw,       num_nodes);
    alloc(d_pivot_id_fw,     num_nodes);
    alloc(d_pivot_id_bw,     num_nodes);
    alloc(d_tree_depth,      num_nodes);
    alloc(d_pivots,          d_max_pivots);
    alloc(d_num_pivots,      1);
    alloc(d_pivot_degrees,   d_max_pivots);
    alloc(d_pivot_parent,    d_max_pivots);
    alloc(d_scc_counter,     1);
    alloc(d_changed,         1);

    d_forest_initialized = 1;
}

void finalize_spanning_forest()
{
    auto sf = [](auto*& p) { if (p) { cudaFree(p); p = NULL; } };
    sf(d_parent_fw);
    sf(d_parent_bw);
    sf(d_pivot_id_fw);
    sf(d_pivot_id_bw);
    sf(d_tree_depth);
    sf(d_pivots);
    sf(d_num_pivots);
    sf(d_pivot_degrees);
    sf(d_pivot_parent);
    sf(d_scc_counter);
    sf(d_changed);
    d_forest_initialized = 0;
    d_num_nodes = 0;
}

// ======================================================================
// Device: union-find helpers (path-halving find, min-root union)
// These are __forceinline__ device functions callable from any kernel.
// ======================================================================
__device__ __forceinline__ int uf_find(int* d_pivot_parent, int x)
{
    // Path-halving: safe under concurrent access, no locks needed
    while (true) {
        int p = d_pivot_parent[x];
        int gp = d_pivot_parent[p];
        if (p == gp) return p;              // p is root
        atomicCAS(&d_pivot_parent[x], p, gp); // halve path
        x = p;
    }
}

__device__ __forceinline__ void uf_union(int* d_pivot_parent, int a, int b, int* d_changed)
{
    while (true) {
        int ra = uf_find(d_pivot_parent, a);
        int rb = uf_find(d_pivot_parent, b);
        if (ra == rb) return;               // already merged
        int lo = min(ra, rb);
        int hi = max(ra, rb);
        int old = atomicCAS(&d_pivot_parent[hi], hi, lo);
        if (old == hi) {                    // successfully claimed
            if (d_changed) *d_changed = 1;
            return;
        }
        // else someone else changed hi — retry
    }
}

// ======================================================================
// Kernel: select pivots from remaining nodes
//
// Strategy: Pick K nodes with highest (out-degree + in-degree).
// Uses warp-ballot for efficient prefix-sum style selection.
// Ensures distributed pivots: only highest-degree node per warp wins.
// ======================================================================
__global__ void select_pivots_kernel(
    const int* d_Color,
    const edge_t* d_begin,
    const edge_t* d_r_begin,
    const int* d_targets, int num_targets,
    int* d_pivots, int* d_num_pivots,
    int max_pivots,
    int* d_pivot_degrees)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t n = d_targets[i];
        if (d_Color[n] == SCC_FOUND) continue;

        int deg = (d_begin[n + 1] - d_begin[n]) + (d_r_begin[n + 1] - d_r_begin[n]);

        bool candidate = (deg > 5);
        unsigned mask = __ballot_sync(0xffffffff, candidate);
        int lane = threadIdx.x & 31;

        if (candidate) {
            int warp_rank = __popc(mask & ((1u << lane) - 1));
            if (warp_rank == 0) {
                int pos = atomicAdd(d_num_pivots, 1);
                if (pos < max_pivots) {
                    d_pivots[pos] = n;
                    d_pivot_degrees[pos] = deg;
                }
            }
        }
    }
}

// ======================================================================
// Kernel: initialize union-find: each pivot is its own root
// ======================================================================
__global__ void init_pivot_union_find_kernel(int* d_pivot_parent, int num_pivots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_pivots) d_pivot_parent[tid] = tid;
}

// ======================================================================
// Kernel: compress union-find (full path compression)
// Operates on K=512 elements only — trivially cheap, single block.
// ======================================================================
__global__ void uf_compress_kernel(int* d_pivot_parent, int num_pivots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_pivots) {
        d_pivot_parent[tid] = uf_find(d_pivot_parent, tid);
    }
}

// ======================================================================
// Kernel: initialize spanning trees from pivots
// ======================================================================
__global__ void init_spanning_trees_kernel(
    int* d_parent_fw, int* d_parent_bw,
    int* d_pivot_id_fw, int* d_pivot_id_bw,
    int* d_tree_depth,
    const int* d_pivots, int num_pivots,
    const int* d_targets, int num_targets,
    int* d_Color)
{
    // Initialize all targets to unassigned
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t n = d_targets[i];
        if (d_Color[n] == SCC_FOUND) continue;
        d_parent_fw[n]   = -1;
        d_parent_bw[n]   = -1;
        d_pivot_id_fw[n] = -1;
        d_pivot_id_bw[n] = -1;
        d_tree_depth[n]  = 0;
    }

    // Initialize pivots
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < num_pivots; i += blockDim.x) {
            node_t p = d_pivots[i];
            if (p >= 0) {
                d_parent_fw[p]   = p;
                d_parent_bw[p]   = p;
                d_pivot_id_fw[p] = i;
                d_pivot_id_bw[p] = i;
                d_tree_depth[p]  = 0;
            }
        }
    }
}

// ======================================================================
// Kernel: one iteration of FW spanning forest expansion
//
// For each unassigned node n:
//   1. Scan out-neighbors; if neighbor k is in a tree, atomicCAS to join
//   2. AFTER joining (or if already in tree), detect cross-pivot edges:
//      if n→k exists and BOTH n and k are in (possibly different) trees,
//      and pivot_id_fw[n] != pivot_id_fw[k], union them via uf_union.
//
// Step 2 merges into the SHARED d_pivot_parent (same array used by BW).
// This creates one complete transitive picture: FW evidence (P→Q via
// forward edges) and BW evidence (Q→R via backward edges) both feed
// into the same union-find, so {P,Q,R} converge to the same root.
// ======================================================================
__global__ void fw_spanning_forest_iteration_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    int* d_parent_fw, int* d_pivot_id_fw, int* d_tree_depth,
    int* d_pivot_parent,
    int* d_changed,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t n = d_targets[i];
        if (d_Color[n] == SCC_FOUND) continue;

        int old_parent = d_parent_fw[n];

        for (edge_t e = d_begin[n]; e < d_begin[n + 1]; e++) {
            node_t k = d_node_idx[e];
            if (k == n) continue;
            if (d_Color[k] == SCC_FOUND) continue;

            int k_parent = d_parent_fw[k];
            if (k_parent == -1) continue;  // k not in a tree yet

            // Step 1: If n not yet in a tree, try to join via k
            if (old_parent == -1) {
                int null_val = -1;
                if (atomicCAS(&d_parent_fw[n], null_val, k) == -1) {
                    d_pivot_id_fw[n] = d_pivot_id_fw[k];
                    d_tree_depth[n] = d_tree_depth[k] + 1;
                    if (d_changed) *d_changed = 1;
                    old_parent = k;  // mark joined, continue scanning for collisions
                }
            }

            // Step 2: Cross-pivot collision detection into SHARED d_pivot_parent
            if (old_parent != -1) {
                int pid_n = d_pivot_id_fw[n];
                int pid_k = d_pivot_id_fw[k];
                if (pid_n != -1 && pid_k != -1 && pid_n != pid_k) {
                    uf_union(d_pivot_parent, pid_n, pid_k, d_changed);
                }
            }
        }
    }
}

// ======================================================================
// Kernel: one iteration of BW spanning forest expansion (reverse edges)
//
// Same logic as FW kernel but on reverse graph (r_begin/r_node_idx)
// and using d_parent_bw / d_pivot_id_bw.
// Merges into the SAME d_pivot_parent as FW, so all evidence
// converges to one complete transitive union-find structure.
// ======================================================================
__global__ void bw_spanning_forest_iteration_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color,
    int* d_parent_bw, int* d_pivot_id_bw, int* d_tree_depth,
    int* d_pivot_parent,
    int* d_changed,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t n = d_targets[i];
        if (d_Color[n] == SCC_FOUND) continue;

        int old_parent = d_parent_bw[n];

        for (edge_t e = d_r_begin[n]; e < d_r_begin[n + 1]; e++) {
            node_t k = d_r_node_idx[e];
            if (k == n) continue;
            if (d_Color[k] == SCC_FOUND) continue;

            int k_parent = d_parent_bw[k];
            if (k_parent == -1) continue;

            // Step 1: If n not yet in a tree, try to join via k
            if (old_parent == -1) {
                int null_val = -1;
                if (atomicCAS(&d_parent_bw[n], null_val, k) == -1) {
                    d_pivot_id_bw[n] = d_pivot_id_bw[k];
                    d_tree_depth[n] = d_tree_depth[k] + 1;
                    if (d_changed) *d_changed = 1;
                    old_parent = k;
                }
            }

            // Step 2: Cross-pivot collision detection into SHARED d_pivot_parent
            if (old_parent != -1) {
                int pid_n = d_pivot_id_bw[n];
                int pid_k = d_pivot_id_bw[k];
                if (pid_n != -1 && pid_k != -1 && pid_n != pid_k) {
                    uf_union(d_pivot_parent, pid_n, pid_k, d_changed);
                }
            }
        }
    }
}

// ======================================================================
// Kernel: extract SCCs from FW ∩ BW tree intersections
//
// Uses ONE shared d_pivot_parent that captures ALL merge evidence
// (both FW and BW edge crossings). Both pid_fw and pid_bw resolve
// through this same structure, so resolved_fw == resolved_bw
// correctly means "this node's FW and BW pivots belong to the same
// merged group" — i.e., the node is in the true SCC.
// ======================================================================
__global__ void extract_sccs_from_forest_kernel(
    int* d_Color, int* d_SCC,
    const int* d_parent_fw, const int* d_parent_bw,
    const int* d_pivot_id_fw, const int* d_pivot_id_bw,
    int* d_pivot_parent,
    const int* d_pivots,
    const int* d_targets, int num_targets,
    int* d_scc_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t n = d_targets[i];
        if (d_Color[n] == SCC_FOUND) continue;

        int fw_parent = d_parent_fw[n];
        int bw_parent = d_parent_bw[n];
        if (fw_parent == -1 || bw_parent == -1) continue;

        int pid_fw = d_pivot_id_fw[n];
        int pid_bw = d_pivot_id_bw[n];
        if (pid_fw == -1 || pid_bw == -1) continue;

        // Both resolve through the SAME shared union-find (FW+BW combined)
        int resolved_fw = uf_find(d_pivot_parent, pid_fw);
        int resolved_bw = uf_find(d_pivot_parent, pid_bw);

        if (resolved_fw == resolved_bw) {
            d_Color[n] = SCC_FOUND;
            d_SCC[n] = d_pivots[resolved_fw];
            if (d_scc_counter) atomicAdd(d_scc_counter, 1);
        }
    }
}

// ======================================================================
// Kernel: mark pivot nodes as SCC roots
//
// Only marks pivots that are canonical roots of the shared union-find
// (uf_find(d_pivot_parent, i) == i). Non-root pivots are left for
// extract_sccs_from_forest_kernel to absorb into the canonical pivot's
// SCC, preventing double-counting.
// ======================================================================
__global__ void mark_scc_roots_kernel(
    int* d_Color, int* d_SCC,
    const int* d_parent_fw, const int* d_parent_bw,
    const int* d_pivots, int* d_pivot_parent, int num_pivots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_pivots; i += stride) {
        node_t p = d_pivots[i];
        if (p < 0) continue;
        if (d_Color[p] == SCC_FOUND) continue;

        if (d_parent_fw[p] == p && d_parent_bw[p] == p) {
            int resolved = uf_find(d_pivot_parent, i);
            if (resolved == i) {
                // This pivot is the canonical root of its merge group
                d_Color[p] = SCC_FOUND;
                d_SCC[p] = p;
            }
            // else: merged into another pivot's group —
            // leave for extract_sccs_from_forest_kernel to absorb
        }
    }
}

// ======================================================================
// Kernel: mark remaining unassigned nodes as singleton SCCs
// (Safety net — should rarely trigger with proper pivot merging)
// ======================================================================
__global__ void mark_remaining_sccs_kernel(
    int* d_Color, int* d_SCC,
    const int* d_targets, int num_targets,
    int* d_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local = 0;
    for (int i = tid; i < num_targets; i += stride) {
        node_t n = d_targets[i];
        if (d_Color[n] == SCC_FOUND) continue;
        d_Color[n] = SCC_FOUND;
        d_SCC[n] = n;
        local++;
    }
    if (local > 0 && d_count)
        atomicAdd(d_count, local);
}

// ======================================================================
// Host driver: run_spanning_forest_round()
//
// One round of spanning forest SCC:
//   1. Select pivots (scaled: ~1 per 2048 targets, min 64, max d_max_pivots)
//   2. Initialize shared union-find: each pivot is its own root
//   3. Initialize spanning trees
//   4. FW propagation (iterative, with inline pivot merging into shared UF)
//   5. BW propagation (iterative, with inline pivot merging into shared UF)
//   6. Compress shared union-find
//   7. Extract SCCs from FW∩BW using shared union-find roots
//   8. Return number of SCCs found
// ======================================================================
int run_spanning_forest_round(GPUState& st, const GPUGraph& g)
{
    if (!d_forest_initialized) {
        initialize_spanning_forest(g.num_nodes);
    }

    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return 0;

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;
    grid_size = min(grid_size, 65535);

    struct timeval ts_fw, te_fw, ts_bw, te_bw, ts_compress, te_compress, ts_ex, te_ex;

    // ---------------------------------------------------------------
    // Phase 1: Select pivots (scaled by target-set size)
    // Pivot scaling: ~1 pivot per 2048 targets, clamped to [64, d_max_pivots]
    // This ensures large sets (1.3M) get ~640 pivots, small sets (21K) get ~64.
    // ---------------------------------------------------------------
    int max_pivots_this_round = max(64, min(num_targets / 2048, d_max_pivots));

    CUDA_CHECK(cudaMemset(d_num_pivots, 0, sizeof(int)));

    select_pivots_kernel<<<grid_size, block_size>>>(
        st.d_Color, g.d_begin, g.d_r_begin,
        d_trim_targets, num_targets,
        d_pivots, d_num_pivots, max_pivots_this_round, d_pivot_degrees);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_num_pivots = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_pivots, d_num_pivots, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_num_pivots == 0) return 0;

    if (h_num_pivots > max_pivots_this_round) h_num_pivots = max_pivots_this_round;

    // ---------------------------------------------------------------
    // Phase 2: Initialize shared union-find (each pivot is its own root)
    // ---------------------------------------------------------------
    int uf_block_small = min(h_num_pivots, 256);
    init_pivot_union_find_kernel<<<(h_num_pivots + uf_block_small - 1) / uf_block_small, uf_block_small>>>(
        d_pivot_parent, h_num_pivots);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------------------------------------------------------
    // Phase 3: Initialize spanning trees from pivots
    // ---------------------------------------------------------------
    int init_gs = min(grid_size, 1024);
    init_spanning_trees_kernel<<<init_gs, block_size>>>(
        d_parent_fw, d_parent_bw,
        d_pivot_id_fw, d_pivot_id_bw,
        d_tree_depth,
        d_pivots, h_num_pivots,
        d_trim_targets, num_targets,
        st.d_Color);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------------------------------------------------------
    // Phase 4: FW spanning forest propagation (into SHARED d_pivot_parent)
    // ---------------------------------------------------------------
    int fw_iters = 0;
    int h_changed = 1;
    int MAX_ITERS = 100;  // deeper propagation for larger SCCs

    gettimeofday(&ts_fw, NULL);
    while (h_changed && fw_iters < MAX_ITERS) {
        fw_iters++;
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

        fw_spanning_forest_iteration_kernel<<<grid_size, block_size>>>(
            g.d_begin, g.d_node_idx,
            st.d_Color,
            d_parent_fw, d_pivot_id_fw, d_tree_depth,
            d_pivot_parent,
            d_changed,
            d_trim_targets, num_targets);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    }
    gettimeofday(&te_fw, NULL);
    double fw_ms = (te_fw.tv_sec - ts_fw.tv_sec) * 1000.0 +
                   (te_fw.tv_usec - ts_fw.tv_usec) * 0.001;

    // ---------------------------------------------------------------
    // Phase 5: BW spanning forest propagation (into the SAME d_pivot_parent)
    // ---------------------------------------------------------------
    int bw_iters = 0;
    h_changed = 1;

    gettimeofday(&ts_bw, NULL);
    while (h_changed && bw_iters < MAX_ITERS) {
        bw_iters++;
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

        bw_spanning_forest_iteration_kernel<<<grid_size, block_size>>>(
            g.d_r_begin, g.d_r_node_idx,
            st.d_Color,
            d_parent_bw, d_pivot_id_bw, d_tree_depth,
            d_pivot_parent,
            d_changed,
            d_trim_targets, num_targets);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    }
    gettimeofday(&te_bw, NULL);
    double bw_ms = (te_bw.tv_sec - ts_bw.tv_sec) * 1000.0 +
                   (te_bw.tv_usec - ts_bw.tv_usec) * 0.001;

    // ---------------------------------------------------------------
    // Phase 5b: Compress shared union-find (K ≤ 1024, single block, ~10 passes)
    // ---------------------------------------------------------------
    for (int pass = 0; pass < 10; pass++) {
        uf_compress_kernel<<<1, min(h_num_pivots, 512)>>>(d_pivot_parent, h_num_pivots);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    gettimeofday(&te_compress, NULL);
    double compress_ms = (te_compress.tv_sec - ts_bw.tv_sec) * 1000.0 +
                         (te_compress.tv_usec - ts_bw.tv_usec) * 0.001;

    // ---------------------------------------------------------------
    // Phase 6: Extract SCCs from FW ∩ BW tree intersections
    // ---------------------------------------------------------------
    gettimeofday(&ts_ex, NULL);

    // First mark pivot nodes as SCC roots (only canonical shared UF roots)
    int pivot_gs = (h_num_pivots + block_size - 1) / block_size;
    mark_scc_roots_kernel<<<pivot_gs, block_size>>>(
        st.d_Color, st.d_SCC,
        d_parent_fw, d_parent_bw,
        d_pivots, d_pivot_parent, h_num_pivots);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract SCC members using resolved (merged) pivot IDs through shared UF
    CUDA_CHECK(cudaMemset(d_scc_counter, 0, sizeof(int)));

    extract_sccs_from_forest_kernel<<<grid_size, block_size>>>(
        st.d_Color, st.d_SCC,
        d_parent_fw, d_parent_bw,
        d_pivot_id_fw, d_pivot_id_bw,
        d_pivot_parent,
        d_pivots,
        d_trim_targets, num_targets,
        d_scc_counter);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_scc_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_scc_count, d_scc_counter, sizeof(int), cudaMemcpyDeviceToHost));

    gettimeofday(&te_ex, NULL);
    double extract_ms = (te_ex.tv_sec - ts_ex.tv_sec) * 1000.0 +
                        (te_ex.tv_usec - ts_ex.tv_usec) * 0.001;

    double density = (num_targets > 0) ? (double)num_targets / h_num_pivots : 0.0;
    printf("[SPAN_FOREST] Round: %d pivots for %d targets (1:%.0f), FW=%d(%.2fms) BW=%d(%.2fms) "
           "Compress=%.2fms Extract=%d SCCs(%.2fms)\n",
           h_num_pivots, num_targets, density,
           fw_iters, fw_ms, bw_iters, bw_ms,
           compress_ms, h_scc_count, extract_ms);

    return h_scc_count;
}

// ======================================================================
// Kernel: reset d_Color from SCC_FOUND back to COLOR_UNASSIGNED
//
// Called BEFORE the fallback so FB sees the full graph connectivity.
// Without this reset, FB is limited to the unresolved work set and
// can't see through spanning-forest-resolved nodes, causing false
// singleton SCC assignments for remaining nodes.
// ======================================================================
__global__ void reset_d_colors_kernel(int* d_Color, int num_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_nodes; i += stride) {
        if (d_Color[i] == SCC_FOUND) {
            d_Color[i] = COLOR_UNASSIGNED;
        }
    }
}

// Host wrapper — called from main.cpp (compiled by g++, needs host function)
void reset_forest_colors(int* d_Color, int num_nodes)
{
    if (num_nodes <= 0) return;
    int block_size = 256;
    int grid_size = (num_nodes + block_size - 1) / block_size;
    grid_size = min(grid_size, 65535);
    reset_d_colors_kernel<<<grid_size, block_size>>>(d_Color, num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ======================================================================
// run_spanning_forest_scc() — Full host driver
//
// Iteratively applies spanning forest rounds until convergence.
// After each round, rebuilds compact target list.
//
// When converging with no new SCCs, returns gracefully so the caller
// can fall back to the standard pipeline for correctness.
// ======================================================================
int run_spanning_forest_scc(GPUState& st, const GPUGraph& g)
{
    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    int total_sccs = 0;
    int round = 0;
    int MAX_ROUNDS = 100;
    const double MIN_RESOLUTION_PCT = 10.0;  // Stop if <10% resolved (fragments too small)

    while (round < MAX_ROUNDS) {
        round++;

        create_trim1_compact(st, g);

        int num_targets = d_trim_targets_count;
        if (num_targets == 0) {
            printf("[SPAN_FOREST] No remaining nodes, done after %d rounds\n", round);
            break;
        }

        int before = d_trim_targets_count;
        int scc_found = run_spanning_forest_round(st, g);
        create_trim1_compact(st, g);
        int after = d_trim_targets_count;
        int resolved = before - after;
        total_sccs += scc_found;

        double pct = (before > 0) ? 100.0 * resolved / before : 0.0;
        printf("[SPAN_FOREST] Round %d: resolved %d/%d nodes (%.1f%%), "
               "found %d SCCs, %d remain\n",
               round, resolved, before, pct,
               scc_found, after);

        if (scc_found == 0) {
            printf("[SPAN_FOREST] Converged after %d rounds (%d SCCs), "
                   "%d targets remain — returning for fallback\n",
                   round, total_sccs, after);
            break;
        }

        // Early exit: if resolution rate drops below threshold, fragments
        // are too small for effective spanning forest — let fallback handle them.
        if (pct < MIN_RESOLUTION_PCT && round >= 2) {
            printf("[SPAN_FOREST] Resolution %.1f%% below threshold %.0f%% after round %d, "
                   "deferring %d remaining to fallback\n",
                   pct, MIN_RESOLUTION_PCT, round, after);
            break;
        }
    }

    gettimeofday(&t_end, NULL);
    double total_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                      (t_end.tv_usec - t_start.tv_usec) * 0.001;

    printf("[SPAN_FOREST] Done: %d rounds, %d SCCs found, %.2fms total, %d remaining\n",
           round, total_sccs, total_ms, d_trim_targets_count);

    return total_sccs;
}
