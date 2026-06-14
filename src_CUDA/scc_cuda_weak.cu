#include "scc_cuda.h"

// ======================================================================
// Device-side global state: analogs of OpenMP static globals
// ======================================================================

// OpenMP:
//   static node_t* G_WCC;
//   static NODE_SET** wcc_sets;
//   static node_t* temp_buf;
//   static int* temp_buf_ptr;

// CUDA: WCC root array (label propagation on device)
static int* d_WCC = NULL;  // [N] WCC root for each node
static int  d_WCC_num_nodes = 0;

// OpenMP: static NODE_SET** wcc_sets; (host-side sets for work creation)
// CUDA: host-side set pointers (mirrored for structure)
static int** h_wcc_sets = NULL;      // [N] array of host-set pointers (each is device node list)
static int*  h_wcc_set_sizes = NULL; // [N] size of each set

// OpenMP: static NODE_SET** wcc_sets — device-side pointer mirror
// CUDA: device-side array of pointers to per-root device buffers
static int** d_wcc_sets = NULL;      // [N] device-side pointer array (mirror of h_wcc_sets)

static int*  d_temp_buf = NULL;      // [max_sets * avg_size] device temp buffer
static int   d_temp_buf_capacity = 0;

// Big buffer for WCC per-root sets (single allocation, sliced by pointer arithmetic)
static int*  d_wcc_big_buffer = NULL;
static int   d_wcc_big_buffer_size = 0;

// OpenMP: static int pool_cnt=0;
//         static NODE_SET** node_set_pool;
// CUDA: pool of pre-allocated device buffers for work sets
static int   pool_cnt = 0;
static int** h_node_set_pool = NULL;  // host array of device pointers
static int*  d_node_set_pool = NULL;  // [pool_size] device-side list handles (not used directly)
static int   node_set_pool_size = 0;

// ======================================================================
// check_WCC()
// OpenMP:
//   void check_WCC() {
//       printf("wcc_sts = %p\n", wcc_sets);
//   }
// ======================================================================
void check_WCC() {
    printf("wcc_sts = %p\n", (void*)h_wcc_sets);
}

// ======================================================================
// get_WCC()
// OpenMP:
//   node_t* get_WCC() {return G_WCC;}
// ======================================================================
int* get_WCC() { return d_WCC; }

// ======================================================================
// init_node_set_pool()
// OpenMP:
//   void init_node_set_pool(int sz) {
//       pool_cnt = sz;
//       node_set_pool = new NODE_SET*[sz];
//       for(int i =0;i<sz;i++)
//           node_set_pool[i] = new NODE_SET();
//   }
// ======================================================================
void init_node_set_pool(int sz) {
    // CUDA: Pool is kept for bookkeeping but per-root buffers are allocated
    //       directly in create_work_items_from_wcc based on actual member counts.
    //       No GPU memory is pre-allocated here (unlike OpenMP which uses NODE_SET
    //       from the pool directly). Allocating 65536 * N device buffers would
    //       consume ~26GB for N=100K — exceeding GPU memory limits.
    pool_cnt = sz;
    node_set_pool_size = sz;
    h_node_set_pool = new int*[sz];
    for (int i = 0; i < sz; i++) {
        h_node_set_pool[i] = NULL;
    }
}

// ======================================================================
// get_node_set_from_pool()
// OpenMP:
//   NODE_SET* get_node_set_from_pool() {
//       int index = __sync_add_and_fetch(&pool_cnt, -1);
//       return node_set_pool[index];
//       assert(pool_cnt > 0);
//   }
// ======================================================================
int* get_node_set_from_pool() {
    int index = __sync_add_and_fetch(&pool_cnt, -1);
    assert(index >= 0);
    return h_node_set_pool[index];
}

// ======================================================================
// initialize_WCC()
// OpenMP:
//   void initialize_WCC() {
//       G_WCC = new node_t[G_num_nodes];
//       wcc_sets = new NODE_SET*[G_num_nodes];
//       assert(wcc_sets != NULL);
//       init_node_set_pool(65536);
//
//       #pragma omp parallel for
//       for (node_t t4 = 0; t4 < G_num_nodes; t4 ++)
//       {
//           G_WCC[t4] = gm_graph::NIL_NODE;
//           wcc_sets[t4] = NULL;
//       }
//
//       int num_threads = gm_rt_get_num_threads();
//       temp_buf = new node_t [num_threads * 2*1024*1024];
//       temp_buf_ptr = new int [num_threads * 32];
//       for(int i=0;i<num_threads;i++)
//           temp_buf_ptr[i*32] = 0;
//   }
// ======================================================================
void initialize_WCC(int num_nodes)
{
    d_WCC_num_nodes = num_nodes;
    CUDA_CHECK(cudaMalloc(&d_WCC, num_nodes * sizeof(int)));
    // Initialize to CUDA_NIL_NODE (-1) — mirror of gm_graph::NIL_NODE
    CUDA_CHECK(cudaMemset(d_WCC, 0xFF, num_nodes * sizeof(int)));

    // OpenMP: wcc_sets = new NODE_SET*[G_num_nodes];
    //         for(t4) wcc_sets[t4] = NULL;
    h_wcc_sets = new int*[num_nodes];
    CUDA_CHECK(cudaMalloc(&d_wcc_sets, num_nodes * sizeof(int*)));
    CUDA_CHECK(cudaMemset(d_wcc_sets, 0, num_nodes * sizeof(int*)));
    h_wcc_set_sizes = new int[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
        h_wcc_sets[i] = NULL;
        h_wcc_set_sizes[i] = 0;
    }

    // OpenMP: init_node_set_pool(65536);
    init_node_set_pool(65536);

    // OpenMP: temp_buf = new node_t [num_threads * 2*1024*1024];
    //         temp_buf_ptr = new int [num_threads * 32];
    d_temp_buf_capacity = 2 * 1024 * 1024;  // simplified: single buffer instead of per-thread
    CUDA_CHECK(cudaMalloc(&d_temp_buf, d_temp_buf_capacity * sizeof(int)));
}

void finalize_WCC()
{
    int saved_num_nodes = d_WCC_num_nodes;
    if (d_WCC) { cudaFree(d_WCC); d_WCC = NULL; }
    d_WCC_num_nodes = 0;

    // Free the single big buffer (replaces thousands of per-root cudaFree calls)
    if (d_wcc_big_buffer) { cudaFree(d_wcc_big_buffer); d_wcc_big_buffer = NULL; }
    d_wcc_big_buffer_size = 0;

    // OpenMP: delete [] wcc_sets;
    if (h_wcc_sets) {
        // Individual h_wcc_sets[i] are slices of d_wcc_big_buffer — do NOT free them.
        // The big buffer is freed above as a single cudaFree call.
        delete[] h_wcc_sets;
        h_wcc_sets = NULL;
    }
    if (d_wcc_sets) { cudaFree(d_wcc_sets); d_wcc_sets = NULL; }
    if (h_wcc_set_sizes) { delete[] h_wcc_set_sizes; h_wcc_set_sizes = NULL; }

    // OpenMP: free node_set_pool
    if (h_node_set_pool) {
        for (int i = 0; i < node_set_pool_size; i++) {
            if (h_node_set_pool[i]) cudaFree(h_node_set_pool[i]);
        }
        delete[] h_node_set_pool;
        h_node_set_pool = NULL;
    }
    pool_cnt = 0;
    node_set_pool_size = 0;

    if (d_temp_buf) { cudaFree(d_temp_buf); d_temp_buf = NULL; }
    d_temp_buf_capacity = 0;
}

// ======================================================================
// OpenMP macros (mirrored as device macros)
// ======================================================================
// OpenMP:
//   #define INIT_WCC_ROOT_LG(t) ((t) | (0x20000000))
//   #define GET_WCC_ROOT(K)     (G_WCC[K]&0x1FFFFFFF)
#define CUDA_INIT_WCC_ROOT_LG(t)  ((t) | (0x20000000))
#define CUDA_GET_WCC_ROOT(K)      ((K) & 0x1FFFFFFF)

// ======================================================================
// WCC initialization kernel
// OpenMP:
//   #pragma omp parallel for
//   for (node_t idx = 0; idx < wcc_candidate.size(); idx++)
//   {
//       node_t t4 = wcc_candidate[idx];
//       if (G_Color[t4] != -2) {
//           assert(t4 < 0x1FFFFFFF);
//           if (G.begin[t4+1] - G.begin[t4] >= 50) {
//               G_WCC[t4] = INIT_WCC_ROOT_LG(t4);
//           }
//           else {
//               G_WCC[t4] = t4;
//           }
//       }
//   }
// ======================================================================
__global__ void wcc_init_kernel(
    int* d_WCC,
    const int* d_Color,
    const edge_t* d_begin,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t4 = d_targets[i];
        if (d_Color[t4] == SCC_FOUND) continue;  // OpenMP: if (G_Color[t4] == -2) continue;

        // OpenMP: assert(t4 < 0x1FFFFFFF);
        // OpenMP: if (G.begin[t4+1] - G.begin[t4] >= 50)
        //             G_WCC[t4] = INIT_WCC_ROOT_LG(t4);
        //         else
        //             G_WCC[t4] = t4;
        if (d_begin[t4 + 1] - d_begin[t4] >= 50) {
            d_WCC[t4] = CUDA_INIT_WCC_ROOT_LG(t4);
        } else {
            d_WCC[t4] = t4;
        }
    }
}

// ======================================================================
// propagate_color() — Phase 1: scan neighbors, find minimum WCC root
//
// OpenMP (propagate_color, Phase 1):
//   #pragma omp parallel for schedule(dynamic, 32)
//   for (int index = 0; index < wcc_candidate.size(); index++)
//   {
//       node_t n = wcc_candidate[index];
//       node_t min_val = G_WCC[n];
//       if (G_Color[n] == -2) continue;
//       for (edge_t k_idx = G.begin[n]; k_idx < G.begin[n+1]; k_idx++)
//       {
//           node_t k = G.node_idx[k_idx];
//           if (G_Color[k] != G_Color[n]) continue;
//           if (G_WCC[k] < min_val) {
//               min_val = G_WCC[k];
//               if (finished) finished = false;
//           }
//       }
//       if (min_val != G_WCC[n]) G_WCC[n] = min_val;
//   }
// ======================================================================
__global__ void wcc_propagate_phase1_kernel(
    int* d_WCC, int* d_changed,
    const int* d_Color,
    const edge_t* d_begin, const node_t* d_node_idx,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid; idx < num_targets; idx += stride) {
        node_t n = d_targets[idx];
        int color_n = d_Color[n];
        if (color_n == SCC_FOUND) continue;  // OpenMP: if (G_Color[n] == -2) continue;

        // OpenMP: node_t min_val = G_WCC[n];
        int min_val = d_WCC[n];

        // OpenMP: for each forward neighbor
        for (edge_t k_idx = d_begin[n]; k_idx < d_begin[n + 1]; k_idx++) {
            node_t k = d_node_idx[k_idx];
            // OpenMP: if (G_Color[k] != G_Color[n]) continue;
            if (d_Color[k] != color_n) continue;
            // OpenMP: if (G_WCC[k] < min_val) { min_val = G_WCC[k]; finished = false; }
            int wcc_k = d_WCC[k];
            if (wcc_k < min_val) {
                min_val = wcc_k;
            }
        }

        // OpenMP: if (min_val != G_WCC[n]) G_WCC[n] = min_val;
        if (min_val != d_WCC[n]) {
            d_WCC[n] = min_val;
            *d_changed = 1;  // OpenMP: finished = false
        }
    }
}

// ======================================================================
// propagate_color() — Phase 2: path compression
//
// OpenMP (propagate_color, Phase 2):
//   #pragma omp parallel for
//   for (int index = 0; index < wcc_candidate.size(); index++)
//   {
//       node_t n = wcc_candidate[index];
//       if (G_Color[n] == -2) continue;
//       if (GET_WCC_ROOT(n) != n)
//       {
//           node_t root = GET_WCC_ROOT(n);
//           if (GET_WCC_ROOT(root) != root) {
//               G_WCC[n] = G_WCC[root];
//               finished = false;
//           }
//       }
//   }
// ======================================================================
__global__ void wcc_propagate_phase2_kernel(
    int* d_WCC, int* d_changed,
    const int* d_Color,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid; idx < num_targets; idx += stride) {
        node_t n = d_targets[idx];
        int color_n = d_Color[n];
        if (color_n == SCC_FOUND) continue;  // OpenMP: if (G_Color[n] == -2) continue;

        // OpenMP: GET_WCC_ROOT(n) — mask off the large-degree flag bit
        int wcc_n    = d_WCC[n];
        int root_n   = CUDA_GET_WCC_ROOT(wcc_n);

        // OpenMP: if (GET_WCC_ROOT(n) != n)
        if (root_n != n) {
            // OpenMP: node_t root = GET_WCC_ROOT(n);
            node_t root = root_n;
            // OpenMP: if (GET_WCC_ROOT(root) != root)
            int wcc_root  = d_WCC[root];
            int root_root = CUDA_GET_WCC_ROOT(wcc_root);
            if (root_root != root) {
                // OpenMP: G_WCC[n] = G_WCC[root]; (unmasked value)
                d_WCC[n] = wcc_root;
                *d_changed = 1;  // OpenMP: finished = false
            }
        }
    }
}

// ======================================================================
// do_global_wcc() — Phase: assign colors to WCC roots
//
// OpenMP (do_global_wcc):
//   // Root color assignment
//   #pragma omp parallel for
//   for (int index = 0; index < wcc_candidate.size(); index++)
//   {
//       node_t t4 = wcc_candidate[index];
//       if (G_Color[t4] == -2) continue;
//       node_t root = GET_WCC_ROOT(t4);
//       if (t4 == root) {
//           G_Color[t4] = get_new_color();
//       }
//   }
//
//   // Propagate colors to members
//   #pragma omp parallel for
//   for (int index = 0; index < wcc_candidate.size(); index++)
//   {
//       node_t t4 = wcc_candidate[index];
//       if (G_Color[t4] == -2) continue;
//       node_t root = GET_WCC_ROOT(t4);
//       if (t4!=root)
//           G_Color[t4] = G_Color[root];
//   }
// ======================================================================
__global__ void wcc_assign_root_colors_kernel(
    int* d_Color, int* d_WCC,
    const int* d_targets, int num_targets,
    int* d_color_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t4 = d_targets[i];
        if (d_Color[t4] == SCC_FOUND) continue;

        // OpenMP: node_t root = GET_WCC_ROOT(t4);
        node_t root = CUDA_GET_WCC_ROOT(d_WCC[t4]);

        // OpenMP: if (t4 == root) G_Color[t4] = get_new_color();
        if (t4 == root) {
            int new_color = atomicAdd(d_color_counter, 1) + 1;
            d_Color[t4] = new_color;
        }
    }
}

__global__ void wcc_propagate_colors_kernel(
    int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t4 = d_targets[i];
        if (d_Color[t4] == SCC_FOUND) continue;

        // OpenMP: node_t root = GET_WCC_ROOT(t4);
        node_t root = CUDA_GET_WCC_ROOT(d_WCC[t4]);

        // OpenMP: if (t4 != root) G_Color[t4] = G_Color[root];
        if (t4 != root) {
            d_Color[t4] = d_Color[root];
        }
    }
}

// ======================================================================
// wcc_insert_members_kernel() — insert members into per-root device buffers
// OpenMP (Pass 2): wcc_sets[root]->insert(t4) with per-node spinlock
// CUDA: atomic position per root, direct write into root's device buffer
// ======================================================================
__global__ void wcc_insert_members_kernel(
    const int* d_Color, const int* d_WCC,
    const int* d_targets, int num_targets,
    int* d_root_pos,      // [N] per-root atomic position counters
    int** d_wcc_sets_dev)  // [N] device-side array of pointers to per-root buffers
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_targets; i += stride) {
        node_t t = d_targets[i];
        if (d_Color[t] == SCC_FOUND) continue;  // OpenMP: if (G_Color[t] == -2) continue;

        int root = d_WCC[t] & 0x1FFFFFFF;  // GET_WCC_ROOT

        // OpenMP: if (root == gm_graph::NIL_NODE) continue;
        if (root < 0) continue;

        // OpenMP: gm_spinlock_acquire_for_node(root);
        //         wcc_sets[root]->insert(t4);
        //         gm_spinlock_release_for_node(root);
        // CUDA: atomic position + direct write (no spinlock needed)
        int* root_buf = d_wcc_sets_dev[root];
        if (root_buf != NULL) {
            int pos = atomicAdd(&d_root_pos[root], 1);
            root_buf[pos] = t;
        }
    }
}

// ======================================================================
// propagate_color()
// OpenMP:
//   void propagate_color(gm_graph& G, std::vector<node_t>& wcc_candidate)
//   {
//       bool finished;
//       do {
//           finished = true;
//
//           #pragma omp parallel for schedule(dynamic, 32)
//           for (int index = 0; index < wcc_candidate.size(); index++)
//               // Phase 1: scan neighbors, find min root
//
//           #pragma omp parallel for
//           for (int index = 0; index < wcc_candidate.size(); index++)
//               // Phase 2: path compression
//
//       } while (!finished);
//   }
// ======================================================================
void propagate_color(GPUState& st, const GPUGraph& g, int num_targets,
    int& cuda_iters)
{
    if (num_targets == 0) return;

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;
    grid_size = min(grid_size, 1024);

    int* d_changed = NULL;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    int max_iterations = 100;  // safety limit
    int iter = 0;
    int h_changed = 1;

    // OpenMP: do { ... } while (!finished);
    while (h_changed && iter < max_iterations) {
        iter++;
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

        // OpenMP: Phase 1 — scan neighbors, find min root
        wcc_propagate_phase1_kernel<<<grid_size, block_size>>>(
            d_WCC, d_changed,
            st.d_Color,
            g.d_begin, g.d_node_idx,
            d_trim_targets, num_targets);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                               cudaMemcpyDeviceToHost));

        if (h_changed) {
            CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

            // OpenMP: Phase 2 — path compression
            wcc_propagate_phase2_kernel<<<grid_size, block_size>>>(
                d_WCC, d_changed,
                st.d_Color,
                d_trim_targets, num_targets);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                                   cudaMemcpyDeviceToHost));
        }
    }
    cuda_iters = iter;
    CUDA_CHECK(cudaFree(d_changed));
    if (iter >= max_iterations) {
        printf("[CUDA WCC] Warning: max iterations reached (%d)\n", max_iterations);
    }
}

// ======================================================================
// do_global_wcc()
// OpenMP:
//   void do_global_wcc(gm_graph& G)
//   {
//       std::vector<node_t>& wcc_candidate = get_compact_trim_targets();
//
//       #pragma omp parallel for
//       for — init WCC roots
//
//       propagate_color(G, wcc_candidate);
//
//       #pragma omp parallel for
//       for — assign colors to roots
//
//       #pragma omp parallel for
//       for — propagate colors to members
//   }
// ======================================================================
void do_global_wcc(GPUState& st, const GPUGraph& g)
{
    int num_targets = d_trim_targets_count;
    if (num_targets == 0) {
        printf("[CUDA WCC] No targets\n");
        return;
    }

    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;
    grid_size = min(grid_size, 1024);

    // ---------------------------------------------------------------
    // Initialize WCC roots
    // OpenMP: #pragma omp parallel for — init with degree-aware roots
    // ---------------------------------------------------------------
    wcc_init_kernel<<<grid_size, block_size>>>(
        d_WCC, st.d_Color, g.d_begin,
        d_trim_targets, num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------------------------------------------------------
    // Label propagation
    // OpenMP: propagate_color(G, wcc_candidate);
    // ---------------------------------------------------------------
    int cuda_iters = 0;
    propagate_color(st, g, num_targets, cuda_iters);

    // ---------------------------------------------------------------
    // Assign colors to WCC roots using the SHARED color counter
    // OpenMP: G_Color[t4] = get_new_color();
    //
    // FIX: Use host-side cuda_get_new_color() to share the SAME color
    // counter with FW-BW phases. The old code used a separate device
    // counter d_color_counter, which re-issued colors 1,2,3,... that
    // had already been used by Global FW-BW — causing DFS to traverse
    // into unrelated nodes (out-of-bounds memory access).
    // ---------------------------------------------------------------
    int* h_WCC = new int[g.num_nodes];
    CUDA_CHECK(cudaMemcpy(h_WCC, d_WCC, g.num_nodes * sizeof(int),
                           cudaMemcpyDeviceToHost));

    int num_roots = 0;
    for (node_t n = 0; n < g.num_nodes; n++) {
        if (h_WCC[n] == -1) continue;                         // NIL_NODE: not in any component
        node_t root = CUDA_GET_WCC_ROOT(h_WCC[n]);
        if (root == n) {
            // OpenMP: G_Color[t4] = get_new_color();
            int new_color = cuda_get_new_color();
            CUDA_CHECK(cudaMemcpy(&st.d_Color[n], &new_color, sizeof(int),
                                   cudaMemcpyHostToDevice));
            num_roots++;
        }
    }
    delete[] h_WCC;

    // OpenMP: #pragma omp parallel for — propagate colors to members
    wcc_propagate_colors_kernel<<<grid_size, block_size>>>(
        st.d_Color, d_WCC,
        d_trim_targets, num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[CUDA WCC] %d components, %d iterations\n", num_roots, cuda_iters);
}

// ======================================================================
// create_work_items_from_wcc()
// OpenMP:
//   void create_work_items_from_wcc(gm_graph& G)
//   {
//       // Phase 1: allocate sets for roots
//       #pragma omp parallel for
//       for — wcc_sets[root] = get_node_set_from_pool();
//
//       // Phase 2: insert members into root sets
//       #pragma omp parallel for schedule(dynamic, 32)
//       for — wcc_sets[root]->insert(t4);
//
//       // Phase 3: create work items
//       #pragma omp parallel
//       {
//           std::vector<my_work*> small_works;
//           #pragma omp for nowait schedule(dynamic,32)
//           for — create my_work for each root set
//           work_q_put_all(tid, small_works);
//       }
//   }
// ======================================================================
void create_work_items_from_wcc(GPUState& st, const GPUGraph& g)
{
    int num_targets = d_trim_targets_count;
    if (num_targets == 0) return;

    int N = g.num_nodes;
    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;
    grid_size = min(grid_size, 1024);

    // ---------------------------------------------------------------
    // Pass 1: count members per root, allocate per-root device buffers
    // OpenMP:
    //   #pragma omp parallel for
    //   for (int index = 0; index < wcc_candidate.size(); index++) {
    //       node_t t4 = wcc_candidate[index];
    //       if (G_Color[t4] == -2) continue;
    //       node_t root = GET_WCC_ROOT(t4);
    //       if (root == t4) {
    //           wcc_sets[t4] = get_node_set_from_pool();
    //       }
    //   }
    // ---------------------------------------------------------------
    int* d_root_counts = NULL;
    CUDA_CHECK(cudaMalloc(&d_root_counts, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_root_counts, 0, N * sizeof(int)));

    count_by_wcc_root_kernel<<<grid_size, block_size>>>(
        st.d_Color, get_WCC(),
        d_trim_targets, num_targets,
        d_root_counts);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy root counts to host to know how many members per root
    int* h_root_counts = new int[N];
    CUDA_CHECK(cudaMemcpy(h_root_counts, d_root_counts,
                           N * sizeof(int), cudaMemcpyDeviceToHost));


    // Build root list and compute prefix sums
    std::vector<int> root_list;
    std::vector<int> offsets;  // prefix sum of root counts
    offsets.push_back(0);
    for (int i = 0; i < N; i++) {
        if (h_root_counts[i] > 0) {
            root_list.push_back(i);
            offsets.push_back(offsets.back() + h_root_counts[i]);
        }
    }

    if (root_list.empty()) {
        delete[] h_root_counts;
        CUDA_CHECK(cudaFree(d_root_counts));
        return;
    }

    // Single large allocation instead of thousands of per-root cudaMalloc calls
    int total_members = offsets.back();
    if (d_wcc_big_buffer) cudaFree(d_wcc_big_buffer);  // guard against re-entrant calls
    CUDA_CHECK(cudaMalloc(&d_wcc_big_buffer, total_members * sizeof(int)));
    d_wcc_big_buffer_size = total_members;

    // Assign slices to each root via pointer arithmetic
    for (size_t ri = 0; ri < root_list.size(); ri++) {
        node_t root = root_list[ri];
        h_wcc_sets[root] = d_wcc_big_buffer + offsets[ri];
    }

    // ---------------------------------------------------------------
    // Pass 2: insert members into per-root buffers
    // OpenMP:
    //   #pragma omp parallel for schedule(dynamic, 32)
    //   for (int index = 0; index < wcc_candidate.size(); index++) {
    //       node_t t4 = wcc_candidate[index];
    //       if (G_Color[t4] == -2) continue;
    //       node_t root = GET_WCC_ROOT(t4);
    //       if (root == gm_graph::NIL_NODE) continue;
    //       gm_spinlock_acquire_for_node(root);
    //       wcc_sets[root]->insert(t4);
    //       gm_spinlock_release_for_node(root);
    //   }
    // ---------------------------------------------------------------
    // CUDA: use per-root position counters and scatter directly into per-root buffers
    // First, zero out the root counts (reuse as position counters)
    CUDA_CHECK(cudaMemset(d_root_counts, 0, N * sizeof(int)));

    // Sync h_wcc_sets → d_wcc_sets (device-side pointer mirror)
    // OpenMP: wcc_sets[root] already populated from Pass 1
    CUDA_CHECK(cudaMemcpy(d_wcc_sets, h_wcc_sets, N * sizeof(int*),
                           cudaMemcpyHostToDevice));

    // Kernel: for each target, find its root and atomically insert into root's buffer
    // (This replaces per-node spinlock + unordered_set::insert)
    wcc_insert_members_kernel<<<grid_size, block_size>>>(
        st.d_Color, get_WCC(),
        d_trim_targets, num_targets,
        d_root_counts,   // per-root position counters
        d_wcc_sets);     // device-side array of pointers to per-root buffers
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_root_counts));

    // ---------------------------------------------------------------
    // Pass 3: create CUDAMyWork for each WCC root and push to queue
    // OpenMP:
    //   #pragma omp parallel
    //   {
    //       std::vector<my_work*> small_works;
    //       #pragma omp for nowait schedule(dynamic,32)
    //       for (int index = 0; index < wcc_candidate.size(); index++) {
    //           node_t i = wcc_candidate[index];
    //           if (G_Color[i] == -2) continue;
    //           node_t root = GET_WCC_ROOT(i);
    //           if (root == i) {
    //               my_work* w1 = new my_work();
    //               w1->color = G_Color[i];
    //               w1->count = wcc_sets[i]->size();
    //               w1->color_set = wcc_sets[i];
    //               small_works.push_back(w1);
    //           }
    //       }
    //       work_q_put_all(tid, small_works);
    //   }
    // ---------------------------------------------------------------
    std::vector<CUDAMyWork*> new_works;
    new_works.reserve(root_list.size());

    // Batch-read entire d_Color array (one D2H instead of N per-root copies)
    int* h_Color = new int[N];
    CUDA_CHECK(cudaMemcpy(h_Color, st.d_Color,
                           N * sizeof(int), cudaMemcpyDeviceToHost));

    for (size_t ri = 0; ri < root_list.size(); ri++) {
        node_t root = root_list[ri];
        int root_count = h_root_counts[root];

        // OpenMP: w1->color = G_Color[i];
        int root_color = h_Color[root];  // host read — zero device calls

        // OpenMP: my_work* w1 = new my_work();
        CUDAMyWork* w1 = new CUDAMyWork();
        w1->color = root_color;
        w1->count = root_count;
        // OpenMP: w1->color_set = wcc_sets[i];
        w1->d_set_nodes = h_wcc_sets[root];  // slice of big buffer
        w1->set_capacity = root_count;
        w1->owns_set = 0;  // CUDA: no per-root cudaFree — part of big buffer
        w1->depth = 0;
        new_works.push_back(w1);
    }

    delete[] h_Color;

    // Clear h_wcc_sets entries (ownership transferred to CUDAMyWork)
    for (size_t ri = 0; ri < root_list.size(); ri++) {
        h_wcc_sets[root_list[ri]] = NULL;
    }

    delete[] h_root_counts;

    // OpenMP: work_q_put_all(tid, small_works);
    if (!new_works.empty()) {
        work_q_put_all(0, new_works);
        new_works.clear();
    }
}
