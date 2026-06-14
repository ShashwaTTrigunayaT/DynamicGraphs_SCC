# Revert fw_bfs_level_kernel and bw_bfs_level_kernel to simple version
# No stride loops, no local buffers — just one thread per queue element

code = open('scc_cuda_fb_global.cu').read()

# --- Revert FW kernel ---
old_fw = '''__global__ void fw_bfs_level_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int base_color)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < queue_size; i += stride) {
        node_t t = d_queue[i];

        // Stage claimed nodes locally before flushing to global queue
        int local_claimed[128];
        int local_cnt = 0;

        for (edge_t nx = d_begin[t]; nx < d_begin[t + 1]; nx++) {
            node_t k = d_node_idx[nx];

            // OpenMP: if (check_navigator(u, nx))
            if (fw_check_navigator_device(d_Color, k, base_color)) {
                // OpenMP: visit_fw(k) { G_Color[k] = fw_color; ... }
                int old = atomicCAS(&d_Color[k], base_color, fw_color);
                if (old == base_color) {
                    local_claimed[local_cnt++] = k;
                    // Flush batch when local buffer is full
                    if (local_cnt == 128) {
                        int base = atomicAdd(d_next_count, 128);
                        for (int j = 0; j < 128; j++)
                            d_next_queue[base + j] = local_claimed[j];
                        local_cnt = 0;
                    }
                }
            }
        }

        // Flush remaining claimed nodes
        if (local_cnt > 0) {
            int base = atomicAdd(d_next_count, local_cnt);
            for (int j = 0; j < local_cnt; j++)
                d_next_queue[base + j] = local_claimed[j];
        }
    }
}'''

new_fw = '''__global__ void fw_bfs_level_kernel(
    const edge_t* d_begin, const node_t* d_node_idx,
    int* d_Color,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int base_color)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= queue_size) return;
    node_t t = d_queue[i];
    for (edge_t nx = d_begin[t]; nx < d_begin[t + 1]; nx++) {
        node_t k = d_node_idx[nx];
        if (fw_check_navigator_device(d_Color, k, base_color)) {
            int old = atomicCAS(&d_Color[k], base_color, fw_color);
            if (old == base_color) {
                int pos = atomicAdd(d_next_count, 1);
                d_next_queue[pos] = k;
            }
        }
    }
}'''

assert old_fw in code, 'FW kernel pattern not found!'
code = code.replace(old_fw, new_fw)
print('FW kernel reverted ✅')

# --- Revert BW kernel ---
old_bw = '''__global__ void bw_bfs_level_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int bw_color, int base_color, node_t pivot,
    int* d_scc_count, int* d_bw_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < queue_size; i += stride) {
        node_t t = d_queue[i];  // OpenMP: t from frontier

        // Stage claimed nodes locally before flushing
        int local_scc[128], local_bw[128];
        int scc_cnt = 0, bw_cnt = 0;

        for (edge_t nx = d_r_begin[t]; nx < d_r_begin[t + 1]; nx++) {
            node_t k = d_r_node_idx[nx];

            // OpenMP: if (check_navigator(k, nx))
            //         return (color == fw_color) || (color == base_color);
            //
            // Read color ONCE to avoid TOCTOU between navigator and visit_fw.
            int k_color = d_Color[k];
            if (k_color == fw_color) {
                // OpenMP: visit_fw(k) — intersection
                //   G_SCC[k] = pivot; G_Color[k] = -2;
                int old = atomicCAS(&d_Color[k], fw_color, SCC_FOUND);
                if (old == fw_color) {
                    d_SCC[k] = pivot;
                    local_scc[scc_cnt++] = k;
                    // Flush when local buffer is full
                    if (scc_cnt == 128) {
                        atomicAdd(d_scc_count, 128);   // SCC count
                        int base_nxt = atomicAdd(d_next_count, 128);
                        for (int j = 0; j < 128; j++)
                            d_next_queue[base_nxt + j] = local_scc[j];
                        scc_cnt = 0;
                    }
                }
            } else if (k_color == base_color) {
                // OpenMP: visit_fw(k) — bw-set
                int old = atomicCAS(&d_Color[k], base_color, bw_color);
                if (old == base_color) {
                    local_bw[bw_cnt++] = k;
                    if (bw_cnt == 128) {
                        // Flush BW batch
                        int base_next = atomicAdd(d_next_count, 128);
                        atomicAdd(d_bw_count, 128);     // BW count
                        for (int j = 0; j < 128; j++)
                            d_next_queue[base_next + j] = local_bw[j];
                        bw_cnt = 0;
                    }
                }
            }
        }

        // Flush remaining SCC nodes
        if (scc_cnt > 0) {
            int base_scc = atomicAdd(d_scc_count, scc_cnt);
            int base_nxt = atomicAdd(d_next_count, scc_cnt);
            for (int j = 0; j < scc_cnt; j++)
                d_next_queue[base_nxt + j] = local_scc[j];
        }

        // Flush remaining BW nodes
        if (bw_cnt > 0) {
            int base_next = atomicAdd(d_next_count, bw_cnt);
            int base_bw   = atomicAdd(d_bw_count, bw_cnt);
            for (int j = 0; j < bw_cnt; j++)
                d_next_queue[base_next + j] = local_bw[j];
        }
    }
}'''

new_bw = '''__global__ void bw_bfs_level_kernel(
    const edge_t* d_r_begin, const node_t* d_r_node_idx,
    int* d_Color, int* d_SCC,
    const int* d_queue, int queue_size,
    int* d_next_queue, int* d_next_count,
    int fw_color, int bw_color, int base_color, node_t pivot,
    int* d_scc_count, int* d_bw_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= queue_size) return;
    node_t t = d_queue[i];
    for (edge_t nx = d_r_begin[t]; nx < d_r_begin[t + 1]; nx++) {
        node_t k = d_r_node_idx[nx];
        int k_color = d_Color[k];
        if (k_color == fw_color) {
            int old = atomicCAS(&d_Color[k], fw_color, SCC_FOUND);
            if (old == fw_color) {
                d_SCC[k] = pivot;
                atomicAdd(d_scc_count, 1);
                int pos = atomicAdd(d_next_count, 1);
                d_next_queue[pos] = k;
            }
        }
        else if (k_color == base_color) {
            int old = atomicCAS(&d_Color[k], base_color, bw_color);
            if (old == base_color) {
                atomicAdd(d_bw_count, 1);
                int pos = atomicAdd(d_next_count, 1);
                d_next_queue[pos] = k;
            }
        }
    }
}'''

assert old_bw in code, 'BW kernel pattern not found!'
code = code.replace(old_bw, new_bw)
print('BW kernel reverted ✅')

open('scc_cuda_fb_global.cu', 'w').write(code)
print('File saved! Run: make clean && make')
