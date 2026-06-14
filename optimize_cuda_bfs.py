#!/usr/bin/env python3
"""Apply batched atomicAdd optimization to CUDA BFS kernels.
Run this on the server: python3 optimize_cuda_bfs.py"""

import os

os.chdir(os.path.expanduser("~/DynamicGraphs_SCC/src_CUDA"))

with open("scc_cuda_fb_global.cu", "r") as f:
    code = f.read()

# Backup
with open("scc_cuda_fb_global.cu.bak", "w") as f:
    f.write(code)
print("Backup saved to scc_cuda_fb_global.cu.bak")

# ---- OPTIMIZE fw_bfs_level_kernel ----
old_fw = """__global__ void fw_bfs_level_kernel(
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
}"""

new_fw = """__global__ void fw_bfs_level_kernel(
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
        int local_buf[128];
        int local_cnt = 0;
        for (edge_t nx = d_begin[t]; nx < d_begin[t + 1]; nx++) {
            node_t k = d_node_idx[nx];
            if (fw_check_navigator_device(d_Color, k, base_color)) {
                int old = atomicCAS(&d_Color[k], base_color, fw_color);
                if (old == base_color) {
                    local_buf[local_cnt++] = k;
                    if (local_cnt == 128) {
                        int base = atomicAdd(d_next_count, 128);
                        for (int j = 0; j < 128; j++) d_next_queue[base + j] = local_buf[j];
                        local_cnt = 0;
                    }
                }
            }
        }
        if (local_cnt > 0) {
            int base = atomicAdd(d_next_count, local_cnt);
            for (int j = 0; j < local_cnt; j++) d_next_queue[base + j] = local_buf[j];
        }
    }
}"""

# ---- OPTIMIZE bw_bfs_level_kernel ----
old_bw = """__global__ void bw_bfs_level_kernel(
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
}"""

new_bw = """__global__ void bw_bfs_level_kernel(
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
        node_t t = d_queue[i];
        int local_scc[128], local_bw[128];
        int scc_cnt = 0, bw_cnt = 0;
        for (edge_t nx = d_r_begin[t]; nx < d_r_begin[t + 1]; nx++) {
            node_t k = d_r_node_idx[nx];
            int k_color = d_Color[k];
            if (k_color == fw_color) {
                int old = atomicCAS(&d_Color[k], fw_color, SCC_FOUND);
                if (old == fw_color) {
                    d_SCC[k] = pivot;
                    local_scc[scc_cnt++] = k;
                    if (scc_cnt == 128) {
                        atomicAdd(d_scc_count, 128);
                        int base_nxt = atomicAdd(d_next_count, 128);
                        for (int j = 0; j < 128; j++) d_next_queue[base_nxt + j] = local_scc[j];
                        scc_cnt = 0;
                    }
                }
            }
            else if (k_color == base_color) {
                int old = atomicCAS(&d_Color[k], base_color, bw_color);
                if (old == base_color) {
                    local_bw[bw_cnt++] = k;
                    if (bw_cnt == 128) {
                        atomicAdd(d_bw_count, 128);
                        int base_nxt = atomicAdd(d_next_count, 128);
                        for (int j = 0; j < 128; j++) d_next_queue[base_nxt + j] = local_bw[j];
                        bw_cnt = 0;
                    }
                }
            }
        }
        if (scc_cnt > 0) {
            atomicAdd(d_scc_count, scc_cnt);
            int base_nxt = atomicAdd(d_next_count, scc_cnt);
            for (int j = 0; j < scc_cnt; j++) d_next_queue[base_nxt + j] = local_scc[j];
        }
        if (bw_cnt > 0) {
            atomicAdd(d_bw_count, bw_cnt);
            int base_nxt = atomicAdd(d_next_count, bw_cnt);
            for (int j = 0; j < bw_cnt; j++) d_next_queue[base_nxt + j] = local_bw[j];
        }
    }
}"""

# Apply replacements
count_fw = code.count(old_fw)
code = code.replace(old_fw, new_fw)
count_bw = code.count(old_bw)
code = code.replace(old_bw, new_bw)

with open("scc_cuda_fb_global.cu", "w") as f:
    f.write(code)

print(f"Replaced {count_fw} fw_bfs_level_kernel")
print(f"Replaced {count_bw} bw_bfs_level_kernel")
print("\nNow rebuild with: cd ~/DynamicGraphs_SCC/src_CUDA && make clean && make")
