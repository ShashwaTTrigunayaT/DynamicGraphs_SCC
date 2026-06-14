#include "scc_cuda.h"

// Upload CSR graph arrays from host to GPU
void graph_upload(GPUGraph& gpu,
    const std::vector<edge_t>& h_begin,
    const std::vector<node_t>& h_node_idx,
    const std::vector<edge_t>& h_r_begin,
    const std::vector<node_t>& h_r_node_idx,
    int N, int M)
{
    gpu.num_nodes = N;
    gpu.num_edges = M;

    CUDA_CHECK(cudaMalloc(&gpu.d_begin,     (N + 1) * sizeof(edge_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_node_idx,   M * sizeof(node_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_r_begin,    (N + 1) * sizeof(edge_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_r_node_idx, M * sizeof(node_t)));

    CUDA_CHECK(cudaMemcpy(gpu.d_begin,     h_begin.data(),    (N + 1) * sizeof(edge_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_node_idx,   h_node_idx.data(),  M * sizeof(node_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_r_begin,    h_r_begin.data(),   (N + 1) * sizeof(edge_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_r_node_idx, h_r_node_idx.data(), M * sizeof(node_t), cudaMemcpyHostToDevice));

    // Prefetch graph arrays to GPU device memory for faster kernel access.
    // This migrates pages to the GPU and avoids cold-cache misses during BFS.
    cudaMemPrefetchAsync(gpu.d_begin,     (N + 1) * sizeof(edge_t), 0, NULL);
    cudaMemPrefetchAsync(gpu.d_node_idx,   M * sizeof(node_t),      0, NULL);
    cudaMemPrefetchAsync(gpu.d_r_begin,    (N + 1) * sizeof(edge_t), 0, NULL);
    cudaMemPrefetchAsync(gpu.d_r_node_idx, M * sizeof(node_t),      0, NULL);
}

void graph_free(GPUGraph& gpu) {
    if (gpu.d_begin)     { cudaFree(gpu.d_begin);     gpu.d_begin = NULL; }
    if (gpu.d_node_idx)  { cudaFree(gpu.d_node_idx);  gpu.d_node_idx = NULL; }
    if (gpu.d_r_begin)   { cudaFree(gpu.d_r_begin);   gpu.d_r_begin = NULL; }
    if (gpu.d_r_node_idx){ cudaFree(gpu.d_r_node_idx); gpu.d_r_node_idx = NULL; }
}

// Allocate GPU state (G_Color, G_SCC arrays)
void state_allocate(GPUState& st, int N) {
    st.num_nodes = N;
    CUDA_CHECK(cudaMalloc(&st.d_Color, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.d_SCC,   N * sizeof(int)));

    // Prefetch state arrays to GPU for faster kernel access
    cudaMemPrefetchAsync(st.d_Color, N * sizeof(int), 0, NULL);
    cudaMemPrefetchAsync(st.d_SCC,   N * sizeof(int), 0, NULL);
}

// Initialize: G_Color = -1, G_SCC = -1 (CUDA_NIL_NODE)
void state_init(GPUState& st) {
    int N = st.num_nodes;
    CUDA_CHECK(cudaMemset(st.d_Color, 0xFF, N * sizeof(int)));  // -1
    CUDA_CHECK(cudaMemset(st.d_SCC,   0xFF, N * sizeof(int)));  // -1
}

void state_free(GPUState& st) {
    if (st.d_Color) { cudaFree(st.d_Color); st.d_Color = NULL; }
    if (st.d_SCC)   { cudaFree(st.d_SCC);   st.d_SCC = NULL; }
}

// ======================================================================
// Dynamic arrays for special methods (7, 9, 11)
// ======================================================================
void dynamic_arrays_allocate(DynamicArrays& da, int N, int num_sccs)
{
    CUDA_CHECK(cudaMalloc(&da.d_scc_list,       N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&da.d_vec_scc_count,  num_sccs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&da.d_level_ver,      N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&da.d_affect_level,   (N + 5) * sizeof(int)));
}

void dynamic_arrays_free(DynamicArrays& da)
{
    if (da.d_scc_list)       { cudaFree(da.d_scc_list);       da.d_scc_list = NULL; }
    if (da.d_vec_scc_count)  { cudaFree(da.d_vec_scc_count);  da.d_vec_scc_count = NULL; }
    if (da.d_level_ver)      { cudaFree(da.d_level_ver);      da.d_level_ver = NULL; }
    if (da.d_affect_level)   { cudaFree(da.d_affect_level);   da.d_affect_level = NULL; }
}

// ======================================================================
// DynamicArrays upload helpers — copies host data into already-allocated
// device buffers. Mirrors the implicit host-to-device transfer that the
// OpenMP version never needs (since it works directly on host memory).
// ======================================================================

void dynamic_arrays_upload_scc_list(DynamicArrays& da,
    const std::vector<int>& h_scc_list, int N)
{
    if (da.d_scc_list && !h_scc_list.empty()) {
        CUDA_CHECK(cudaMemcpy(da.d_scc_list, h_scc_list.data(),
                               N * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void dynamic_arrays_upload_vec_scc_count(DynamicArrays& da,
    const std::vector<int>& h_vec_scc_count, int num_sccs)
{
    if (da.d_vec_scc_count && !h_vec_scc_count.empty()) {
        CUDA_CHECK(cudaMemcpy(da.d_vec_scc_count, h_vec_scc_count.data(),
                               num_sccs * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void dynamic_arrays_upload_level_ver(DynamicArrays& da,
    const std::vector<int>& h_level_ver, int N)
{
    if (da.d_level_ver && !h_level_ver.empty()) {
        CUDA_CHECK(cudaMemcpy(da.d_level_ver, h_level_ver.data(),
                               N * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void dynamic_arrays_upload_affect_level(DynamicArrays& da,
    const std::vector<int>& h_affect_level, int size)
{
    if (da.d_affect_level && !h_affect_level.empty()) {
        CUDA_CHECK(cudaMemcpy(da.d_affect_level, h_affect_level.data(),
                               size * sizeof(int), cudaMemcpyHostToDevice));
    }
}
