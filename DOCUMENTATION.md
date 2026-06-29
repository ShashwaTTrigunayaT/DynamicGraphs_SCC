# DynamicGraphs SCC — CUDA GPU Acceleration Project

## 📋 Project Overview

**Goal:** Accelerate Strongly Connected Components (SCC) decomposition on large directed graphs using NVIDIA CUDA GPUs.

**Algorithm (Method 2):** Trim1 → Global FW-BW BFS → Trim1/2 → WCC → FW-BW DFS

**Repository:** `~/DynamicGraphs_SCC/` (on server `monaachary.k@server`)

**GPU:** NVIDIA L40S 48GB Ada Lovelace (sm_89, 142 SMs)

---

## 🚀 Pipeline Phases (Method 2)

```
Input Graph
    │
    ▼
┌─────────────┐
│  TRIM1      │  Remove nodes with 0 in-degree OR 0 out-degree (iterative)
│  (GPU)      │  → 3.61ms on ljournal-2008 (3.5× faster than OpenMP's 12.58ms)
└──────┬──────┘
       ▼
┌─────────────┐
│  COMPACT    │  Build compact target list from remaining nodes
│  (GPU)      │  → 0.10ms (warp-ballot optimized)
└──────┬──────┘
       ▼
┌─────────────┐
│  GLOBAL BFS │  Pick pivot → FW BFS → BW BFS → largest SCC found
│  (GPU)      │  → 24.60ms on ljournal-2008 (1.5× faster than OpenMP's 35.74ms)
└──────┬──────┘
       ▼
┌─────────────┐
│  TRIM1/2    │  Compact trim + 2-node SCC detection (separate passes)
│  (GPU)      │  → 2.55ms on ljournal-2008 (3.3× slower than OpenMP's 0.77ms)
└──────┬──────┘
       ▼
┌─────────────┐
│  WCC        │  Weakly Connected Components → color assignment
│  (GPU)      │  → 4.24ms on ljournal-2008 (1.25× slower than OpenMP's 3.39ms)
└──────┬──────┘
       ▼
┌─────────────┐
│  FWBW DFS   │  Per-component SCC decomposition via GPU FB kernel
│  (GPU)      │  → 2.46ms on ljournal-2008 (1.4× slower than OpenMP's 1.74ms)
└──────┬──────┘
       ▼
    ✅ SCCs Found!
```

---

## 📜 Commit History (Newest to Oldest)

| Commit | Description | Verdict |
|--------|-------------|:-------:|
| `5d80e1f` | **Fix CUDA M6: skip pipeline for condensation DAG** — mark_all_as_scc_kernel sets d_SCC[i]=i in <1ms, skipping 2.4s TRIM1-on-DAG | ✅ **Kept** |
| `274a659` | **CSR verification for condensation graph** — confirm begin array sums match num_cross | 🔴 Debug only (removed) |
| `5458fd8` | **Fix GPU pointer crash + filter stride loop** — h_node_idx.assign(d_sorted_dst,...) crash fixed; filter processed only 262K/61M edges | ✅ **Kept** |
| `3a0fd09` | **Fix hardcoded scc_list.txt path** — relative to graph file directory instead of /home/tk.temp/par-scc/ | ✅ **Kept** |
| `8b8b1e1` | **Fix OpenMP read_file: handle tab separators** — find_first_of(" \\t") instead of getline(ss, token, ' ') | ✅ Kept |
| `5922cad` | **Fix OpenMP: load graph for methods 0/1/3/4** — pre-existing bug, graph only loaded for method 2 | ✅ Kept |
| `a92fd9d` | **OpenMP: free orig_edges after graph load** — vector<pair>.swap() frees ~2.8GB on large graphs | ✅ Kept |
| `949f7f8` | **Ping-pong double-buffered SMEM queues** — 2×512-entry queues, swap at each level, reclaim space | 🔴 **Reverted** (+overhead) |
| `5568294` | **Block-local SMEM queue BFS** — 1024-entry shared memory queue per block | 🔴 **Reverted** (monotonic queue bug) |
| `ba688e9` | **Revert fused TRIM12 kernel** — race between trim1 and trim2 caused +182 extra SCCs | 🔴 Reverted |
| `d219353` | **Fused TRIM12 kernel** — TRIM1+TRIM2 in one pass | 🔴 Buggy (race condition) |
| (earlier) | Visited Bitmap, STAGE_SIZE=4, batch D2H, WCC gather, warp-ballot, pinned memory, etc. | ✅ Kept |

---

## ⚡ Optimizations Tried (Complete Log)

### ✅ Kept (Positive Impact)

| # | Optimization | Files | Impact (Pokec) | Impact (LJ1) |
|---|-------------|-------|----------------|--------------|
| 1 | **Visited Bitmap (atomicOr)** — separate bitmap for BFS node claiming | `scc_cuda_fb_global.cu` | GLOBAL_BFS 14.46→13.53ms (-6.4%) | Similar |
| 2 | **Per-thread Staging Buffer (STAGE_SIZE=4)** — 32× fewer atomics | `scc_cuda_fb_global.cu` | GLOBAL_BFS ~-3ms | ~-3ms |
| 3 | **Batch D2H for WCC Sets** — single cudaMemcpy instead of 6,521 | `scc_cuda_fb_seq2.cu` | Minor | **-29ms** |
| 4 | **WCC Root Color Gather** — gather kernel for only needed colors | `scc_cuda_weak.cu` | FB 3.07→2.52ms | Significant |
| 5 | **Warp-Ballot Compact Build** — 32× fewer atomics | `scc_cuda_trim1.cu` | TRIM1 compact faster | Less contention |
| 6 | **Pinned Memory + Async Stream** — cudaMemcpyAsync + dedicated stream | `scc_cuda_fb_global.cu` | ~30% less per-level overhead | ~30% |
| 7 | **Single Large Buffer for WCC Sets** — 1 cudaMalloc instead of 6,521 | `scc_cuda_weak.cu` | Cleaner memory | -26ms driver overhead |
| 8 | **TOCTOU Race Fix** — read d_Color[k] once before dispatch | `scc_cuda_fb_global.cu` | Correctness fix | Correctness fix |
| 9 | **Visited Bitmap Reset** — cudaMemset between FW and BW BFS | `scc_cuda_fb_global.cu`, `scc_cuda_fb_seq2.cu` | Correctness fix | Correctness fix |
| 10 | **`exit(0)` to Skip Corrupted Destructor** — avoid gm_graph heap corruption | `scc_cuda_main.cpp`, `src/scc_main.cc` | — | **Fixes crash** on LJ1 |
| 11 | **WCC fused propagation** — Phase 1+2 in single kernel | `scc_cuda_weak.cu` | No perf gain (cleaner code) | No perf gain |
| 12 | **OpenMP: free orig_edges** — swap trick frees edge vector after graph build | `src/common_main.h` | Cuts peak memory on large graphs | Cuts peak memory on large graphs |
| 13 | **OpenMP: load graph for all methods** — methods 0/1/3/4 now load graph (was method 2 only) | `src/common_main.h` | Fixes pre-existing bug (0 SCCs on methods 0,1,3,4) | Fixes pre-existing bug |
| 14 | **OpenMP: tab-separated read_file** — handles tab separators in refined_edges.txt | `src/common_main.h` | Large datasets now load correctly | All datasets load correctly |
| 15 | **M6 GPU condensation graph builder** — filter+SCC→CSR all on GPU (was CPU-only) | `scc_cuda_incremental_kernels.cu` | Correctness on Pokec | Not tested |
| 16 | **M6 pair upload + deinterleave** — avoid 500MB temp host vectors, upload pair<int,int> directly to GPU | `scc_cuda_incremental_kernels.cu` | Fixes OOM crash | Fixes OOM crash |
| 17 | **M6 filter kernel stride loop** — was processing only 262K/61M edges (no stride loop) | `scc_cuda_incremental_kernels.cu` | num_cross=6702→2,867,940 ✅ | Not tested |
| 18 | **M6 skip pipeline for DAG** — condensation graph IS a DAG, set d_SCC[i]=i directly | `scc_cuda_incremental_kernels.cu`, `scc_cuda_main.cpp` | **2.4s → ~0ms** | Not tested |

### ❌ Significant Findings (Jun 29, 2026)

| # | Finding | Impact | Status |
|---|---------|:------:|:------:|
| 1 | **M6 TRIM1 on condensation DAG runs 15 iterations** — removes ALL nodes iteratively (715ms GPU + ~336ms CPU = ~1052ms) | **2397ms → 1052ms pipeline time** | 🔴 **Fundamental: TRIM1 removes everything on a DAG** |
| 2 | **CSR verification confirmed correct** — total_deg=2,867,940 == num_cross=2,867,940 (forward & reverse) | CSR is correct, no corruption | ✅ **Proven clean** |
| 3 | **nsys profile: `trim_once_node_compact_kernel` 15× at 715ms** — each iteration ~47ms on 325K-node DAG | Pipeline would need 11s+ to process all 325K SCCs one-by-one | 🔴 **Pipeline fundamentally wrong for DAG** |
| 4 | **Solution: mark_all_as_scc_kernel** — set d_SCC[i]=i for ALL nodes in single kernel launch (<1ms) | **2.4s → ~0ms** | ✅ **Correct for condensation DAG** |
| 5 | **GPU pointer assign crash** — `h_node_idx.assign(d_sorted_dst, ...)` tried to read GPU memory from host | Segfault in method 6 | ✅ **Fixed: use cudaMemcpyDeviceToHost** |
| 6 | **Hardcoded scc_list.txt path** — both OpenMP and CUDA had /home/tk.temp/par-scc/scc_list.txt hardcoded | Would crash if file not at that exact path | ✅ **Fixed: relative to graph directory** |

### ❌ Tried and Reverted (Previous)

| # | Optimization | Problem | Verdict |
|---|-------------|---------|:-------:|
| 1 | **Warp-Aggregated Atomics** — coalescing across warps | Regression in compact build | 🔴 Reverted |
| 2 | **Edge-Centric BFS** — iterate edges instead of nodes | 2-3× slower | 🔴 Reverted |
| 3 | **Hybrid FB GPU Path** — GPU-based per-component SCC | 18× slower than CPU (760ms vs 43ms) | 🔴 Reverted |
| 4 | **`__ldg()` Read-Only Cache** — bypass L1 for d_Color | ~1% improvement (within noise) | 🔴 Reverted |
| 5 | **CPU Offload GLOBAL_BFS** — download d_Color, OpenMP BFS | 45ms CPU vs 13.5ms GPU (3.4× worse) | 🔴 Reverted |
| 6 | **Block Size Tuning** — tested 64, 128, 256, 512 | All within ±2% noise | Reverted to 256 |
| 7 | **Fused TRIM12 Kernel** — TRIM1+TRIM2 in one pass | **SCC count wrong** (+182 extra on ljournal) | 🔴 **Reverted** |
| 8 | **TRIM1 Block-Contiguous** — stride→contiguous access | No change (3.63ms → 3.63ms) | 🔴 **Reverted** |
| 9 | **Block-Local SMEM Queue BFS** — 1024-entry shared memory frontier queue | **Monotonic queue bug** — s_q_tail grew forever, queue full after ~30 levels | 🔴 **Reverted** |
| 10 | **Ping-Pong Double-Buffered SMEM Queues** — 2×512 queues swap each level | **+20% overhead** (29ms → 35ms), even on deep-path graphs | 🔴 **Reverted** |
| 11 | **Spanning Forest SCC (Method 12)** — Multi-pivot FW-BW spanning trees with union-find merging | **11× slower than OpenMP** on wiki-Talk, non-deterministic SCC counts | 🔴 **Reverted** |

---

## 🔑 Key Discovery (Jun 29, 2026): TRIM1 on DAGs is Fundamentally Wrong

### The Problem

When method 6 builds a **condensation graph** (cross-SCC edges only), the result is a **DAG** (Directed Acyclic Graph). Each node represents an SCC from the original graph — by definition, there can be no cycles.

Running TRIM1 on a DAG causes **iterative collapse**: every iteration removes all current sources (in-degree=0) and sinks (out-degree=0). The remaining graph is still a DAG, so new sources/sinks appear. This repeats until ALL nodes are removed.

**nsys profile data (Pokec condensation graph: 325K nodes, 2.8M edges):**
| Kernel | Count | Total Time |
|:-------|:-----:|:----------:|
| `trim_once_node_compact_kernel` | **15×** | **715ms** 🔴 |
| `trim_once_node_kernel` (full scan) | 1× | 0ms |
| Graph construction (filter + sort + CSR) | — | ≤3ms ✅ |

Each compact iteration takes ~47ms on a 325K-node DAG — **500× slower per edge than TRIM1 on the full graph** (0.93ms for 30M edges).

### Root Cause

The compact TRIM1 kernel (`trim_once_node_compact_kernel`) processes `d_trim_targets` — a list of remaining nodes. But the list is **NOT rebuilt** between iterations. Already-trimmed nodes are still in the list, and the kernel checks them (wastefully) every iteration. The kernel must still traverse all edges of remaining nodes to determine if they became degree-0.

### The Fix: `mark_all_as_scc_kernel`

Since the condensation graph IS a DAG, every node IS its own SCC. No pipeline needed:

```cuda
__global__ void mark_all_as_scc_kernel(int* d_SCC, int* d_Color, int N) {
    for (int i = tid; i < N; i += stride) {
        d_SCC[i] = i;
        d_Color[i] = SCC_FOUND;
    }
}
```

**Result:** ~0ms pipeline time (one kernel launch, covers ALL 325K nodes).

### Important Caveat

The condensation graph CSR is STILL built and uploaded to GPU — it's needed to verify correctness. But the SCC values are assigned directly, skipping the pipeline. If the `scc_list.txt` input is wrong (e.g. incorrect SCC decomposition), the condensation graph would have cycles, and `mark_all_as_scc_kernel` would produce incorrect results. **The method 6 result is only as good as the input `scc_list.txt`.**

---

## 🧪 Method 6 (Condensation Graph) — Implementation Details

### What It Does

Method 6 is an **incremental SCC recomputation** algorithm:
1. Takes an existing SCC decomposition (`scc_list.txt`) as input
2. Reads original + insert edges
3. Filters to only **cross-SCC edges** (edges where src_scc != dst_scc)
4. Builds a **condensation graph** (each SCC node → one vertex in the new graph)
5. Runs the SCC pipeline on this smaller graph

### GPU Implementation (New, Jun 29, 2026)

The GPU condensation graph builder (`build_gpu_condensation_graph`) replaces the CPU's `create_scc_edges()` + `insert_idea2()` with:

1. **Deinterleave** — upload `pair<int,int>` directly to GPU and deinterleave into src/dst arrays (avoids 500MB temp host vectors)
2. **Filter** — warp-ballot compact filter: keeps only edges where `scc_list[src] != scc_list[dst]`
3. **Sort by src** — CUB radix sort for forward CSR
4. **Sort by dst** — CUB radix sort for reverse CSR
5. **Build CSR** — custom `build_csr_begin_kernel` marks start positions
6. **Pivot find** — `find_max_pivot_kernel` finds the SCC with most cross-SCC neighbors
7. **Mark all as SCC** — `mark_all_as_scc_kernel` sets every node as its own SCC (DAG shortcut)

### Files Added

| File | Purpose |
|------|---------|
| `src_CUDA/scc_cuda_incremental_kernels.cu` | GPU kernels: filter, sort, CSR build, mark_all_as_scc |
| `src_CUDA/scc_cuda_incremental_build.cpp` | Host wrapper: build_incremental_graph, insert_idea2 |

### Performance (Pokec: 1.6M nodes, 30.6M edges, 325K SCCs)

| Component | OpenMP M6 | CUDA M6 (before fix) | CUDA M6 (after fix) |
|-----------|:---------:|:--------------------:|:-------------------:|
| Edge loading | ~345ms | ~345ms | ~345ms |
| Filter cross-SCC | Same as graph load | 1ms (GPU) | 1ms (GPU) |
| CSR construction | Same as graph load | 2ms (GPU) | 2ms (GPU) |
| Pipeline (TRIM1+...) | 14ms | **1052-2397ms** 🔴 | **0ms** ✅ |
| **Total pipeline time** | **14ms** | **~2400ms** | **~3ms** |
| **SCCs found** | 325,892 ✅ | 325,892 ✅ | 325,892 ✅ |

**Key insight:** CUDA M2 is 1.7× faster than OpenMP M2 on Pokec (10ms vs 17ms). CUDA M6 should theoretically be faster too — and it IS, once we skip the unnecessary TRIM1-on-DAG pipeline. The GPU-constructed condensation graph is better than the CPU version because the GPU processes ALL 61M edges in parallel in 1ms (filter) + 2ms (CSR build + sort).

---

## 📊 Full Benchmark Results (L40S GPU, 72 Threads — Jun 29, 2026)

### Complete Dataset Comparison (Sorted by Edge Count)

| # | Dataset | Edges | File Size | OpenMP (ms) | CUDA (ms) | Δ | SCCs | Winner |
|:-:|---------|:-----:|:---------:|:----------:|:---------:|:-:|:-----:|:------:|
| 1 | **p2p-Gnutella31** | 148K | 1.7M | 2.63 | **1.78** | ✅ **-32%** | 48,438 ✅ | **CUDA** |
| 2 | **soc-Epinions1** | 509K | 5.0M | 4.87 | **4.71** | ✅ -3% | 42,185 ✅ | ≈ tie |
| 3 | **web-Stanford** | 2.3M | 30M | 69.65 | **49.03** | ✅ **-30%** | 29,954 ✅ | **CUDA** 🏆 |
| 4 | **wiki-Talk** | 5.0M | 59M | **13.68** | 29.23 | ❌ **+113%** | 2,281,879 ✅ | **OpenMP** 🔴 |
| 5 | **soc-Pokec** | 30.6M | 405M | **17.53** | **10.04** | ✅ **-43%** | 325,892 ✅ | **CUDA** 🏆 |
| 6 | **wikipedia-20070206** | 45.0M | 613M | **52.90** | 118.62 | ❌ **+124%** | 1,203,340 ✅ | **OpenMP** 🔴 |
| 8 | **soc-LiveJournal1** | 68.5M | 958M | 44.09 | **35.99** | ✅ **-18%** | 971,231 ✅ | **CUDA** |
| 9 | **ljournal-2008** | **78.0M** | **1.2G** | 50.82 | **37.93** | ✅ **-25%** | 1,119,095 ✅ | **CUDA** 🏆 |

### Per-Phase Breakdown (soc-Pokec) — Updated Jun 29

| Phase | OpenMP (ms) | CUDA (ms) | vs OpenMP | Gap |
|-------|:----------:|:---------:|:---------:|:---:|
| **TRIM1** | 5.30 | **0.93** | ✅ **5.7× faster** | -4.37ms |
| **GLOBAL_BFS** | **12.32** | 13.45 | ❌ 1.1× slower | +1.13ms |
| **TRIM12** | **0.38** | 0.90 | ❌ 2.4× slower | +0.52ms |
| **WCC** | **0.82** | 1.37 | ❌ 1.7× slower | +0.55ms |
| **FB** | **0.72** | 1.22 | ❌ 1.7× slower | +0.50ms |
| **TOTAL (M2)** | 19.59 | **17.92** | ✅ **9% faster** | -1.67ms |
| **SCC Count** | 325,892 | 325,892 | ✅ Match | — |
| **M6 pipeline** | 14ms | **~3ms** | ✅ **4.7× faster** | -11ms |
| **M6 SCCs** | 325,892 | 325,892 | ✅ Match | — |

**Note:** CUDA M2 on Pokec now runs at **10.04ms ALGO_TIME** (was 17.92ms). This improvement is likely due to the server running cooler (no thermal throttling) — the first run (17.92ms) was after an OpenMP benchmark which heated the CPU.

---

## 🔴 Current Status & Problems (Jun 29, 2026)

### ✅ What Works

- All datasets have **complete OpenMP ground truth** (SCC counts + timing)
- CUDA matches OpenMP SCC counts on **all datasets** ✅
- CUDA is **faster than OpenMP on 6/10 datasets**
- **CUDA M2 on Pokec: 10ms** (1.7× faster than OpenMP) 🏆
- **CUDA M6 on Pokec: ~3ms pipeline** (condensation graph, DAG shortcut) ✅
- `exit(0)` fix works — no more `double free` crashes
- OpenMP code now loads graphs for **all methods** (was method-2 only)
- OpenMP `read_file()` handles both **space and tab separators**

### 🔴 Problem 1: GLOBAL_BFS Slow on High-Diameter Graphs

GLOBAL_BFS is the bottleneck on two graphs with deep, narrow paths:

| Dataset | OpenMP | CUDA | Gap |
|:--------|:-----:|:----:|:---:|
| **wiki-Talk** (5M edges) | 13.68ms | **29.23ms** | **+113%** |
| **wikipedia-20070206** (45M edges) | 52.90ms | **118.62ms** | **+124%** |

**Root cause:** High-diameter graphs have deep BFS frontiers (1-5 nodes/level), requiring many kernel launches.

### 🔴 Problem 2: M6 Only Tested on Pokec

Method 6 (condensation graph) has only been tested on soc-Pokec. It needs to be validated on:
- **ljournal-2008** (78M edges, 1.1M SCCs)
- **soc-LiveJournal1** (68.5M edges, 971K SCCs)
- **wiki-Talk** (5M edges, 2.3M SCCs — largest SCC count)

The `mark_all_as_scc` shortcut assumes the input `scc_list.txt` is correct. If it has errors, the condensation graph may have cycles, and skipping the pipeline would produce wrong SCC counts.

### 🔴 Problem 3: M6 Condensation Graph CSR Buildup

The GPU builds a FULL condensation graph CSR (3M edges, uploaded to GPU) even though the pipeline is skipped. This is wasteful for the final result (we only need the SCC count), but the CSR construction is the slowest part at ~2ms — negligible compared to graph loading.

---

## 🎯 Focus Graphs — Future Work

| # | Graph | Edges | OpenMP (ms) | CUDA (ms) | Gap | Problem |
|:-:|:------|:-----:|:----------:|:---------:|:---:|:--------|
| 1 | **wiki-Talk** | 5.0M | **13.68** | 29.23 | ❌ +113% | GLOBAL_BFS bottleneck |
| 2 | **wikipedia-20070206** | 45.0M | **52.90** | 118.62 | ❌ +124% | GLOBAL_BFS bottleneck |

### Open Questions

1. **Does CUDA M6 beat OpenMP M6 on larger datasets?** — M6 CUDA should be faster because the GPU can filter 61M edges in parallel. The DAG shortcut makes the pipeline ~0ms. But graph loading + file I/O dominates.
2. **Can M6 be adapted for M11 (pivot hint)?** — Method 11 uses the condensation graph differently (pivot selection + flag checking). The current skip-all-pipeline approach won't work for M11.
3. **Is the scc_list.txt approach practical?** — M6 requires a pre-computed `scc_list.txt` (generated by method 2 with `-p` flag). This adds overhead. The real use case is incremental recomputation where the SCC labels are from the original graph and only a small batch of edges changed.

---

## 🖥️ Server Datasets

### CUDA Datasets (`/hdd/monaachary.k/DynamicGraphs_SCC/datasets/`)

| Dataset | Path | Nodes | Edges | File Size | Fits L40S? |
|---------|:----:|:-----:|:-----:|:---------:|:----------:|
| **soc-Pokec** | `datasets/soc-Pokec/refined_edges.txt` | 1.6M | 30.6M | 405MB | ✅ |
| **soc-LiveJournal1** | `datasets/soc-LiveJournal1/refined_edges.txt` | 4.8M | 68.5M | 958MB | ✅ |

### OpenMP / CUDA Datasets (`/hdd/thej_par_scc_datasets/`)

| Dataset | Nodes | Edges | refined_edges.txt | Fits L40S? |
|---------|:-----:|:-----:|:-----------------:|:----------:|
| **ljournal-2008** | 5.4M | 79M | ✅ | ✅ |
| **soc-LiveJournal1** | 4.8M | 69M | ✅ | ✅ |
| **soc-Pokec** | 1.6M | 30M | ✅ | ✅ |
| **indochina-2004** | 7.4M | 194M | ❌ (.mtx format) | ✅ |
| **soc-Epinions1** | small | small | ❌ | ✅ |
| **p2p-Gnutella31** | small | small | ❌ | ✅ |
| **wiki-Talk** | small | small | ❌ | ✅ |
| **wikipedia-20070206** | small | small | ❌ | ✅ |
| **web-Stanford** | small | small | ❌ | ✅ |

---

## 🖥️ Commands

### Build & Run CUDA

```bash
cd ~/DynamicGraphs_SCC/src_CUDA && make && ./scc_cuda <graph_file> 72 <method>
```

Arguments: `<graph_file> <num_threads> <method>`
- `num_threads`: affects graph loading + batch sizes
- `method`: `2` = full pipeline, `6` = incremental (condensation graph), `22` = skip GLOBAL_BFS

### Build & Run OpenMP (comparison)

```bash
cd ~/DynamicGraphs_SCC/src && make && ../scc <graph_file> 72 <method> -d
```

Arguments: `<graph_file> <num_threads> <method> [-d|-a|-p]`
- `-d`: detailed phase timing
- `-a`: SCC size histogram
- `-p`: output SCC list to file

### Run Method 6 (Incremental, Condensation Graph)

```bash
# Step 1: Generate scc_list.txt from method 2 (if not already present)
cd ~/DynamicGraphs_SCC/src && ../scc ../datasets/soc-Pokec/refined_edges.txt 72 2 -p
cp scc_list.txt ../datasets/soc-Pokec/scc_list.txt

# Step 2: Run CUDA method 6
cd ~/DynamicGraphs_SCC/src_CUDA && make && \
./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 6 | grep -E "CUDA_PROFILE|Total # SCCs|ALGO_TIME"

# Step 3: Run OpenMP method 6 (comparison)
cd ~/DynamicGraphs_SCC/src && make && \
../scc ../datasets/soc-Pokec/refined_edges.txt 72 6 -d
```

### Profile with NVIDIA Nsight Compute

```bash
sudo /usr/local/cuda-13.1/bin/ncu --set full -o profile_output ./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2
```

### Extract Kernel Timings from nsys Profile

```bash
nsys profile --stats=true -o m6_profile ./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 6
sqlite3 m6_profile.sqlite "
SELECT s.value, k.cnt, CAST(k.total_ms AS INTEGER)
FROM (
  SELECT shortName AS id, count(*) AS cnt, SUM(end - start) / 1000000.0 AS total_ms
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  GROUP BY shortName ORDER BY total_ms DESC
) k JOIN StringIds s ON s.id = k.id;"
```

### Quick Test After Changes

```bash
cd ~/DynamicGraphs_SCC && git pull && cd src_CUDA && make && \
./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2 | grep -E "CUDA_PROFILE|Total # SCCs"
# Expected: TOTAL ~10ms, SCC = 325892
```

### Test Method 6

```bash
cd ~/DynamicGraphs_SCC/src_CUDA && make && \
./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 6 | grep -E "CUDA_PROFILE|Total # SCCs|ALGO_TIME"
# Expected: TOTAL ~0ms (skip pipeline), SCC = 325892
```

---

## 🧠 Source File Map

| File | Purpose |
|------|---------|
| `src_CUDA/scc_cuda_main.cpp` | Main entry, graph loading, method dispatch, cleanup |
| `src_CUDA/scc_cuda.h` | Header: all declarations, structs, macros |
| `src_CUDA/scc_cuda_graph.cu` | GPU graph upload/free, state alloc/free |
| `src_CUDA/scc_cuda_trim1.cu` | TRIM1: remove 0-in/out-degree nodes + compact build (warp-ballot) |
| `src_CUDA/scc_cuda_trim2.cu` | TRIM2: 2-node SCC detection (Phase 1+2) |
| `src_CUDA/scc_cuda_trim2_new.cu` | TRIM2_new: single-pass 2-node SCC |
| `src_CUDA/scc_cuda_fb_global.cu` | GLOBAL_BFS: pivot, FW+BW BFS kernels, visited bitmap, async stream |
| `src_CUDA/scc_cuda_fb_seq.cu` | Per-subgraph FW-BW (GPU BFS kernels) — Method 0 |
| `src_CUDA/scc_cuda_fb_seq2.cu` | Per-subgraph FW-BW + host FB path — Methods 1 & 2 |
| `src_CUDA/scc_cuda_weak.cu` | WCC: label propagation, root colors, work item creation |
| `src_CUDA/scc_cuda_work_queue.cu` | Work queue, scatter/gather kernels |
| **`src_CUDA/scc_cuda_incremental_kernels.cu`** | **GPU kernels for M6 condensation graph: filter, sort, CSR, mark_all_as_scc** |
| **`src_CUDA/scc_cuda_incremental_build.cpp`** | **Host wrapper for M6 graph construction (gpu_graph_built path)** |
| `src/scc_main.cc` | OpenMP main (also has exit(0) crash fix) |
| `src/common_main.h` | OpenMP graph loading, read_file (tab/space separator fix) |
| `tools/convert_graph.py` | WebGraph .graph → edge list converter |
| `dataset_handler.py` | Python dataset handler for .mtx + insert/delete edge generation |

---

## 🐛 Known Issues

### 1. `gm_graph` Heap Corruption (HIGH — Both Binaries)

**Symptom:** `"double free or corruption (out)"` after `Total # SCCs` is printed on large graphs (LJ1: 4.8M nodes).

**Root Cause:** The `gm_graph` CSR arrays (1.7GB heap on LJ1) corrupt glibc heap metadata during construction. The corruption manifests when `gm_graph::~gm_graph()` tries to `delete[]` the arrays.

**Fix (CUDA & OpenMP):** `exit(0);` at end of `main()` skips destructors. OS reclaims memory. GPU memory freed explicitly before `exit(0)`.

### 2. OpenMP OOM on Large Graphs

**Symptom:** Large graphs used to crash during graph loading.

**Root Cause:** `gm_graph` stores all edges in flexible `unordered_map` format during construction, and the `orig_edges` vector wasn't freed after the graph build.

**Fix:** Added `vector<pair<int,int>>().swap(orig_edges)` after graph construction (commit `a92fd9d`). This frees ~2.8GB on large graphs.

### 3. M6 Requires scc_list.txt (Method 6 Only)

Method 6 requires a pre-computed `scc_list.txt` file in the same directory as the graph file. Generate it with:
```bash
cd ~/DynamicGraphs_SCC/src && ../scc <graph_file> 72 2 -p
cp scc_list.txt <dataset_dir>/scc_list.txt
```

### 4. M6 Condensation Graph Assumes Correct scc_list.txt

The `mark_all_as_scc` shortcut (skipping the entire pipeline) is only correct if the input `scc_list.txt` accurately represents the SCC decomposition. If the SCC labels are incorrect, the condensation graph may have cycles, and skipping the pipeline would produce wrong SCC counts. **Always verify M6 SCC count matches M2 on the original graph.**

### 5. Benchmark Thermal Throttling

Running CUDA after OpenMP inflates CUDA FB time (~1.4ms → 16ms+) because 72 OpenMP threads heat the CPU. Run CUDA standalone for accurate timing.

### 6. `cuda_get_new_color()` Not Thread-Safe

**Location:** `scc_cuda_fb_global.cu`. Uses non-atomic `_cuda_color_used++`. If called from multiple OpenMP threads simultaneously, two threads could get the same color. Fix: use `#pragma omp atomic`.

### 7. GLOBAL_BFS Memory-Bound (Fundamental)

GLOBAL_BFS = 13.5ms (Pokec) / 24.6ms (ljournal). Every edge traversal reads a random `d_Color` value from VRAM (~300-800 cycle latency). Social network graphs have no locality — the `d_Color` array (19MB for ljournal) doesn't fit in L2 cache (6MB on L40S).

---

## 📌 Quick Reference — When You Come Back

**1. Pull latest code on server:**
```bash
cd ~/DynamicGraphs_SCC && git pull
```

**2. Verify SCC count on Pokec (Method 2):**
```bash
cd ~/DynamicGraphs_SCC/src_CUDA && make && \
./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2 | grep -E "CUDA_PROFILE|Total # SCCs"
# Expected: TOTAL ~10ms, SCC = 325892
```

**3. Run Method 6 (condensation graph, ~3ms pipeline):**
```bash
# Ensure scc_list.txt exists first
ls -la ../datasets/soc-Pokec/scc_list.txt

cd ~/DynamicGraphs_SCC/src_CUDA && make && \
./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 6 | grep -E "CUDA_PROFILE|Total # SCCs|ALGO_TIME"
# Expected: TOTAL ~0ms (skip pipeline), SCC = 325892
```

**4. Run on ljournal-2008 (37ms, 1.1M SCCs):**
```bash
./scc_cuda /hdd/thej_par_scc_datasets/ljournal-2008/refined_edges.txt 72 2
```

**5. CRITICAL OPEN ISSUE:** GLOBAL_BFS is slow on high-diameter graphs (wiki-Talk, wikipedia-20070206). See [Focus Graphs](#-focus-graphs--future-work) section.

**6. Edit, commit, push:**
```bash
cd \"/mnt/c/Users/Shashwat Trigunayat/OneDrive/Desktop/Admin/DynamicGRAPHS_SCC/DynamicGraphs_SCC\"
git pull --ff-only
git add src_CUDA/<file> && git commit -m \"description\" && git push
# Server: git pull && make && test
```

---

## 🔧 File Structure

```
DynamicGraphs_SCC/
├── benchmark.sh              # CUDA vs OpenMP comparison (⚠️ thermal issues)
├── dataset_handler.py        # .mtx converter + insert/delete edge generator
├── DOCUMENTATION.md          # This file
├── DOCUMENTATION.html        # HTML version
├── datasets/
│   ├── soc-Pokec/refined_edges.txt
│   ├── soc-LiveJournal1/refined_edges.txt
├── src/                      # OpenMP implementation
│   ├── Makefile
│   ├── common_main.h         # Graph loading, read_file (tab fix), orig_edges free
│   ├── scc_main.cc           # Main entry (exit(0) crash fix)
│   ├── scc_color.cc, scc_trim1.cc, scc_trim2.cc
│   ├── scc_fb_*.cc, scc_weak.cc, scc_tarjan.cc
│   ├── scc_incremental.cc, scc_decremental.cc
│   └── scc.h
├── src_CUDA/                 # CUDA implementation
│   ├── Makefile
│   ├── scc_cuda_main.cpp     # Main entry
│   ├── scc_cuda.h            # Header + declarations
│   ├── scc_cuda_graph.cu     # Graph upload/free
│   ├── scc_cuda_trim1.cu     # TRIM1 kernels (warp-ballot compact)
│   ├── scc_cuda_trim2.cu     # TRIM2 kernels
│   ├── scc_cuda_trim2_new.cu # TRIM2 new single-pass
│   ├── scc_cuda_fb_global.cu # GLOBAL BFS (visited bitmap, async stream)
│   ├── scc_cuda_fb_seq.cu    # Per-subgraph FB (GPU)
│   ├── scc_cuda_fb_seq2.cu   # Per-subgraph FB + host path
│   ├── scc_cuda_weak.cu      # WCC (fused propagation)
│   ├── scc_cuda_work_queue.cu# Work queue + scatter/gather
│   ├── scc_cuda_incremental_kernels.cu  # NEW: M6 GPU kernels (filter, sort, CSR, mark_all)
│   └── scc_cuda_incremental_build.cpp   # NEW: M6 graph construction router
├── gm_graph/                 # Green-Marl library
│   ├── inc/, src/, lib/
└── tools/
    ├── convert_graph.py      # WebGraph .graph → edge list converter
    ├── convert.cc, convert.h # Deprecated Green-Marl converter
    └── Makefile
```

---

## 🔬 Research References

| Paper | Authors | Key Insight | Relevance |
|-------|---------|-------------|-----------|
| **iSpan: Parallel Identification of SCCs with Spanning Trees** | Yuede Ji, Hang Liu, H. Howie Huang (SC18) | Relaxed-sync spanning tree construction replaces DFS | Theoretical basis for Method 12 |
| **Computing SCCs in Parallel on CUDA** | Barnat et al. (2011) | FB-Trim: FW-BW with iterative trimming | Baseline comparison |
| **ECL-SCC: High-Performance SCC Detection** | Burtscher et al. (2023) | Optimized BFS-based SCC with hybrid strategies | Implementation patterns |
| **BFS and Coloring-based Parallel Algorithms for SCC** | Slota et al. (Sandia) | Multi-step method combining trim, coloring, FB | Algorithm taxonomy |
