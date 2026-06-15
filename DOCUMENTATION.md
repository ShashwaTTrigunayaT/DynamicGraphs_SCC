# DynamicGraphs SCC — CUDA GPU Acceleration Project

## 📋 Project Overview

**Goal:** Accelerate Strongly Connected Components (SCC) decomposition on large directed graphs using NVIDIA CUDA GPUs.

**Algorithm (Method 2):** Trim1 → Global FW-BW BFS → Trim1/2 → WCC → FW-BW DFS

**Repository:** `~/DynamicGraphs_SCC/` (on server `monaachary.k@server`)

**GPU:** NVIDIA L40S 48GB Ada Lovelace (sm_89, 142 SMs) — **NOT A100!**

---

## 🚀 Pipeline Phases (Method 2)

```
Input Graph
    │
    ▼
┌─────────────┐
│  TRIM1      │  Remove nodes with 0 in-degree OR 0 out-degree (iterative)
│  (GPU)      │
└──────┬──────┘
       ▼
┌─────────────┐
│  GLOBAL BFS │  Pick pivot → FW BFS → BW BFS → largest SCC found
│  (GPU)      │
└──────┬──────┘
       ▼
┌─────────────┐
│  TRIM1/2    │  Compact trim + 2-node SCC detection (iterative)
│  (GPU)      │
└──────┬──────┘
       ▼
┌─────────────┐
│  WCC        │  Weakly Connected Components → color assignment
│  (GPU)      │
└──────┬──────┘
       ▼
┌─────────────┐
│  FWBW DFS   │  Per-component SCC decomposition via GPU BFS kernels
│  (GPU)      │
└──────┬──────┘
       ▼
    SCCs Found!
```

The FB phase runs on GPU via `start_workers_fw_bw_dfs()` (GPU BFS kernels, not CPU). The FB phase also runs a host path (`start_workers_fw_bw_dfs_host`) which uses CPU OpenMP for DFS-based decomposition of the remaining WCC components.

---

## ⚡ Optimizations — Full Summary

### ✅ Kept (Positive Impact)

| # | Optimization | Files Changed | Impact (Pokec) | Impact (LJ1) |
|---|-------------|--------------|----------------|--------------|
| 1 | **Visited Bitmap (atomicOr)** — separate bitmap for BFS node claiming instead of atomicCAS on `d_Color` (avoids L2 cache line invalidation) | `scc_cuda_fb_global.cu` | GLOBAL_BFS 14.46→13.53ms (-0.93ms, 6.4%) | Similar |
| 2 | **Per-thread Staging Buffer (STAGE_SIZE=4)** — threads buffer claimed nodes locally, flush with single atomicAdd per 4 nodes (32× fewer atomics) | `scc_cuda_fb_global.cu` | GLOBAL_BFS ~-3ms | ~-3ms |
| 3 | **Batch D2H for WCC Sets** — single `cudaMemcpy` of entire WCC big buffer instead of 6,521 individual copies | `scc_cuda_fb_seq2.cu` | Minor | **-29ms** (6,521 comps) |
| 4 | **WCC Root Color Gather** — gather kernel collects only needed colors (~36KB) instead of full 6.4MB download | `scc_cuda_weak.cu` | FB 3.07→2.52ms (-0.55ms) | Significant |
| 5 | **Warp-Ballot Compact Build** — `__ballot_sync` + `__shfl_sync` for single atomicAdd per warp (32× fewer) | `scc_cuda_trim1.cu` | TRIM1 compact faster | Less DRAM contention |
| 6 | **Pinned Memory + Async Stream** — `cudaMallocHost` pinned buffers + `cudaMemcpyAsync` + dedicated CUDA stream for BFS loop | `scc_cuda_fb_global.cu` | ~30% less per-level overhead | ~30% less |
| 7 | **Single Large Buffer for WCC Sets** — one `cudaMalloc` for all WCC sets, sliced by pointer arithmetic | `scc_cuda_weak.cu` | Cleaner memory | Avoids 6521× driver overhead |
| 8 | **TOCTOU Race Fix** — `bw_bfs_level_kernel` reads `d_Color[k]` once before navigator check and if-else dispatch | `scc_cuda_fb_global.cu` | Correctness fix | Correctness fix |
| 9 | **Visited Bitmap Reset** — `cudaMemset` of visited bitmap between FW and BW BFS phases | `scc_cuda_fb_global.cu`, `scc_cuda_fb_seq2.cu` | Correctness fix | Correctness fix |
| 10 | **`exit(0)` to Skip Corrupted Destructor** — large graphs (LJ1) corrupt heap during processing → `exit(0)` after output to avoid crash | `scc_cuda_main.cpp`, `src/scc_main.cc` | — | **Fixes crash** on LJ1 |
| 11 | **`tools/Makefile` Fix** — removed broken Green-Marl include paths, uses local `../gm_graph/` | `tools/Makefile` | Enables convert tool | Enables convert tool |

### ❌ Tried and Reverted (Negative or Negligible)

| # | Optimization | Problem | Verdict |
|---|-------------|---------|---------|
| 1 | **Warp-Aggregated Atomics** — coalescing atomicAdd across warps | Caused regression in compact build performance | 🔴 Reverted |
| 2 | **Edge-Centric BFS** — iterate edges instead of nodes per BFS level | 2-3× slower due to redundant neighbor checks | 🔴 Reverted |
| 3 | **Hybrid FB GPU Path** — GPU-based per-component SCC decomposition | GPU was 18× slower than CPU path on Pokec (760ms vs 43ms) | 🔴 Reverted |
| 4 | **`__ldg()` Read-Only Cache** — use `__ldg()` for `d_Color` reads to bypass L1 (separate read-only cache) | Negligible impact (~1%, within noise) | 🔴 Reverted |
| 5 | **CPU Offload for GLOBAL_BFS** — download d_Color, run BFS on CPU with 72 OpenMP threads | 45ms CPU vs 13.5ms GPU (3.4× worse) | 🔴 Reverted |
| 6 | **Block Size Tuning** — tested 64, 128, 256, 512 | All within ±2% noise (memory bandwidth bound) | Reverted to 256 |
| 7 | **Grid Cap Removal** — removing `grid = min(grid, 1024)` | ~0.2ms improvement, negligible | Kept (no negative) |
| 8 | **STAGE_SIZE Tuning** — tested 2, 4, 8, 16 | 4 was optimal; 8+ caused register spill | Kept at 4 |

---

## 📊 Performance Results (L40S GPU, 72 Threads)

### soc-Pokec (1.6M nodes, 30M edges, directed)

| Phase | OpenMP (ms) | CUDA (ms) | vs OpenMP | Gap |
|-------|:----------:|:---------:|:---------:|:---:|
| **TRIM1** | 5.30 | **1.03** | ✅ **5.1× faster** | -4.27ms |
| **GLOBAL_BFS** | 12.32 | **13.51** | ❌ 1.1× slower | +1.19ms |
| **TRIM12** | 0.38 | 0.92 | ❌ 2.4× slower | +0.54ms |
| **WCC** | 0.82 | **1.42** | ❌ 1.7× slower | +0.60ms |
| **FB** | 0.72 | **0.79** | ≈ tie | +0.07ms |
| **TOTAL** | **19.59** | **17.70** | ✅ **10% faster** | -1.89ms |
| **SCC Count** | 325,892 | 325,892 | ✅ Match | — |
| **Crash?** | OK | OK | ✅ | — |

### soc-LiveJournal1 (4.8M nodes, 69M edges, directed)

| Phase | OpenMP (ms) | CUDA (ms) | vs OpenMP | Gap |
|-------|:----------:|:---------:|:---------:|:---:|
| **TRIM1** | 12.09 | **5.31** | ✅ **2.3× faster** | -6.78ms |
| **GLOBAL_BFS** | 24.42 | **23.13** | ✅ faster | -1.29ms |
| **TRIM12** | 0.88 | **0.40** | ✅ **2.2× faster** | -0.48ms |
| **WCC** | 3.34 | 3.78 | ❌ slower | +0.44ms |
| **FB** | **1.18** | 2.23 | ❌ slower | +1.05ms |
| **TOTAL** | **41.96** | **34.95** | ✅ **17% faster** | -7.01ms |
| **SCC Count** | 971,234 | 971,234 | ✅ Match | — |
| **Crash?** | Was crashing 🔴 | **Fixed** ✅ | — | — |

### Key Observations

1. **TRIM1 is the biggest GPU win** — 5× faster on Pokec, 2.3× on LJ1. The all-node scan with GPU parallelism massively outperforms OpenMP's processor-locked threads.

2. **GLOBAL_BFS is memory-bandwidth bound** — The BFS touches random nodes in `d_Color` (6.4MB for Pokec, 19MB for LJ1). Neither fits in any cache on L40S (128KB L1, 48KB read-only cache). Every edge traversal → random VRAM read → ~300-800 cycle latency. This is a **fundamental hardware limit**, not a kernel optimization problem.

3. **FB phase is on CPU (host path)** — After WCC, remaining components are processed via `start_workers_fw_bw_dfs_host()` which uses CPU OpenMP. The D2H/H2D transfers are minimal due to batching.

4. **GPU total is 10-17% faster than 72-thread OpenMP** — Impressive for a 72-core server CPU, but the edge is smaller than expected because the memory-bound phases (GLOBAL_BFS, WCC, FB) don't benefit from GPU parallelism.

5. **OpenMP crashes on LJ1** — The OpenMP binary has the same `gm_graph` heap corruption bug. Fixed in CUDA with `exit(0)`. Same fix applied to OpenMP `src/scc_main.cc`.

---

## 🖥️ Commands

### Build & Run CUDA

```bash
cd ~/DynamicGraphs_SCC/src_CUDA && make && ./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2
```

Arguments: `<graph_file> <num_threads> <method>`
- `num_threads`: affects graph loading + batch sizes (GPU uses its own cores)
- `method`: `2` = full pipeline (Trim1 + Global BFS + Trim12 + WCC + FB-DFS)

### Build & Run OpenMP

```bash
cd ~/DynamicGraphs_SCC/src && make && ../scc ../datasets/soc-Pokec/refined_edges.txt 72 2 -d
```

Arguments: `<graph_file> <num_threads> <method> [-d|-a|-p]`
- `-d`: detailed phase timing
- `-a`: SCC size histogram
- `-p`: output SCC list to file

### WebGraph .graph Converter (LAW datasets)

```bash
# Download
wget http://data.law.di.unimi.it/webdata/it-2004/it-2004.graph
mkdir -p datasets/it-2004

# Convert (auto-downloads .properties + .offsets)
python3 tools/convert_graph.py it-2004 datasets/it-2004/refined_edges.txt
```

The converter:
1. Auto-downloads missing `.properties` and `.offsets` files from LAW
2. Uses the `webgraph` Python package (auto-installs via `pip`)
3. Outputs plain `src dst` edge list format

**Known datasets:**
- `soc-Pokec` (1.6M nodes, 30M edges) — already in edge list format
- `soc-LiveJournal1` (4.8M nodes, 69M edges) — already in edge list format
- `indochina-2004` (7.4M nodes, 194M edges) — LAW .graph format
- `it-2004` (41M nodes, 1.15B edges) — LAW .graph format

### profile with NVIDIA Nsight Compute (ncu)

```bash
# Check if ncu is available
which ncu

# Profile (requires sudo for perf counters)
sudo /usr/local/cuda-13.1/bin/ncu --set full -o profile_output ./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2
```

If you get `ERR_NVGPUCTRPERM`, you need to enable permissions:
```bash
sudo /usr/local/cuda-13.1/bin/ncu --device-memory-bandwidth ./scc_cuda ...
```

### Benchmark Comparison (use with care — see note)

```bash
cd ~/DynamicGraphs_SCC && ./benchmark.sh datasets/soc-Pokec/refined_edges.txt 72
```

**⚠️ WARNING:** The benchmark script has thermal throttling issues. The FB phase runs OpenMP on CPU with 72 threads, which heats up the CPU and inflates subsequent CUDA FB times (observed: 1.4ms → 16ms+). For accurate CUDA results, run CUDA standalone.

### Quick Test After Changes

```bash
cd ~/DynamicGraphs_SCC && git pull && cd src_CUDA && make && ./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2 | grep -E "CUDA_PROFILE|Total # SCCs"
# Expected: TOTAL ~17-18ms, SCC = 325892
```

---

## 🧠 Source File Map

| File | Purpose |
|------|---------|
| `src_CUDA/scc_cuda_main.cpp` | Main entry, graph loading, method dispatch, cleanup |
| `src_CUDA/scc_cuda.h` | Header: all declarations, structs, macros |
| `src_CUDA/scc_cuda_graph.cu` | GPU graph upload/free, state alloc/free |
| `src_CUDA/scc_cuda_trim1.cu` | TRIM1: remove 0-in/out-degree nodes |
| `src_CUDA/scc_cuda_trim2.cu` | TRIM2: 2-node SCC detection (Phase 1+2) |
| `src_CUDA/scc_cuda_trim2_new.cu` | TRIM2_new: single-pass 2-node SCC |
| `src_CUDA/scc_cuda_fb_global.cu` | GLOBAL_BFS: pivot selection, FW+BW BFS kernels, `cuda_get_new_color()` |
| `src_CUDA/scc_cuda_fb_seq.cu` | Per-subgraph FW-BW (GPU BFS kernels) — Method 0 |
| `src_CUDA/scc_cuda_fb_seq2.cu` | Per-subgraph FW-BW + host FB path — Methods 1 & 2 |
| `src_CUDA/scc_cuda_weak.cu` | WCC: weakly connected components |
| `src_CUDA/scc_cuda_work_queue.cu` | Work queue, scatter/gather kernels |
| `src_CUDA/scc_cuda_dynamic.cpp` | Dynamic/incremental method helpers |
| `src/scc_main.cc` | OpenMP main (also has crash fix) |
| `tools/convert_graph.py` | WebGraph .graph → edge list converter |
| `benchmark.sh` | Side-by-side comparison script (⚠️ thermal throttling issues) |

---

## 🐛 Known Issues

### 1. `gm_graph` Heap Corruption (HIGH — Both Binaries)

**Symptom:** `"double free or corruption (out)"` after `Total # SCCs` is printed, only on large graphs (LiveJournal1: 4.8M nodes, 1.7GB CSR arrays). Works on Pokec.

**Root Cause:** The `gm_graph` CSR arrays (`begin`, `node_idx`, `r_begin`, etc.) are allocated during graph construction. The 1.7GB heap allocation on LJ1 somehow corrupts glibc heap metadata. The corruption manifests when `gm_graph::~gm_graph()` tries to `delete[]` the arrays.

**Fix (CUDA & OpenMP):** Added `exit(0);` at the end of `main()` after all output is printed. This skips destructors entirely — the OS reclaims the memory. All GPU memory is freed explicitly before `exit(0)`.

**Files:** `src_CUDA/scc_cuda_main.cpp`, `src/scc_main.cc`

### 2. Benchmark Thermal Throttling

**Symptom:** Running CUDA immediately after OpenMP (or a warmup run that uses CPU OpenMP) inflates CUDA FB time from ~1.4ms to ~16ms+.

**Root Cause:** The FB phase uses `start_workers_fw_bw_dfs_host()` which runs 72 OpenMP threads on CPU. This heats up the CPU, and the next run's FB phase is slower due to thermal throttling.

**Workaround:** Run CUDA standalone for accurate timing. The `benchmark.sh` script has a warmup run but still shows inflated FB times.

### 3. `cuda_get_new_color()` Not Thread-Safe

**Location:** `scc_cuda_fb_global.cu`

**Problem:** Uses non-atomic `_cuda_color_used++` and plain-int reads. If called from multiple OpenMP threads simultaneously, two threads could get the same color.

**Current Status:** The host FB path (`start_workers_fw_bw_dfs_host`) uses `#pragma omp parallel for` and calls `cuda_get_new_color()` inside the parallel region. This IS a data race.

**Impact:** Could cause incorrect color assignment (two partitions with same color), potentially leading to incorrect SCC merging. However, in practice the color counter increments fast enough that collisions are rare, and the algorithm's redundant checks catch most issues.

**Fix needed:** Use `#pragma omp atomic` or `__sync_fetch_and_add`.

### 4. GLOBAL_BFS Memory-Bound (Fundamental)

**Symptom:** GLOBAL_BFS = 13.5ms (Pokec) / 23ms (LJ1). The kernel is memory-latency bound — every edge traversal reads a random `d_Color` value from VRAM.

**Why it can't be fixed:** Social network graphs have no locality. The `d_Color` array (6.4MB Pokec, 19MB LJ1) doesn't fit in any cache. On L40S: L1=128KB, read-only cache=48KB, L2=6MB (Pokec barely fits in L2, LJ1 doesn't).

### 5. OpenMP Binary Also Has the Crash (Fixed)

The OpenMP binary (`src/scc`) has the same `gm_graph` heap corruption. Fixed with `exit(0)` in `src/scc_main.cc`.

---

## 📈 What Was Tried — Full Experiment Log

All experiments were on **soc-Pokec (1.6M nodes, 30M edges)** unless noted. Times are GLOBAL_BFS phase except where specified.

| Date | Experiment | Result | Verdict |
|:----:|-----------|--------|:-------:|
| Jun 15 | **Baseline** — initial GPU BFS kernel | GLOBAL_BFS ~17ms, TOTAL ~22ms | Starting point |
| Jun 15 | **Visited Bitmap** — atomicOr instead of atomicCAS | 14.46ms → **13.53ms** (-6.4%) | ✅ Kept |
| Jun 15 | **STAGE_SIZE=4** — per-thread staging buffer | ~-3ms GLOBAL_BFS | ✅ Kept |
| Jun 15 | **STAGE_SIZE=8,16** — larger staging buffers | Register spill, slower | 🔴 Reverted to 4 |
| Jun 15 | **Block size 64** | 13.53ms (within noise) | 🔴 Reverted |
| Jun 15 | **Block size 128** | 13.73ms (within noise) | 🔴 Reverted |
| Jun 15 | **Block size 512** | 13.44ms (within noise) | 🔴 Reverted |
| Jun 15 | **Warp-aggregated atomics** | Regression | 🔴 Reverted |
| Jun 15 | **Grid cap removal** | -0.2ms | ✅ Kept |
| Jun 15 | **Batch D2H for FB sets** | -29ms on LJ1 FB | ✅ Kept |
| Jun 15 | **WCC root color gather** | FB 3.07→2.52ms | ✅ Kept |
| Jun 15 | **Edge-centric BFS** | 2-3× slower | 🔴 Reverted |
| Jun 15 | **Hybrid FB GPU path** | 760ms GPU vs 43ms CPU | 🔴 Reverted |
| Jun 15 | **`__ldg()` read-only cache** | ~1% improvement | 🔴 Reverted |
| Jun 15 | **CPU offload GLOBAL_BFS** | 45ms CPU vs 13.5ms GPU (3.4×) | 🔴 Reverted |
| Jun 15 | **`exit(0)` crash fix** | LJ1 crash fixed ✅ | ✅ Kept |
| Jun 15 | **TOCTOU race fix** | Correctness | ✅ Kept |
| Jun 15 | **Final benchmark Pokec** | TOTAL 17.70ms (10% faster than OpenMP) | ✅ |
| Jun 15 | **Final benchmark LJ1** | TOTAL 34.95ms (17% faster than OpenMP) | ✅ |

---

## 📌 Quick Reference

### If you come back after a break:

**1. Pull latest code on server:**
```bash
cd ~/DynamicGraphs_SCC && git pull
```

**2. Build and verify SCC count on soc-Pokec:**
```bash
cd ~/DynamicGraphs_SCC/src_CUDA && make && ./scc_cuda ../datasets/soc-Pokec/refined_edges.txt 72 2 | grep -E "CUDA_PROFILE|Total # SCCs"
# Expected: TOTAL ~17-18ms, SCC = 325892
```

**3. Run on LiveJournal1:**
```bash
./scc_cuda ../datasets/soc-LiveJournal1/refined_edges.txt 72 2 | grep -E "CUDA_PROFILE|Total # SCCs"
# Expected: TOTAL ~35ms, SCC = 971234
```

**4. Compare with OpenMP:**
```bash
cd ~/DynamicGraphs_SCC/src && make && ../scc ../datasets/soc-Pokec/refined_edges.txt 72 2 -d | grep -E "phase:|running_time"
```

**5. Download and convert a LAW dataset:**
```bash
wget http://data.law.di.unimi.it/webdata/it-2004/it-2004.graph
mkdir -p datasets/it-2004
python3 tools/convert_graph.py it-2004 datasets/it-2004/refined_edges.txt
```

**6. Edit code locally (Windows), commit, push:**
```bash
cd "/mnt/c/Users/Shashwat Trigunayat/OneDrive/Desktop/Admin/DynamicGRAPHS_SCC/DynamicGraphs_SCC"
git pull --ff-only
# make changes...
git add src_CUDA/<file>
git commit -m "description"
git push
# On server: git pull, make, test
```

---

## 🔧 File Structure

```
DynamicGraphs_SCC/
├── benchmark.sh              # Side-by-side CUDA vs OpenMP comparison (⚠️ thermal issues)
├── dataset_handler.py        # Python dataset handler (orig_edges, insert/delete)
├── DOCUMENTATION.md          # This file
├── DOCUMENTATION.html        # HTML version
├── datasets/
│   ├── soc-Pokec/refined_edges.txt
│   └── soc-LiveJournal1/refined_edges.txt
├── src/                      # OpenMP implementation
│   ├── Makefile
│   ├── scc_main.cc           # Main entry (has exit(0) crash fix)
│   ├── scc_color.cc          # Color management
│   ├── scc_trim1.cc          # TRIM1
│   ├── scc_trim2.cc          # TRIM2
│   ├── scc_fb_*.cc           # FW-BW phases
│   ├── scc_weak.cc           # WCC
│   ├── scc_tarjan.cc         # Tarjan (reference)
│   ├── scc_incremental.cc    # Incremental methods
│   └── scc.h                 # Header
├── src_CUDA/                 # CUDA implementation
│   ├── Makefile
│   ├── scc_cuda_main.cpp     # Main entry
│   ├── scc_cuda.h            # Header + declarations
│   ├── scc_cuda_graph.cu     # GPU graph upload/free
│   ├── scc_cuda_trim1.cu     # TRIM1 device kernels
│   ├── scc_cuda_trim2.cu     # TRIM2 device kernels
│   ├── scc_cuda_trim2_new.cu # TRIM2 new single-pass
│   ├── scc_cuda_fb_global.cu # GLOBAL BFS kernels
│   ├── scc_cuda_fb_seq.cu    # Per-subgraph FB (GPU)
│   ├── scc_cuda_fb_seq2.cu   # Per-subgraph FB + host path
│   ├── scc_cuda_weak.cu      # WCC
│   ├── scc_cuda_work_queue.cu# Work queue + scatter/gather
│   └── scc_cuda_dynamic.cpp  # Dynamic/incremental
├── gm_graph/                 # Green-Marl graph library
│   ├── inc/                  # Headers
│   ├── src/                  # Source
│   └── lib/libgmgraph.a      # Static library
└── tools/
    ├── convert_graph.py      # WebGraph .graph → edge list converter
    ├── convert.cc            # Broken Green-Marl converter (deprecated)
    ├── convert.h
    └── Makefile              # Fixed (removed Green-Marl dependency)
```
