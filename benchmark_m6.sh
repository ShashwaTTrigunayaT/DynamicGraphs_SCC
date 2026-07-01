#!/bin/bash
#===============================================================================
# benchmark_m6.sh — CUDA vs OpenMP Method 6 (Condensation Graph) Comparison
#
# This script runs Method 6 on all available datasets for both CUDA and OpenMP,
# then prints a comparison table. Method 6 requires pre-computed scc_list.txt
# files in the scc_lists/ directory.
#
# Prerequisites:
#   1. Build both binaries: cd src && make && cd src_CUDA && make
#   2. Generate scc_list.txt files (run this script with --gen-scc-lists)
#   3. Have insert_edges files in dataset directories
#
# Usage:
#   ./benchmark_m6.sh [--gen-scc-lists] [dataset_name_filter]
#
# Examples:
#   ./benchmark_m6.sh                                          # Run on all datasets
#   ./benchmark_m6.sh soc-Pokec                                 # Run on soc-Pokec only
#   ./benchmark_m6.sh --gen-scc-lists                           # Generate scc_list.txt for all datasets
#   ./benchmark_m6.sh --gen-scc-lists soc-Pokec                 # Generate scc_list.txt for one dataset
#===============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMP_BIN="$SCRIPT_DIR/scc"
CUDA_BIN="$SCRIPT_DIR/src_CUDA/scc_cuda"
DATASETS_DIR="/hdd/thej_par_scc_datasets"
SCC_LISTS_DIR="$SCRIPT_DIR/scc_lists"
THREADS=72

# Parse args
GEN_SCC=0
DATASET_FILTER=""
for arg in "$@"; do
    if [ "$arg" = "--gen-scc-lists" ]; then
        GEN_SCC=1
    elif [[ "$arg" != --* ]]; then
        DATASET_FILTER="$arg"
    fi
done

# ==============================================================================
# Helper: format duration as ms with 2 decimal places
# ==============================================================================
format_ms() {
    printf "%.2f" "$1"
}

# ==============================================================================
# Step 1: Generate scc_list.txt files
# ==============================================================================
if [ "$GEN_SCC" = "1" ]; then
    echo "========================================================================="
    echo "  Generating scc_list.txt files from Method 2 (OpenMP)"
    echo "========================================================================="
    echo ""

    if [ ! -f "$OMP_BIN" ]; then
        echo "[ERROR] OpenMP binary not found at $OMP_BIN. Build it first."
        echo "  cd $SCRIPT_DIR/src && make"
        exit 1
    fi

    mkdir -p "$SCC_LISTS_DIR"

    for dir in "$DATASETS_DIR"/*/; do
        name=$(basename "$dir")
        [ "$name" = "indochina-2004" ] && { echo "  Skipping $name (no refined_edges.txt)"; continue; }
        [ "$name" = "syn_datasets" ] && { echo "  Skipping $name (synthetic)"; continue; }

        # Apply filter if specified
        [ -n "$DATASET_FILTER" ] && [[ "$name" != *"$DATASET_FILTER"* ]] && continue

        refined="$dir/refined_edges.txt"
        if [ ! -f "$refined" ]; then
            echo "  [SKIP] $name — no refined_edges.txt"
            continue
        fi

        out="$SCC_LISTS_DIR/$name.txt"
        if [ -f "$out" ]; then
            echo "  [EXISTS] $name → $out (delete to regenerate)"
            continue
        fi

        echo "  [GEN] $name → $out ..."
        cd "$SCRIPT_DIR" && "$OMP_BIN" "$refined" "$THREADS" 2 -p 2>&1 | grep -E "running_time|Total # SCCs|TRIM1|GLOBAL_BFS|WCC|FB"
        if [ -f "$SCRIPT_DIR/src/scc_list.txt" ]; then
            mv "$SCRIPT_DIR/src/scc_list.txt" "$out"
            echo "    ✅ $name: $(wc -l < "$out") lines"
        else
            echo "    ❌ $name: scc_list.txt not generated (check for crashes)"
        fi
        echo ""
    done

    echo "  Done generating SCC lists."
    echo "  Files stored in $SCC_LISTS_DIR"
    echo ""
    # Exit if only generating — don't run benchmarks
    # (but continue if there's a dataset filter too)
    [ "$GEN_SCC" = "1" ] && [ -z "$DATASET_FILTER" ] && exit 0
fi

# ==============================================================================
# Validate prerequisites
# ==============================================================================
if [ ! -f "$OMP_BIN" ]; then
    echo "[ERROR] OpenMP binary not found at $OMP_BIN"
    echo "  cd $SCRIPT_DIR/src && make"
    exit 1
fi
if [ ! -f "$CUDA_BIN" ]; then
    echo "[ERROR] CUDA binary not found at $CUDA_BIN"
    echo "  cd $SCRIPT_DIR/src_CUDA && make"
    exit 1
fi
if ! command -v bc &>/dev/null; then
    echo "[WARN] bc not found — install with: apt-get install bc"
fi

if [ ! -d "$DATASETS_DIR" ]; then
    echo "[ERROR] Datasets directory not found: $DATASETS_DIR"
    echo "  This script must be run on the server (monaachary.k@server)"
    exit 1
fi

echo "========================================================================="
echo "  CUDA vs OpenMP — Method 6 (Condensation Graph) Benchmark"
echo "  Threads: $THREADS | GPU: L40S | Date: $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================================="
echo ""

# ==============================================================================
# Find insert_edges files for each dataset
# ==============================================================================
declare -a BENCHMARKS

for dir in "$DATASETS_DIR"/*/; do
    name=$(basename "$dir")
    [ "$name" = "indochina-2004" ] && continue
    [ "$name" = "syn_datasets" ] && continue

    [ -n "$DATASET_FILTER" ] && [[ "$name" != *"$DATASET_FILTER"* ]] && continue

    # Check for scc_list.txt
    scc_list="$SCC_LISTS_DIR/$name.txt"
    if [ ! -f "$scc_list" ]; then
        echo "  [SKIP] $name — no scc_list.txt (run with --gen-scc-lists first)"
        continue
    fi

    # Find insert_edges files
    for insert_file in "$dir"/*insert_edges.txt; do
        [ -f "$insert_file" ] || continue
        batch=$(basename "$insert_file" | sed 's/_insert_edges.txt//')
        BENCHMARKS+=("$name|$batch|$insert_file")
    done

    # If no insert files, try the refined_edges.txt itself (edge case: M6 on full graph)
    # refined="$dir/refined_edges.txt"
    # if [ -f "$refined" ] && [ -z "$DATASET_FILTER" ]; then
    #     BENCHMARKS+=("$name|full|$refined")
    # fi
done

if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    echo "[ERROR] No benchmarks to run. Check that:"
    echo "  1. Datasets exist in $DATASETS_DIR"
    echo "  2. scc_list.txt files exist in $SCC_LISTS_DIR (run with --gen-scc-lists)"
    echo "  3. insert_edges files exist in dataset directories"
    exit 1
fi

echo "  Found ${#BENCHMARKS[@]} benchmarks to run"
echo ""

# ==============================================================================
# Results storage
# ==============================================================================
RESULTS_FILE="/tmp/m6_results_$(date +%s).csv"
echo "dataset,batch,omp_status,omp_runtime,omp_sccs,cuda_status,cuda_runtime,cuda_sccs" > "$RESULTS_FILE"

HEADER_PRINTED=0

# ==============================================================================
# Run benchmarks
# ==============================================================================
for bench in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name batch insert_path <<< "$bench"

    # --- Print header ---
    echo "────────────────────────────────────────────────────────────────────────"
    echo "  Dataset: $name  |  Batch: $batch"
    echo "  Insert file: $insert_path"
    echo "────────────────────────────────────────────────────────────────────────"
    echo ""

    OMP_STATUS="?"
    OMP_RUNTIME="?"
    OMP_SCCS="?"
    CUDA_STATUS="?"
    CUDA_RUNTIME="?"
    CUDA_SCCS="?"

    # --- Run OpenMP Method 6 ---
    echo "  [OMP M6] Running..."
    OMP_OUTPUT=$(cd "$SCRIPT_DIR" && timeout 300 "$OMP_BIN" "$insert_path" "$THREADS" 6 -d 2>&1) || true

    OMP_RUNTIME=$(echo "$OMP_OUTPUT" | awk -F'=' '/running_time/ {print $2}' | awk '{print $1}' | tail -1)
    OMP_SCCS=$(echo "$OMP_OUTPUT" | awk '/Total # SCCs/ {print $NF}')
    if echo "$OMP_OUTPUT" | grep -q "double free\|Aborted\|core dump\|Segmentation\|SEGV"; then
        OMP_STATUS="CRASHED"
    else
        OMP_STATUS="OK"
    fi

    # Print summary
    echo "  [OMP M6] Runtime: ${OMP_RUNTIME:-N/A}ms  |  SCCs: ${OMP_SCCS:-N/A}  |  Status: $OMP_STATUS"
    # Print phase timings if available
    echo "$OMP_OUTPUT" | grep -E "TRIM1|GLOBAL_BFS|WCC|FB" | sed 's/^/           /'
    echo ""

    # --- Run CUDA Method 6 ---
    echo "  [CUDA M6] Running..."
    CUDA_OUTPUT=$(cd "$SCRIPT_DIR" && timeout 300 "$CUDA_BIN" "$insert_path" "$THREADS" 6 2>&1) || true

    CUDA_ALGO_TIME=$(echo "$CUDA_OUTPUT" | awk -F': ' '/ALGO_TIME/ {print $2}' | awk '{print $1}')
    CUDA_SCCS=$(echo "$CUDA_OUTPUT" | awk '/Total # SCCs/ {print $NF}')
    if echo "$CUDA_OUTPUT" | grep -q "double free\|Aborted\|core dump\|Segmentation\|SEGV\|CUDA error"; then
        CUDA_STATUS="CRASHED"
    else
        CUDA_STATUS="OK"
    fi

    echo "  [CUDA M6] ALGO_TIME: ${CUDA_ALGO_TIME:-N/A}ms  |  SCCs: ${CUDA_SCCS:-N/A}  |  Status: $CUDA_STATUS"
    echo "$CUDA_OUTPUT" | grep -E "CUDA_PROFILE|condensation|graph loading" | sed 's/^/           /'
    echo ""

    # --- Store results ---
    omp_r="${OMP_RUNTIME:-?}"
    omp_s="${OMP_SCCS:-?}"
    cuda_r="${CUDA_ALGO_TIME:-?}"
    cuda_s="${CUDA_SCCS:-?}"
    echo "$name,$batch,$OMP_STATUS,$omp_r,$omp_s,$CUDA_STATUS,$cuda_r,$cuda_s" >> "$RESULTS_FILE"

    # --- Print live comparison ---
    echo "  ─────── LIVE COMPARISON ───────"
    printf "  %-12s %-15s %-15s %s\n" "Metric" "OpenMP M6" "CUDA M6" "Δ"
    echo "  ─────────────────────────────────────────────────────────────"
    if [ "$omp_r" != "?" ] && [ "$cuda_r" != "?" ]; then
        gap=$(echo "scale=2; $omp_r - $cuda_r" | bc -l 2>/dev/null || echo "?")
        pct=$(echo "scale=1; 100 * $gap / $omp_r" | bc -l 2>/dev/null || echo "?")
        printf "  %-12s %-15s %-15s %s\n" "Runtime (ms)" "$omp_r" "$cuda_r" "${gap}ms (${pct}%)"
    else
        printf "  %-12s %-15s %-15s\n" "Runtime (ms)" "${omp_r}" "${cuda_r}"
    fi
    if [ "$omp_s" != "?" ] && [ "$cuda_s" != "?" ]; then
        match="✅ MATCH" 
        [ "$omp_s" != "$cuda_s" ] && match="❌ MISMATCH"
        printf "  %-12s %-15s %-15s %s\n" "SCCs" "$omp_s" "$cuda_s" "$match"
    else
        printf "  %-12s %-15s %-15s\n" "SCCs" "${omp_s}" "${cuda_s}"
    fi
    echo ""

done

# ==============================================================================
# Summary Table
# ==============================================================================
echo ""
echo "========================================================================="
echo "  COMPARISON TABLE — Method 6 (Condensation Graph)"
echo "========================================================================="
printf "  %-20s %-8s %-12s %-12s %-10s %-10s\n" "Dataset" "Batch" "OMP (ms)" "CUDA (ms)" "Speedup" "SCC Match"
echo "  -----------------------------------------------------------------------"

while IFS=',' read -r ds batch omp_st omp_rt omp_sc cuda_st cuda_rt cuda_sc; do
    [ "$ds" = "dataset" ] && continue  # skip header
    speedup=""
    match=""
    if [ "$omp_rt" != "?" ] && [ "$cuda_rt" != "?" ] && [ "$(echo "$cuda_rt > 0" | bc -l)" = "1" ]; then
        speedup=$(echo "scale=2; $omp_rt / $cuda_rt" | bc -l)
        speedup="${speedup}x"
    fi
    if [ "$omp_sc" != "?" ] && [ "$cuda_sc" != "?" ]; then
        [ "$omp_sc" = "$cuda_sc" ] && match="✅" || match="❌"
    fi
    omp_display="${omp_rt}${omp_st}"
    [ "$omp_st" = "OK" ] && omp_display="$omp_rt" || omp_display="${omp_rt}[${omp_st}]"
    cuda_display="${cuda_rt}${cuda_st}"
    [ "$cuda_st" = "OK" ] && cuda_display="$cuda_rt" || cuda_display="${cuda_rt}[${cuda_st}]"
    printf "  %-20s %-8s %-12s %-12s %-10s %-10s\n" "$ds" "$batch" "$omp_display" "$cuda_display" "$speedup" "$match"
done < "$RESULTS_FILE"

echo "  -----------------------------------------------------------------------"
echo ""
echo "  Results saved to: $RESULTS_FILE"
echo "========================================================================="
echo ""
