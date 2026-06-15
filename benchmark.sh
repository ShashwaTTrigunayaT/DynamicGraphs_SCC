#!/bin/bash
#===============================================================================
# benchmark.sh — Run CUDA + OpenMP side-by-side and print comparison table
#
# Usage:
#   ./benchmark.sh <dataset_path> [num_threads]
#
# Order: CUDA (with warmup) → OpenMP
#   Running OpenMP (72 threads, CPU-intensive) just before CUDA can cause
#   GPU power-state lag and CPU thermal throttling, making CUDA appear 2x
#   slower than its true steady-state performance. A CUDA warmup run is
#   added to stabilize GPU clocks before the measured run.
#
# Examples:
#   ./benchmark.sh datasets/soc-Pokec/refined_edges.txt 72
#===============================================================================
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [num_threads]"
    echo "  $0 datasets/soc-Pokec/refined_edges.txt 72"
    exit 1
fi

DATASET="$1"
THREADS="${2:-72}"

# Resolve paths
DATASET="$(cd "$(dirname "$DATASET")" 2>/dev/null && pwd)/$(basename "$DATASET")"
DS_NAME=$(basename "$(dirname "$DATASET")")
[ "$DS_NAME" = "." ] && DS_NAME="$DATASET"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMP_BIN="$SCRIPT_DIR/scc"
CUDA_BIN="$SCRIPT_DIR/src_CUDA/scc_cuda"

# Validate
[ ! -f "$OMP_BIN" ]  && { echo "[ERROR] OpenMP binary not found at $OMP_BIN" ; exit 1; }
[ ! -f "$CUDA_BIN" ] && { echo "[ERROR] CUDA binary not found at $CUDA_BIN" ; exit 1; }
[ ! -f "$DATASET" ]  && { echo "[ERROR] Dataset not found: $DATASET" ; exit 1; }

if ! command -v bc &>/dev/null; then
    echo "[WARN] bc not found — gap/PCT columns will show '?'"
fi

# --- Header ---
echo ""
echo "========================================================================"
echo "  SCC Benchmark Comparison"
echo "========================================================================"
echo "  Dataset : $DS_NAME — Threads : $THREADS — $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================================="
echo ""

# ==============================================================================
# Warmup: run CUDA once to stabilize GPU clocks
# ==============================================================================
echo "--- Warmup: GPU clock stabilization + CPU cooldown ---"
(cd "$SCRIPT_DIR/src_CUDA" && "$CUDA_BIN" "$DATASET" "$THREADS" 2 >/dev/null 2>&1) || true
# Warmup's FB phase uses 72 OpenMP threads — let CPU cool before timed run
sleep 1
echo "  Done."
echo ""

# ==============================================================================
# 1) Run CUDA (warm GPU)
# ==============================================================================
echo "--- [1/2] Running CUDA (Method 2) ---"
echo ""

CUDA_OUTPUT=$(cd "$SCRIPT_DIR/src_CUDA" && "$CUDA_BIN" "$DATASET" "$THREADS" 2 2>&1) || true

CUDA_PROFILE_LINE=$(echo "$CUDA_OUTPUT" | grep ">>>CUDA_PROFILE:" | head -1)

if [ -n "$CUDA_PROFILE_LINE" ]; then
    CUDA_TRIM1=$(echo "$CUDA_PROFILE_LINE" | awk '{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="TRIM1") print a[2]}}' | sed 's/ms//')
    CUDA_GLOBAL_BFS=$(echo "$CUDA_PROFILE_LINE" | awk '{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="GLOBAL_BFS") print a[2]}}' | sed 's/ms//')
    CUDA_TRIM12=$(echo "$CUDA_PROFILE_LINE" | awk '{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="TRIM12") print a[2]}}' | sed 's/ms//')
    CUDA_WCC=$(echo "$CUDA_PROFILE_LINE" | awk '{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="WCC") print a[2]}}' | sed 's/ms//')
    CUDA_FB=$(echo "$CUDA_PROFILE_LINE" | awk '{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="FB") print a[2]}}' | sed 's/ms//')
    CUDA_TOTAL=$(echo "$CUDA_PROFILE_LINE" | awk '{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="TOTAL") print a[2]}}' | sed 's/ms//')
fi

CUDA_SCC=$(echo "$CUDA_OUTPUT" | awk '/Total # SCCs/ {print $NF}')

for v in CUDA_TRIM1 CUDA_GLOBAL_BFS CUDA_TRIM12 CUDA_WCC CUDA_FB CUDA_TOTAL CUDA_SCC; do
    [ -z "${!v}" ] && eval "$v=?"
done

if echo "$CUDA_OUTPUT" | grep -q "double free\|Aborted\|core dumped"; then
    CUDA_CRASH="CRASHED"
else
    CUDA_CRASH="OK"
fi

echo "  CUDA done. Total: ${CUDA_TOTAL}ms  SCCs: ${CUDA_SCC}  Status: ${CUDA_CRASH}"
echo ""

# ==============================================================================
# 2) Run OpenMP
# ==============================================================================
echo "--- [2/2] Running OpenMP (Method 2, -d) ---"
echo ""

OMP_OUTPUT=$(cd "$SCRIPT_DIR/src" && "$OMP_BIN" "$DATASET" "$THREADS" 2 -d 2>&1) || true

OMP_TRIM1=$(echo "$OMP_OUTPUT" | awk -F': ' '/TRIM1 phase/ && !done {print $2; done=1}' | awk '{print $1}')
OMP_GLOBAL_BFS=$(echo "$OMP_OUTPUT" | awk -F': ' '/GLOBAL_BFS phase/ {print $2}' | awk '{print $1}')
OMP_TRIM12=$(echo "$OMP_OUTPUT" | awk -F': ' '/TRIM1 phase/ {last=$2} END {print last}' | awk '{print $1}')
OMP_WCC=$(echo "$OMP_OUTPUT" | awk -F': ' '/WCC phase/ {print $2}' | awk '{print $1}')
OMP_FB=$(echo "$OMP_OUTPUT" | awk -F': ' '/FB phase/ {print $2}' | awk '{print $1}')
OMP_TOTAL=$(echo "$OMP_OUTPUT" | awk -F'=' '/running_time/ {print $2}' | awk '{print $1}')
OMP_SCC=$(echo "$OMP_OUTPUT" | awk '/Total # SCCs/ {print $NF}')

for v in OMP_TRIM1 OMP_GLOBAL_BFS OMP_TRIM12 OMP_WCC OMP_FB OMP_TOTAL OMP_SCC; do
    [ -z "${!v}" ] && eval "$v=?"
done

if echo "$OMP_OUTPUT" | grep -q "double free\|Aborted\|core dumped"; then
    OMP_CRASH="CRASHED"
else
    OMP_CRASH="OK"
fi

echo "  OpenMP done. Total: ${OMP_TOTAL}ms  SCCs: ${OMP_SCC}  Status: ${OMP_CRASH}"
echo ""

# ==============================================================================
# 3) Comparison Table
# ==============================================================================
echo "========================================================================"
echo "  COMPARISON TABLE — $DS_NAME (${THREADS} threads)"
echo "========================================================================"
printf "  %-20s %-12s %-12s %s\n" "Phase" "CUDA (ms)" "OpenMP (ms)" "Gap"
echo "  ----------------------------------------------------------------------"

cmp_phase() {
    local label="$1"
    local cuda="$2"
    local omp="$3"

    if [ "$cuda" = "?" ] || [ "$omp" = "?" ]; then
        printf "  %-20s %-12s %-12s %s\n" "$label" "$cuda" "$omp" "N/A"
        return
    fi

    local gap=$(echo "scale=3; $cuda - $omp" | bc -l 2>/dev/null || echo "?")
    [ "$gap" = "?" ] && { printf "  %-20s %-12s %-12s %s\n" "$label" "${cuda}" "${omp}" "?" ; return; }

    local gap_sign=""
    [ "$(echo "$gap > 0.005" | bc -l 2>/dev/null)" = "1" ] && gap_sign="+"

    local gap_pct=$(echo "scale=1; 100 * $gap / $omp" | bc -l 2>/dev/null)
    if [ -n "$gap_pct" ]; then
        printf "  %-20s %-10s %-10s  %s%sms (%s%%)\n" "$label" "${cuda}" "${omp}" "$gap_sign" "$gap" "$gap_pct"
    else
        printf "  %-20s %-10s %-10s  %s%sms\n" "$label" "${cuda}" "${omp}" "$gap_sign" "$gap"
    fi
}

# Note: column order is CUDA first, OpenMP second (matching execution order)
cmp_phase "TRIM1"          "$CUDA_TRIM1" "$OMP_TRIM1"
cmp_phase "GLOBAL_BFS"     "$CUDA_GLOBAL_BFS" "$OMP_GLOBAL_BFS"
cmp_phase "TRIM12"         "$CUDA_TRIM12" "$OMP_TRIM12"
cmp_phase "WCC"            "$CUDA_WCC" "$OMP_WCC"
cmp_phase "FB"             "$CUDA_FB" "$OMP_FB"
cmp_phase "TOTAL"          "$CUDA_TOTAL" "$OMP_TOTAL"

echo "  ----------------------------------------------------------------------"
printf "  %-20s %-12s %-12s\n" "SCC Count" "$CUDA_SCC" "$OMP_SCC"

if [ "$CUDA_SCC" != "?" ] && [ "$OMP_SCC" != "?" ]; then
    if [ "$CUDA_SCC" = "$OMP_SCC" ]; then
        echo "  SCC Match : ✅ YES ($CUDA_SCC == $OMP_SCC)"
    else
        echo "  SCC Match : ❌ NO ($CUDA_SCC vs $OMP_SCC)"
    fi
fi

printf "  %-20s %-12s %-12s\n" "Crash?" "$CUDA_CRASH" "$OMP_CRASH"
echo "========================================================================="
echo ""
