#!/bin/bash
#===============================================================================
# benchmark.sh — Run OpenMP + CUDA side-by-side and print comparison table
#
# Usage:
#   ./benchmark.sh <dataset_path> [num_threads]
#
# Examples:
#   ./benchmark.sh datasets/soc-Pokec/refined_edges.txt 72
#   ./benchmark.sh datasets/it-2004/refined_edges.txt 72
#   ./benchmark.sh datasets/indochina-2004/refined_edges.txt 72
#
# Output: a formatted table comparing each phase (TRIM1, GLOBAL_BFS, TRIM12,
#         WCC, FB) across the two implementations, plus SCC count verification.
#===============================================================================
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [num_threads]"
    echo ""
    echo "Examples:"
    echo "  $0 datasets/soc-Pokec/refined_edges.txt 72"
    exit 1
fi

DATASET="$1"
THREADS="${2:-72}"

# Resolve dataset name for display
DS_NAME=$(basename "$(dirname "$DATASET")")
[ "$DS_NAME" = "." ] && DS_NAME="$DATASET"

# Resolve project root (script lives in project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Validate binaries exist ---
OMP_BIN="$SCRIPT_DIR/scc"
CUDA_BIN="$SCRIPT_DIR/src_CUDA/scc_cuda"

if [ ! -f "$OMP_BIN" ]; then
    echo "[ERROR] OpenMP binary not found at $OMP_BIN"
    echo "        Build it first: cd src && make"
    exit 1
fi

if [ ! -f "$CUDA_BIN" ]; then
    echo "[ERROR] CUDA binary not found at $CUDA_BIN"
    echo "        Build it first: cd src_CUDA && make"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "[ERROR] Dataset not found: $DATASET"
    exit 1
fi

# Check for bc (used for gap computation)
if ! command -v bc &>/dev/null; then
    echo "[WARN] bc not found — gap/PCT columns will show '?'"
    echo "       Install bc: apt-get install bc (Ubuntu) or sudo yum install bc (CentOS)"
    echo ""
fi

# --- Header ---
echo ""
echo "========================================================================"
echo "  SCC Benchmark Comparison"
echo "========================================================================"
echo "  Dataset : $DS_NAME"
echo "  Threads : $THREADS"
echo "  Date    : $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================================="
echo ""

# --- Helper to extract a numeric value from text ---
# Usage: extract_float "text" "pattern_with_capture"
# e.g. extract_float "$output" 'TRIM1 phase: *\([0-9.]*\) ms'
extract_float() {
    local text="$1"
    local pattern="$2"
    echo "$text" | sed -n "s/.*$pattern.*/\1/p" | head -1
}

# --- Helper to extract key=value from CUDA_PROFILE line ---
extract_cuda_val() {
    local text="$1"
    local key="$2"
    echo "$text" | sed -n "s/.*$2=\([0-9.]*\)ms.*/\1/p" | head -1
}

# ==============================================================================
# 1) Run OpenMP
# ==============================================================================
echo "--- [1/2] Running OpenMP (Method 2, -d) ---"
echo ""

# Capture OpenMP output (stdout + stderr)
OMP_OUTPUT=$(cd "$SCRIPT_DIR/src" && "$OMP_BIN" "$DATASET" "$THREADS" 2 -d 2>&1) || true

# Parse OpenMP phases
# Format: \t[TRIM1 phase: 12.094000 ms]
OMP_TRIM1=$(echo "$OMP_OUTPUT" | sed -n '/TRIM1 phase/p' | head -1 | sed 's/.*: \([0-9.]*\) ms.*/\1/')
OMP_GLOBAL_BFS=$(echo "$OMP_OUTPUT" | sed -n 's/.*GLOBAL_BFS phase: \([0-9.]*\) ms.*/\1/p')
OMP_TRIM12=$(echo "$OMP_OUTPUT" | sed -n '/TRIM1 phase/p' | tail -1 | sed 's/.*: \([0-9.]*\) ms.*/\1/')
OMP_WCC=$(echo "$OMP_OUTPUT" | sed -n 's/.*WCC phase: \([0-9.]*\) ms.*/\1/p')
OMP_FB=$(echo "$OMP_OUTPUT" | sed -n 's/.*FB phase: \([0-9.]*\) ms.*/\1/p')
OMP_TOTAL=$(echo "$OMP_OUTPUT" | sed -n 's/.*running_time(ms)=\([0-9.]*\).*/\1/p')
OMP_SCC=$(echo "$OMP_OUTPUT" | sed -n 's/Total # SCCs = \([0-9]*\).*/\1/p')

# Fallback: if empty, try grep alternative
[ -z "$OMP_TRIM1" ] && OMP_TRIM1="?"
[ -z "$OMP_GLOBAL_BFS" ] && OMP_GLOBAL_BFS="?"
[ -z "$OMP_TRIM12" ] && OMP_TRIM12="?"
[ -z "$OMP_WCC" ] && OMP_WCC="?"
[ -z "$OMP_FB" ] && OMP_FB="?"
[ -z "$OMP_TOTAL" ] && OMP_TOTAL="?"
[ -z "$OMP_SCC" ] && OMP_SCC="?"

# Check if OpenMP crashed
if echo "$OMP_OUTPUT" | grep -q "double free\|Aborted\|core dumped"; then
    OMP_CRASH="CRASHED"
else
    OMP_CRASH="OK"
fi

echo "  OpenMP done. Total: ${OMP_TOTAL}ms  SCCs: ${OMP_SCC}  Status: ${OMP_CRASH}"
echo ""

# ==============================================================================
# 2) Run CUDA
# ==============================================================================
echo "--- [2/2] Running CUDA (Method 2) ---"
echo ""

CUDA_OUTPUT=$(cd "$SCRIPT_DIR/src_CUDA" && "$CUDA_BIN" "$DATASET" "$THREADS" 2 2>&1) || true

# Parse CUDA profile line
# Format: >>>>CUDA_PROFILE: TRIM1=5.31ms COMPACT_BUILD=0.04ms GLOBAL_BFS=23.20ms TRIM12=0.38ms WCC=3.76ms FB=2.10ms TOTAL=34.84ms
CUDA_PROFILE_LINE=$(echo "$CUDA_OUTPUT" | grep "CUDA_PROFILE:" | grep -v "STDERR" | head -1)

CUDA_TRIM1=$(echo "$CUDA_PROFILE_LINE" | sed 's/.*TRIM1=\([0-9.]*\)ms.*/\1/')
CUDA_GLOBAL_BFS=$(echo "$CUDA_PROFILE_LINE" | sed 's/.*GLOBAL_BFS=\([0-9.]*\)ms.*/\1/')
CUDA_TRIM12=$(echo "$CUDA_PROFILE_LINE" | sed 's/.*TRIM12=\([0-9.]*\)ms.*/\1/')
CUDA_WCC=$(echo "$CUDA_PROFILE_LINE" | sed 's/.*WCC=\([0-9.]*\)ms.*/\1/')
CUDA_FB=$(echo "$CUDA_PROFILE_LINE" | sed 's/.*FB=\([0-9.]*\)ms.*/\1/')
CUDA_TOTAL=$(echo "$CUDA_PROFILE_LINE" | sed 's/.*TOTAL=\([0-9.]*\)ms.*/\1/')
CUDA_SCC=$(echo "$CUDA_OUTPUT" | sed -n 's/Total # SCCs = \([0-9]*\).*/\1/p')

[ -z "$CUDA_TRIM1" ] && CUDA_TRIM1="?"
[ -z "$CUDA_GLOBAL_BFS" ] && CUDA_GLOBAL_BFS="?"
[ -z "$CUDA_TRIM12" ] && CUDA_TRIM12="?"
[ -z "$CUDA_WCC" ] && CUDA_WCC="?"
[ -z "$CUDA_FB" ] && CUDA_FB="?"
[ -z "$CUDA_TOTAL" ] && CUDA_TOTAL="?"
[ -z "$CUDA_SCC" ] && CUDA_SCC="?"

if echo "$CUDA_OUTPUT" | grep -q "double free\|Aborted\|core dumped"; then
    CUDA_CRASH="CRASHED"
else
    CUDA_CRASH="OK"
fi

echo "  CUDA done. Total: ${CUDA_TOTAL}ms  SCCs: ${CUDA_SCC}  Status: ${CUDA_CRASH}"
echo ""

# ==============================================================================
# 3) Comparison Table
# ==============================================================================
echo "========================================================================"
echo "  COMPARISON TABLE — $DS_NAME (${THREADS} threads)"
echo "========================================================================"
printf "  %-20s %-12s %-12s %s\n" "Phase" "OpenMP (ms)" "CUDA (ms)" "Gap"
echo "  ----------------------------------------------------------------------"

# For each phase, compute gap
# Gap = CUDA - OpenMP (negative = CUDA faster)
cmp_phase() {
    local label="$1"
    local omp="$2"
    local cuda="$3"
    local unit="$4"

    if [ "$omp" = "?" ] || [ "$cuda" = "?" ]; then
        printf "  %-20s %-12s %-12s %s\n" "$label" "$omp" "$cuda" "N/A"
    else
        local gap=$(echo "$cuda - $omp" | bc -l 2>/dev/null || echo "?")
        local gap_sign=""
        local faster=""
        if [ "$gap" != "?" ]; then
            if [ "$(echo "$gap > 0.01" | bc -l 2>/dev/null)" = "1" ]; then
                gap_sign="+"
                faster="OpenMP"
            elif [ "$(echo "$gap < -0.01" | bc -l 2>/dev/null)" = "1" ]; then
                gap_sign=""
                faster="CUDA"
            else
                faster="tie"
            fi
            local gap_pct=""
            if [ "$(echo "$omp > 0" | bc -l 2>/dev/null)" = "1" ]; then
                gap_pct=$(echo "scale=1; 100 * $gap / $omp" | bc -l 2>/dev/null)
                printf "  %-20s %-12s %-12s %s%sms (%s%%)\n" "$label" "${omp}${unit}" "${cuda}${unit}" "$gap_sign$gap" "$gap_pct"
            else
                printf "  %-20s %-12s %-12s %s%sms\n" "$label" "${omp}${unit}" "${cuda}${unit}" "$gap_sign$gap"
            fi
        else
            printf "  %-20s %-12s %-12s %s\n" "$label" "${omp}${unit}" "${cuda}${unit}" "?"
        fi
    fi
}

cmp_phase "TRIM1"          "$OMP_TRIM1" "$CUDA_TRIM1" ""
cmp_phase "GLOBAL_BFS"     "$OMP_GLOBAL_BFS" "$CUDA_GLOBAL_BFS" ""
cmp_phase "TRIM12"         "$OMP_TRIM12" "$CUDA_TRIM12" ""
cmp_phase "WCC"            "$OMP_WCC" "$CUDA_WCC" ""
cmp_phase "FB"             "$OMP_FB" "$CUDA_FB" ""
cmp_phase "TOTAL"          "$OMP_TOTAL" "$CUDA_TOTAL" ""

echo "  ----------------------------------------------------------------------"
printf "  %-20s %-12s %-12s\n" "SCC Count" "$OMP_SCC" "$CUDA_SCC"

# SCC match check
if [ "$OMP_SCC" != "?" ] && [ "$CUDA_SCC" != "?" ]; then
    if [ "$OMP_SCC" = "$CUDA_SCC" ]; then
        echo "  SCC Match : ✅ YES ($OMP_SCC == $CUDA_SCC)"
    else
        echo "  SCC Match : ❌ NO ($OMP_SCC vs $CUDA_SCC)"
    fi
else
    echo "  SCC Match : N/A"
fi

printf "  %-20s %-12s %-12s\n" "Crash?" "$OMP_CRASH" "$CUDA_CRASH"

echo "========================================================================="
echo ""

# --- Show any warnings/errors ---
if echo "$OMP_OUTPUT" | grep -qi "error\|warning: too many arguments"; then
    echo "[OMP warnings/errors]"
    echo "$OMP_OUTPUT" | grep -i "error\|warning" | head -5
    echo ""
fi

if echo "$CUDA_OUTPUT" | grep -qi "error"; then
    echo "[CUDA errors]"
    echo "$CUDA_OUTPUT" | grep -i "error" | head -5
    echo ""
fi
