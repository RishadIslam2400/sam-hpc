#!/bin/bash
# Topology optimization benchmark sweep.
#
# Usage: ./run_benchmark.sh [MATRIX_DIR] [RHS_DIR] [MAX_REPS]
#
# Environment overrides:
#   THREADS   space separated thread counts (default: 1 2 4 8 16 24)
#   VARIANTS  pattern variants to sweep     (default: "source union")
#   ROWSETS   row set variants to sweep     (default: "pattern")
#   NUMA      set to 0 to disable numactl pinning
#
# Two things this script deliberately enforces, both of which were wrong before:
#
#   1. The thread list stops at 24. The target cluster is 4x Xeon Platinum 8160 with 24
#      cores per socket. Every large array in this code is created with resize(n, 0) and so
#      is first-touched by the master thread, putting the whole working set on one NUMA
#      node. Past 24 threads the numbers measure remote memory latency, not scaling. The
#      previous list ran to 180 threads on a 96-core machine, which is not a scaling curve.
#
#   2. Runs are pinned with numactl. Without it the scheduler is free to migrate threads
#      across sockets mid-run, which is the most likely explanation for the non-monotonic
#      totals in the current Fig. 9a.

set -euo pipefail

MATRIX_DIR="${1:-$(dirname "$0")/top_opt_matrices_small_csr/}"
RHS_DIR="${2:-$(dirname "$0")/top_opt_rhs_small/}"
MAX_REPS="${3:-10}"

THREADS="${THREADS:-1 2 4 8 16 24}"
VARIANTS="${VARIANTS:-source union}"
ROWSETS="${ROWSETS:-pattern}"
NUMA="${NUMA:-1}"

BENCHMARK_EXE="./build/benchmarks/top_opt_bench"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_FILE="top_opt_results_${TIMESTAMP}.txt"
CSV_FILE="top_opt_results_${TIMESTAMP}.csv"

for d in "$MATRIX_DIR" "$RHS_DIR"; do
    [ -d "$d" ] || { echo "error: no such directory: $d" >&2; exit 1; }
done

build_if_needed() {
    if [ ! -x "$BENCHMARK_EXE" ]; then
        echo "Building SAM HPC..."
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=off -DENABLE_BENCHMARKS=on
        cmake --build build -j --target top_opt_bench
    else
        echo "Build artifacts found. Skipping build."
    fi
}

# Pin to one NUMA node so the sweep measures scaling rather than interconnect traffic.
runner() {
    local threads=$1; shift
    if [ "$NUMA" != "0" ] && command -v numactl >/dev/null 2>&1; then
        SAM_PINNED="numactl --cpunodebind=0 --membind=0" \
            numactl --cpunodebind=0 --membind=0 "$@"
    else
        [ "$NUMA" != "0" ] && echo "warning: numactl not found, running unpinned" >&2
        "$@"
    fi
}

run_sweep() {
    for variant in $VARIANTS; do
        for rowset in $ROWSETS; do
            for threads in $THREADS; do
                echo "==> variant=$variant rowset=$rowset threads=$threads"
                {
                    printf "\n========== variant %s | rowset %s | threads %s ==========\n" \
                        "$variant" "$rowset" "$threads"
                } >> "$OUTPUT_FILE"

                runner "$threads" "$BENCHMARK_EXE" \
                    -n "top_opt" -x "$MATRIX_DIR" -y "$RHS_DIR" \
                    -k "$MAX_REPS" -t "$threads" -u "$variant" -j "$rowset" \
                    | tee -a "$OUTPUT_FILE" \
                    | grep '^CSV,' >> "$CSV_FILE" || true
            done
        done
    done
}

{
    echo "Topology Optimization Benchmark Results"
    echo "Date:      $(date)"
    echo "Host:      $(hostname)"
    echo "Matrices:  $MATRIX_DIR"
    echo "RHS:       $RHS_DIR"
    echo "Max reps:  $MAX_REPS"
    echo "Threads:   $THREADS"
    echo "Variants:  $VARIANTS"
    echo "Row sets:  $ROWSETS"
    echo "NUMA pin:  $NUMA"
    command -v numactl >/dev/null 2>&1 && numactl --hardware | head -3
} > "$OUTPUT_FILE"

echo "pattern,variant,row_set,threads,map_nnz,map_min_ms,map_med_ms,solve_min_ms,solve_med_ms,total_min_ms,iters,residual,map_reps,solve_reps" > "$CSV_FILE"

build_if_needed
run_sweep

echo "Benchmark finished."
echo "  log: $OUTPUT_FILE"
echo "  csv: $CSV_FILE"
