#!/bin/bash
# Topology optimization benchmark sweep: pipeline_bench -d top_opt over threads x variants.
#
# Usage: scripts/run_benchmark.sh [MATRIX_DIR] [RHS_DIR] [MAX_REPS]
#
# Environment overrides:
#   THREADS   space separated thread counts (default: 1 2 4 8 16 24)
#   VARIANTS  pattern variants to sweep     (default: "source union")
#   NUMA      set to 0 to disable numactl pinning
#   BUILD     build directory               (default: build-release)
#
# Three things this script deliberately enforces:
#
#   1. The thread list stops at 24. The target cluster is 4x Xeon Platinum 8160 with 24
#      cores per socket. Past 24 threads the numbers measure remote memory latency, not
#      scaling.
#
#   2. Runs are pinned with numactl. Without it the scheduler is free to migrate threads
#      across sockets mid-run.
#
#   3. It always runs a Release build. The build directory is configured with
#      -DCMAKE_BUILD_TYPE=Release every time, because a cached Debug build links
#      AddressSanitizer and LeakSanitizer discards the benchmark's entire stdout.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATRIX_DIR="${1:-$ROOT/data/top_opt_matrices_small_csr/}"
RHS_DIR="${2:-$ROOT/data/top_opt_rhs_small/}"
MAX_REPS="${3:-10}"

THREADS="${THREADS:-1 2 4 8 16 24}"
VARIANTS="${VARIANTS:-source union}"
NUMA="${NUMA:-1}"
BUILD="${BUILD:-$ROOT/build-release}"

BENCHMARK_EXE="$BUILD/benchmarks/pipeline_bench"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_FILE="top_opt_results_${TIMESTAMP}.txt"
CSV_FILE="top_opt_results_${TIMESTAMP}.csv"

for d in "$MATRIX_DIR" "$RHS_DIR"; do
    [ -d "$d" ] || { echo "error: no such directory: $d" >&2; exit 1; }
done

build() {
    echo "Building pipeline_bench (Release) in $BUILD..."
    cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=off -DENABLE_BENCHMARKS=on >/dev/null
    cmake --build "$BUILD" -j --target pipeline_bench
}

# Pin to one NUMA node so the sweep measures scaling rather than interconnect traffic.
runner() {
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
        for threads in $THREADS; do
            echo "==> variant=$variant threads=$threads"
            printf "\n========== variant %s | threads %s ==========\n" "$variant" "$threads" >> "$OUTPUT_FILE"

            runner "$BENCHMARK_EXE" -d top_opt \
                -n "top_opt" -x "$MATRIX_DIR" -y "$RHS_DIR" \
                -k "$MAX_REPS" -t "$threads" -u "$variant" \
                | tee -a "$OUTPUT_FILE"
        done
    done
}

{
    echo "Topology Optimization Benchmark Results"
    echo "Date:      $(date)"
    echo "Host:      $(hostname)"
    echo "Commit:    $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Matrices:  $MATRIX_DIR"
    echo "RHS:       $RHS_DIR"
    echo "Max reps:  $MAX_REPS"
    echo "Threads:   $THREADS"
    echo "Variants:  $VARIANTS"
    echo "NUMA pin:  $NUMA"
    if command -v numactl >/dev/null 2>&1; then numactl --hardware | head -3; fi
} > "$OUTPUT_FILE"

build
run_sweep

# The CSV header is the one the binary prints, so the columns cannot drift from the rows.
grep -m1 '^# CSV,' "$OUTPUT_FILE" | sed 's/^# CSV,//' > "$CSV_FILE" || true
grep '^CSV,' "$OUTPUT_FILE" | sed 's/^CSV,//' >> "$CSV_FILE" || true

echo "Benchmark finished."
echo "  log: $OUTPUT_FILE"
echo "  csv: $CSV_FILE"
