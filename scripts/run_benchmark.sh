#!/bin/bash
# Topology optimization benchmark sweep.
#
# Usage: scripts/run_benchmark.sh [MATRIX_DIR] [RHS_DIR] [MAX_REPS]
#
# Experiment parameters:
#   THREADS   thread counts                  (default: 1 12 24 48 72 96)
#   VARIANTS  pattern variants to sweep      (default: "source union")
#   NUMA      set to 0 to run without the numactl memory policy
#   BUILD     build directory                (default: build-release)
#   CXX       C++ compiler, must be GCC 11+  (default: g++; CMakeLists.txt rejects others)
#
# Output goes to the current directory: top_opt_results_<timestamp>.txt, the full log with
# a machine/config header, and a .csv built from the benchmark's CSV rows. Move a run into
# results/ to keep it.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATRIX_DIR="${1:-$ROOT/data/top_opt_matrices_small_csr/}"
RHS_DIR="${2:-$ROOT/data/top_opt_rhs_small/}"
MAX_REPS="${3:-10}"
THREADS="${THREADS:-1 12 24 48 72 96}"
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

# Distinct (core, socket) pairs = physical cores, whatever the SMT sibling numbering.
PHYS_CORES=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
for threads in $THREADS; do
  if [ "$threads" -gt "$PHYS_CORES" ]; then
    echo "error: $threads threads requested but this machine has $PHYS_CORES physical cores;" \
      "set THREADS to counts <= $PHYS_CORES" >&2
    exit 1
  fi
done

if [ "$NUMA" != "0" ] && ! command -v numactl >/dev/null 2>&1; then
  echo "error: numactl not found; install it or set NUMA=0 to run without interleaving" >&2
  exit 1
fi

build() {
  echo "Building pipeline_bench (Release) in $BUILD..."
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
        -DENABLE_TESTS=off -DENABLE_BENCHMARKS=on >/dev/null
  cmake --build "$BUILD" -j --target pipeline_bench
}

runner() {
  if [ "$NUMA" != "0" ]; then
    SAM_PINNED="numactl --interleave=all" numactl --interleave=all "$@"
  else
    "$@"
  fi
}

run_sweep() {
  for variant in $VARIANTS; do
    for threads in $THREADS; do
      echo "==> variant=$variant threads=$threads"
      printf "\n========== variant %s | threads %s ==========\n" "$variant" "$threads" >> "$OUTPUT_FILE"

      runner "$BENCHMARK_EXE" \
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
  echo "Cores:     $PHYS_CORES physical, $(nproc --all) logical (SMT siblings unused)"
  echo "Memory:    $([ "$NUMA" != "0" ] && echo "numactl --interleave=all" || echo "default policy")"
  if command -v numactl >/dev/null 2>&1; then numactl --hardware | head -3; fi
} > "$OUTPUT_FILE"

build
run_sweep

grep -m1 '^# CSV,' "$OUTPUT_FILE" | sed 's/^# CSV,//' > "$CSV_FILE" || true
grep '^CSV,' "$OUTPUT_FILE" | sed 's/^CSV,//' >> "$CSV_FILE" || true

echo "Benchmark finished."
echo "  log: $OUTPUT_FILE"
echo "  csv: $CSV_FILE"