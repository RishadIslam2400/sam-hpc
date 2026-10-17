# SAM-HPC

Parallel C++ implementation of **Sparse Approximate Map (SAM)** preconditioner updates for
sequences of linear systems `A_k x_k = b_k`. Instead of building a new preconditioner for
every `A_k`, SAM computes a sparse map `N_k ≈ argmin ||N A_k − A_0||_F` onto a target matrix
`A_0` whose preconditioner `P_0` already exists, and applies `P_0 N_k` as the preconditioner
for `A_k`. This code builds the map's sparsity pattern from a sparsified source matrix, or a
sparsified union of source and target, and computes it in parallel with TBB.

Based on Carr, de Sturler & Gugercin, *Preconditioning Parametrized Linear Systems*,
SIAM J. Sci. Comput. 43(3), 2021. That paper's results use the column-wise map
`||A_k N − A_0||_F`; this code solves the row-wise form above, one small least squares
problem per row of `N_k`.

## Layout

```
include/sam_hpc/        header-only library, one CMake target: sam_hpc
  core/                 CSRMatrix, index types, SpGEMM, linear algebra, matrix reader
  sam/                  sparsity patterns, pattern extension (S^k), the map itself
  dense/                Householder QR for the per-row least squares problems
  krylov/               GMRES
  precond/              AMG, skyline LU, ILU(0)/ILU(k)/ILUTP, Jacobi, Gauss-Seidel
    coarsening/         Ruge-Stuben and aggregation
  util/                 CLI config, timing, thread control and pinning
benchmarks/             pipeline_bench (end to end), phase_bench (per-phase timing)
tests/                  unit tests (ctest) with small fixtures in tests/data;
                        tests/integration for runs against the full matrix sets
scripts/                run_benchmark.sh: pinned thread sweep, CSV output
results/                recorded experiment output and the scripts that summarize it
data/                   matrix sets, not in git (see below)
```

Everything is included as `"sam_hpc/<module>/<header>.hpp"`.

## Build and test

Requires a C++20 compiler, CMake ≥ 3.14, Intel TBB (developed against TBB 2020.3) and
Eigen 3.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release
```

The four unit tests (`csr_test`, `qr_test`, `sam_test`, `solver_test`) run under `ctest`.
For a correctness run under AddressSanitizer and UBSan, use a Debug build:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-debug
```

Take timings from Release builds only. Debug links AddressSanitizer, which distorts timings
and, through LeakSanitizer, can swallow a benchmark's output entirely. A build type already
in a build directory's cache survives a plain reconfigure, so pass `-DCMAKE_BUILD_TYPE`
explicitly.

`tests/integration` builds `sam_integration`, which runs SAM + GMRES on real matrix sets and
reports iteration counts rather than asserting. It is not part of `ctest`:

```bash
./build-release/tests/sam_integration [cd2d|top_opt|weather|all]   # default cd2d
```

## Data

The matrix sets are text CSR files (`rows cols nnz`, then row pointers, column indices and
values, each on one line) and add up to tens of GB, so they live in `data/`, which git
ignores. Each directory holds `matrix_<i>.txt`, and the matching rhs directory `rhs_<i>.txt`:

```
data/top_opt_matrices_small_csr/   data/top_opt_rhs_small/      topology optimization, n = 132,300
data/weather_matrices_18x27/       data/weather_rhs_18x27/      NWP (WxFactory),    n = 48,600
data/cd2d_data/                    data/cd2d_rhs/               2D convection-diffusion, n = 202,500
```

The tests find the data through `SAM_DATA_DIR`; configure with `-DSAM_DATA_DIR=<path>` to use
a copy stored somewhere else. The benchmarks take their paths on the command line instead.

## Benchmarks

```bash
# per-phase timing across a thread sweep (-t 0), CSV output
./build-release/benchmarks/phase_bench -x data/cd2d_data/ -k 10 -t 0 -p column,fixed

# pattern + map + GMRES solve; -d picks the preconditioner, GMRES settings and pattern list
./build-release/benchmarks/pipeline_bench -d top_opt \
    -x data/top_opt_matrices_small_csr/ -y data/top_opt_rhs_small/ -k 5 -t 16 -u union

# the full top_opt sweep: one process per thread count, threads pinned to physical cores,
# memory interleaved across NUMA nodes; log and CSV land in the current directory
scripts/run_benchmark.sh
```

Common options: `-x` matrix directory, `-y` rhs directory, `-k` maximum repetitions,
`-t` threads, `-a` / `-b` target and source matrix index, `-p` patterns (phase_bench),
`-u source|union` which matrix is sparsified, `-d top_opt|weather` (pipeline_bench).
`-h` prints the full list, but does not exit.

`run_benchmark.sh` sweeps 1, 12, 24, 48, 72 and 96 threads by default and refuses a count
above the machine's physical cores; set `THREADS` (and `VARIANTS`, `NUMA=0`, `BUILD`) in the
environment to change it. It needs `numactl` unless `NUMA=0`. `results/e1/summarize.py`
turns saved pipeline_bench logs into comparison tables.
