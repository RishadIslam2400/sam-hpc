# SAM-HPC

Parallel C++ implementation of **Sparse Approximate Map (SAM)** preconditioner updates for
sequences of linear systems `A_k x_k = b_k`. Instead of building a new preconditioner for
every `A_k`, SAM computes a sparse map `N_k ≈ argmin ||N A_k − A_0||_F` onto a target matrix
`A_0` whose preconditioner `P_0` already exists, and applies `P_0 N_k` as the preconditioner
for `A_k`. This code builds the map's sparsity pattern from a sparsified source matrix, or a
sparsified union of source and target, and computes it in parallel with TBB.

Based on Carr, de Sturler & Gugercin, *Preconditioning Parametrized Linear Systems*,
SIAM J. Sci. Comput. 43(3), 2021.

## Layout

```
include/sam_hpc/        header-only library, one CMake target: sam_hpc
  core/                 CSRMatrix, index types, SpGEMM, linear algebra, matrix reader
  sam/                  sparsity patterns, pattern extension (S^k), the map itself
  dense/                Householder QR for the per-row least squares problems
  krylov/               GMRES
  precond/              AMG, ILU(0)/ILU(k)/ILUTP, Jacobi, Gauss-Seidel
    coarsening/         Ruge-Stuben, smoothed aggregation
  util/                 CLI config, timing, thread control
benchmarks/             pipeline_bench (end to end), phase_bench (per-phase timing)
tests/                  unit tests (ctest) with small fixtures in tests/data;
                        tests/integration for runs against the full matrix sets
scripts/                run_benchmark.sh: pinned thread sweep, CSV output
results/                recorded experiment output and the scripts that summarize it
data/                   matrix sets, not in git (see below)
```

## Build and test

Requires a C++20 compiler, CMake ≥ 3.14, oneTBB and Eigen 3.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release
```

Take timings from Release builds only. Debug links AddressSanitizer, which distorts timings
and, through LeakSanitizer, can swallow a benchmark's output entirely.

## Data

The matrix sets are text CSR files (`rows cols nnz`, then row pointers, column indices and
values, each on one line) and add up to tens of GB, so they live in `data/`, which git
ignores:

```
data/top_opt_matrices_small_csr/   data/top_opt_rhs_small/      topology optimization, n = 132,300
data/weather_matrices_18x27/       data/weather_rhs_18x27/      NWP (WxFactory),    n = 48,600
data/cd2d_data/                    data/cd2d_rhs/               2D convection-diffusion, n = 202,500
```

Configure with `-DSAM_DATA_DIR=<path>` to use a copy stored somewhere else.

## Benchmarks

```bash
# per-phase timing across a thread sweep (-t 0), CSV output
./build-release/benchmarks/phase_bench -x data/cd2d_data/ -k 10 -t 0 -p column,fixed

# pattern + map + GMRES solve; -d picks the preconditioner, GMRES settings and pattern list
./build-release/benchmarks/pipeline_bench -d top_opt \
    -x data/top_opt_matrices_small_csr/ -y data/top_opt_rhs_small/ -k 5 -t 16 -u union

# the full top_opt sweep, pinned to one NUMA node
scripts/run_benchmark.sh
```

Pass `-h` to either benchmark to list the options.
