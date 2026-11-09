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
./build-release/tests/sam_integration [cd2d|top_opt|all]   # default cd2d
```

## Data

The matrix sets are text CSR files (`rows cols nnz`, then row pointers, column indices and
values, each on one line) and add up to tens of GB, so they live in `data/`, which git
ignores. Each directory holds `matrix_<i>.txt`, and the matching rhs directory `rhs_<i>.txt`:

```
data/top_opt_matrices_small_csr/      data/top_opt_rhs_small/     topology optimization, n = 132,300, 1..50
data/top_opt_matrices_paper_big_csr/  data/top_opt_paper_rhs_big/ topology optimization, n = 432,450, 1..31
data/cd2d_seq/                        data/cd2d_seq_rhs/          2D convection-diffusion, n = 202,500, 1..21
data/elast3d_seq/                     data/elast3d_seq_rhs/       3D nonlinear elasticity (Newton), n = 180,000, 1..21
```

`cd2d_seq` and `elast3d_seq` are generated, not shipped: `scripts/gen_cd2d.py` and
`scripts/gen_elast3d.py` (needs `pyamg`). `scripts/run_experiments.sh` generates either one if
it is missing. `scripts/gen_rail.py` converts Carr et al.'s Rail model-reduction MAT file into
the same format; that set is not part of the benchmark (see CLAUDE.md).

The tests find the data through `SAM_DATA_DIR`; configure with `-DSAM_DATA_DIR=<path>` to use
a copy stored somewhere else. The benchmarks take their paths on the command line instead.

## Problem description: 3D nonlinear elasticity with damage (for the manuscript)

The text below describes the `elast3d` set in the style of the manuscript, ready to paste
into the draft (math is LaTeX). Values marked † are measured from the generated sequence
(`data/elast3d_seq/sequence_info.txt`); regenerate with `scripts/gen_elast3d.py`, whose
defaults are exactly the parameters stated here.

---

**Nonlinear elasticity with progressive damage.** Our second test problem comes from the
large-deformation analysis of a heterogeneous elastic body that is progressively damaged as
the load is applied incrementally. Every load increment requires a Newton iteration, and
each Newton step needs the solution of a linear system with the current tangent stiffness
matrix. These matrices keep a fixed sparsity pattern, but their entries change for two
reasons. The deformation changes them throughout the body. Damage softens the most strained
elements by up to a factor of 20, which creates a localized, high-contrast change that grows
from one load level to the next, much like the evolving densities in topology optimization.

Let $\Omega_0 = (0,4)\times(0,1)\times(0,1)$ be the reference configuration of a cantilever
beam, clamped on $\Gamma_D = \{X_1 = 0\}$ and traction-free on the rest of the boundary
$\Gamma_N$. The displacement $u$ satisfies

$$
-\nabla_X \cdot P(F) = \lambda\, b \quad \text{in } \Omega_0, \qquad
u = 0 \ \text{on } \Gamma_D, \qquad
P(F)\,N = 0 \ \text{on } \Gamma_N,
$$

where $F = I + \nabla_X u$ is the deformation gradient, $N$ is the outward unit normal,
$b = -g\, e_3$ is a dead body load with $g = 4\times 10^{-3}$, and $\lambda \in (0,1]$ is the
load factor. The material is compressible neo-Hookean, with stored energy density and first
Piola-Kirchhoff stress

$$
W(F) = \frac{\mu}{2}\left(\operatorname{tr}(F^T F) - 3\right) - \mu \ln J
       + \frac{\Lambda}{2} (\ln J)^2, \qquad
P(F) = \frac{\partial W}{\partial F} = \mu\left(F - F^{-T}\right) + \Lambda \ln J\, F^{-T},
$$

with $J = \det F$ and Lamé parameters $\mu = E/(2(1+\nu))$ and
$\Lambda = E\nu/((1+\nu)(1-2\nu))$. The body is heterogeneous: Young's modulus is
$E = E_0 = 1$ in the matrix and $E = \kappa E_0$, with contrast $\kappa = 10$, inside 57
spherical inclusions of radius $0.15$ at random (seeded) positions. The inclusions occupy
17.2% of the volume. The Poisson ratio is $\nu = 0.3$ in both phases.

Damage is modelled by a scalar, isotropic, element-wise damage variable
$d_e \in [0, d_{\max}]$ that scales the stored energy, $W_e(F) = (1 - d_e)\, W(F)$. It is
driven by the history maximum $Y_e^{\max}$ of the element's undamaged strain-energy density
$Y_e = \frac{1}{|\Omega_e|}\int_{\Omega_e} W(F)\, dX$, through an exponential softening law

$$
d_e = \min\left\{ d_{\max},\ 1 - \exp\left( -\frac{\langle Y_e^{\max} - Y_0 \rangle_+}{Y_c} \right) \right\},
$$

with threshold $Y_0 = 3\times 10^{-3}$, softening scale $Y_c = 5\times 10^{-3}$ and
$d_{\max} = 0.95$, so a fully damaged element keeps 5% of its stiffness. Damage is
irreversible ($d_e$ never decreases). It is updated in a staggered fashion: $d$ is held fixed
while Newton solves a load level, and then updated from the converged displacement before
the next level.

We discretize with trilinear ($Q_1$) hexahedral finite elements on a uniform
$96 \times 24 \times 24$ grid ($h = 1/24$), with $2\times2\times2$ Gauss quadrature. The
clamped nodes are eliminated. This leaves $n = 3 \cdot 96 \cdot 25 \cdot 25 = 180{,}000$
unknowns: three displacement components per node, interleaved, with the nodes ordered
lexicographically. An interior node couples to its 27 neighbouring nodes, so a typical row
has 81 nonzeros; the matrix has $13{,}716{,}846$ nonzeros, 76.2 per row on average. The
discrete equilibrium equations are

$$
R(u;\lambda, d) = f_{\mathrm{int}}(u; d) - \lambda f_{\mathrm{ext}} = 0, \qquad
f_{\mathrm{int}}(u; d) = \frac{\partial}{\partial u} \sum_e \int_{\Omega_e} (1 - d_e)\, W(F(u))\, dX .
$$

The load is applied by continuation, $\lambda_j = j/10$ for $j = 1, \dots, 10$. At each
level we solve $R(u;\lambda_j, d) = 0$ by Newton's method with the current damage $d$,
starting from the previous level's solution. The Newton correction solves

$$
K(u^{(i)}; d)\, \delta u = -R(u^{(i)};\lambda_j, d), \qquad
K(u; d) = \frac{\partial f_{\mathrm{int}}}{\partial u}(u; d),
$$

with a backtracking (Armijo) line search on the total potential energy
$\Pi(u) = \sum_e \int_{\Omega_e} (1-d_e) W\, dX - \lambda_j\, f_{\mathrm{ext}}^T u$. Newton is
stopped when $\|R\|_2 \le 10^{-8} \lambda_j \|f_{\mathrm{ext}}\|_2$. With $d$ fixed, the
tangent $K$ is the Hessian of the stored energy, so it is symmetric, and it is positive
definite at every stable equilibrium along the path. It changes through the material and
geometric stiffness at every Newton step, and through the damage at every load level.

The sequence of linear systems is $A_k = K(u_k)$, $b_k = -R(u_k)$, taken over the Newton
iterates $u_k$ of all load levels in order. A tangent whose relative change from the
previously recorded one, $\|A - A_{\mathrm{prev}}\|_F / \|A_{\mathrm{prev}}\|_F$, is below
$10^{-4}$ is not recorded again; this happens only at the levels before damage starts,
where a level's first tangent repeats the previous level's last one. The resulting sequence
has 35† systems. The relative change $\|A_k - A_1\|_F/\|A_1\|_F$ grows to 0.394† at full load.
There, damage exceeds 0.1 in 13.0†% of the elements and 0.9 in 6.6†%, and the maximum
deflection is 2.56†, about 64% of the beam length. The target
matrix is $A_0 := A_1$, the first Newton system.

This problem has the properties that make preconditioner updates attractive:
- **Wide, fixed sparsity pattern.** Rows have about 76 nonzeros, so the simple pattern
  $S(A_0)$ gives an expensive map, and sparsification has something to remove.
- **Expensive preconditioner.** An incomplete factorization of a 3D elasticity operator
  generates substantial fill.
- **Localized, growing, high-contrast change.** Damage lowers element stiffness by up to
  20× in a growing band. A reused preconditioner degrades quickly, while the change is
  concentrated where a sparse map can represent it.
- **Ill-conditioning.** The beam is slender, so bending modes are far softer than membrane
  modes. Together with the inclusions and the damage contrast, this makes the problem hard
  for incomplete factorizations.

---

## Benchmarks

```bash
# per-phase timing across a thread sweep (-t 0), CSV output
./build-release/benchmarks/phase_bench -x data/cd2d_seq/ -k 10 -t 0 -p column,fixed

# pattern + map + GMRES solve; every pattern value and solver setting is a flag (-h)
./build-release/benchmarks/pipeline_bench \
    -x data/top_opt_matrices_small_csr/ -y data/top_opt_rhs_small/ -k 5 -t 16 -u union

# the full top_opt sweep: one process per thread count, threads pinned to physical cores,
# memory interleaved across NUMA nodes; log and CSV land in the current directory
scripts/run_benchmark.sh
```

Common options: `-x` matrix directory, `-y` rhs directory, `-k` maximum repetitions,
`-t` threads, `-a` / `-b` target and source matrix index, `-p` patterns (phase_bench),
`-u source|union` which matrix is sparsified, `-m left|right` which map (pipeline_bench).
`-h` prints the full list, but does not exit.

`run_benchmark.sh` sweeps 1, 12, 24, 48, 72 and 96 threads by default and refuses a count
above the machine's physical cores; set `THREADS` (and `VARIANTS`, `NUMA=0`, `BUILD`) in the
environment to change it. It needs `numactl` unless `NUMA=0`. `results/e1/summarize.py`
turns saved pipeline_bench logs into comparison tables.
