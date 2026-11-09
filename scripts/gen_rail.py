#!/usr/bin/env python3
"""Generate the Rail sequence: shifted systems from IRKA model reduction of the rail model.

IRKA needs, at every iteration j and for each of its r = 6 shifts sigma_ij,
  (sigma_ij E - A) x = B(:, 1)
which is system k = (j - 1) r + i of the sequence written here:
  matrix_k = sigma_ij E - A,   rhs_k = B(:, 1)  (the same right-hand side for every k)

A and E are symmetric, A is negative definite and E positive definite, and every shift is
real and positive, so each matrix is SPD. supp(A) is contained in supp(E), so every matrix
has the structure of E.
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def write_csr(path, A):
  A = A.tocsr()
  A.sort_indices()
  with open(path, "w") as fh:
    fh.write(f"{A.shape[0]} {A.shape[1]} {A.nnz}\n")
    fh.write(" ".join(map(str, A.indptr.tolist())) + "\n")
    fh.write(" ".join(map(str, A.indices.tolist())) + "\n")
    fh.write(" ".join(f"{v:.17g}" for v in A.data.tolist()) + "\n")


def write_vec(path, b):
  np.savetxt(path, b, fmt="%.17g")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--mat", required=True, help="Rail_IRKA_shifts_data.mat")
  p.add_argument("--out-matrix", required=True, help="directory for matrix_<k>.txt")
  p.add_argument("--out-rhs", required=True, help="directory for rhs_<k>.txt")
  p.add_argument("--steps", type=int, default=5,
                 help="IRKA iterations to write, initial shifts included (default 5; max 41)")
  p.add_argument("--dry-run", action="store_true", help="print the sequence, write nothing")
  return p.parse_args()


def main():
  a = parse_args()
  t_start = time.time()
  m = sio.loadmat(a.mat)
  for key in ("A", "E", "B", "S2_initial", "Siter"):
    if key not in m:
      sys.exit(f"error: {a.mat} has no variable '{key}'")

  A = sp.csr_matrix(m["A"], dtype=np.float64)
  E = sp.csr_matrix(m["E"], dtype=np.float64)
  b = np.asarray(m["B"].toarray() if sp.issparse(m["B"]) else m["B"], dtype=np.float64)[:, 0]
  n = A.shape[0]
  if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,):
    sys.exit(f"error: inconsistent sizes A {A.shape}, E {E.shape}, B {b.shape}")

  # Exactly the driver's shift table: sorted initial shifts, then the IRKA iterates.
  shifts = np.hstack([np.sort(np.asarray(m["S2_initial"], dtype=np.float64).ravel())[:, None],
                      np.asarray(m["Siter"], dtype=np.float64)])
  r, total_steps = shifts.shape
  if not 1 <= a.steps <= total_steps:
    sys.exit(f"error: --steps must be in 1..{total_steps}")
  if np.iscomplexobj(shifts) or np.any(shifts <= 0):
    sys.exit("error: expected real positive shifts")

  asym = max(abs(A - A.T).max(), abs(E - E.T).max())
  print(f"# n = {n}, nnz(A) = {A.nnz}, nnz(E) = {E.nnz}, r = {r}, max |X - X^T| = {asym:.1e}")

  if not a.dry_run:
    os.makedirs(a.out_matrix, exist_ok=True)
    os.makedirs(a.out_rhs, exist_ok=True)

  # Every matrix gets the structure supp(A) | supp(E), explicit zeros kept, so the whole
  # sequence shares one structure even if a shift cancels an entry exactly. A and E are
  # aligned to that skeleton once; each matrix is then sigma * eE - eA on it.
  struct = (abs(A) + abs(E)).tocsr()
  struct.sort_indices()
  ri, ci = np.repeat(np.arange(n), np.diff(struct.indptr)), struct.indices
  eA = np.asarray(A[ri, ci]).ravel()
  eE = np.asarray(E[ri, ci]).ravel()

  def assemble(sigma):
    return sp.csr_matrix((sigma * eE - eA, struct.indices.copy(), struct.indptr.copy()), shape=(n, n))

  K1 = assemble(shifts[0, 0])
  norm_K1 = sp.linalg.norm(K1)
  rows, prev = [], None
  for j in range(a.steps):
    for i in range(r):
      k = j * r + i + 1
      sigma = shifts[i, j]
      K = assemble(sigma)
      drift = sp.linalg.norm(K - K1) / norm_K1
      change = sp.linalg.norm(K - prev) / sp.linalg.norm(prev) if prev is not None else 0.0
      rows.append((k, j + 1, i + 1, sigma, drift, change))
      if not a.dry_run:
        write_csr(os.path.join(a.out_matrix, f"matrix_{k}.txt"), K)
        write_vec(os.path.join(a.out_rhs, f"rhs_{k}.txt"), b)
      prev = K
      print(f"  matrix_{k:<4d} step {j + 1:2d} shift {i + 1}  sigma {sigma:.6e}  drift {drift:.3e}", flush=True)

  summary = [
    f"# Rail IRKA shifted systems, generated {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t_start:.0f} s",
    f"# source {os.path.basename(a.mat)}: matrix_k = sigma E - A, rhs_k = B(:,1); Carr et al. arXiv:1601.05883 v1",
    f"# n = {n}, nnz = {K1.nnz} ({K1.nnz / n:.2f}/row), r = {r} shifts per IRKA step, {a.steps} of {total_steps} steps",
    f"# ||b|| = {np.linalg.norm(b):.6e}  (the original GMRES stops on the absolute ||r|| <= 1e-10)",
    "# k  irka_step  shift_index  sigma  drift_to_A1  change_from_prev",
  ] + [f"{k} {j} {i} {s:.10e} {d:.6e} {c:.6e}" for k, j, i, s, d, c in rows]
  print("\n".join(summary))
  if not a.dry_run:
    with open(os.path.join(a.out_matrix, "sequence_info.txt"), "w") as fh:
      fh.write("\n".join(summary) + "\n")


if __name__ == "__main__":
  main()
