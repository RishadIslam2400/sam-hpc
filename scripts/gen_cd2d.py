#!/usr/bin/env python3
"""Generate the CD2D sequence: Newton Jacobians of a nonlinear 2D convection-diffusion problem.

PDE, on the unit square with homogeneous Dirichlet boundary conditions:
  -div((eta + gamma u^2) grad u) + r u_x + s u_y + t u = lambda f

discretized with second-order central finite differences on an N x N grid of interior
points (h = 1/(N+1), n = N^2, lexicographic ordering). The diffusion term is
conservative: the coefficient on the face between two nodes is the mean of k(u) = eta +
gamma u^2 at those nodes, so k_face = eta + gamma (u_P^2 + u_Q^2) / 2.

The nonlinearity is handled with damped Newton. System k of the sequence is the Newton
system at iterate u_k:
  matrix_k = J(u_k),   rhs_k = -F(u_k)

so every file is a linear system that a real Newton solve has to solve. J is a 5-point
stencil with the same structure at every step and values that change with u: the diffusion
coefficients through k(u), and the Newton terms k'(u) (u_Q - u_P) make J nonsymmetric
even before convection does.
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--grid", type=int, default=450, help="interior points per side N (n = N^2)")
  p.add_argument("--eta", type=float, default=1.0, help="linear diffusion coefficient")
  p.add_argument("--gamma", type=float, default=0.5, help="nonlinear diffusion coefficient (k = eta + gamma u^2)")
  p.add_argument("--r", type=float, default=200.0, help="convection velocity in x")
  p.add_argument("--s", type=float, default=100.0, help="convection velocity in y")
  p.add_argument("--t", type=float, default=0.0, help="reaction coefficient")
  p.add_argument("--f", type=float, default=3000.0, help="amplitude of the source term")
  p.add_argument("--load-steps", type=int, default=10, help="load continuation levels (1 = plain Newton)")
  p.add_argument("--newton-tol", type=float, default=1e-8, help="relative ||F|| at which Newton stops")
  p.add_argument("--max-newton", type=int, default=30, help="Newton steps per load level")
  p.add_argument("--min-change", type=float, default=1e-4,
                  help="skip a Jacobian whose relative change from the last written one is below this")
  p.add_argument("--max-systems", type=int, default=0, help="stop after this many systems (0 = no limit)")
  p.add_argument("--out-matrix", required=True, help="directory for matrix_<k>.txt")
  p.add_argument("--out-rhs", required=True, help="directory for rhs_<k>.txt")
  p.add_argument("--dry-run", action="store_true", help="report the sequence without writing files")
  return p.parse_args()


class Problem:
  def __init__(self, a):
    self.N = a.grid
    self.h = 1.0 / (a.grid + 1)
    self.eta, self.gamma, self.r, self.s, self.t = a.eta, a.gamma, a.r, a.s, a.t
    x = np.arange(1, self.N + 1) * self.h
    X, Y = np.meshgrid(x, x)
    self.f = a.f * np.exp(-((X - 0.35) ** 2 + (Y - 0.6) ** 2) / 0.05)

  # Compute the nonlinear diffusion coefficient k(u) at each grid point.
  def k(self, u):
    return self.eta + self.gamma * u * u

  # Convert one dimensional vector of interior unknowns into a two dimensional grid with
  # the boundary layer satisfying homogenous Dirichlet boundary condition (u = 0).
  def _pad(self, u):
    U = np.zeros((self.N + 2, self.N + 2))
    U[1:-1, 1:-1] = u.reshape(self.N, self.N)
    return U

  # Compute the nonlinear residual vector F(u) at a given solution vector u and
  # load parameter lam
  def residual(self, u, lam):
    U = self._pad(u)
    K = self.eta + self.gamma * U * U
    P = U[1:-1, 1:-1]  # interior grid points
    Kp = K[1:-1, 1:-1] # diffusion coefficients
    h2 = self.h * self.h
    F = np.zeros_like(P) # discrete residual vector
    for sl in ((slice(1, -1), slice(2, None)), (slice(1, -1), slice(0, -2)),
                (slice(2, None), slice(1, -1)), (slice(0, -2), slice(1, -1))):
      Q, Kq = U[sl], K[sl]
      # Update the residual based on the diffusion coefficient for each neighboring points
      F -= 0.5 * (Kp + Kq) * (Q - P) / h2
    # Update the residual based on the convection and reaction terms
    F += self.r * (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * self.h)
    F += self.s * (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * self.h)
    F += self.t * P - lam * self.f
    return F.ravel()

  # Construct the sparse Jacobian matrix J(u) containing the derivative of every
  # residual equation with respect to every unknown.
  def jacobian(self, u):
    N, h = self.N, self.h
    h2 = h * h
    U = self._pad(u)
    K = self.eta + self.gamma * U * U
    P = U[1:-1, 1:-1]  # interior grid points
    Kp = K[1:-1, 1:-1] # diffusion coefficients
    g = self.gamma
    diag = np.full((N, N), self.t) # diagonal Jacobian entry for each grid point
    nbrs = { # four neighbors with their corresponding convection term contribution
      "E": ((slice(1, -1), slice(2, None)), self.r / (2 * h)),
      "W": ((slice(1, -1), slice(0, -2)), -self.r / (2 * h)),
      "N": ((slice(2, None), slice(1, -1)), self.s / (2 * h)),
      "S": ((slice(0, -2), slice(1, -1)), -self.s / (2 * h)),
    }
    
    off = {}
    for name, (sl, conv) in nbrs.items():
      Q, Kq = U[sl], K[sl] # neighbor of every interior node (N X N)
      kf = 0.5 * (Kp + Kq) # face coefficient between P and Q
      diag += (kf - g * P * (Q - P)) / h2 # delta F_P / delta u_P 
      off[name] = (-(kf + g * Q * (Q - P)) / h2) + conv # delta F_P / delta u_Q
    idx = np.arange(N * N).reshape(N, N)
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    
    # Neighbours that are boundary nodes are Dirichlet zeros: drop those couplings.
    for name, di, dj in (("E", 0, 1), ("W", 0, -1), ("N", 1, 0), ("S", -1, 0)):
      i0, i1 = max(0, -di), N - max(0, di)
      j0, j1 = max(0, -dj), N - max(0, dj)
      r_ = idx[i0:i1, j0:j1]
      c_ = idx[i0 + di:i1 + di, j0 + dj:j1 + dj]
      rows.append(r_.ravel())
      cols.append(c_.ravel())
      vals.append(off[name][i0:i1, j0:j1].ravel())
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N * N, N * N))
    J.sort_indices()
    return J


# Write a sparse matrix to disk in CSR format
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


def main():
  a = parse_args()
  prob = Problem(a)
  n = prob.N * prob.N
  if not a.dry_run:
    os.makedirs(a.out_matrix, exist_ok=True)
    os.makedirs(a.out_rhs, exist_ok=True)

  u = np.zeros(n)
  written = []
  A_first = A_last = None
  norm_first = None
  k = 0
  t_start = time.time()
  done = False
  
  # Solve the non-linear PDE using load continuation and damped Newton iterations 
  for j in range(1, a.load_steps + 1):
    lam = j / a.load_steps    # gradually increasing the forcing
    F = prob.residual(u, lam) # start with the solution from the previous load step
    F0 = np.linalg.norm(lam * prob.f.ravel())
    for it in range(a.max_newton):
      fn = np.linalg.norm(F)
      # Convergence test
      if fn <= a.newton_tol * F0:
        break
      
      # Compute the Jacobian matrix for the current iterate and write it if the change
      # compared to the last written matrix is significant
      J = prob.jacobian(u)
      change = np.inf if A_last is None else spla.norm(J - A_last) / spla.norm(A_last)
      if change >= a.min_change:
        k += 1
        if A_first is None:
          A_first, norm_first = J, spla.norm(J) # record the first Jacobian matrix
        drift = spla.norm(J - A_first) / norm_first
        written.append((k, lam, it, fn / F0, drift, change))
        if not a.dry_run:
          write_csr(os.path.join(a.out_matrix, f"matrix_{k}.txt"), J)
          write_vec(os.path.join(a.out_rhs, f"rhs_{k}.txt"), -F)
        A_last = J
        print(f"  system {k:3d}: load {lam:.3f} newton {it:2d}  ||F||/||f|| {fn / F0:.3e}"
              f"  drift to A_1 {drift:.3e}  change {change:.3e}  max u {np.abs(u).max():.3f}",
              flush=True)
        if a.max_systems and k >= a.max_systems:
          done = True
          break
      
      # Newton step
      delta = spla.spsolve(J.tocsc(), -F)
      alpha = 1.0
      while True:
        u_new = u + alpha * delta
        F_new = prob.residual(u_new, lam)
        if np.linalg.norm(F_new) < fn or alpha < 1e-4:
          break
        alpha *= 0.5
      u, F = u_new, F_new
    else:
      print(f"warning: load {lam:.3f} did not converge in {a.max_newton} Newton steps", file=sys.stderr)
    if done:
      break

  asym = spla.norm(A_last - A_last.T) / spla.norm(A_last)
  summary = [
    f"# CD2D Newton sequence, generated {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t_start:.0f} s",
    f"# -div((eta + gamma u^2) grad u) + r u_x + s u_y + t u = lambda f on (0,1)^2, u = 0 on the boundary",
    f"# grid {prob.N} x {prob.N} (n = {n}), h = 1/{prob.N + 1}",
    f"# eta {a.eta}  gamma {a.gamma}  r {a.r}  s {a.s}  t {a.t}  f {a.f}"
    f"  load steps {a.load_steps}  newton tol {a.newton_tol}  min change {a.min_change}",
    f"# cell Peclet (eta only) {max(abs(a.r), abs(a.s)) * prob.h / (2 * a.eta):.3f}"
    f"   ||A - A^T||/||A|| of the last matrix {asym:.3e}",
    "# k  load  newton_step  rel_F  drift_to_A1  change_from_prev",
  ] + [f"{k} {lam:.4f} {it} {rf:.6e} {d:.6e} {c:.6e}" for k, lam, it, rf, d, c in written]
  print("\n".join(summary))
  if not a.dry_run:
    with open(os.path.join(a.out_matrix, "sequence_info.txt"), "w") as fh:
      fh.write("\n".join(summary) + "\n")


if __name__ == "__main__":
  main()
