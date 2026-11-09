#!/usr/bin/env python3
"""Generate the ELAST3D sequence: Newton tangents of a 3D nonlinear elasticity problem.

A cantilever [0, nx h] x [0, ny h] x [0, nz h] (h = 1/nz), clamped at x = 0, deforms under a
dead body load b = -lambda g e_z in a compressible neo-Hookean material

    W(F) = mu/2 (tr F^T F - 3) - mu ln J + lam/2 (ln J)^2,
    P(F) = mu (F - F^-T) + lam ln J F^-T,

discretized with trilinear (Q1) hexahedra on a uniform nx x ny x nz grid, 2x2x2 Gauss
quadrature, 3 displacement DOFs per node. The material is high contrast: spherical stiff
inclusions (Young's modulus `contrast` x the matrix's) at seeded random positions.

The load is applied by continuation, lambda = j / L for j = 1..L, with damped Newton at
each level. System k of the sequence is the Newton system at iterate u_k:

    matrix_k = K(u_k) = dR/du,   rhs_k = -R(u_k),   R(u) = f_int(u) - lambda f_ext

so every file is a linear system that a real Newton solve has to solve. K is symmetric (it
is the Hessian of the energy) and positive definite while the beam is stable; its structure
is fixed, and its values change through the deformation (geometric and material stiffness).

Ordering: free nodes (i = 1..nx, j = 0..ny, k = 0..nz; i = 0 is clamped and eliminated) are
numbered lexicographically, x fastest, with the 3 DOFs of a node interleaved (u, v, w):
    dof = 3 * ((i - 1) + nx * (j + (ny + 1) * k)) + c
so the node strides are 3 / 3 nx / 3 nx (ny + 1), exactly the top_opt layout. An interior row
has 27 nodes x 3 = 81 nonzeros.

The Newton systems are solved here with CG preconditioned by smoothed-aggregation AMG
(pyamg) using the six rigid-body modes, to a relative 1e-10.
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (sp.linalg.norm, eigsh)

try:
  import pyamg
except ImportError:  # pragma: no cover
  pyamg = None


def write_csr(path, A):
  A = A.tocsr()
  A.sort_indices()
  # repr() is the shortest string that round-trips a double, and much faster than %.17g.
  with open(path, "w") as fh:
    fh.write(f"{A.shape[0]} {A.shape[1]} {A.nnz}\n")
    fh.write(" ".join(map(str, A.indptr.tolist())) + "\n")
    fh.write(" ".join(map(str, A.indices.tolist())) + "\n")
    fh.write(" ".join(map(repr, A.data.tolist())) + "\n")


def write_vec(path, b):
  with open(path, "w") as fh:
    fh.write("\n".join(map(repr, b.tolist())) + "\n")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--grid", type=int, nargs=3, default=[96, 24, 24], metavar=("NX", "NY", "NZ"),
                 help="elements per direction (default 96 24 24: n = 180,000)")
  p.add_argument("--young", type=float, default=1.0, help="Young's modulus of the matrix phase")
  p.add_argument("--poisson", type=float, default=0.3, help="Poisson ratio (both phases)")
  p.add_argument("--contrast", type=float, default=10.0,
                 help="inclusion / matrix stiffness ratio (at 100 even tuned SA-AMG stalls)")
  p.add_argument("--inclusion-radius", type=float, default=0.15, help="inclusion radius (beam depth = 1)")
  p.add_argument("--inclusion-frac", type=float, default=0.2, help="target inclusion volume fraction")
  p.add_argument("--seed", type=int, default=1, help="seed for the inclusion positions")
  p.add_argument("--gravity", type=float, default=0.004,
                 help="body-force density at full load (with the default damage: ~50%% deflection)")
  p.add_argument("--load-steps", type=int, default=10, help="load-continuation levels L")
  p.add_argument("--newton-tol", type=float, default=1e-8, help="||R|| / ||lambda f_ext|| to stop Newton")
  p.add_argument("--newton-maxit", type=int, default=25)
  p.add_argument("--damage-y0", type=float, default=3e-3,
                 help="damage threshold on the element strain-energy density (inf = no damage)")
  p.add_argument("--damage-yc", type=float, default=5e-3,
                 help="damage growth scale: d = 1 - exp(-(Y_max - Y0) / Yc)")
  p.add_argument("--damage-max", type=float, default=0.95,
                 help="cap on d; the residual stiffness 1 - d keeps equilibrium solvable")
  p.add_argument("--min-change", type=float, default=1e-4,
                 help="skip a tangent whose relative change from the last written one is below this")
  p.add_argument("--chunk", type=int, default=8192, help="elements per assembly chunk (memory)")
  p.add_argument("--out-matrix", help="directory for matrix_<k>.txt")
  p.add_argument("--out-rhs", help="directory for rhs_<k>.txt")
  p.add_argument("--dry-run", action="store_true", help="solve and report, write nothing")
  p.add_argument("--check", action="store_true",
                 help="finite-difference check of the tangent on a small mesh, then exit")
  a = p.parse_args()
  if not a.check and not a.dry_run and not (a.out_matrix and a.out_rhs):
    p.error("--out-matrix and --out-rhs are required unless --dry-run or --check")
  return a


class Beam:
  """Mesh, material, and vectorized Q1 assembly of f_int and the tangent."""

  def __init__(self, a):
    self.nx, self.ny, self.nz = a.grid
    nx, ny, nz = self.nx, self.ny, self.nz
    self.h = 1.0 / nz
    h = self.h
    self.n = 3 * nx * (ny + 1) * (nz + 1)
    self.chunk = a.chunk

    def node_id(i, j, k):
      """Free node number of grid node (i, j, k); clamped nodes (i = 0) give -1."""
      return np.where(i == 0, -1, (i - 1) + nx * (j + (ny + 1) * k))

    ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
    self.ne = ex.size
    # Local node a = (ax, ay, az), ax fastest; reference coordinate xi_a = 2 a - 1.
    loc = np.array([(ax, ay, az) for az in (0, 1) for ay in (0, 1) for ax in (0, 1)])
    nodes = node_id(ex[:, None] + loc[None, :, 0], ey[:, None] + loc[None, :, 1],
                    ez[:, None] + loc[None, :, 2])
    edof = np.where(nodes[:, :, None] >= 0, 3 * nodes[:, :, None] + np.arange(3)[None, None, :], -1)
    self.edof = edof.reshape(self.ne, 24)                       # -1 = clamped
    self.edof_pad = np.where(self.edof >= 0, self.edof, self.n)  # clamped -> a zero pad slot

    # Gauss points and shape-function gradients: the same for every element (uniform grid).
    xi_l = loc * 2.0 - 1.0
    g = 1.0 / np.sqrt(3.0)
    gps = np.array([(x, y, z) for z in (-g, g) for y in (-g, g) for x in (-g, g)])
    self.G = np.empty((8, 8, 3))  # [q, a, J] = dN_a / dX_J at Gauss point q
    for q, xq in enumerate(gps):
      for a_ in range(8):
        s = xi_l[a_]
        f = 1.0 + s * xq
        self.G[q, a_, 0] = s[0] * f[1] * f[2] / 8.0 * (2.0 / h)
        self.G[q, a_, 1] = f[0] * s[1] * f[2] / 8.0 * (2.0 / h)
        self.G[q, a_, 2] = f[0] * f[1] * s[2] / 8.0 * (2.0 / h)
    self.wdet = (h ** 3) / 8.0  # quadrature weight 1 x det(dX/dxi)
    self.GG = np.einsum("qaJ,qbJ->qab", self.G, self.G)

    # Material: inclusion centres (beam depth 1), Lame parameters per element.
    rng = np.random.default_rng(a.seed)
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    r = a.inclusion_radius
    n_inc = max(0, int(round(a.inclusion_frac * Lx * Ly * Lz / (4.0 / 3.0 * np.pi * r ** 3))))
    centres = rng.uniform([0, 0, 0], [Lx, Ly, Lz], size=(n_inc, 3))
    xc = np.stack([(ex + 0.5) * h, (ey + 0.5) * h, (ez + 0.5) * h], axis=1)
    inside = np.zeros(self.ne, dtype=bool)
    for c in centres:
      inside |= np.sum((xc - c) ** 2, axis=1) <= r * r
    E = np.where(inside, a.young * a.contrast, a.young)
    nu = a.poisson
    self.mu0 = E / (2.0 * (1.0 + nu))
    self.lam0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # Damage d_e in [0, damage_max] scales the element's stored energy by (1 - d_e).
    self.d = np.zeros(self.ne)
    self.Ymax = np.zeros(self.ne)
    self.mu, self.lam = self.mu0.copy(), self.lam0.copy()
    self.inc_frac = inside.mean()
    self.n_inc = n_inc

    # External load per unit lambda: body force -g e_z, consistent nodal weights vol / 8.
    fe = np.zeros(self.n + 1)
    np.add.at(fe, self.edof_pad[:, 2::3].ravel(), -a.gravity * h ** 3 / 8.0)
    self.f_ext = fe[: self.n]

    # The CSR structure is fixed: map every (free, free) element entry to its slot once.
    rows = np.repeat(self.edof, 24, axis=1)
    cols = np.tile(self.edof, (1, 24))
    self.keep = (rows >= 0) & (cols >= 0)  # [e, 576]
    key = rows[self.keep].astype(np.int64) * self.n + cols[self.keep]
    del rows, cols
    ukeys, self.slot = np.unique(key, return_inverse=True)
    del key
    self.indices = (ukeys % self.n).astype(np.int32)
    counts = np.bincount(ukeys // self.n, minlength=self.n)
    self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    self.nnz = ukeys.size
    del ukeys
    self.kept_start = np.concatenate([[0], np.cumsum(self.keep.sum(axis=1))])

    # Free-node coordinates, in DOF order, for the rigid-body modes.
    fi, fj, fk = np.meshgrid(np.arange(1, nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    ids = node_id(fi, fj, fk).ravel()
    self.xyz = np.empty((ids.size, 3))
    self.xyz[ids] = np.stack([fi.ravel() * h, fj.ravel() * h, fk.ravel() * h], axis=1)

  def _kinematics(self, ue, q):
    """F, F^-1 and ln J at Gauss point q for a chunk of elements."""
    F = np.eye(3)[None] + np.einsum("eai,aJ->eiJ", ue, self.G[q])
    detF = np.linalg.det(F)
    if np.any(detF <= 0):
      raise FloatingPointError("element inverted (det F <= 0)")
    return F, np.linalg.inv(F), np.log(detF)

  def element_energy_density(self, u):
    """Undamaged strain-energy density W0(F) per element, averaged over its Gauss points."""
    up = np.append(u, 0.0)
    Y = np.zeros(self.ne)
    for e0 in range(0, self.ne, self.chunk):
      e1 = min(self.ne, e0 + self.chunk)
      ue = up[self.edof_pad[e0:e1]].reshape(-1, 8, 3)
      mu, lam = self.mu0[e0:e1], self.lam0[e0:e1]
      for q in range(8):
        F, _, lnJ = self._kinematics(ue, q)
        trC = np.einsum("eiJ,eiJ->e", F, F)
        Y[e0:e1] += (0.5 * mu * (trC - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2) / 8.0
    return Y

  def update_damage(self, u, y0, yc, dmax):
    """Staggered, irreversible damage update from the history maximum of Y (after a level)."""
    self.Ymax = np.maximum(self.Ymax, self.element_energy_density(u))
    d_new = np.minimum(dmax, 1.0 - np.exp(-np.maximum(0.0, self.Ymax - y0) / yc))
    self.d = np.maximum(self.d, d_new)
    self.mu = (1.0 - self.d) * self.mu0
    self.lam = (1.0 - self.d) * self.lam0

  def energy(self, u):
    """Stored energy sum_e int W(F) dX (no load term)."""
    up = np.append(u, 0.0)
    total = 0.0
    for e0 in range(0, self.ne, self.chunk):
      e1 = min(self.ne, e0 + self.chunk)
      ue = up[self.edof_pad[e0:e1]].reshape(-1, 8, 3)
      mu, lam = self.mu[e0:e1], self.lam[e0:e1]
      for q in range(8):
        F, _, lnJ = self._kinematics(ue, q)
        trC = np.einsum("eiJ,eiJ->e", F, F)
        total += self.wdet * np.sum(0.5 * mu * (trC - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2)
    return total

  def internal_force(self, u, tangent=False):
    """f_int(u), and with tangent=True also K(u) = d f_int / du as CSR."""
    up = np.append(u, 0.0)
    f = np.zeros(self.n + 1)
    data = np.zeros(self.nnz) if tangent else None
    eye3 = np.eye(3)
    for e0 in range(0, self.ne, self.chunk):
      e1 = min(self.ne, e0 + self.chunk)
      ue = up[self.edof_pad[e0:e1]].reshape(-1, 8, 3)
      mu, lam = self.mu[e0:e1], self.lam[e0:e1]
      fe = np.zeros((e1 - e0, 8, 3))
      Ke = np.zeros((e1 - e0, 8, 3, 8, 3)) if tangent else None
      for q in range(8):
        F, Finv, lnJ = self._kinematics(ue, q)    # Finv[e, J, i] = (F^-1)_{Ji}
        FinvT = np.transpose(Finv, (0, 2, 1))     # F^-T, [e, i, J]
        P = mu[:, None, None] * (F - FinvT) + (lam * lnJ)[:, None, None] * FinvT
        fe += self.wdet * np.einsum("eiJ,aJ->eai", P, self.G[q])
        if tangent:
          # A_iJkL = mu d_ik d_JL + (mu - lam ln J) Finv_Li Finv_Jk + lam Finv_Ji Finv_Lk.
          # With H[e, a, k] = sum_J G[a, J] Finv[J, k], contracting with G on both sides:
          # K[a i, b k] = mu GG[a, b] d_ik + c1 H[a, k] H[b, i] + lam H[a, i] H[b, k].
          H = np.einsum("aJ,eJk->eak", self.G[q], Finv)
          c1 = mu - lam * lnJ
          Ke += self.wdet * (
            mu[:, None, None, None, None] * np.einsum("ab,ik->aibk", self.GG[q], eye3)[None]
            + c1[:, None, None, None, None] * np.einsum("eak,ebi->eaibk", H, H)
            + lam[:, None, None, None, None] * np.einsum("eai,ebk->eaibk", H, H))
      np.add.at(f, self.edof_pad[e0:e1].ravel(), fe.reshape(-1))
      if tangent:
        vals = Ke.reshape(e1 - e0, 576)[self.keep[e0:e1]]
        s = self.slot[self.kept_start[e0]: self.kept_start[e1]]
        data += np.bincount(s, weights=vals, minlength=self.nnz)
    f = f[: self.n]
    if not tangent:
      return f
    return f, sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

  def rigid_body_modes(self):
    """Three translations and three rotations of the free nodes (near-nullspace for SA)."""
    x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]
    B = np.zeros((3 * x.size, 6))
    for c in range(3):
      B[c::3, c] = 1.0
    B[0::3, 3], B[1::3, 3] = -y, x
    B[1::3, 4], B[2::3, 4] = -z, y
    B[0::3, 5], B[2::3, 5] = z, -x
    return B


def solve(K, rhs, B):
  """CG + smoothed-aggregation AMG on 3x3 node blocks; GMRES if CG falls short.

  pyamg's defaults stall on this problem (300+ iterations); evolution strength, energy-
  minimizing prolongation smoothing and symmetric block Gauss-Seidel converge in ~16.
  """
  ml = pyamg.smoothed_aggregation_solver(
    K.tobsr(blocksize=(3, 3)), B=B, symmetry="symmetric", max_coarse=500,
    strength=("evolution", {"k": 2, "proj_type": "l2", "epsilon": 4.0}),
    smooth=("energy", {"krylov": "cg", "maxiter": 3, "degree": 2}),
    presmoother=("block_gauss_seidel", {"sweep": "symmetric"}),
    postsmoother=("block_gauss_seidel", {"sweep": "symmetric"}))
  nb = np.linalg.norm(rhs)
  for accel in ("cg", "gmres"):
    res = []
    x = ml.solve(rhs, tol=1e-10, accel=accel, maxiter=500, residuals=res)
    rel = np.linalg.norm(rhs - K @ x) / nb
    if rel <= 1e-8:
      break
  return x, len(res) - 1, rel


def check(a):
  """Finite-difference check of the tangent, and symmetry / zero-strain sanity, small mesh."""
  a.grid = [4, 2, 2]
  a.inclusion_frac = 0.3
  a.inclusion_radius = 0.4
  beam = Beam(a)
  rng = np.random.default_rng(0)
  u = 0.05 * rng.standard_normal(beam.n)
  _, K = beam.internal_force(u, tangent=True)
  v = rng.standard_normal(beam.n)
  eps = 1e-6
  fd = (beam.internal_force(u + eps * v) - beam.internal_force(u - eps * v)) / (2 * eps)
  err = np.linalg.norm(K @ v - fd) / np.linalg.norm(fd)
  asym = sp.linalg.norm(K - K.T) / sp.linalg.norm(K)
  f0, K0 = beam.internal_force(np.zeros(beam.n), tangent=True)
  lmin = sp.linalg.eigsh(K0, k=1, sigma=0, which="LM")[0][0]
  print(f"n = {beam.n}, nnz = {beam.nnz}, {beam.n_inc} inclusions (stiff fraction {beam.inc_frac:.2f})")
  print(f"tangent vs central differences {err:.2e}, ||K - K^T||/||K|| {asym:.2e}")
  print(f"||f_int(0)|| = {np.linalg.norm(f0):.2e}, smallest eigenvalue of K(0) = {lmin:.3e}")
  ok = err < 1e-6 and asym < 1e-12 and np.linalg.norm(f0) < 1e-12 and lmin > 0
  print("OK" if ok else "FAILED")
  sys.exit(0 if ok else 1)


def main():
  a = parse_args()
  if pyamg is None:
    sys.exit("error: gen_elast3d.py needs pyamg (pip install pyamg)")
  if a.check:
    check(a)
  t_start = time.time()
  beam = Beam(a)
  n = beam.n
  print(f"# grid {beam.nx} x {beam.ny} x {beam.nz} elements, n = {n}, nnz = {beam.nnz} "
        f"({beam.nnz / n:.2f}/row), {beam.n_inc} inclusions, stiff fraction {beam.inc_frac:.3f}, "
        f"setup {time.time() - t_start:.1f} s", flush=True)
  if not a.dry_run:
    os.makedirs(a.out_matrix, exist_ok=True)
    os.makedirs(a.out_rhs, exist_ok=True)

  B = beam.rigid_body_modes()
  u = np.zeros(n)
  written, K1, nK1, last = [], None, None, None
  k = 0
  tip = 0.0
  for j in range(1, a.load_steps + 1):
    lam_load = j / a.load_steps
    fext = lam_load * beam.f_ext
    nref = np.linalg.norm(fext)
    for it in range(a.newton_maxit + 1):
      t0 = time.time()
      fint, K = beam.internal_force(u, tangent=True)
      R = fint - fext
      rel = np.linalg.norm(R) / nref
      t_asm = time.time() - t0
      if rel <= a.newton_tol:
        break
      if it == a.newton_maxit:
        sys.exit(f"error: Newton did not converge at load {lam_load:.2f} (rel R {rel:.3e})")
      if K1 is None:
        K1, nK1 = K, sp.linalg.norm(K)
      change = sp.linalg.norm(K - last) / sp.linalg.norm(last) if last is not None else 1.0
      if change >= a.min_change:
        k += 1
        drift = sp.linalg.norm(K - K1) / nK1
        if not a.dry_run:
          write_csr(os.path.join(a.out_matrix, f"matrix_{k}.txt"), K)
          write_vec(os.path.join(a.out_rhs, f"rhs_{k}.txt"), -R)
        written.append((k, lam_load, it, rel, drift, change))
        last = K
        tag = f"matrix_{k}"
      else:
        tag = "(skipped)"
      t1 = time.time()
      du, lin_it, lin_rel = solve(K, -R, B)
      t_sol = time.time() - t1
      # Backtracking (Armijo) on the potential energy Pi(u) = W(u) - f_ext . u, which Newton
      # minimizes. Not on ||R||: the beam is slender (cond(K) ~ 1e5-1e6), so a good step leaves
      # large membrane forces that inflate ||R|| while moving u almost nothing.
      # Near convergence the energy decrease drops below roundoff in Pi, so a step that
      # reduces ||R|| is accepted as well.
      pi0 = beam.energy(u) - fext @ u
      slope = R @ du  # dPi/dalpha at 0, < 0 for an SPD tangent
      nR = np.linalg.norm(R)
      alpha = 1.0
      while True:
        try:
          ut = u + alpha * du
          if (beam.energy(ut) - fext @ ut <= pi0 + 1e-4 * alpha * slope
              or np.linalg.norm(beam.internal_force(ut) - fext) < nR or alpha < 1.0 / 64):
            break
        except FloatingPointError:
          if alpha < 1.0 / 64:
            sys.exit("error: line search could not avoid an inverted element")
        alpha /= 2
      u = u + alpha * du
      print(f"  load {lam_load:.2f} newton {it}: rel R {rel:.3e}  {tag:12s} assemble {t_asm:5.1f} s,"
            f" solve {t_sol:5.1f} s ({lin_it} it, {lin_rel:.1e}), step {alpha:g}", flush=True)
    tip = -u[2::3].min()
    print(f"load {lam_load:.2f} converged after {it} Newton steps, max deflection {tip:.4f}"
          f" (length {beam.nx * beam.h:.2f}, depth 1)", flush=True)
    Y = beam.element_energy_density(u)
    if np.isfinite(a.damage_y0):
      beam.update_damage(u, a.damage_y0, a.damage_yc, a.damage_max)
    q = np.percentile(Y, [50, 90, 99, 99.9])
    print(f"  energy density Y: median {q[0]:.3e} p90 {q[1]:.3e} p99 {q[2]:.3e} p99.9 {q[3]:.3e} max {Y.max():.3e};"
          f" damage: d>0.1 in {np.mean(beam.d > 0.1):.2%}, d>0.9 in {np.mean(beam.d > 0.9):.2%}"
          f" of elements, max d {beam.d.max():.3f}", flush=True)

  asym = sp.linalg.norm(last - last.T) / sp.linalg.norm(last)
  summary = [
    f"# ELAST3D Newton sequence, generated {time.strftime('%Y-%m-%d %H:%M:%S')} in {time.time() - t_start:.0f} s",
    "# compressible neo-Hookean cantilever, Q1 hexahedra, clamped at x = 0, dead body load -g e_z",
    f"# grid {beam.nx} x {beam.ny} x {beam.nz} elements (h = 1/{beam.nz}), n = {n}, nnz = {beam.nnz}"
    f" ({beam.nnz / n:.2f}/row); node strides 3 / {3 * beam.nx} / {3 * beam.nx * (beam.ny + 1)}",
    f"# E {a.young}  nu {a.poisson}  contrast {a.contrast}  inclusions {beam.n_inc} (r {a.inclusion_radius},"
    f" stiff fraction {beam.inc_frac:.3f}, seed {a.seed})  g {a.gravity}  load steps {a.load_steps}"
    f"  newton tol {a.newton_tol}  min change {a.min_change}",
    f"# damage Y0 {a.damage_y0}  Yc {a.damage_yc}  d_max {a.damage_max}: at the end d>0.1 in"
    f" {np.mean(beam.d > 0.1):.2%} and d>0.9 in {np.mean(beam.d > 0.9):.2%} of elements",
    f"# max deflection at full load {tip:.4f};  ||K - K^T||/||K|| of the last matrix {asym:.2e}",
    "# k  load  newton_step  rel_R  drift_to_A1  change_from_prev",
  ] + [f"{k_} {l:.4f} {i} {r:.6e} {d:.6e} {c:.6e}" for k_, l, i, r, d, c in written]
  print("\n".join(summary))
  if not a.dry_run:
    with open(os.path.join(a.out_matrix, "sequence_info.txt"), "w") as fh:
      fh.write("\n".join(summary) + "\n")


if __name__ == "__main__":
  main()
