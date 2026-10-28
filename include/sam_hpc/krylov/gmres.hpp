#pragma once

#include <algorithm>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/krylov/givensRotations.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/krylov/precondSide.hpp"
#include "sam_hpc/core/utilities.hpp"

/// Restarted GMRES solver for sparse linear systems.
/// The preconditioner must provide `apply(in, out)`, computing `out = M^-1 in`.
/// A solver instance owns its Krylov workspace and supports one active solve at a time.
template <typename T>
class GMRES {
 public:
  /// Parameters controlling restart, stopping, and preconditioning.
  struct params {
    /// Krylov-subspace dimension before restarting.
    unsigned M;

    /// Selects left or right preconditioning.
    precondSide pside;

    /// Maximum total number of Arnoldi steps.
    unsigned maxIter;

    /// Relative tolerance for the residual in the iteration space.
    T tol;

    /// Absolute tolerance for the residual in the iteration space.
    T absTol;

    /// Use a unit norm scale instead of returning for a zero right-hand side.
    bool ns_search;

    /// Print every fifth iteration and its residual estimate when enabled.
    bool verbose;

    params() : M(30), pside(precondSide::right), maxIter(10000), tol(1e-8),
               absTol(std::numeric_limits<T>::min()), ns_search(false), verbose(false) {}
  };
  params prm;

  /// Constructs a solver for a system with `n` unknowns.
  GMRES(size_t n, const params& prm = params())
    : prm(prm), n(n), H(prm.M + 1, prm.M), s(prm.M + 1, T()), cs(prm.M + 1, T()),
      sn(prm.M + 1, T()), r(n, T())
  {
    v.resize(prm.M + 1);
    for (unsigned i = 0; i < prm.M + 1; ++i) {
      v[i].resize(n, T(0));
    }
  }

  /// Solves `A x = rhs` from the initial guess in `x`.
  /// Returns the total Arnoldi steps and the true relative residual.
  template <class Precond>
  std::tuple<size_t, T> solve(const csr::CSRMatrix<T>& A, const Precond& P,
                              const std::vector<T>& rhs, std::vector<T>& x)
  {
    T norm_rhs = norm(rhs);
    if (norm_rhs < eps<T>(1)) {
      if (prm.ns_search) {
        norm_rhs = T(1);
      } else {
        std::fill(x.begin(), x.end(), T(0));
        return std::make_tuple(0, norm_rhs);
      }
    }

    // Scale the tolerance in the same space as the iteration residual.
    const T norm_ref = precond_norm(P, rhs, norm_rhs);
    T tolerance = std::max(prm.tol * norm_ref, prm.absTol);
    T norm_r = T(0);

    size_t iter = 0;
    while (true) {
      if (prm.pside == left) {
        residual(rhs, A, x, v[0]);
        P.apply(v[0], r);
      } else {
        residual(rhs, A, x, r);
      }

      norm_r = norm(r);
      if (norm_r < tolerance || iter >= prm.maxIter)
        break;

      axpby((1 / norm_r), r, T(0), v[0]);
      std::fill(s.begin(), s.end(), T(0));
      s[0] = norm_r;
      unsigned j = 0;
      while (true) {
        auto& v_new = v[j + 1];
        precond_spmv(prm.pside, P, A, v[j], v_new, r);

        for (size_t k = 0; k <= j; ++k) {
          H(k, j) = innerProduct(v_new, v[k]);
          axpby(-H(k, j), v[k], T(1), v_new);
        }
        H(j + 1, j) = norm(v_new);

        // A zero subdiagonal means that the Krylov space is invariant.
        const bool breakdown = H(j + 1, j) == T(0);
        if (!breakdown)
          axpby((1 / H(j + 1, j)), v_new, T(0), v_new);

        for (unsigned k = 0; k < j; ++k) {
          applyPlaneRotation(H(k, j), H(k + 1, j), cs[k], sn[k]);
        }
        generatePlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(s[j], s[j + 1], cs[j], sn[j]);

        T inner_res = std::abs(s[j + 1]);
        if (prm.verbose && iter % 5 == 0) {
          std::cout << iter << "\t" << std::scientific << inner_res / norm_ref << std::endl;
        }

        ++j, ++iter;
        if (iter >= prm.maxIter || j >= prm.M || inner_res <= tolerance || breakdown)
          break;
      }

      // Solve the rotated upper-triangular system for the Krylov coefficients.
      for (unsigned i = j; i-- > 0;) {
        s[i] /= H(i, i);
        for (unsigned k = 0; k < i; ++k) {
          s[k] -= H(k, i) * s[i];
        }
      }

      std::vector<T>& dx = r;
      linComb(j, s, v, T(0), dx);

      if (prm.pside == left) {
        axpby(T(1), dx, T(1), x);
      } else {
        std::vector<T>& tmp = v[0];
        P.apply(dx, tmp);
        axpby(T(1), tmp, T(1), x);
      }
    }

    residual(rhs, A, x, r);
    return std::make_tuple(iter, norm(r) / norm_rhs);
  }

  /// Solves `A x = b` using a left SAM map satisfying `N A ≈ A_0`.
  /// The iteration uses `N A x = N b`; the returned residual is computed
  /// against the original system `A x = b`.
  template <class Precond>
  std::tuple<size_t, T> solve(const csr::CSRMatrix<T>& A, const Precond& P,
                              const csr::CSRMatrix<T>& N, const std::vector<T>& b,
                              std::vector<T>& x)
  {
    std::vector<T> Nb(N.m_rows, T(0));
    spmv(T(1), N, b, T(0), Nb);

    T norm_b = norm(b);
    if (norm_b < eps<T>(1)) {
      if (prm.ns_search) {
        norm_b = T(1);
      } else {
        std::fill(x.begin(), x.end(), T(0));
        return std::make_tuple(0, norm_b);
      }
    }
    T norm_rhs = norm(Nb);
    if (norm_rhs < eps<T>(1)) {
      if (prm.ns_search) {
        norm_rhs = T(1);
      } else {
        residual(b, A, x, r);
        return std::make_tuple(0, norm(r) / norm_b);
      }
    }

    // Scale the tolerance in the transformed iteration space.
    const T norm_ref = precond_norm(P, Nb, norm_rhs);
    T tolerance = std::max(prm.tol * norm_ref, prm.absTol);
    T norm_r = T(0);

    size_t iter = 0;
    while (true) {
      std::vector<T> temp(x.size(), T(0));
      if (prm.pside == left) {
        spmv(T(1), A, x, T(0), temp);
        residual(Nb, N, temp, v[0]);
        P.apply(v[0], r);
      } else {
        spmv(T(1), A, x, T(0), temp);
        residual(Nb, N, temp, r);
      }

      norm_r = norm(r);
      if (norm_r < tolerance || iter >= prm.maxIter)
        break;

      axpby((1 / norm_r), r, T(0), v[0]);
      std::fill(s.begin(), s.end(), T(0));
      s[0] = norm_r;
      unsigned j = 0;
      while (true) {
        auto& v_new = v[j + 1];
        precond_spmv_sam(prm.pside, P, A, N, v[j], v_new, r, temp);

        for (size_t k = 0; k <= j; ++k) {
          H(k, j) = innerProduct(v_new, v[k]);
          axpby(-H(k, j), v[k], T(1), v_new);
        }
        H(j + 1, j) = norm(v_new);

        // A zero subdiagonal means that the Krylov space is invariant.
        const bool breakdown = H(j + 1, j) == T(0);
        if (!breakdown)
          axpby((1 / H(j + 1, j)), v_new, T(0), v_new);

        for (unsigned k = 0; k < j; ++k) {
          applyPlaneRotation(H(k, j), H(k + 1, j), cs[k], sn[k]);
        }
        generatePlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(s[j], s[j + 1], cs[j], sn[j]);

        T inner_res = std::abs(s[j + 1]);
        if (prm.verbose && iter % 5 == 0) {
          std::cout << iter << "\t" << std::scientific << inner_res / norm_ref << std::endl;
        }

        ++j, ++iter;
        if (iter >= prm.maxIter || j >= prm.M || inner_res <= tolerance || breakdown)
          break;
      }

      for (unsigned i = j; i-- > 0;) {
        s[i] /= H(i, i);
        for (unsigned k = 0; k < i; ++k) {
          s[k] -= H(k, i) * s[i];
        }
      }

      std::vector<T>& dx = r;
      linComb(j, s, v, T(0), dx);

      if (prm.pside == left) {
        axpby(T(1), dx, T(1), x);
      } else {
        std::vector<T>& tmp = v[0];
        P.apply(dx, tmp);
        axpby(T(1), tmp, T(1), x);
      }
    }

    residual(b, A, x, r);
    return std::make_tuple(iter, norm(r) / norm_b);
  }

  /// Solves `A x = b` using a right SAM map satisfying `A N ≈ A_0`.
  /// The map transforms the correction space rather than the right-hand
  /// side. The returned residual is computed against the original system.
  template <class Precond>
  std::tuple<size_t, T> solveRightMap(const csr::CSRMatrix<T>& A, const Precond& P,
                                      const csr::CSRMatrix<T>& N, const std::vector<T>& b,
                                      std::vector<T>& x)
  {
    T norm_rhs = norm(b);
    if (norm_rhs < eps<T>(1)) {
      if (prm.ns_search) {
        norm_rhs = T(1);
      } else {
        std::fill(x.begin(), x.end(), T(0));
        return std::make_tuple(0, norm_rhs);
      }
    }

    // Scale the tolerance in the iteration space.
    const T norm_ref = precond_norm(P, b, norm_rhs);
    T tolerance = std::max(prm.tol * norm_ref, prm.absTol);
    T norm_r = T(0);

    std::vector<T> temp(x.size(), T(0));
    std::vector<T> corr(x.size(), T(0));

    size_t iter = 0;
    while (true) {
      if (prm.pside == left) {
        residual(b, A, x, v[0]);
        P.apply(v[0], r);
      } else {
        residual(b, A, x, r);
      }

      norm_r = norm(r);
      if (norm_r < tolerance || iter >= prm.maxIter)
        break;

      axpby((1 / norm_r), r, T(0), v[0]);
      std::fill(s.begin(), s.end(), T(0));
      s[0] = norm_r;
      unsigned j = 0;
      while (true) {
        auto& v_new = v[j + 1];
        precond_spmv_sam_right(prm.pside, P, A, N, v[j], v_new, r, temp);

        for (size_t k = 0; k <= j; ++k) {
          H(k, j) = innerProduct(v_new, v[k]);
          axpby(-H(k, j), v[k], T(1), v_new);
        }
        H(j + 1, j) = norm(v_new);

        // A zero subdiagonal means that the Krylov space is invariant.
        const bool breakdown = H(j + 1, j) == T(0);
        if (!breakdown)
          axpby((1 / H(j + 1, j)), v_new, T(0), v_new);

        for (unsigned k = 0; k < j; ++k) {
          applyPlaneRotation(H(k, j), H(k + 1, j), cs[k], sn[k]);
        }
        generatePlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(s[j], s[j + 1], cs[j], sn[j]);

        T inner_res = std::abs(s[j + 1]);
        if (prm.verbose && iter % 5 == 0) {
          std::cout << iter << "\t" << std::scientific << inner_res / norm_ref << std::endl;
        }

        ++j, ++iter;
        if (iter >= prm.maxIter || j >= prm.M || inner_res <= tolerance || breakdown)
          break;
      }

      for (unsigned i = j; i-- > 0;) {
        s[i] /= H(i, i);
        for (unsigned k = 0; k < i; ++k) {
          s[k] -= H(k, i) * s[i];
        }
      }

      std::vector<T>& dx = r;
      linComb(j, s, v, T(0), dx);

      if (prm.pside == left) {
        spmv(T(1), N, dx, T(0), corr);
      } else {
        std::vector<T>& tmp = v[0];
        P.apply(dx, tmp);
        spmv(T(1), N, tmp, T(0), corr);
      }
      axpby(T(1), corr, T(1), x);
    }

    residual(b, A, x, r);
    return std::make_tuple(iter, norm(r) / norm_rhs);
  }

  /// Writes the solver type, restart length, and problem size.
  friend std::ostream& operator<<(std::ostream& os, const GMRES& s) {
    return os << "Type:             GMRES(" << s.prm.M << ")"
              << "\nUnknowns:         " << s.n << std::endl;
  }

 private:
  size_t n;

  /// Returns the right-hand-side norm in the iteration space.
  /// Under left preconditioning this is `||M^-1 rhs||`; otherwise it is `unprec`.
  /// The residual workspace `r` is reused for the preconditioned vector.
  template <class Precond>
  T precond_norm(const Precond& P, const std::vector<T>& rhs, T unprec) {
    if (prm.pside != left || norm(rhs) < eps<T>(1))
      return unprec;
    P.apply(rhs, r);
    const T nr = norm(r);
    return nr < eps<T>(1) ? unprec : nr;
  }

  mutable multiArray<T, 2> H;     // Rotated upper Hessenberg matrix.
  std::vector<T> s;               // Least-squares right-hand side and coefficients.
  std::vector<T> cs, sn;          // Givens rotation coefficients.
  std::vector<T> r;               // Residual and correction workspace.
  std::vector<std::vector<T>> v;  // Krylov basis vectors.
};