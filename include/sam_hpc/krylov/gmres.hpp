#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/krylov/givensRotations.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/krylov/precondSide.hpp"
#include "sam_hpc/core/utilities.hpp"

/**
 * @class GMRES
 * @brief Implements the Generalized Minimal Residual (GMRES) iterative solver.
 *
 * This class solves large, sparse, non-symmetric linear systems of the form Ax = b.
 * It is a templated class, allowing for different floating-point precisions
 * (e.g., float, double). The method uses a Krylov subspace approach and can be restarted
 * to limit memory consumption.
 *
 * @tparam T The floating-point type for calculations (e.g., double).
 */
template <typename T>
class GMRES {
 public:
  // solver parameters
  struct params {
    // Number of iterations before restart
    unsigned M;

    // Preconditioning kind (left/right)
    precondSide pside;

    // Maximum number of iterations
    unsigned maxIter;

    // Target relative residual error
    T tol;

    // Target absolute residual error
    T absTol;

    // Ignore the trivial solution when x = 0 when rhs is zero search for null space vectors
    bool ns_search;

    // Verbose output
    bool verbose;

    params() : M(30), pside(precondSide::right), maxIter(10000), tol(1e-8),
               absTol(std::numeric_limits<T>::min()), ns_search(false), verbose(false) {}
  };
  params prm;

  /**
   * @brief Constructor for the GMRES solver.
   * @param n The size of the linear system (number of unknowns).
   * @param prm An optional parameters object to configure the solver.
   */
  GMRES(size_t n, const params& prm = params())
    : prm(prm), n(n), H(prm.M + 1, prm.M), s(prm.M + 1, T()), cs(prm.M + 1, T()),
      sn(prm.M + 1, T()), r(n, T())
  {
    // The Krylov basis `v` is a vector of vectors.
    v.resize(prm.M + 1);
    for (unsigned i = 0; i < prm.M + 1; ++i) {
      v[i].resize(n, T(0));
    }
  }

  /* Computes the solution for the given system matrix \p A and the right-hand side \p rhs.
   * Returns the number of iterations made and the achieved residual as a ``std::tuple``.
   * The solution vector \p x provides initial approximation in input and holds the
   * computed solution on output.
   *
   * The system matrix may differ from the matrix used during initialization. This may be
   * used for the solution of non-stationary problems with slowly changing coefficients.
   * There is a strong chance that a preconditioner built for a time step will act as a
   * reasonably good preconditioner for several subsequent time steps [DeSh12]_.
   */
  template <class Precond>
  std::tuple<size_t, T> solve(const CSRMatrix<T>& A, const Precond& P,
                              const std::vector<T>& rhs, std::vector<T>& x)
  {
    // Calculate the norm of the right-hand side for relative residual checks.
    T norm_rhs = norm(rhs);
    if (norm_rhs < eps<T>(1)) {
      if (prm.ns_search) {
        norm_rhs = T(1);
      } else {
        x.clear();
        return std::make_tuple(0, norm_rhs);
      }
    }

    T tolerance = std::max(prm.tol * norm_rhs, prm.absTol);
    T norm_r = T(0);

    size_t iter = 0;
    while (true) {  // outer loop runs until restart
      // Calculate the initial preconditioned residual
      if (prm.pside == left) {
        // For left preconditioning, r = P⁻¹(b - Ax).
        residual(rhs, A, x, v[0]);
        P.apply(v[0], r);
      } else {
        // For right preconditioning, r = b - Ax.
        residual(rhs, A, x, r);
      }

      // Check for immediate convergence
      norm_r = norm(r);
      if (norm_r < tolerance || iter >= prm.maxIter)
        break;

      // Inner GMRES iteration
      // Normalize the residual to get the first Krylov basis vector v₀.
      axpby((1 / norm_r), r, T(0), v[0]);
      // The vector `s` stores the RHS of the least-squares problem. s₀ = ||r||.
      s[0] = norm_r;
      unsigned j = 0;  // inner iteration counter
      while (true) {
        // -- Arnoldi process: Generate a new basis vector
        // Build an orthonormal basis V and a Hessenberg matrix H such that A*V = V*H.
        auto& v_new = v[j + 1];  // next basis vector
        // Compute v_new = A*v[j] (with preconditioning).
        precond_spmv(prm.pside, P, A, v[j], v_new, r);

        // Gram-Schmidt Orthogonalization
        // Make v_new orthogonal to all previous basis vectors v[0]...v[j].
        for (size_t k = 0; k <= j; ++k) {
          H(k, j) = innerProduct(v_new, v[k]);  // H(k,j) is the projection of v_new onto v[k].
          axpby(-H(k, j), v[k], T(1), v_new);
        }
        // H(j+1,j) is the norm of the now-orthogonalized vector.
        H(j + 1, j) = norm(v_new);

        // Add a check for the "lucky break" or breakdown.
        /* if (H(j + 1, j) < std::numeric_limits<T>::epsilon()) {
            break;
        } */

        // Normalize the new basis vector to have length 1.
        axpby((1 / H(j + 1, j)), v_new, T(0), v_new);

        // Solve the least-squares problem using Givens Rotations
        // Apply all previous rotations to the new column of H.
        for (unsigned k = 0; k < j; ++k) {
          applyPlaneRotation(H(k, j), H(k + 1, j), cs[k], sn[k]);
        }
        // Generate and apply a new rotation to zero out the subdiagonal
        // H(j+1,j).
        generatePlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        // Apply the same rotation to the right-hand-side vector `s`.
        applyPlaneRotation(s[j], s[j + 1], cs[j], sn[j]);

        // The absolute value of s[j+1] is now the residual of the subproblem.
        T inner_res = std::abs(s[j + 1]);
        if (prm.verbose && iter % 5 == 0) {
          std::cout << iter << "\t" << std::scientific << inner_res / norm_rhs
                    << std::endl;
        }

        // Check for inner loop termination
        ++j, ++iter;
        if (iter >= prm.maxIter || j >= prm.M || inner_res <= tolerance)
          break;
      }

      // Back Substitution to find the solution coefficients
      // The inner loop has terminated. Now, solve the upper triangular system
      // Hy = s. The result `y` is stored back into `s`.
      for (unsigned i = j; i-- > 0;) {
        s[i] /= H(i, i);
        for (unsigned k = 0; k < i; ++k) {
          s[k] -= H(k, i) * s[i];
        }
      }

      // Compute and apply the solution update
      // The solution update is dx = V*y (where y is stored in s).
      std::vector<T>& dx = r;  // Reuse residual vector `r` for the update `dx`.
      linComb(j, s, v, T(0), dx);

      if (prm.pside == left) {
        // For left preconditioning, update is x = x + dx.
        axpby(T(1), dx, T(1), x);
      } else {
        // For right preconditioning, update is x = x + P⁻¹(dx).
        std::vector<T>& tmp = v[0];
        P.apply(dx, tmp);
        axpby(T(1), tmp, T(1), x);
      }

      // Added for more robustness
      // residual(rhs, A, x, r); // This calculates r = b - A*x
      // norm_r = norm(r);       // This updates norm_r for the check at the top
      // of the next loop
    }

    return std::make_tuple(iter, norm_r / norm_rhs);
  }

  /* Computes the solution for the given system matrix \p A and the
   * right-hand side \p rhs with the provided map N.  Returns the number of
   * iterations made and the achieved residual as a ``std::tuple``. The solution
   * vector
   * \p x provides initial approximation in input and holds the computed
   * solution on output.
   */
  template <class Precond>
  std::tuple<size_t, T> solve(const CSRMatrix<T>& A, const Precond& P,
                              const CSRMatrix<T>& N, const std::vector<T>& rhs,
                              std::vector<T>& x)
  {
    // Calculate the norm of the right-hand side for relative residual checks.
    T norm_rhs = norm(rhs);
    if (norm_rhs < eps<T>(1)) {
      if (prm.ns_search) {
        norm_rhs = T(1);
      } else {
        x.clear();
        return std::make_tuple(0, norm_rhs);
      }
    }

    T tolerance = std::max(prm.tol * norm_rhs, prm.absTol);
    T norm_r = T(0);

    size_t iter = 0;
    while (true) {  // outer loop runs until restart
      // Calculate the initial preconditioned residual
      std::vector<T> temp(x.size(), T(0));
      if (prm.pside == left) {
        // For left preconditioning, r = P⁻¹(b - Ax).
        spmv(T(1), A, x, T(0), temp);
        residual(rhs, N, temp, v[0]);
        P.apply(v[0], r);
      } else {
        // For right preconditioning, r = b - Ax.
        spmv(T(1), A, x, T(0), temp);
        residual(rhs, N, temp, r);
      }

      // Check for immediate convergence
      norm_r = norm(r);
      if (norm_r < tolerance || iter >= prm.maxIter)
        break;

      // Inner GMRES iteration
      // Normalize the residual to get the first Krylov basis vector v₀.
      axpby((1 / norm_r), r, T(0), v[0]);
      // The vector `s` stores the RHS of the least-squares problem. s₀ = ||r||.
      s[0] = norm_r;
      unsigned j = 0;  // inner iteration counter
      while (true) {
        // -- Arnoldi process: Generate a new basis vector
        // Build an orthonormal basis V and a Hessenberg matrix H such that A*V = V*H.
        auto& v_new = v[j + 1];  // next basis vector
        // Compute v_new = A*v[j] (with preconditioning).
        precond_spmv_sam(prm.pside, P, A, N, v[j], v_new, r, temp);

        // Gram-Schmidt Orthogonalization
        // Make v_new orthogonal to all previous basis vectors v[0]...v[j].
        for (size_t k = 0; k <= j; ++k) {
          H(k, j) = innerProduct(v_new, v[k]);  // H(k,j) is the projection of v_new onto v[k].
          axpby(-H(k, j), v[k], T(1), v_new);
        }
        // H(j+1,j) is the norm of the now-orthogonalized vector.
        H(j + 1, j) = norm(v_new);
        // Normalize the new basis vector to have length 1.
        axpby((1 / H(j + 1, j)), v_new, T(0), v_new);

        // Solve the least-squares problem using Givens Rotations
        // Apply all previous rotations to the new column of H.
        for (unsigned k = 0; k < j; ++k) {
          applyPlaneRotation(H(k, j), H(k + 1, j), cs[k], sn[k]);
        }
        // Generate and apply a new rotation to zero out the subdiagonal
        // H(j+1,j).
        generatePlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        applyPlaneRotation(H(j, j), H(j + 1, j), cs[j], sn[j]);
        // Apply the same rotation to the right-hand-side vector `s`.
        applyPlaneRotation(s[j], s[j + 1], cs[j], sn[j]);

        // The absolute value of s[j+1] is now the residual of the subproblem.
        T inner_res = std::abs(s[j + 1]);
        if (prm.verbose && iter % 5 == 0) {
          std::cout << iter << "\t" << std::scientific << inner_res / norm_rhs
                    << std::endl;
        }

        // Check for inner loop termination
        ++j, ++iter;
        if (iter >= prm.maxIter || j >= prm.M || inner_res <= tolerance)
          break;
      }

      // Back Substitution to find the solution coefficients
      // The inner loop has terminated. Now, solve the upper triangular system
      // Hy = s. The result `y` is stored back into `s`.
      for (unsigned i = j; i-- > 0;) {
        s[i] /= H(i, i);
        for (unsigned k = 0; k < i; ++k) {
          s[k] -= H(k, i) * s[i];
        }
      }

      // Compute and apply the solution update
      // The solution update is dx = V*y (where y is stored in s).
      std::vector<T>& dx = r;  // Reuse residual vector `r` for the update `dx`.
      linComb(j, s, v, T(0), dx);

      if (prm.pside == left) {
        // For left preconditioning, update is x = x + dx.
        axpby(T(1), dx, T(1), x);
      } else {
        // For right preconditioning, update is x = x + P⁻¹(dx).
        std::vector<T>& tmp = v[0];
        P.apply(dx, tmp);
        axpby(T(1), tmp, T(1), x);
      }
    }

    return std::make_tuple(iter, norm_r / norm_rhs);
  }

  //@todo: implement bytes() method

  friend std::ostream& operator<<(std::ostream& os, const GMRES& s) {
    return os << "Type:             GMRES(" << s.prm.M << ")"
              << "\nUnknowns:         " << s.n << std::endl;
  }

 private:
  size_t n;

  mutable multiArray<T, 2> H;     // upper Hessenberg matrix
  std::vector<T> s;               // rhs for least squares problem
  std::vector<T> cs, sn;          // sine and cosine coeffs for Givens rotation
  std::vector<T> r;               // residual for each iteration
  std::vector<std::vector<T>> v;  // orthonormal basis of the Krylov subspace
};