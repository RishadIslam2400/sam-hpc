#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

/// Damped Jacobi relaxation:
/// `x <- x + damping * D^{-1} * (b - A x)`, where `D = diag(A)`.
template <typename T>
struct damped_jacobi {
  /// Relaxation parameters.
  struct params {
    /// Relaxation weight.
    double damping;

    params(double damping = 0.72) : damping(damping) { }
  } prm;

  /// Inverse diagonal of the system matrix.
  std::vector<T> dia;

  /// Constructs the smoother and stores only the inverse diagonal of A. For a null entry
  /// in the diagonal the inverse contains 1.
  /// @param A System matrix used to compute the inverse diagonal.
  /// @param prm Relaxation parameters.
  damped_jacobi(const csr::CSRMatrix<T> &A, const params &prm = params())
    : prm(prm), dia(csr::diagonal<csr::diagonalType::inverse>(A)) { }

  /// Applies one damped Jacobi step before coarse-grid correction.
  /// @param A System matrix.
  /// @param rhs Right-hand side.
  /// @param x Solution vector, updated in place.
  /// @param tmp Scratch vector, overwritten with `rhs - A * x`.
  void apply_pre(const csr::CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
    residual(rhs, A, x, tmp);
    vmul(prm.damping, dia, tmp, T(1), x);
  }

  /// Applies one damped Jacobi step after coarse-grid correction.
  /// Jacobi is independent of sweep direction, so this performs the same operation as
  /// apply_pre.
  void apply_post(const csr::CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
    residual(rhs, A, x, tmp);
    vmul(prm.damping, dia, tmp, T(1), x);
  }

  /// Applies undamped diagonal scaling: `x = D^{-1} * rhs`.
  /// The matrix argument is unused and is present for compatibility with the smoother
  /// interface. The output vector must already have the correct length.
  void apply(const csr::CSRMatrix<T>&, const std::vector<T>& rhs, std::vector<T>& x) const {
    vmul(T(1), dia, rhs, T(0), x);
  }
};