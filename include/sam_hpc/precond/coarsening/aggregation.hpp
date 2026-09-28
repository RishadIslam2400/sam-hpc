#pragma once

#include "sam_hpc/precond/coarsening/pointwise_aggregates.hpp"
#include "sam_hpc/precond/coarsening/tentative_prolongation.hpp"

/// Coarsening strategies

/**
 * A coarsener in AMGCL is a class that takes a system matrix and returns three
 * operators:
 *
 * 1. Restriction operator R that downsamples the residual error to a
 *    coarser level in AMG hierarchy,
 * 2. Prolongation operator P that interpolates a correction computed on a
 *    coarser grid into a finer grid,
 * 3. System matrix \f$A^H\f$ at a coarser level that is usually computed as a
 *    Galerkin operator \f$A^H = R A^h P\f$.
 *
 * The AMG hierarchy is constructed by recursive invocation of the selected
 * coarsener.
 */

/// Non-smoothed aggregation.
template <class T>
struct aggregation {
  /// Coarsening parameters.
  struct params {
    /// Aggregation parameters.
    pointwise_aggregates::params aggr;

    /// Near nullspace parameters.
    nullspace_params nullspace;

    /// Over-interpolation factor \f$\alpha\f$.
    /**
     * In case of aggregation coarsening, coarse-grid
     * correction of smooth error, and by this the overall convergence, can
     * often be substantially improved by using "over-interpolation", that is,
     * by multiplying the actual correction (corresponding to piecewise
     * constant interpolation) by some factor \f$\alpha > 1\f$. Equivalently,
     * this means that the coarse-level Galerkin operator is re-scaled by
     * \f$1 / \alpha\f$:
     * \f[I_h^HA_hI_H^h \to \frac{1}{\alpha}I_h^HA_hI_H^h.\f]
     */
    float over_interp;

    params() : over_interp(2.0f) {}
  } prm;

  aggregation(const params& prm = params()) : prm(prm) {}

  /// Creates transfer operators for the given system matrix.
  /**
   * \param A   The system matrix.
   * \param prm Coarsening parameters.
   * \returns   A tuple of prolongation and restriction operators.
   */
  std::tuple<std::shared_ptr<CSRMatrix<T>>, std::shared_ptr<CSRMatrix<T>>>
  transfer_operators(const CSRMatrix<T>& A) {
    const size_t n = A.m_rows;

    pointwise_aggregates aggr(A, prm.aggr, prm.nullspace.cols);

    std::shared_ptr<CSRMatrix<T>> P = 
      tentative_prolongation<T>(n, aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);

    return std::make_tuple(P, transpose(*P));
  }

  /// Creates system matrix for the coarser level.
  /**
   * \param A The system matrix at the finer level.
   * \param P Prolongation operator returned by transfer_operators().
   * \param R Restriction operator returned by transfer_operators().
   * \returns System matrix for the coarser level.
   */
  std::shared_ptr<CSRMatrix<T>> coarse_operator(const CSRMatrix<T>& A,
                                                const CSRMatrix<T>& P,
                                                const CSRMatrix<T>& R) const
  {
    return scaled_galerkin(A, P, R, 1 / prm.over_interp);
  }
};