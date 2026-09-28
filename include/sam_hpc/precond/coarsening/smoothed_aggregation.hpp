#pragma once

#include "sam_hpc/precond/coarsening/pointwise_aggregates.hpp"
#include "sam_hpc/precond/coarsening/tentative_prolongation.hpp"

template <typename T>
struct smoothed_aggregation {
  /// Coarsening parameters
  struct params {
    /// Aggregation parameters
    pointwise_aggregates::params aggr;

    /// Near nullspace parameters
    nullspace_params nullspace;

    /// Relaxation factor
    /**
     * Used as a scaling for the damping factor omega.
     * When estimate_spectral_radius is set, then
     *   omega = relax * (4/3) / rho.
     * Otherwise
     *   omega = relax * (2/3).
     *
     * Piecewise constant prolongation \f$\tilde P\f$ from non-smoothed
     * aggregation is improved by a smoothing to get the final prolongation
     * matrix \f$P\f$. Simple Jacobi smoother is used here, giving the
     * prolongation matrix
     * \f[P = \left( I - \omega D^{-1} A^F \right) \tilde P.\f]
     * Here \f$A^F = (a_{ij}^F)\f$ is the filtered matrix given by
     * \f[
     * a_{ij}^F =
     * \begin{cases}
     * a_{ij} \quad \text{if} \; j \in N_i\\
     * 0 \quad \text{otherwise}
     * \end{cases}, \quad \text{if}\; i \neq j,
     * \quad a_{ii}^F = a_{ii} - \sum\limits_{j=1,j\neq i}^n
     * \left(a_{ij} - a_{ij}^F \right),
     * \f]
     * where \f$N_i\f$ is the set of variables, strongly coupled to
     * variable \f$i\f$, and \f$D\f$ denotes the diagonal of \f$A^F\f$.
     */
    float relax;

    // Estimate the matrix spectral radius.
    // This usually improves convergence rate and results in faster solves,
    // but costs some time during setup.
    bool estimate_spectral_radius;

    // Number of power iterations to apply for the spectral radius
    // estimation. Use Gershgorin disk theorem when power_iters = 0.
    int power_iters;

    params() : relax(1.0f), estimate_spectral_radius(false), power_iters(0) {}

  } prm;

  smoothed_aggregation(const params& prm = params()) : prm(prm) {}

  std::tuple<std::shared_ptr<CSRMatrix<T>>, std::shared_ptr<CSRMatrix<T>>>
  transfer_operators(const CSRMatrix<T>& A) {
    const size_t n = A.m_rows;

    pointwise_aggregates aggr(A, prm.aggr, prm.nullspace.cols);
    prm.aggr.eps_strong *= 0.5;

    std::shared_ptr<CSRMatrix<T>> P_tent =
      tentative_prolongation<T>(n, aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);
    std::shared_ptr<CSRMatrix<T>> P = std::make_shared<CSRMatrix<T>>();
    P->m_rows = P_tent->m_rows;
    P->m_cols = P_tent->m_cols;
    P->m_row_pointers.resize(P->m_rows + 1, 0);

    T omega = prm.relax;
    if (prm.estimate_spectral_radius) {
      // For now spectral radius implementation is not needed
    } else {
      omega *= static_cast<T>(2.0 / 3);
    }

    // Smoothing
    tbb::enumerable_thread_specific<std::vector<ptrdiff_t>> local_marker_1(P->m_cols, -1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      std::vector<ptrdiff_t>& marker = local_marker_1.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t nnz = 0;
        for (size_t ja = A.m_row_pointers[i], ea = A.m_row_pointers[i + 1]; ja < ea; ++ja) {
          size_t ca = A.m_col_indices[ja];

          // Skip weak off-diagonal connections
          if (ca != i && !aggr.strong_connection[ja])
            continue;

          for (size_t jp = P_tent->m_row_pointers[ca], ep = P_tent->m_row_pointers[ca + 1]; jp < ep; ++jp) {
            size_t cp = P_tent->m_col_indices[jp];

            if (marker[cp] != static_cast<ptrdiff_t>(i)) {
              marker[cp] = i;
              nnz++;
            }
          }
        }
        P->m_row_pointers[i + 1] = nnz;
      }
    });

    P->m_nnz = P->scanRowSize();
    P->m_col_indices.resize(P->m_nnz, 0);
    P->m_vals.resize(P->m_nnz, T(0));

    local_marker_1.clear();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      std::vector<ptrdiff_t>& marker = local_marker_1.local();

      // Fill the interpolation matrix
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Diagonal of the filtered matrix is the original matrix
        // diagonal minus its weak connections.
        T dia = T(0);
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          if (A.m_col_indices[j] == i || !aggr.strong_connection[j])
            dia += A.m_vals[j];
        }
        if (dia != T(0))
          dia = -omega * (1 / dia);

        size_t row_beg = P->m_row_pointers[i];
        size_t row_end = row_beg;
        for (size_t ja = A.m_row_pointers[i], ea = A.m_row_pointers[i + 1]; ja < ea; ++ja) {
          size_t ca = A.m_col_indices[ja];

          // Skip weak off-diagonal connections.
          if (ca != i && !aggr.strong_connection[ja])
            continue;

          T va = (ca == i) ? static_cast<T>(static_cast<T>(1 - omega) * T(1)) : dia * A.m_vals[ja];
          for (size_t jp = P_tent->m_row_pointers[ca], ep = P_tent->m_row_pointers[ca + 1]; jp < ep; ++jp) {
            size_t cp = P_tent->m_col_indices[jp];
            T vp = P_tent->m_vals[jp];

            if (marker[cp] < static_cast<ptrdiff_t>(row_beg)) {
              marker[cp] = row_end;
              P->m_col_indices[row_end] = cp;
              P->m_vals[row_end] = va * vp;
              ++row_end;
            } else {
              P->m_vals[marker[cp]] += va * vp;
            }
          }
        }

        for (size_t k = row_beg; k < row_end; ++k) {
          marker[P->m_col_indices[k]] = -1;
        }
      }
    });

    return std::make_tuple(P, transpose(*P));
  }

  std::shared_ptr<CSRMatrix<T>> coarse_operator(const CSRMatrix<T>& A,
                                                const CSRMatrix<T>& P,
                                                const CSRMatrix<T>& R) const
  {
    return galerkin(A, P, R);
  }
};