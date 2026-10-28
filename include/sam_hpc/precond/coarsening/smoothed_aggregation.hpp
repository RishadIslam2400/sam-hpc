#pragma once

#include "sam_hpc/precond/coarsening/galerkin.hpp"
#include "sam_hpc/precond/coarsening/pointwise_aggregates.hpp"
#include "sam_hpc/precond/coarsening/tentative_prolongation.hpp"

/// Smoothed aggregation with Jacobi-smoothed tentative prolongation.
template <typename T>
struct smoothed_aggregation {
  /// Parameters controlling aggregation, smoothing, and near-nullspace data.
  struct params {
    /// Aggregation and strong-connection parameters.
    /// The strength threshold is halved after each transfer-operator construction.
    pointwise_aggregates::params aggr;

    /// Near-nullspace vectors used to construct tentative prolongation.
    nullspace_params nullspace;

    /// User scaling for the Jacobi smoothing factor.
    float relax;

    /// Selects spectral-radius-based damping when that estimator is implemented.
    bool estimate_spectral_radius;

    /// Number of power iterations reserved for spectral-radius estimation.
    int power_iters;

    params() : relax(1.0f), estimate_spectral_radius(false), power_iters(0) {}

  } prm;

  smoothed_aggregation(const params& prm = params()) : prm(prm) {}

  /// Builds smoothed prolongation and its transpose.
  /// \param A Square system matrix with sorted rows.
  /// \returns `(P, R)` where `R = P^T` and `P` is the smoothed prolongation.
  ///          Output rows are not guaranteed to be sorted.
  /// \throws std::runtime_error If no aggregate can be formed.
  std::tuple<std::shared_ptr<csr::CSRMatrix<T>>, std::shared_ptr<csr::CSRMatrix<T>>>
  transfer_operators(const csr::CSRMatrix<T>& A) {
    const size_t n = A.m_rows;

    // Find strong connections based on current strength threshold.
    pointwise_aggregates aggr(A, prm.aggr, prm.nullspace.cols);
    // Update the threshold for the next coarsening level.
    prm.aggr.eps_strong *= 0.5;

    // Generate P_tent with n rows and aggr.count x nullspace.cols columns
    std::shared_ptr<csr::CSRMatrix<T>> P_tent =
      tentative_prolongation<T>(n, aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);
    std::shared_ptr<csr::CSRMatrix<T>> P = std::make_shared<csr::CSRMatrix<T>>();
    P->m_rows = P_tent->m_rows;
    P->m_cols = P_tent->m_cols;
    P->m_row_pointers.resize(P->m_rows + 1, 0);

    // Damping factor damped-jacobi smoothing
    T omega = prm.relax;
    if (prm.estimate_spectral_radius) {
      throw std::runtime_error("smoothed_aggregation: spectral radius estimation is not implemented");
    } else {
      omega *= static_cast<T>(2.0 / 3);
    }

    // Count the distinct coarse columns in each row of A_F P_tent.
    tbb::enumerable_thread_specific<std::vector<ptrdiff_t>> markers(P->m_cols, -1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      std::vector<ptrdiff_t>& marker = markers.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t nnz = 0;
        for (size_t ja = A.m_row_pointers[i], ea = A.m_row_pointers[i + 1]; ja < ea; ++ja) {
          size_t ca = A.m_col_indices[ja];

          if (ca != i && !aggr.strong_connection[ja])
            continue;

          // For each entry (i, ca) of A that is diagonal or a strong connection
          for (size_t jp = P_tent->m_row_pointers[ca], ep = P_tent->m_row_pointers[ca + 1]; jp < ep; ++jp) {
            size_t cp = P_tent->m_col_indices[jp];

            // For each entry (ca, cp) of P_tent, count the entry not yet seen.
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

    markers.clear();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      std::vector<ptrdiff_t>& marker = markers.local();

      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Lump weak off-diagonals into the filtered diagonal.
        T dia = T(0);
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          if (A.m_col_indices[j] == i || !aggr.strong_connection[j])
            dia += A.m_vals[j];
        }

        // Compute the scaling factor -omega / D_ii.
        if (dia != T(0))
          dia = -omega * (1 / dia);

        size_t row_beg = P->m_row_pointers[i];
        size_t row_end = row_beg;
        for (size_t ja = A.m_row_pointers[i], ea = A.m_row_pointers[i + 1]; ja < ea; ++ja) {
          size_t ca = A.m_col_indices[ja];

          if (ca != i && !aggr.strong_connection[ja])
            continue;

          // Entry (i, ca) of the smoother matrix S = I − ω D_F⁻¹ A_F:
          // On the diagonal, 1 − ω · A_F,ii / D_F,ii = 1 − ω.
          // For a strong off-diagonal, the value is −ω · A_ica / D_F,ii.
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

    return std::make_tuple(P, std::make_shared<csr::CSRMatrix<T>>(transpose(*P)));
  }

  /// Forms the coarse operator `R A P`.
  std::shared_ptr<csr::CSRMatrix<T>> coarse_operator(const csr::CSRMatrix<T>& A,
                                                     const csr::CSRMatrix<T>& P,
                                                     const csr::CSRMatrix<T>& R) const
  {
    return galerkin(A, P, R);
  }
};