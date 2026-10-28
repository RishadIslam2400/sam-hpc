#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/dense/qr.hpp"

/// Orders fine-grid indices by aggregate block, placing excluded variables last.
struct skip_negative {
  const std::vector<ptrdiff_t>& key;
  int block_size;

  skip_negative(const std::vector<ptrdiff_t>& key, int block_size)
    : key(key), block_size(block_size) {}

  bool operator()(ptrdiff_t i, ptrdiff_t j) const {
    // Negative aggregate IDs convert to large unsigned values and sort last.
    return static_cast<size_t>(key[i]) / block_size < static_cast<size_t>(key[j]) / block_size;
  }
};

/// Describes near-nullspace vectors propagated through the aggregation hierarchy.
struct nullspace_params {
  /// Number of near-nullspace vectors.
  int cols;

  /// Row-major near-nullspace vectors, with `B[i * cols + k]` representing vector `k` at
  /// fine variable `i`. The aggregation hierarchy replaces this data with coarse-level vectors.
  std::vector<double> B;

  nullspace_params() : cols(0) {}
};

/// Builds the tentative prolongation operator.
/// \param n Number of fine-level variables.
/// \param naggr Number of aggregate IDs.
/// \param aggr Aggregate ID for each fine-level variable; negative IDs are excluded.
/// \param nullspace In/out near-nullspace data. Its vectors are replaced by the
///                  coarse vectors when `cols > 0`.
/// \param block_size Aggregate block size used by the nullspace path.
/// \returns A CSR prolongation matrix. Nullspace rows occupy contiguous coarse
///          column blocks; rows are not necessarily sorted.
template <typename T>
std::shared_ptr<csr::CSRMatrix<T>> tentative_prolongation(size_t n, size_t naggr,
                                                          const std::vector<ptrdiff_t> aggr,
                                                          nullspace_params& nullspace,
                                                          int block_size)
{
  std::shared_ptr<csr::CSRMatrix<T>> P = std::make_shared<csr::CSRMatrix<T>>();

  // With a near-nullspace, each aggregate’s piece of the nullspace is QR-factored.
  // Q becomes the rows of P, and R becomes the coarse-level nullspace.
  if (nullspace.cols > 0) {
    size_t nba = naggr / block_size;

    // Group fine variables by aggregate block.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) {
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), skip_negative(aggr, block_size));

    // Store the range of sorted variables belonging to each aggregate block.
    std::vector<ptrdiff_t> aggr_ptr(nba + 1, 0);
    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t a = aggr[order[i]];
      if (a < 0)
        break;
      ++aggr_ptr[a / block_size + 1];
    }
    std::partial_sum(aggr_ptr.begin(), aggr_ptr.end(), aggr_ptr.begin());

    // Each aggregated row receives one entry per near-nullspace vector.
    P->m_rows = n;
    P->m_cols = nullspace.cols * nba;
    P->m_row_pointers.resize(n + 1, 0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        P->m_row_pointers[i + 1] = (aggr[i] < 0) ? 0 : nullspace.cols;
      }
    });

    P->m_nnz = P->scanRowSize();
    P->m_col_indices.resize(P->m_nnz);
    P->m_vals.resize(P->m_nnz);

    // Build Q blocks of P and the corresponding coarse nullspace blocks.
    std::vector<double> Bnew;
    Bnew.resize(nba * nullspace.cols * nullspace.cols);

    struct thread_local_solver {
      QR<double> qr;
      std::vector<double> Bpart;
    };
    tbb::enumerable_thread_specific<thread_local_solver> local_solver;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nba),
    [&](const tbb::blocked_range<size_t>& r) {
      thread_local_solver& ls = local_solver.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        ptrdiff_t aggr_beg = aggr_ptr[i];
        ptrdiff_t aggr_end = aggr_ptr[i + 1];
        auto d = aggr_end - aggr_beg;

        ls.Bpart.resize(d * nullspace.cols);

        for (ptrdiff_t j = aggr_beg, jj = 0; j < aggr_end; ++j, ++jj) {
          ptrdiff_t ib = nullspace.cols * order[j];
          for (int k = 0; k < nullspace.cols; ++k) {
            ls.Bpart[jj + d * k] = nullspace.B[ib + k];
          }
        }

        ls.qr.factorize(d, nullspace.cols, &ls.Bpart[0], col_major);

        // Copy the aggregate's R factor into its coarse nullspace block.
        for (int ii = 0, kk = 0; ii < nullspace.cols; ++ii) {
          for (int jj = 0; jj < nullspace.cols; ++jj, ++kk) {
            Bnew[i * nullspace.cols * nullspace.cols + kk] = ls.qr.R(ii, jj);
          }
        }

        for (ptrdiff_t j = aggr_beg, ii = 0; j < aggr_end; ++j, ++ii) {
          csr::index_t* c = &P->m_col_indices[P->m_row_pointers[order[j]]];
          T* v = &P->m_vals[P->m_row_pointers[order[j]]];

          for (int jj = 0; jj < nullspace.cols; ++jj) {
            c[jj] = i * nullspace.cols + jj;
            v[jj] = ls.qr.Q(ii, jj) * T(1);
          }
        }
      }
    });

    std::swap(nullspace.B, Bnew);
  } else {
    // P is piecewise constant, every fine variable interpolates from its aggregate with weight 1.
    P->m_rows = n;
    P->m_cols = naggr;
    P->m_row_pointers.resize(n + 1, 0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        P->m_row_pointers[i + 1] = (aggr[i] >= 0);
      }
    });

    P->m_nnz = P->scanRowSize();
    P->m_col_indices.resize(P->m_nnz);
    P->m_vals.resize(P->m_nnz);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        if (aggr[i] >= 0) {
          size_t row_start = P->m_row_pointers[i];
          P->m_col_indices[row_start] = aggr[i];
          P->m_vals[row_start] = T(1);
        }
      }
    });
  }

  return P;
}