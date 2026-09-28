#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/dense/qr.hpp"

struct skip_negative {
  const std::vector<ptrdiff_t>& key;
  int block_size;

  skip_negative(const std::vector<ptrdiff_t>& key, int block_size)
    : key(key), block_size(block_size) {}

  bool operator()(ptrdiff_t i, ptrdiff_t j) const {
    // Cast to unsigned type to keep negative values at the end
    return static_cast<size_t>(key[i]) / block_size < static_cast<size_t>(key[j]) / block_size;
  }
};

struct nullspace_params {
  /// Number of vectors in near nullspace
  int cols;

  /// Near nullspace vectors
  /**
   * The vectors are represented as columns of a 2D matrix stored in row major order
   */
  std::vector<double> B;

  nullspace_params() : cols(0) {}
};

/// Tentative prolongation operator
/**
 * If near nullspace vectors are not provided, returns piecewise-constant
 * prolongation opearator. If user provides near nullspace vectors, those are
 * used to improve the prlongation operator.
 */
template <typename T>
std::shared_ptr<CSRMatrix<T>> tentative_prolongation(size_t n, size_t naggr, 
                                                    const std::vector<ptrdiff_t> aggr,
                                                    nullspace_params& nullspace,
                                                    int block_size)
{
  std::shared_ptr<CSRMatrix<T>> P = std::make_shared<CSRMatrix<T>>();

  if (nullspace.cols > 0) {
    size_t nba = naggr / block_size;  // total number of aggregates

    // Sort fine points by aggregate ids.
    // Put points not belonging to any aggregate to the end of the list
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) {
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), skip_negative(aggr, block_size));

    // aggr_ptr will store the start and end positions of each aggregates nodes
    // within the sorted order vector
    std::vector<ptrdiff_t> aggr_ptr(nba + 1, 0);
    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t a = aggr[order[i]];  // get the aggregate id for the current node
      if (a < 0)
        break;
      ++aggr_ptr[a / block_size + 1];
    }
    std::partial_sum(aggr_ptr.begin(), aggr_ptr.end(), aggr_ptr.begin());

    // Precompute the shape of the prolongation operator.
    // Each row contains exactly nullspace.cols non-zero entries.
    // Rows that do not belong to any aggregate are empty.
    P->m_rows = n;  // number of fine grid nodes
    P->m_cols = nullspace.cols * nba;  // number of coarse grid nodes = numnber of aggregates * number of
                                       // near nullspace vectors (DOF for each coarse grid node)
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

    // Compute the tentative prolongation opreator and nullspace vectors for the coarser level
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
        auto d = aggr_end - aggr_beg;  // the number of fine-grid nodes within the current aggregate.

        // Bpart has d rows and nullspace.cols columns
        ls.Bpart.resize(d * nullspace.cols);

        // Iterate through each node in the current aggregate
        // Populate Bpart from the global nullspace.B
        for (ptrdiff_t j = aggr_beg, jj = 0; j < aggr_end; ++j, ++jj) {
          // starting index for node j in global B matrix
          ptrdiff_t ib = nullspace.cols * order[j];
          // Iterate through all the columns
          for (int k = 0; k < nullspace.cols; ++k) {
            // Nullspace.B[ib + k] accesses the row for node j
            // Bpart will be a column major, transposed acces to B
            ls.Bpart[jj + d * k] = nullspace.B[ib + k];
          }
        }

        ls.qr.factorize(d, nullspace.cols, &ls.Bpart[0], col_major);

        // R matrix dimension: nullspace.cols x nullspace.cols
        // ii is the row index in the R matrix
        // kk is the 1d index into Bnew
        for (int ii = 0, kk = 0; ii < nullspace.cols; ++ii) {
          // jj iterates through the column of R matrix
          for (int jj = 0; jj < nullspace.cols; ++jj, ++kk) {
            // i * nullspace.cols * nullspace.cols is the block offset for the current aggregate
            Bnew[i * nullspace.cols * nullspace.cols + kk] = ls.qr.R(ii, jj);
          }
        }

        // Populate the prolongation operator with the Q matrix
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