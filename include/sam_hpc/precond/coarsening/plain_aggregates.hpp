#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"

/// Builds coarse aggregates using greedy plain aggregation. Off-diagonal entries are
/// classified as strong connections using a symmetric, diagonal-scaled threshold.
/// Variables without strong neighbors are excluded. The remaining variables are assigned
/// during a serial seed pass; later seeds may replace provisional assignments made by
/// earlier seeds.
struct plain_aggregates {
  /// Parameters controlling strong-connection detection.
  struct params {
    /// Threshold used to classify an off-diagonal entry as strong.
    float eps_strong;

    params() : eps_strong(0.08f) {}
  };

  /// Temporary construction state for a variable not yet assigned to an aggregate.
  static const ptrdiff_t undefined = -1;

  /// State for a variable excluded because it has no strong connection.
  static const ptrdiff_t removed = -2;

  /// Number of nonempty aggregates created by the constructor.
  size_t count;

  /// Strong-connection flags corresponding to the nonzeros of the input matrix.
  std::vector<char> strong_connection;

  /// Aggregate identifier assigned to each fine-level variable. `removed` denotes a
  /// variable excluded from coarsening. Surviving identifiers are contiguous in `[0, count)`.
  std::vector<ptrdiff_t> id;

  /// Constructs aggregates for a square CSR system matrix.
  /// \param A System matrix used to determine strong connections.
  /// \param prm Parameters for strong-connection detection.
  /// \throws std::runtime_error If no aggregate can be formed.
  template <typename T>
  plain_aggregates(const csr::CSRMatrix<T>& A, const params& prm)
    : count(0), strong_connection(A.m_nnz), id(A.m_rows)
  {
    T eps_squared = prm.eps_strong * prm.eps_strong;
    const size_t n = A.m_rows;
    std::vector<T> dia = csr::diagonal<csr::diagonalType::simple>(A);

    // Classify strong connections
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // compute the strength threshold for row i
        T eps_dia_i = eps_squared * dia[i];
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          size_t c = A.m_col_indices[j];
          T v = A.m_vals[j];
          // An off-diagonal entry is strong when eps^2 * aii * acc < aic^2
          strong_connection[j] = (c != i) && (eps_dia_i * dia[c] < v * v);
        }
      }
    });

    // Exclude variables without any strong neighbor.
    size_t max_neib = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1];
      max_neib = std::max(max_neib, e - j);

      // Initialize every variable as removed
      ptrdiff_t state = removed;
      for (; j < e; ++j) {
        // If a strong entry is found in the row it is upgraded to undefined
        if (strong_connection[j]) {
          state = undefined;
          break;
        }
      }

      id[i] = state;
    }

    // Growing aggregates from seed order sequentially.
    std::vector<size_t> neib;
    neib.reserve(max_neib);

    for (size_t i = 0; i < n; ++i) {
      // Only an undefined variable can start an aggregate
      if (id[i] != undefined)
        continue;

      ptrdiff_t curr_id = static_cast<ptrdiff_t>(count++);
      id[i] = curr_id;

      // Claim the seed's strong neighbors excluding removed and undefined variables.
      neib.clear();
      for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
        size_t c = A.m_col_indices[j];
        // A neighbor already assigned to an earlier aggregate is taken over
        if (strong_connection[j] && id[static_cast<ptrdiff_t>(c)] != removed) {
          id[static_cast<ptrdiff_t>(c)] = curr_id;
          neib.push_back(c);
        }
      }

      // For each direct neighbor c, scan row c and take c's strong neighbors. A later
      // seed that has cc as a direct neighbor will steal it in its aggregate.
      for (size_t c : neib) {
        for (size_t j = A.m_row_pointers[c], e = A.m_row_pointers[c + 1]; j < e; ++j) {
          size_t cc = A.m_col_indices[j];
          if (strong_connection[j] && id[cc] == undefined) {
            id[cc] = curr_id;
          }
        }
      }
    }

    // If every variable was removed, no aggregate was created. True for diagonal matrix.
    if (!count)
      throw std::runtime_error("The aggregate level is empty");

    // Remove empty aggregates and compact the remaining identifiers.
    std::vector<ptrdiff_t> cnt(count, 0);
    for (ptrdiff_t i : id) {
      if (i >= 0) {
        cnt[i] = 1;
      }
    }
    std::partial_sum(cnt.begin(), cnt.end(), cnt.begin());

    if (static_cast<ptrdiff_t>(count) > cnt.back()) {
      count = cnt.back();
      for (size_t i = 0; i < n; ++i) {
        if (id[i] >= 0) {
          id[i] = cnt[id[i]] - 1;
        }
      }
    }
  }
};
