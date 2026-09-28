#pragma once

/// Plain aggregation
/**
 * Modification of a greedy aggregation scheme from smoothed aggregation paper.
 * Connectivity is defined in a symmetric way, that is, two variables i and j
 * are considered to be connected to each other if a_{ij}^2 / a_{ii}a_{jj} >
 * eps_strong. Variables without neighbors (resulting from Dirichlet conditions)
 * are excluded from aggregation process. The aggregation is completed in a
 * single pass over variables: variables adjacent to a new aggregate are
 * temporarily marked belonging to this aggregate. Later they may be claimed by
 * other aggregates; if nobody claims them, then they just stay in their initial
 * aggregate.
 */

#include "sam_hpc/core/CSRMatrix.hpp"

struct plain_aggregates {
  /// Aggregation parameters.
  struct params {
    /// Parameter defining strong couplings
    /**
     * Connectivity is defined in a symmetric way, that is, two variables i and
     * j are considered to be connected to each other if a_{ij}^2 / a_{ii}a_{jj}
     * > eps_strong with fixed 0 < eps_strong < 1.
     */
    float eps_strong;
    params() : eps_strong(0.08f) {}
  };

  static const ptrdiff_t undefined = -1;
  static const ptrdiff_t removed = -2;

  /// Number of aggregates
  size_t count;

  /// Strong connectivity matrix
  /**
   * This is just 'values' part of CSR matrix. 'col' and 'ptr' arrays are
   * borrowed from the system matrix.
   */
  std::vector<char> strong_connection;

  /// Aggregate id that each fine level variable belongs to
  /** When id[i] < 0, then variable i stays at the fine level (this could be
   * the case for a Dirichelt condition variable).*/
  std::vector<ptrdiff_t> id;

  /// Constructs aggregates for a given matrix.
  /**
   * \param A   The system matrix.
   * \param prm Aggregation parameters.
   */
  template <typename T>
  plain_aggregates(const CSRMatrix<T>& A, const params& prm)
    : count(0), strong_connection(A.m_nnz), id(A.m_rows)
  {
    T eps_squared = prm.eps_strong * prm.eps_strong;
    const size_t n = A.m_rows;

    // 1. Get strong connections
    std::vector<T> dia = diagonal<diagonalType::simple>(A);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        T eps_dia_i = eps_squared * dia[i];
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          size_t c = A.m_col_indices[j];
          T v = A.m_vals[j];
          strong_connection[j] = (c != i) && (eps_dia_i * dia[c] < v * v);
        }
      }
    });

    // 2. Get aggregate ids
    // Remove lonely nodes
    size_t max_neib = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1];
      max_neib = std::max(max_neib, e - j);

      ptrdiff_t state = removed;
      for (; j < e; ++j) {
        // check if node i has at least one strong connection otherwise the node is lonely
        if (strong_connection[j]) {
          state = undefined;
          break;
        }
      }

      id[i] = state;  // currently id holds the initial state for every node
    }

    std::vector<size_t> neib;
    neib.reserve(max_neib);

    // Perform plain aggregation
    for (size_t i = 0; i < n; ++i) {
      // checks if current node has already been assigned to an aggregate or marked as removed
      if (id[i] != undefined) continue;

      // The point is not adjacent to a core of any previous aggregate:
      // so its a seed of a new aggregate
      ptrdiff_t curr_id = static_cast<ptrdiff_t>(count++);
      id[i] = curr_id;  // node i is assigned to the new aggregate

      // (*) Include its neighbors as well
      neib.clear();
      for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
        size_t c = A.m_col_indices[j];
        // include the neighbor c of node i in the aggregate if they are
        // strongly coupled and not marked as lonely node
        if (strong_connection[j] && id[static_cast<ptrdiff_t>(c)] != removed) {
          id[static_cast<ptrdiff_t>(c)] = curr_id;  // the neighbor c is assigned to new aggregate id
          neib.push_back(c);
        }
      }

      // Temporarily mark undefined points adjacent to the new aggregate as
      // members of the aggregate. If nobody claims them later, they will stay there
      for (size_t c : neib) {
        for (size_t j = A.m_row_pointers[c], e = A.m_row_pointers[c + 1]; j < e; ++j) {
          size_t cc = A.m_col_indices[j];
          if (strong_connection[j] && id[cc] == undefined) {
            id[cc] = curr_id;
          }
        }
      }
    }

    if (!count)
      throw std::runtime_error("The aggregate level is empty");

    // Some of the aggregates could potentially vanish during expansion step (*)
    // above. We need to exclude those and renumber the rest.
    std::vector<ptrdiff_t> cnt(count, 0);
    for (ptrdiff_t i : id) {
      // checks if aggregate id is valid
      if (i >= 0) {
        cnt[i] = 1;
      }
    }

    std::partial_sum(cnt.begin(), cnt.end(), cnt.begin());

    // cnt.back() equals to total number of aggregates
    if (static_cast<ptrdiff_t>(count) > cnt.back()) {
      count = cnt.back();  // update the aggregate count
      // update aggregate id assignment of each node
      for (size_t i = 0; i < n; ++i) {
        if (id[i] >= 0) {
          id[i] = cnt[id[i]] - 1;  // convert to 0-based indexing
        }
      }
    }
  }
};
