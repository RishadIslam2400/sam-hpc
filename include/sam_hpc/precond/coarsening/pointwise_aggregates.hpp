#pragma once

#include "sam_hpc/precond/coarsening/plain_aggregates.hpp"

/// Pointwise aggregation.
/**
 * The system matrix should have block structure. It is reduced to a single
 * value per block and is subjected to coarsening::plain_aggregation.
 */

class pointwise_aggregates {
 public:
  /// Aggregation parameter
  struct params : plain_aggregates::params {
    /// Block size for the system matrix
    /**
     * When block_size = 1, the scheme is equivalent to plain_aggregates
     */
    unsigned block_size;
    params() : block_size(1) {}
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
  pointwise_aggregates(const CSRMatrix<T>& A, const params& prm, unsigned min_aggregate)
    : count(0)
  {
    if (prm.block_size == 1) {
      plain_aggregates aggr(A, prm);  // aggr contains the greedy plain aggregation
      remove_small_aggregates(A.m_rows, 1, min_aggregate, aggr);

      count = aggr.count;
      strong_connection.swap(aggr.strong_connection);
      id.swap(aggr.id);
    } else {
      // this is the pointwise version, not required for now
    }
  }

  static void remove_small_aggregates(size_t n, unsigned block_size,
                                      unsigned min_aggregate, plain_aggregates& aggr)
  {
    if (min_aggregate <= 1)
      return;  // nothing to do

    // Count entries in each of the aggregates
    std::vector<ptrdiff_t> count(aggr.count, 0);

    // Iterate through the nodes and count how many nodes are in each aggregate
    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t id = aggr.id[i];
      if (id != removed) {
        ++count[id];
      }
    }

    // If any aggregate has less entries than required, remove it.
    // Renumber the rest of the aggregates to leave no gaps.
    size_t m = 0;
    for (size_t i = 0; i < aggr.count; ++i) {
      // If any aggreagate is less than threshold mark it for removal
      if (block_size * count[i] < min_aggregate) {
        count[i] = removed;
      } else {
        // count contains the new id of the aggregates
        count[i] = m++;
      }
    }

    // Update aggregate count and aggregate ids
    aggr.count = m;
    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t id = aggr.id[i];
      if (id != removed) {
        aggr.id[i] = count[id];  // Update the id to the new id
      }
    }
  }
};