#pragma once

#include <stdexcept>

#include "sam_hpc/precond/coarsening/plain_aggregates.hpp"

/// Builds pointwise aggregates for AMG coarsening. The current implementation supports
/// scalar systems only (`block_size == 1`). It applies plain aggregation and removes
/// aggregates smaller than the requested minimum size.
class pointwise_aggregates {
 public:
  /// Parameters controlling pointwise aggregation.
  struct params : plain_aggregates::params {
    /// Number of scalar variables represented by each matrix block.
    unsigned block_size;

    params() : block_size(1) {}
  };

  /// Temporary construction state for an unassigned variable.
  static const ptrdiff_t undefined = -1;

  /// State for a variable excluded from coarsening.
  static const ptrdiff_t removed = -2;

  /// Number of surviving aggregates.
  size_t count;

  /// Strong-connection flags corresponding to the input matrix's nonzeros.
  std::vector<char> strong_connection;

  /// Aggregate identifier assigned to each fine-level variable. `removed` denotes a
  /// variable excluded from coarsening. Surviving identifiers are contiguous in `[0, count)`.
  std::vector<ptrdiff_t> id;

  /// Constructs and filters pointwise aggregates. The matrix must be square with sorted
  /// rows. Aggregates smaller than `min_aggregate` are dissolved and their variables are
  /// marked `removed`.
  /// \param A System matrix used to determine strong connections.
  /// \param prm Aggregation and block-size parameters.
  /// \param min_aggregate Minimum aggregate size after block scaling.
  /// \throws std::runtime_error If no aggregate can be formed or block aggregation is requested.
  template <typename T>
  pointwise_aggregates(const csr::CSRMatrix<T>& A, const params& prm, unsigned min_aggregate) : count(0) {
    if (prm.block_size == 1) {
      plain_aggregates aggr(A, prm);
      remove_small_aggregates(A.m_rows, 1, min_aggregate, aggr);
      if (aggr.count == 0)
        throw std::runtime_error("The aggregate level is empty");

      count = aggr.count;
      strong_connection.swap(aggr.strong_connection);
      id.swap(aggr.id);
    } else {
      throw std::runtime_error(
        "pointwise_aggregates: block_size > 1 is not implemented; only block_size == 1 "
        "is supported (it reduces to plain_aggregates + remove_small_aggregates)"
      );
    }
  }

  /// Removes aggregates whose block-scaled size is below the threshold. Members of
  /// dissolved aggregates become `removed`. Surviving aggregates are renumbered in their
  /// original order. The input identifiers must be either `removed` or valid aggregate IDs.
  static void remove_small_aggregates(size_t n, unsigned block_size, unsigned min_aggregate,
                                      plain_aggregates& aggr)
  {
    if (min_aggregate <= 1)
      return;

    std::vector<ptrdiff_t> count(aggr.count, 0);

    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t id = aggr.id[i];
      if (id != removed) {
        ++count[id];
      }
    }

    // Mark undersized aggregates and assign compact IDs to survivors.
    size_t m = 0;
    for (size_t i = 0; i < aggr.count; ++i) {
      if (block_size * count[i] < min_aggregate) {
        count[i] = removed;
      } else {
        count[i] = m++;
      }
    }

    aggr.count = m;
    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t id = aggr.id[i];
      if (id != removed) {
        aggr.id[i] = count[id];
      }
    }
  }
};