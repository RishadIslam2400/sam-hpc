#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "CSRMatrix.hpp"
#include "eigenQRSolve.hpp"
#include "householderQR.hpp"
#include "launchThreads.hpp"
#include "mgsQR.hpp"
#include "qr.hpp"
#include "sparsityPattern.hpp"

/// Per-phase breakdown of computeMap, collected only by computeMapProfiled().
struct sam_phase_times {
  uint64_t symbolic_ns = 0;   // row set J, its ordering, and the dense index map
  uint64_t rhs_ns = 0;        // gather of the target row into the dense rhs
  uint64_t submatrix_ns = 0;  // scatter of the source rows into the dense local matrix
  uint64_t solve_ns = 0;      // the least squares solve itself

  size_t rows = 0;            // rows with a non-empty pattern
  size_t sum_I = 0, max_I = 0;
  size_t sum_J = 0, max_J = 0;
  double lsq_flops = 0;       // 2|J||I|^2 - (2/3)|I|^3, summed over rows

  uint64_t total_ns() const {
    return symbolic_ns + rhs_ns + submatrix_ns + solve_ns;
  }

  void merge(const sam_phase_times& o) {
    symbolic_ns += o.symbolic_ns;
    rhs_ns += o.rhs_ns;
    submatrix_ns += o.submatrix_ns;
    solve_ns += o.solve_ns;
    rows += o.rows;
    sum_I += o.sum_I;
    sum_J += o.sum_J;
    max_I = std::max(max_I, o.max_I);
    max_J = std::max(max_J, o.max_J);
    lsq_flops += o.lsq_flops;
  }
};

/// Least squares residual of the map, summed over rows, collected by computeMapResidual().
///
/// This is the residual the local problems actually minimize, ||N A_k - A_0||_F.
struct sam_residual {
  double sum_sq = 0.0;

  double frobenius() const { return std::sqrt(sum_sq); }
};

/// Relative accuracy of a computed map, from the explicit product.
struct sam_accuracy {
  double residual_frobenius = 0.0;  // ||N A_k - A_0||_F
  double target_frobenius = 0.0;    // ||A_0||_F

  double relative() const {
    return target_frobenius > 0.0 ? residual_frobenius / target_frobenius : 0.0;
  }
};

/// Exact ||N A_k - A_0||_F and ||A_0||_F.
///
/// Does not compute N A_k explicitly. Each row is accumulated into a thread-local dense
/// scratch, compared against the matching row of A_0, and discarded.
template <typename T>
inline sam_accuracy sam_residual_exact(const CSRMatrix<T>& map,
                                       const CSRMatrix<T>& sourceMatrix,
                                       const CSRMatrix<T>& targetMatrix)
{
  struct scratch {
    std::vector<double> acc;
    std::vector<char> touched;
    std::vector<size_t> hits;
    scratch(size_t n) : acc(n, 0.0), touched(n, 0) {}
  };
  tbb::enumerable_thread_specific<scratch> local(sourceMatrix.m_cols);

  auto acc = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, targetMatrix.m_rows),
    std::pair<double, double>(0.0, 0.0),
    [&](const tbb::blocked_range<size_t>& r, std::pair<double, double> sums) {
      scratch& s = local.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        s.hits.clear();

        // Row i of the left map: (N A_k)(i,:) = sum_j N(i,j) * A_k(j,:)
        for (size_t j = map.m_row_pointers[i]; j < map.m_row_pointers[i + 1]; ++j) {
          const size_t srcRow = map.m_col_indices[j];
          const double coeff = map.m_vals[j];
          for (size_t k = sourceMatrix.m_row_pointers[srcRow];
               k < sourceMatrix.m_row_pointers[srcRow + 1]; ++k)
          {
            const size_t c = sourceMatrix.m_col_indices[k];
            if (!s.touched[c]) {
              s.touched[c] = 1;
              s.acc[c] = 0.0;
              s.hits.push_back(c);
            }
            s.acc[c] += coeff * sourceMatrix.m_vals[k];
          }
        }

        // ... minus row i of A_0
        for (size_t k = targetMatrix.m_row_pointers[i];
             k < targetMatrix.m_row_pointers[i + 1]; ++k)
        {
          const size_t c = targetMatrix.m_col_indices[k];
          const double b = targetMatrix.m_vals[k];
          if (!s.touched[c]) {
            s.touched[c] = 1;
            s.acc[c] = 0.0;
            s.hits.push_back(c);
          }
          s.acc[c] -= b;
          sums.second += b * b;
        }

        for (const size_t c : s.hits) {
          sums.first += s.acc[c] * s.acc[c];
          s.touched[c] = 0;
        }
      }
      return sums;
    },
    [](std::pair<double, double> a, std::pair<double, double> b) {
      return std::pair<double, double>(a.first + b.first, a.second + b.second);
    });

  return {std::sqrt(acc.first), std::sqrt(acc.second)};
}

template <typename T>
struct sam_internal_storage {
  csr::vec<csr::index_t> J;
  std::vector<T> rhs;
  std::vector<T> submatrix;
  std::vector<int> marker;
  QR<T> qr;

  sam_internal_storage(const size_t marker_size) : marker(marker_size, -1) {}
};

template <typename T, typename SparsityPatternType>
class SparseApproximateMap {
 public:
  // Computes the map for an already extracted sparsity pattern matrix.
  static void computeMap(const CSRMatrix<T>& targetMatrix, const CSRMatrix<T>& sourceMatrix,
                         const CSRMatrix<int>& pattern, CSRMatrix<T>& mappingMatrix)
  {
    computeMapImpl<false, false>(targetMatrix, sourceMatrix, pattern, mappingMatrix,
                                 nullptr, nullptr);
  }

  static void computeMap(const CSRMatrix<T>& targetMatrix, const CSRMatrix<T>& sourceMatrix,
                         const SparsityPattern<T, SparsityPatternType>& sparsityPattern,
                         CSRMatrix<T>& mappingMatrix)
  {
    computeMap(targetMatrix, sourceMatrix, *sparsityPattern.getPattern(), mappingMatrix);
  }

  /// Same computation, with a per-phase timing breakdown. The extra clock reads
  /// cost a few percent of the loop, so keep this out of the numbers you report
  /// as SAM's cost.
  static void computeMapProfiled(const CSRMatrix<T>& targetMatrix, const CSRMatrix<T>& sourceMatrix,
                                 const CSRMatrix<int>& pattern, CSRMatrix<T>& mappingMatrix,
                                 sam_phase_times& profile)
  {
    computeMapImpl<true, false>(targetMatrix, sourceMatrix, pattern, mappingMatrix,
                                &profile, nullptr);
  }

  /// Same computation, additionally accumulating the least squares residual of
  /// every row. The QR has already produced it, so this reads O(|J| - |I|)
  /// values per row against O(|J||I|^2) of factorization work - cheap enough to
  /// leave on in measurement runs. Exact - J covers the whole residual support.
  static void computeMapResidual(const CSRMatrix<T>& targetMatrix, const CSRMatrix<T>& sourceMatrix,
                                 const CSRMatrix<int>& pattern, CSRMatrix<T>& mappingMatrix,
                                 sam_residual& residual)
  {
    computeMapImpl<false, true>(targetMatrix, sourceMatrix, pattern, mappingMatrix,
                                nullptr, &residual);
  }

 private:
  template <bool Profile, bool Residual>
  static void computeMapImpl(const CSRMatrix<T>& targetMatrix, const CSRMatrix<T>& sourceMatrix,
                             const CSRMatrix<int>& pattern, CSRMatrix<T>& mappingMatrix,
                             sam_phase_times* profile, sam_residual* residual)
  {
    using clock = std::chrono::steady_clock;
    const auto ns = [](clock::time_point a, clock::time_point b) {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                                   .count());
    };

    // Initialize the mapping matrix
    mappingMatrix.m_rows = targetMatrix.m_rows;
    mappingMatrix.m_cols = targetMatrix.m_cols;
    mappingMatrix.m_nnz = pattern.m_nnz;
    mappingMatrix.m_row_pointers = pattern.m_row_pointers;
    mappingMatrix.m_col_indices = pattern.m_col_indices;
    mappingMatrix.m_vals.resize(mappingMatrix.m_nnz);

    // Start SAM computation
    // J holds column indices of the source and of the target row; size the marker for
    // whichever operand is widest.
    tbb::enumerable_thread_specific<sam_internal_storage<T>> local_storages(std::max({mappingMatrix.m_cols, sourceMatrix.m_cols, static_cast<size_t>(pattern.m_cols)}));
    tbb::enumerable_thread_specific<sam_phase_times> local_profiles;
    tbb::enumerable_thread_specific<double> local_residuals(0.0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mappingMatrix.m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      sam_internal_storage<T>& ls = local_storages.local();
      [[maybe_unused]] sam_phase_times& lp = local_profiles.local();
      [[maybe_unused]] double& lr = local_residuals.local();
      [[maybe_unused]] clock::time_point t0, t1;
      for (size_t i = r.begin(); i < r.end(); ++i) {
        if constexpr (Profile) t0 = clock::now();
        // Compute the submatrix indices for each row
        const size_t rowStart = pattern.m_row_pointers[i];
        const size_t rowEnd = pattern.m_row_pointers[i + 1];
        size_t iSize = rowEnd - rowStart;

        if (iSize == 0) {
          // No map entries in this row, so (N A_k)(i,:) is zero and the
          // whole target row survives as residual.
          if constexpr (Residual) {
            for (size_t k = targetMatrix.m_row_pointers[i]; k < targetMatrix.m_row_pointers[i + 1]; ++k) {
              lr += static_cast<double>(targetMatrix.m_vals[k]) * targetMatrix.m_vals[k];
            }
          }
          continue;
        }

        ls.J.clear();
        const auto add = [&ls](const csr::index_t col) {
          if (ls.marker[col] == -1) {
            ls.marker[col] = 1;
            ls.J.push_back(col);
          }
        };

        // I indexes rows of the source matrix, and choosing it is the sparsity
        // pattern's entire job. J is not a second choice: once I is fixed, row i of
        // the residual is the determined vector n_i^T A_k - a_0i^T, so J is simply
        // its support - the columns those source rows reach, plus the target row's.
        for (size_t j = rowStart; j < rowEnd; ++j) {
          const size_t k = pattern.m_col_indices[j];
          for (size_t p = sourceMatrix.m_row_pointers[k]; p < sourceMatrix.m_row_pointers[k + 1]; ++p) {
            add(sourceMatrix.m_col_indices[p]);
          }
        }

        // Target entries outside the reach of those source rows are irreducible
        // residual. Carrying them as all-zero rows of the local matrix leaves the
        // minimizer untouched - they add a constant b^2 for every x - and makes the
        // QR's own residual exactly ||N A_k - A_0||_F.
        for (size_t p = targetMatrix.m_row_pointers[i]; p < targetMatrix.m_row_pointers[i + 1]; ++p) {
          add(targetMatrix.m_col_indices[p]);
        }
        std::sort(ls.J.begin(), ls.J.end());
        const size_t jSize = ls.J.size();

        // Use the marker as a map from sparse index to dense index (0 to |J|-1)
        for (size_t k = 0; k < jSize; ++k) {
          ls.marker[ls.J[k]] = k;
        }

        if constexpr (Profile) {
          t1 = clock::now();
          lp.symbolic_ns += ns(t0, t1);
          t0 = t1;
          ++lp.rows;
          lp.sum_I += iSize;
          lp.max_I = std::max(lp.max_I, iSize);
          lp.sum_J += jSize;
          lp.max_J = std::max(lp.max_J, jSize);
          const double I = static_cast<double>(iSize), J = static_cast<double>(jSize);
          lp.lsq_flops += 2.0 * J * I * I - (2.0 / 3.0) * I * I * I;
        }

        // Compute the RHS vector from the target matrix
        ls.rhs.resize(jSize);
        std::fill(ls.rhs.begin(), ls.rhs.end(), T(0));
        for (size_t k = targetMatrix.m_row_pointers[i]; k < targetMatrix.m_row_pointers[i + 1]; ++k) {
          const size_t col = targetMatrix.m_col_indices[k];
          const int dense_idx = ls.marker[col];
          if (dense_idx != -1) {
            ls.rhs[dense_idx] = targetMatrix.m_vals[k];
          }
        }

        if constexpr (Profile) {
          t1 = clock::now();
          lp.rhs_ns += ns(t0, t1);
          t0 = t1;
        }

        // Compute the submatrix
        ls.submatrix.resize(jSize * iSize);
        std::fill(ls.submatrix.begin(), ls.submatrix.end(), T(0));
        for (size_t j = 0; j < iSize; ++j) {
          const size_t source_row_idx = pattern.m_col_indices[rowStart + j];

          for (size_t k = sourceMatrix.m_row_pointers[source_row_idx]; k < sourceMatrix.m_row_pointers[source_row_idx + 1]; ++k) {
            const size_t col = sourceMatrix.m_col_indices[k];
            const int dense_idx = ls.marker[col];
            if (dense_idx != -1) {
              ls.submatrix[dense_idx + jSize * j] = sourceMatrix.m_vals[k];
            }
          }
        }

        // Cleanup the marker for the next iteration
        for (size_t col : ls.J) {
          ls.marker[col] = -1;
        }

        if constexpr (Profile) {
          t1 = clock::now();
          lp.submatrix_ns += ns(t0, t1);
          t0 = t1;
        }

        ls.qr.solve(jSize, iSize, ls.submatrix.data(), ls.rhs.data(),
                    &mappingMatrix.m_vals[rowStart], storage_order::col_major);

        if constexpr (Residual)
          lr += static_cast<double>(ls.qr.residual_sq());

        if constexpr (Profile) {
          t1 = clock::now();
          lp.solve_ns += ns(t0, t1);
        }
      }
    });

    if constexpr (Profile) {
      for (const sam_phase_times& lp : local_profiles)
        profile->merge(lp);
    }
    if constexpr (Residual) {
      for (const double lr : local_residuals)
        residual->sum_sq += lr;
    }
  }

 public:
  static void post_filtration(const CSRMatrix<T>& map, CSRMatrix<T>& final_map, T threshold) {
    final_map.m_rows = map.m_rows;
    final_map.m_cols = map.m_cols;
    final_map.m_row_pointers.resize(final_map.m_rows + 1, 0);

    // Filter on magnitude: a large negative entry carries as much of the map as
    // a large positive one. Both passes below share this predicate deliberately
    // - if they ever disagree, the fill pass writes past the space the count
    // pass sized.
    const auto keep = [threshold](const T val) {
      return std::abs(val) >= threshold;
    };

    // count non-zero in the final map
    tbb::parallel_for(tbb::blocked_range<size_t>(0, final_map.m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        int count = 0;
        for (size_t j = map.m_row_pointers[i], e = map.m_row_pointers[i + 1]; j < e; ++j) {
          if (keep(map.m_vals[j])) {
            count++;
          }
        }

        final_map.m_row_pointers[i + 1] = count;
      }
    });

    final_map.m_nnz = final_map.scanRowSize();
    final_map.m_col_indices.resize(final_map.m_nnz);
    final_map.m_vals.resize(final_map.m_nnz);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, final_map.m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t current_pos = final_map.m_row_pointers[i];

        for (size_t j = map.m_row_pointers[i]; j < map.m_row_pointers[i + 1]; ++j) {
          if (keep(map.m_vals[j])) {
            final_map.m_col_indices[current_pos] = map.m_col_indices[j];
            final_map.m_vals[current_pos] = map.m_vals[j];
            current_pos++;
          }
        }
      }
    });
  }
};