#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/util/parallel.hpp"
#include "sam_hpc/dense/qr.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"

/// Per-phase counters collected by computeMapProfiled(). Time fields are aggregate
/// worker-thread time. Size fields count work across all processed rows.
struct sam_phase_times {
  uint64_t symbolic_ns = 0;
  uint64_t rhs_ns = 0;
  uint64_t submatrix_ns = 0;
  uint64_t solve_ns = 0;

  size_t rows = 0;
  size_t sum_I = 0, max_I = 0;
  size_t sum_J = 0, max_J = 0;
  double lsq_flops = 0;
  size_t pivoted_rows = 0;

  uint64_t total_ns() const {
    return symbolic_ns + rhs_ns + submatrix_ns + solve_ns;
  }

  /// Merges another worker's counters into this one.
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
    pivoted_rows += o.pivoted_rows;
  }
};

/// Squared Frobenius residual accumulated during map computation. The QR least sqaures
/// residual of each row provides the residual.
struct sam_residual {
  double sum_sq = 0.0;

  // Compute the absolute error ||N A_k - A_0||_F
  double frobenius() const { return std::sqrt(sum_sq); }
};

/// Frobenius norms used to report map accuracy.
struct sam_accuracy {
  double residual_frobenius = 0.0; // ||N A_k - A_0||_F 
  double target_frobenius = 0.0;   // ||A_0||_F 

  /// Relative error, ||N A_k - A_0||_F / ||A_0||_F
  double relative() const {
    return target_frobenius > 0.0 ? residual_frobenius / target_frobenius : 0.0;
  }
};

/// Computes ||N A_k - A_0||_F and ||A_0||_F independently after map computation.
template <typename T>
inline sam_accuracy sam_residual_exact(const csr::CSRMatrix<T>& map,
                                       const csr::CSRMatrix<T>& sourceMatrix,
                                       const csr::CSRMatrix<T>& targetMatrix)
{
  struct temp_storage {
    std::vector<double> acc;  // dense accumulator for each row
    std::vector<char> marked; // marks the column indices used for the current row 
    std::vector<size_t> hits; // records the column indices used for the current row
    temp_storage(size_t n) : acc(n, 0.0), marked(n, 0) {}
  };
  tbb::enumerable_thread_specific<temp_storage> local(sourceMatrix.m_cols);

  auto acc = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, targetMatrix.m_rows),
    std::pair<double, double>(0.0, 0.0),
    [&](const tbb::blocked_range<size_t>& r, std::pair<double, double> sums) {
      temp_storage& s = local.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        s.hits.clear();

        // Accumulate row i of N A_k.
        for (size_t j = map.m_row_pointers[i]; j < map.m_row_pointers[i + 1]; ++j) {
          const size_t srcRow = map.m_col_indices[j];
          const double coeff = map.m_vals[j];
          for (size_t k = sourceMatrix.m_row_pointers[srcRow]; k < sourceMatrix.m_row_pointers[srcRow + 1]; ++k) {
            const size_t c = sourceMatrix.m_col_indices[k];
            if (!s.marked[c]) {
              s.marked[c] = 1;
              s.acc[c] = 0.0;
              s.hits.push_back(c);
            }
            s.acc[c] += coeff * sourceMatrix.m_vals[k];
          }
        }

        // Subtract row i of A_0.
        for (size_t k = targetMatrix.m_row_pointers[i]; k < targetMatrix.m_row_pointers[i + 1]; ++k) {
          const size_t c = targetMatrix.m_col_indices[k];
          const double b = targetMatrix.m_vals[k];
          if (!s.marked[c]) {
            s.marked[c] = 1;
            s.acc[c] = 0.0;
            s.hits.push_back(c);
          }
          s.acc[c] -= b;
          sums.second += b * b;
        }

        // Reset only entries marked by this row.
        for (const size_t c : s.hits) {
          sums.first += s.acc[c] * s.acc[c];
          s.marked[c] = 0;
        }
      }
      return sums;
    },
    [](std::pair<double, double> a, std::pair<double, double> b) {
      return std::pair<double, double>(a.first + b.first, a.second + b.second);
    }
  );

  return {std::sqrt(acc.first), std::sqrt(acc.second)};
}

/// Per-thread scratch reused across map rows.
template <typename T>
struct sam_internal_storage {
  csr::vec<csr::index_t> J;
  std::vector<T> rhs;
  std::vector<T> submatrix;
  // -1 means absent; otherwise the value is the dense index in J.
  std::vector<int> marker;
  QR<T> qr;

  sam_internal_storage(const size_t marker_size) : marker(marker_size, -1) {}
};

/// Selects the left or right SAM formulation.
/// The left form is row-wise and works directly with CSR rows. The right form is column-wise
/// and is implemented through transposes. They differ only in the objective being minimized.
enum class mapSide {
  left,
  right,
};

inline const char* to_string(mapSide side) {
  return side == mapSide::left ? "left" : "right";
}

/// Time spent transposing inputs and output for the right map.
struct sam_transpose_times {
  uint64_t source_ns = 0;
  uint64_t target_ns = 0;
  uint64_t pattern_ns = 0;
  uint64_t result_ns = 0;
  uint64_t total_ns() const { return source_ns + target_ns + pattern_ns + result_ns; }
  double total_ms() const { return total_ns() / 1e6; }
  void merge(const sam_transpose_times& o) {
    source_ns += o.source_ns; target_ns += o.target_ns;
    pattern_ns += o.pattern_ns; result_ns += o.result_ns;
  }
};

/// Computes the right-map residual without materializing A_k N.
template <typename T>
inline sam_accuracy sam_residual_exact_right(const csr::CSRMatrix<T>& map,
                                             const csr::CSRMatrix<T>& sourceMatrix,
                                             const csr::CSRMatrix<T>& targetMatrix)
{
  return sam_residual_exact(transpose(map), transpose(sourceMatrix), transpose(targetMatrix));
}

/// Computes sparse approximate maps using independent row-wise least-squares problems.
template <typename T, typename SparsityPatternType>
class SparseApproximateMap {
 public:
  /// Computes the map for an already extracted sparsity pattern matrix. Each row builds
  /// the column indices set J, assembles a dense least squares problem, and solves it with QR.
  static void computeMap(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                         const csr::CSRMatrix<int>& pattern, csr::CSRMatrix<T>& mappingMatrix)
  {
    computeMapImpl<false, false>(targetMatrix, sourceMatrix, pattern, mappingMatrix, nullptr, nullptr);
  }

  /// Computes a map from a previously computed SparsityPattern.
  static void computeMap(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                         const SparsityPattern<T, SparsityPatternType>& sparsityPattern,
                         csr::CSRMatrix<T>& mappingMatrix)
  {
    computeMap(targetMatrix, sourceMatrix, *sparsityPattern.getPattern(), mappingMatrix);
  }

  /// Computes the right map N = argmin_{N in S} ||A_k N - A_0||_F through transposed CSR
  /// inputs and the row-wise kernel.
  static void computeMapRight(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                              const csr::CSRMatrix<int>& pattern, csr::CSRMatrix<T>& mappingMatrix,
                              sam_transpose_times* transposes = nullptr)
  {
    using clock = std::chrono::steady_clock;
    const auto ns = [](clock::time_point a, clock::time_point b) {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };

    const auto t0 = clock::now();
    auto sourceT = transpose(sourceMatrix);
    const auto t1 = clock::now();
    auto targetT = transpose(targetMatrix);
    const auto t2 = clock::now();
    auto patternT = transpose(pattern);
    const auto t3 = clock::now();

    csr::CSRMatrix<T> mapT;
    computeMapImpl<false, false>(targetT, sourceT, patternT, mapT, nullptr, nullptr);

    const auto t4 = clock::now();
    mappingMatrix = transpose(mapT);
    const auto t5 = clock::now();

    if (transposes) {
      transposes->source_ns += ns(t0, t1);
      transposes->target_ns += ns(t1, t2);
      transposes->pattern_ns += ns(t2, t3);
      transposes->result_ns += ns(t4, t5);
    }
  }

  /// Dispatches to the selected map formulation.
  static void computeMapSided(mapSide side,
                              const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                              const csr::CSRMatrix<int>& pattern, csr::CSRMatrix<T>& mappingMatrix,
                              sam_transpose_times* transposes = nullptr)
  {
    if (side == mapSide::left) {
      computeMap(targetMatrix, sourceMatrix, pattern, mappingMatrix);
    } else {
      computeMapRight(targetMatrix, sourceMatrix, pattern, mappingMatrix, transposes);
    }
  }

  /// Computes a map and accumulates per-phase timing counters.
  static void computeMapProfiled(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                 const csr::CSRMatrix<int>& pattern, csr::CSRMatrix<T>& mappingMatrix,
                                 sam_phase_times& profile)
  {
    computeMapImpl<true, false>(targetMatrix, sourceMatrix, pattern, mappingMatrix, &profile, nullptr);
  }

  /// Computes a map and accumulates its least-squares residual.
  static void computeMapResidual(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                 const csr::CSRMatrix<int>& pattern, csr::CSRMatrix<T>& mappingMatrix,
                                 sam_residual& residual)
  {
    computeMapImpl<false, true>(targetMatrix, sourceMatrix, pattern, mappingMatrix, nullptr, &residual);
  }

  /// Computes the map for base^level without materializing the extended pattern.
  static void computeMapFused(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                              const csr::CSRMatrix<int>& base, const int level,
                              csr::CSRMatrix<T>& mappingMatrix)
  {
    computeMapFusedDispatch<false>(targetMatrix, sourceMatrix, base, level, mappingMatrix, nullptr);
  }

  /// Fused map computation with accumulated least-squares residual.
  static void computeMapFusedResidual(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                      const csr::CSRMatrix<int>& base, const int level,
                                      csr::CSRMatrix<T>& mappingMatrix, sam_residual& residual)
  {
    computeMapFusedDispatch<true>(targetMatrix, sourceMatrix, base, level, mappingMatrix, &residual);
  }

  /// Computes the right map with fused extension by transposing the unextended base pattern.
  static void computeMapRightFused(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                   const csr::CSRMatrix<int>& base, const int level,
                                   csr::CSRMatrix<T>& mappingMatrix,
                                   sam_transpose_times* transposes = nullptr)
  {
    using clock = std::chrono::steady_clock;
    const auto ns = [](clock::time_point a, clock::time_point b) {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };

    const auto t0 = clock::now();
    auto sourceT = transpose(sourceMatrix);
    const auto t1 = clock::now();
    auto targetT = transpose(targetMatrix);
    const auto t2 = clock::now();
    auto baseT = transpose(base);
    const auto t3 = clock::now();

    csr::CSRMatrix<T> mapT;
    computeMapFusedDispatch<false>(targetT, sourceT, baseT, level, mapT, nullptr);

    const auto t4 = clock::now();
    mappingMatrix = transpose(mapT);
    const auto t5 = clock::now();

    if (transposes) {
      transposes->source_ns += ns(t0, t1);
      transposes->target_ns += ns(t1, t2);
      transposes->pattern_ns += ns(t2, t3);
      transposes->result_ns += ns(t4, t5);
    }
  }

  /// Dispatches fused extension to the selected map formulation.
  static void computeMapSidedFused(mapSide side,
                                   const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                   const csr::CSRMatrix<int>& base, const int level,
                                   csr::CSRMatrix<T>& mappingMatrix,
                                   sam_transpose_times* transposes = nullptr)
  {
    if (side == mapSide::left) {
      computeMapFused(targetMatrix, sourceMatrix, base, level, mappingMatrix);
    } else {
      computeMapRightFused(targetMatrix, sourceMatrix, base, level, mappingMatrix, transposes);
    }
  }

 private:
  /// Solves row i of N given its pattern row I[0, iSize). Builds J, M and b, runs the QR,
  /// and writes the iSize coefficients to out.
  template <bool Profile, bool Residual>
  static void solve_row(size_t i, const csr::index_t* I, size_t iSize,
                        const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                        sam_internal_storage<T>& ls, [[maybe_unused]] sam_phase_times& lp,
                        [[maybe_unused]] double& lr, T* out)
  {
    using clock = std::chrono::steady_clock;
    [[maybe_unused]] const auto ns = [](clock::time_point a, clock::time_point b) {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    [[maybe_unused]] clock::time_point t0, t1;
    if constexpr (Profile)
      t0 = clock::now();


    if (iSize == 0) {
      // An empty pattern produces a zero map row.
      if constexpr (Residual) {
        for (size_t k = targetMatrix.m_row_pointers[i]; k < targetMatrix.m_row_pointers[i + 1]; ++k) {
          lr += static_cast<double>(targetMatrix.m_vals[k]) * targetMatrix.m_vals[k];
        }
      }
      return;
    }

    ls.J.clear();
    const auto add = [&ls](const csr::index_t col) {
      if (ls.marker[col] == -1) {
        ls.marker[col] = 1;
        ls.J.push_back(col);
      }
    };

    // I selects the rows from source matrix. J is the column indices reached by the 
    // these rows and the corresponding row in the target matrix.
    for (size_t j = 0; j < iSize; ++j) {
      const size_t k = I[j];
      for (size_t p = sourceMatrix.m_row_pointers[k]; p < sourceMatrix.m_row_pointers[k + 1]; ++p) {
        add(sourceMatrix.m_col_indices[p]);
      }
    }
    
    for (size_t p = targetMatrix.m_row_pointers[i]; p < targetMatrix.m_row_pointers[i + 1]; ++p) {
      add(targetMatrix.m_col_indices[p]);
    }
    std::sort(ls.J.begin(), ls.J.end());
    const size_t jSize = ls.J.size();

    // Convert global column indices to dense local indices.
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

    // Gather b = A_0(i, J).
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

    // Assemble M = A_k(I, J)^T in column-major storage.
    const auto build_submatrix = [&]() {
      ls.submatrix.resize(jSize * iSize);
      std::fill(ls.submatrix.begin(), ls.submatrix.end(), T(0));
      for (size_t j = 0; j < iSize; ++j) {
        const size_t source_row_idx = I[j];

        for (size_t k = sourceMatrix.m_row_pointers[source_row_idx]; k < sourceMatrix.m_row_pointers[source_row_idx + 1]; ++k) {
          const size_t col = sourceMatrix.m_col_indices[k];
          const int dense_idx = ls.marker[col];
          if (dense_idx != -1) {
            ls.submatrix[dense_idx + jSize * j] = sourceMatrix.m_vals[k];
          }
        }
      }
    };
    build_submatrix();

    if constexpr (Profile) {
      t1 = clock::now();
      lp.submatrix_ns += ns(t0, t1);
      t0 = t1;
    }

    // Solve the local least-squares problem, using pivoting only when needed.
    if (!ls.qr.try_solve_unpivoted(jSize, iSize, ls.submatrix.data(), ls.rhs.data(),
                                   out, storage_order::col_major))
    {
      build_submatrix();
      ls.qr.solve(jSize, iSize, ls.submatrix.data(), ls.rhs.data(),
                  out, storage_order::col_major);
      if constexpr (Profile)
        ++lp.pivoted_rows;
    }

    // Reset markers used by this row.
    for (size_t col : ls.J) {
      ls.marker[col] = -1;
    }

    if constexpr (Residual)
      lr += static_cast<double>(ls.qr.residual_sq());

    if constexpr (Profile) {
      t1 = clock::now();
      lp.solve_ns += ns(t0, t1);
    }
  }

  /// Shared implementation for the plain, profiled, and residual map paths.
  template <bool Profile, bool Residual>
  static void computeMapImpl(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                             const csr::CSRMatrix<int>& pattern, csr::CSRMatrix<T>& mappingMatrix,
                             sam_phase_times* profile, sam_residual* residual)
  {
    mappingMatrix.m_rows = targetMatrix.m_rows;
    mappingMatrix.m_cols = targetMatrix.m_cols;
    mappingMatrix.m_nnz = pattern.m_nnz;
    mappingMatrix.m_row_pointers = pattern.m_row_pointers;
    mappingMatrix.m_col_indices = pattern.m_col_indices;
    mappingMatrix.m_vals.resize(mappingMatrix.m_nnz);

    // Size per-thread markers for the widest column space.
    tbb::enumerable_thread_specific<sam_internal_storage<T>> local_storages(std::max({mappingMatrix.m_cols, sourceMatrix.m_cols, static_cast<size_t>(pattern.m_cols)}));
    tbb::enumerable_thread_specific<sam_phase_times> local_profiles;
    tbb::enumerable_thread_specific<double> local_residuals(0.0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mappingMatrix.m_rows),
      [&](const tbb::blocked_range<size_t>& r) {
        sam_internal_storage<T>& ls = local_storages.local();
        [[maybe_unused]] sam_phase_times& lp = local_profiles.local();
        [[maybe_unused]] double& lr = local_residuals.local();
        for (size_t i = r.begin(); i < r.end(); ++i) {
          const size_t rowStart = pattern.m_row_pointers[i];
          const size_t iSize = pattern.m_row_pointers[i + 1] - rowStart;
          solve_row<Profile, Residual>(i, pattern.m_col_indices.data() + rowStart, iSize,
                                      targetMatrix, sourceMatrix, ls, lp, lr,
                                      mappingMatrix.m_vals.data() + rowStart);
        }
      }
    );

    // Merge thread-local counters after the parallel loop.
    if constexpr (Profile) {
      for (const sam_phase_times& lp : local_profiles)
        profile->merge(lp);
    }
    if constexpr (Residual) {
      for (const double lr : local_residuals)
        residual->sum_sq += lr;
    }
  }

  /// Computes the map for S^level while generating each extended pattern row.
  template <bool Residual, class Acc>
  static void computeMapFusedImpl(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                  const csr::CSRMatrix<int>& base, const int level,
                                  csr::CSRMatrix<T>& mappingMatrix, sam_residual* residual)
  {
    assert(base.m_rows == base.m_cols && "Fused extension needs a square pattern.");
    const size_t n = targetMatrix.m_rows;
    const bool frontier = sam::has_full_diagonal(base);
    const size_t marker_size = std::max({targetMatrix.m_cols, sourceMatrix.m_cols, static_cast<size_t>(base.m_cols)});

    mappingMatrix.m_rows = n;
    mappingMatrix.m_cols = targetMatrix.m_cols;
    mappingMatrix.m_row_pointers.assign(n + 1, 0);

    struct local_t {
      Acc acc;
      std::vector<csr::index_t> walk; // storage for graph traversal
      sam::row_buffer rows;
      std::vector<T> vals;
      sam_internal_storage<T> ls;
      sam_phase_times lp;
      double lr = 0;
    };
    tbb::enumerable_thread_specific<local_t> locals([&] {
      return local_t{Acc(base.m_cols), {}, {}, {}, sam_internal_storage<T>(marker_size), {}, 0.0};
    });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
      local_t& L = locals.local();
      L.rows.pieces.push_back({r.begin(), r.end(), L.rows.buf.size()});
      for (size_t i = r.begin(); i < r.end(); ++i) {
        sam::pattern_power_row(base, i, level, frontier, L.acc, L.walk);
        auto& I = L.acc.members;
        std::sort(I.begin(), I.end());

        const size_t offset = L.rows.buf.size();
        L.rows.buf.insert(L.rows.buf.end(), I.begin(), I.end());
        L.vals.resize(offset + I.size());
        solve_row<false, Residual>(i, L.rows.buf.data() + offset, I.size(), targetMatrix,
                                   sourceMatrix, L.ls, L.lp, L.lr, L.vals.data() + offset);
        mappingMatrix.m_row_pointers[i + 1] = I.size();
      }
    });

    mappingMatrix.m_nnz = mappingMatrix.scanRowSize();
    mappingMatrix.m_col_indices.resize(mappingMatrix.m_nnz);
    mappingMatrix.m_vals.resize(mappingMatrix.m_nnz);

    // Copy each recorded piece into its final CSR range.
    std::vector<std::pair<const local_t*, sam::row_buffer::piece>> all;
    for (const local_t& L : locals)
      for (const auto& p : L.rows.pieces) all.push_back({&L, p});
    tbb::parallel_for(tbb::blocked_range<size_t>(0, all.size()), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t t = r.begin(); t < r.end(); ++t) {
        const auto& [L, p] = all[t];
        const size_t dst = mappingMatrix.m_row_pointers[p.row_begin];
        const size_t len = mappingMatrix.m_row_pointers[p.row_end] - dst;
        std::copy_n(L->rows.buf.begin() + p.offset, len, mappingMatrix.m_col_indices.begin() + dst);
        std::copy_n(L->vals.begin() + p.offset, len, mappingMatrix.m_vals.begin() + dst);
      }
    });

    if constexpr (Residual) {
      for (const local_t& L : locals)
        residual->sum_sq += L.lr;
    }
  }

  template <bool Residual>
  static void computeMapFusedDispatch(const csr::CSRMatrix<T>& targetMatrix, const csr::CSRMatrix<T>& sourceMatrix,
                                      const csr::CSRMatrix<int>& base, const int level,
                                      csr::CSRMatrix<T>& mappingMatrix, sam_residual* residual)
  {
    if (spgemm_acc == spgemm_accumulator::hash) {
      computeMapFusedImpl<Residual, sam::detail::hash_index_set>(targetMatrix, sourceMatrix, base, level, mappingMatrix, residual);
    } else {
      computeMapFusedImpl<Residual, sam::detail::dense_index_set>(targetMatrix, sourceMatrix, base, level, mappingMatrix, residual);
    }
  }

 public:
  /// Removes map entries whose absolute value is below `threshold`.
  ///
  /// The threshold is applied to unscaled values. Rows may become empty, and the output
  /// must not alias the input.
  static void post_filtration(const csr::CSRMatrix<T>& map, csr::CSRMatrix<T>& final_map, T threshold) {
    final_map.m_rows = map.m_rows;
    final_map.m_cols = map.m_cols;
    final_map.m_row_pointers.resize(final_map.m_rows + 1, 0);

    // Use the same predicate for counting and filling.
    const auto keep = [threshold](const T val) {
      return std::abs(val) >= threshold;
    };

    // Count retained entries.
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
    // The fill pass initializes every allocated element.
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