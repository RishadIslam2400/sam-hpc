#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/sam/extendPattern.hpp"
#include "sam_hpc/util/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

/// Wall-clock phases of computePattern(), measured in nanoseconds.
struct pattern_phase_times {
  uint64_t merge_ns = 0;   // Candidate merge for patternMatrix::union_max.
  uint64_t filter_ns = 0;  // Candidate scaling and sparsification.
  uint64_t extend_ns = 0;  // Pattern extension; zero for SimplePattern.

  uint64_t total_ns() const { return merge_ns + filter_ns + extend_ns; }
};

/// Uses the candidate structure without dropping entries or extending the pattern.
struct SimplePattern {};

/// Keeps off-diagonal entries with magnitude greater than a global threshold. 
/// The diagonal is always kept.
struct GlobalThresholdPattern { double globalThreshold; };

/// Keeps off-diagonal entries above a row-relative magnitude threshold.
struct ColumnThresholdPattern { double columnThreshold; };

/// Keeps up to `fixedNNZ` largest-magnitude entries in each candidate row.
struct FixedNNZPattern { size_t fixedNNZ; };

/// Keeps off-diagonal entries that pass both global and row-relative thresholds.
struct combinedThresholdPattern {
  double global_thresh;
  double column_thresh;
};

/// Fixed-offset (stencil) pattern. The offsets are mesh-specific.
struct StencilPattern {
  std::vector<long> offsets;
  bool restrict_to_structure = false;
};

/// The offsets of the transposed stencil.
inline StencilPattern transposed(const StencilPattern &s) {
  StencilPattern t = s;
  for (long &d : t.offsets)
    d = -d;
  return t;
}

/// A caller-supplied pattern used exactly as given
struct ExplicitPattern { 
  const csr::CSRMatrix<int> *pattern; 
};

/// Whether a pattern type is extended to S^level. Simple, stencil and explicit patterns are
/// used as they are and ignore the level argument.
template <typename PatternType>
inline constexpr bool pattern_is_extended_v =
  !std::is_same_v<PatternType, SimplePattern> && !std::is_same_v<PatternType, StencilPattern> &&
  !std::is_same_v<PatternType, ExplicitPattern>;


/// Selects the matrix used to construct the sparsification candidate.
enum class patternMatrix {
  source,
  union_max
};

/// Builds, owns and caches one sparsity pattern for the map from a source to a target.
/// Construction is cheap and does no work; computePattern() builds the pattern once and
/// every later call is a no-op, even if the matrices behind the references have changed
/// since - build a new object for a new source.
template <typename T, typename PatternType>
class SparsityPattern {
public:
  SparsityPattern() = delete;

  /// @param originalMatrix  The source A_k. Must be square with ascending column indices.
  /// @param targetMatrix    The target A_0.
  /// @param type            The drop rule and its parameters.
  /// @param level           Extension power: the filtered pattern S becomes S^level.
  /// @param pm              Which candidate is sparsified, source or union_max.
  SparsityPattern(const csr::CSRMatrix<T> &originalMatrix,
                 const csr::CSRMatrix<T> &targetMatrix, const PatternType &type,
                 int level = 2, patternMatrix pm = patternMatrix::source)
    : m_originalMatrix(originalMatrix), m_targetMatrix(targetMatrix), m_type(type),
      m_level(level),m_pattern_matrix(pm), m_pattern(nullptr) {}
  
  SparsityPattern(const SparsityPattern &other) = delete;
  SparsityPattern &operator=(const SparsityPattern &other) = delete;

  // Reference members prevent move assignment.
  SparsityPattern(SparsityPattern&&) = default;
  SparsityPattern& operator=(SparsityPattern&&) = default;

  template <typename X, typename Type>
  friend std::ostream &operator<<(std::ostream &os, const SparsityPattern<X, Type> &p);

  /// Builds the pattern
  void computePattern() {
    if (m_pattern)
      return;

    m_phase_times = pattern_phase_times{};
    const auto t_begin = std::chrono::steady_clock::now();

    if constexpr (std::is_same_v<PatternType, SimplePattern>)
      computeSimplePattern();
    else if constexpr (std::is_same_v<PatternType, GlobalThresholdPattern>)
      computeGlobalThresholdPattern(m_type.globalThreshold);
    else if constexpr (std::is_same_v<PatternType, ColumnThresholdPattern>)
      computeColumnThresholdPattern(m_type.columnThreshold);
    else if constexpr (std::is_same_v<PatternType, FixedNNZPattern>)
      computeFixedNNZPattern(m_type.fixedNNZ);
    else if constexpr (std::is_same_v<PatternType, combinedThresholdPattern>)
      computeCombinedPattern(m_type.global_thresh, m_type.column_thresh);
    else if constexpr (std::is_same_v<PatternType, StencilPattern>)
      computeStencilPattern(m_type.offsets, m_type.restrict_to_structure);
    else if constexpr (std::is_same_v<PatternType, ExplicitPattern>)
      computeExplicitPattern(m_type.pattern);
    else
      static_assert(!std::is_same_v<T,T>, "Unsupported pattern type");

    const auto total = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_begin).count());
    m_phase_times.filter_ns = total - m_phase_times.extend_ns - m_phase_times.merge_ns;
  }

  /// The finished pattern S
  const csr::CSRMatrix<int> *getPattern() const {
    assert(m_pattern != nullptr && "Compute the sparsity pattern first.");
    return m_pattern.get();
  }
  
  size_t getNNZ() const {
    return m_pattern ? m_pattern->m_nnz : 0;
  }

  const pattern_phase_times& getPhaseTimes() const {
    return m_phase_times;
  }

  patternMatrix getPatternMatrix() const {
    return m_pattern_matrix;
  }

private:
  const csr::CSRMatrix<T> &m_originalMatrix;
  const csr::CSRMatrix<T> &m_targetMatrix;
  PatternType m_type;
  int m_level;
  patternMatrix m_pattern_matrix;
  std::unique_ptr<csr::CSRMatrix<int>> m_pattern;
  pattern_phase_times m_phase_times;

  csr::vec<T> m_scaled;
  csr::CSRMatrix<T> m_H;

  // Non-ownning view of the CSR matrix arrays
  struct candidateView {
    const csr::offset_t *row_pointers = nullptr;
    const csr::index_t *col_indices = nullptr;
    const T *vals = nullptr;
    size_t rows = 0;
    size_t cols = 0;
  };

  // Builds the selected candidate.
  candidateView prepareCandidate() {
    assert(m_originalMatrix.m_rows == m_originalMatrix.m_cols && "Sparsity patterns are defined for square matrices only.");
    if (m_pattern_matrix == patternMatrix::union_max) {
      const auto t0 = std::chrono::steady_clock::now();
      buildUnionMatrix();
      m_phase_times.merge_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
      return {m_H.m_row_pointers.data(), m_H.m_col_indices.data(), m_H.m_vals.data(), m_H.m_rows, m_H.m_cols};
    }

    const std::vector<T> diag = csr::diagonal<csr::diagonalType::forScaling>(m_originalMatrix);
    m_scaled.assign(m_originalMatrix.m_vals.begin(), m_originalMatrix.m_vals.end());
    diagonalScaling(m_originalMatrix.m_row_pointers, m_originalMatrix.m_col_indices, m_scaled, diag);
    return {m_originalMatrix.m_row_pointers.data(), m_originalMatrix.m_col_indices.data(), m_scaled.data(), m_originalMatrix.m_rows, m_originalMatrix.m_cols};
  }

  /// Merge diagonally scaled source and target rows.
  void buildUnionMatrix() {
    const csr::CSRMatrix<T> &A = m_originalMatrix;
    const csr::CSRMatrix<T> &B = m_targetMatrix;
    assert(A.m_rows == B.m_rows && A.m_cols == B.m_cols && "Source and target must have matching dimensions to merge.");

    const std::vector<T> dA = csr::diagonal<csr::diagonalType::forScaling>(A);
    const std::vector<T> dB = csr::diagonal<csr::diagonalType::forScaling>(B);

    m_H.m_rows = A.m_rows;
    m_H.m_cols = A.m_cols;
    m_H.m_row_pointers.assign(m_H.m_rows + 1, 0);

    const csr::offset_t *ap = A.m_row_pointers.data();
    const csr::index_t *ac = A.m_col_indices.data();
    const csr::offset_t *bp = B.m_row_pointers.data();
    const csr::index_t *bc = B.m_col_indices.data();

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_H.m_rows), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        m_H.m_row_pointers[i + 1] = sam::mergeRowsCount(ac + ap[i], ac + ap[i + 1], bc + bp[i], bc + bp[i + 1]);
      }
    });

    m_H.m_nnz = m_H.scanRowSize();
    m_H.m_col_indices.resize(m_H.m_nnz);
    m_H.m_vals.resize(m_H.m_nnz);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_H.m_rows), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t out = m_H.m_row_pointers[i];
        size_t ja = ap[i], ea = ap[i + 1];
        size_t jb = bp[i], eb = bp[i + 1];

        while (ja < ea && jb < eb) {
          const size_t ca = ac[ja];
          const size_t cb = bc[jb];
          if (ca < cb) {
            m_H.m_col_indices[out] = ca;
            m_H.m_vals[out++] = A.m_vals[ja] * dA[i] * dA[ca];
            ++ja;
          } else if (cb < ca) {
            m_H.m_col_indices[out] = cb;
            m_H.m_vals[out++] = B.m_vals[jb] * dB[i] * dB[cb];
            ++jb;
          } else {
            const T va = A.m_vals[ja] * dA[i] * dA[ca];
            const T vb = B.m_vals[jb] * dB[i] * dB[cb];
            m_H.m_col_indices[out] = ca;
            m_H.m_vals[out++] = (std::abs(va) >= std::abs(vb)) ? va : vb;
            ++ja;
            ++jb;
          }
        }
        for (; ja < ea; ++ja) {
          m_H.m_col_indices[out] = ac[ja];
          m_H.m_vals[out++] = A.m_vals[ja] * dA[i] * dA[ac[ja]];
        }
        for (; jb < eb; ++jb) {
          m_H.m_col_indices[out] = bc[jb];
          m_H.m_vals[out++] = B.m_vals[jb] * dB[i] * dB[bc[jb]];
        }
      }
    });
  }

  // Copies the candidate structure without filtering or extension.
  void computeSimplePattern() {
    const candidateView c = prepareCandidate();

    m_pattern = std::make_unique<csr::CSRMatrix<int>>();
    m_pattern->m_rows = c.rows;
    m_pattern->m_cols = c.cols;
    m_pattern->m_nnz = c.row_pointers[c.rows];
    m_pattern->m_row_pointers.assign(c.row_pointers, c.row_pointers + c.rows + 1);
    m_pattern->m_col_indices.assign(c.col_indices, c.col_indices + m_pattern->m_nnz);
    m_pattern->m_vals.assign(m_pattern->m_nnz, 1);
  }

  // Row s keeps {s + d : d in offsets} intersected with [0, n), and with the columns of row s
  // of the source when `restricted` is set.
  void computeStencilPattern(std::vector<long> offsets, bool restricted) {
    assert(m_originalMatrix.m_rows == m_originalMatrix.m_cols && "Sparsity patterns are defined for square matrices only.");
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

    const long n = static_cast<long>(m_originalMatrix.m_rows);
    const csr::CSRMatrix<T> &A = m_originalMatrix;
    const auto keep = [&](long s, long d) {
      const long c = s + d;
      if (c < 0 || c >= n)
        return false;
      if (!restricted)
        return true;
      const auto first = A.m_col_indices.begin() + A.m_row_pointers[s];
      const auto last = A.m_col_indices.begin() + A.m_row_pointers[s + 1];
      return std::binary_search(first, last, static_cast<csr::index_t>(c));
    };
    m_pattern = std::make_unique<csr::CSRMatrix<int>>();
    m_pattern->m_rows = m_originalMatrix.m_rows;
    m_pattern->m_cols = m_originalMatrix.m_cols;
    m_pattern->m_row_pointers.resize(m_pattern->m_rows + 1, 0);

    tbb::parallel_for(tbb::blocked_range<long>(0, n), [&](const tbb::blocked_range<long> &r) {
      for (long s = r.begin(); s < r.end(); ++s) {
        size_t count = 0;
        for (const long d : offsets)
          count += keep(s, d);
        m_pattern->m_row_pointers[s + 1] = count;
      }
    });

    m_pattern->m_nnz = m_pattern->scanRowSize();
    m_pattern->m_col_indices.resize(m_pattern->m_nnz);
    m_pattern->m_vals.resize(m_pattern->m_nnz, 1);

    tbb::parallel_for(tbb::blocked_range<long>(0, n), [&](const tbb::blocked_range<long> &r) {
      for (long s = r.begin(); s < r.end(); ++s) {
        size_t out = m_pattern->m_row_pointers[s];
        for (const long d : offsets) {
          if (keep(s, d))
            m_pattern->m_col_indices[out++] = static_cast<csr::index_t>(s + d);
        }
      }
    });
  }

  // Copies a validated caller-supplied pattern.
  void computeExplicitPattern(const csr::CSRMatrix<int> *p) {
    if (p == nullptr)
      throw std::invalid_argument("ExplicitPattern: null pattern");
    const size_t n = m_originalMatrix.m_rows;
    if (p->m_rows != n || p->m_cols != m_originalMatrix.m_cols)
      throw std::invalid_argument("ExplicitPattern: pattern is " + std::to_string(p->m_rows) + " x " +
                                  std::to_string(p->m_cols) + ", source is " + std::to_string(n) +
                                  " x " + std::to_string(m_originalMatrix.m_cols));
    if (p->m_row_pointers.size() != n + 1 || p->m_row_pointers[0] != 0 ||
        p->m_row_pointers[n] != p->m_nnz || p->m_col_indices.size() != p->m_nnz)
      throw std::invalid_argument("ExplicitPattern: inconsistent CSR arrays");
    for (size_t i = 0; i < n; ++i) {
      const size_t b = p->m_row_pointers[i], e = p->m_row_pointers[i + 1];
      if (e < b)
        throw std::invalid_argument("ExplicitPattern: row pointers decrease at row " + std::to_string(i));
      for (size_t k = b; k < e; ++k) {
        if (p->m_col_indices[k] >= p->m_cols || (k > b && p->m_col_indices[k] <= p->m_col_indices[k - 1]))
          throw std::invalid_argument("ExplicitPattern: row " + std::to_string(i) +
                                      " is not strictly ascending and in range");
      }
    }

    m_pattern = std::make_unique<csr::CSRMatrix<int>>(*p);
    m_pattern->m_vals.assign(m_pattern->m_nnz, 1);
  }

  // Genrate the pattern based on the filtering rule.
  template<typename Func>
  void buildPattern(const candidateView &c, Func filter) {
    m_pattern = std::make_unique<csr::CSRMatrix<int>>();
    m_pattern->m_rows = c.rows;
    m_pattern->m_cols = c.cols;
    m_pattern->m_row_pointers.resize(m_pattern->m_rows + 1, 0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_pattern->m_rows), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        m_pattern->m_row_pointers[i + 1] = filter(i, nullptr);
      }
    });

    m_pattern->m_nnz = m_pattern->scanRowSize();
    m_pattern->m_col_indices.resize(m_pattern->m_nnz);
    m_pattern->m_vals.resize(m_pattern->m_nnz, 1);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_pattern->m_rows), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        csr::index_t *dest = m_pattern->m_col_indices.data() + m_pattern->m_row_pointers[i];
        filter(i, dest);
        std::sort(dest, m_pattern->m_col_indices.data() + m_pattern->m_row_pointers[i + 1]);
      }
    });

    const auto t_filtered = std::chrono::steady_clock::now();
    sam::extend_pattern(*m_pattern, m_level);
    m_phase_times.extend_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_filtered).count();
  }

  // Applies a global magnitude threshold and always retains the diagonal.
  void computeGlobalThresholdPattern(const double globalThreshold) {
    const candidateView c = prepareCandidate();

    auto filter = [&](size_t i, csr::index_t* dest) {
      size_t count = 0;
      bool diagonal_found = false;
      for (size_t j = c.row_pointers[i]; j < c.row_pointers[i + 1]; ++j) {
        const size_t colIdx = c.col_indices[j];
        bool keep = false;
        if (colIdx == i) {
          keep = true;
          diagonal_found = true;
        } else if (std::abs(c.vals[j]) > globalThreshold) {
          keep = true;
        }
        if (keep) {
          if (dest)
            dest[count] = colIdx;
          count++;
        }
      }
      if (!diagonal_found) {
          if (dest)
            dest[count] = i;
          count++;
      }
      return count;
    };

    buildPattern(c, filter);
  }

  // Applies a row-relative magnitude threshold and always retains the diagonal.
  void computeColumnThresholdPattern(const double tau) {
    const candidateView c = prepareCandidate();

    auto filter = [&](size_t i, csr::index_t* dest) {
      const size_t rowStart = c.row_pointers[i];
      const size_t rowEnd = c.row_pointers[i + 1];

      T maxVal = 0;
      for (size_t j = rowStart; j < rowEnd; ++j) {
        maxVal = std::max(maxVal, std::abs(c.vals[j]));
      }
      const T threshold = (1 - tau) * maxVal;

      size_t count = 0;
      bool diagonal_found = false;
      for (size_t j = rowStart; j < rowEnd; ++j) {
        const size_t colIdx = c.col_indices[j];
        bool keep = false;
        if (colIdx == i) {
          keep = true;
          diagonal_found = true;
        } else if (std::abs(c.vals[j]) > threshold) {
          keep = true;
        }
        if (keep) {
          if (dest)
            dest[count] = colIdx;
          count++;
        }
      }
      if (!diagonal_found) {
        if (dest)
          dest[count] = i;
        count++;
      }
      return count;
    };

    buildPattern(c, filter);
  }

  // Keeps the lfil largest-magnitude entries in each row.
  void computeFixedNNZPattern(const size_t lfil) {
    const candidateView c = prepareCandidate();

    // Scratch is local to each TBB worker because rows are filtered concurrently.
    tbb::enumerable_thread_specific<std::vector<std::pair<T, size_t>>> local_entries;
    auto filter = [&](size_t i, csr::index_t *dest) {
      const size_t rowStart = c.row_pointers[i];
      const size_t rowEnd = c.row_pointers[i + 1];
      const size_t nnz = rowEnd - rowStart;

      size_t count = std::min(lfil, nnz);

      if (count == 0)
        return size_t(0);

      if (dest) {
        auto &col_entries = local_entries.local();
        col_entries.clear();
        col_entries.reserve(nnz);

        for (size_t j = rowStart; j < rowEnd; ++j) {
          col_entries.emplace_back(std::abs(c.vals[j]), c.col_indices[j]);
        }

        std::nth_element(col_entries.begin(), col_entries.begin() + count - 1, col_entries.end(), std::greater<>{});

        for (size_t k = 0; k < count; ++k) {
          dest[k] = col_entries[k].second;
        }
      }

      return count;
    };

    buildPattern(c, filter);
  }

  // Applies both row-relative and global thresholds and always retains the diagonal.
  void computeCombinedPattern(const double global_thresh, const double column_thresh) {
    const candidateView c = prepareCandidate();

    auto filter = [&](size_t i, csr::index_t* dest) {
      const size_t rowStart = c.row_pointers[i];
      const size_t rowEnd = c.row_pointers[i + 1];

      T maxVal = 0;
      for (size_t j = rowStart; j < rowEnd; ++j) {
        maxVal = std::max(maxVal, std::abs(c.vals[j]));
      }
      const T threshold = (1 - column_thresh) * maxVal;

      size_t count = 0;
      bool diagonal_found = false;
      for (size_t j = rowStart; j < rowEnd; ++j) {
        const size_t colIdx = c.col_indices[j];
        bool keep = false;
        if (colIdx == i) {
          keep = true;
          diagonal_found = true;
        } else if (std::abs(c.vals[j]) > threshold && std::abs(c.vals[j]) > global_thresh) {
          keep = true;
        }
        if (keep) {
          if (dest)
            dest[count] = colIdx;
          count++;
        }
      }
      if (!diagonal_found) {
        if (dest)
          dest[count] = i;
        count++;
      }
      return count;
    };

    buildPattern(c, filter);
  }

  // Applies D^-1/2 A D^-1/2 in place, where diagonal contains the scale factors.
  void diagonalScaling(const csr::vec<csr::offset_t> &rowPointers,
                       const csr::vec<csr::index_t> &colIndices, csr::vec<T> &values,
                       const std::vector<T> &diagonal)
  {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, rowPointers.size() - 1),
      [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          const size_t rowStart = rowPointers[i];
          const size_t rowEnd = rowPointers[i + 1];
          for (size_t j = rowStart; j < rowEnd; ++j) {
            values[j] *= diagonal[i] * diagonal[colIndices[j]];
          }
        }
      }
    );
  }
};

/// Prints the computed pattern. The pattern must have been built before this operator
/// is called; output is intended for small matrices.
template <typename X, typename Type>
std::ostream &operator<<(std::ostream &os, const SparsityPattern<X, Type> &p) {
  os << "Sparsity Pattern: " << std::endl;
  os << *(p.m_pattern);
  return os;
}