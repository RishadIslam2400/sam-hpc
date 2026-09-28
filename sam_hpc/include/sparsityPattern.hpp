#pragma once

#include "CSRMatrix.hpp"
#include "extendPattern.hpp"
#include "launchThreads.hpp"

#include <chrono>
#include <queue>

/// Wall-clock split of computePattern(). Filled in on every call; two clock reads per
/// pattern build, so it costs nothing worth measuring.
struct pattern_phase_times {
    uint64_t merge_ns = 0;   // building H_k, the source/target magnitude-max merge (union only)
    uint64_t filter_ns = 0;  // diagonal scaling, count pass, scan, fill pass
    uint64_t extend_ns = 0;  // sam::extend_pattern (the symbolic S^k product)

    uint64_t total_ns() const { return merge_ns + filter_ns + extend_ns; }
};

struct SimplePattern {};
struct GlobalThresholdPattern { double globalThreshold; };
struct ColumnThresholdPattern { double columnThreshold; };
struct FixedNNZPattern { size_t fixedNNZ; };
struct combinedThresholdPattern {
    double global_thresh;
    double column_thresh;
};

/// Which matrix the sparsification is applied to.
///
///   source     Sparsify the source matrix A_k alone.
///
///   union_max  Build H_k first by merging S(A_k) and S(A_0), keeping the larger-magnitude
///              entry at shared locations (both symmetrically diagonally scaled so the two are
///              numerically comparable), then sparsify and extend H_k. This is the union step
///              as described in the manuscript: target information influences *which* entries
///              survive rather than being appended wholesale afterwards.
///
/// There is deliberately no "union afterwards" variant. Merging the sparsified pattern with
/// the raw S(A_0) forces nnz(N) >= nnz(A_0), which undoes the sparsification this project
/// exists to study, and it is not the scheme the manuscript describes.
enum class patternMatrix {
    source,
    union_max
};

// @todo: get rid of private
template <typename T, typename PatternType>
class SparsityPattern {
public:
    SparsityPattern() = delete;

    SparsityPattern(const CSRMatrix<T> &originalMatrix, const CSRMatrix<T> &targetMatrix, const PatternType &type,
                    int level = 2, patternMatrix pm = patternMatrix::source)
        : m_originalMatrix(originalMatrix), m_targetMatrix(targetMatrix), m_type(type), m_level(level),
          m_pattern_matrix(pm), m_pattern(nullptr) {}
    
    SparsityPattern(const SparsityPattern &other) = delete;
    SparsityPattern &operator=(const SparsityPattern &other) = delete;

    SparsityPattern(SparsityPattern&&) = default;
    SparsityPattern& operator=(SparsityPattern&&) = default;

    template <typename X, typename Type>
    friend bool operator==(const SparsityPattern<X, Type> &lhs, const SparsityPattern<X, Type> &rhs);

    template <typename X, typename Type>
    friend bool operator!=(const SparsityPattern<X, Type> &lhs, const SparsityPattern<X, Type> &rhs);

    template <typename X, typename Type>
    friend std::ostream &operator<<(std::ostream &os, const SparsityPattern<X, Type> &p);

    void computePattern() {
        if (m_pattern) return; // Pattern already computed

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
        else
            static_assert(!std::is_same_v<T,T>, "Unsupported pattern type");

        // The merge and the extension time themselves; whatever is left in the call
        // (diagonal scaling, the count pass, the scan, the fill pass) is the filter cost.
        const auto total = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_begin).count());
        m_phase_times.filter_ns = total - m_phase_times.extend_ns - m_phase_times.merge_ns;
    }

    const CSRMatrix<int> *getPattern() const {
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
    const CSRMatrix<T> &m_originalMatrix;      // source matrix
    const CSRMatrix<T> &m_targetMatrix;        // target matrix
    PatternType m_type;                        // sparsification technique
    int m_level;                               // level of pattern extension
    patternMatrix m_pattern_matrix;            // which matrix gets sparsified
    std::unique_ptr<CSRMatrix<int>> m_pattern; // computed pattern
    pattern_phase_times m_phase_times;         // filter vs extension split of the last build

    csr::vec<T> m_scaled;                      // diagonally scaled source values (patternMatrix::source)
    CSRMatrix<T> m_H;                          // merged source/target matrix (patternMatrix::union_max)

    // ================ Candidate matrix ================
    // The matrix the drop rules are applied to: its structure supplies the candidate nonzero
    // locations and its values are already symmetrically diagonally scaled, so every filter
    // below can threshold on them directly.
    struct candidateView {
        const csr::offset_t *row_pointers = nullptr;
        const csr::index_t *col_indices = nullptr;
        const T *vals = nullptr;
        size_t rows = 0;
        size_t cols = 0;
    };

    candidateView prepareCandidate() {
        if (m_pattern_matrix == patternMatrix::union_max) {
            const auto t0 = std::chrono::steady_clock::now();
            buildUnionMatrix();
            m_phase_times.merge_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
            return {m_H.m_row_pointers.data(), m_H.m_col_indices.data(), m_H.m_vals.data(),
                    m_H.m_rows, m_H.m_cols};
        }

        const std::vector<T> diag = diagonal<diagonalType::forScaling>(m_originalMatrix);
        m_scaled.assign(m_originalMatrix.m_vals.begin(), m_originalMatrix.m_vals.end());
        diagonalScaling(m_originalMatrix.m_row_pointers, m_originalMatrix.m_col_indices, m_scaled, diag);
        return {m_originalMatrix.m_row_pointers.data(), m_originalMatrix.m_col_indices.data(), m_scaled.data(),
                m_originalMatrix.m_rows, m_originalMatrix.m_cols};
    }

    /// H_k = merge of the diagonally scaled source and target, keeping the entry with the
    /// larger magnitude wherever both are nonzero. Both operands store rows with ascending
    /// column indices, so each row is a two-way merge and no marker array is needed.
    void buildUnionMatrix() {
        const CSRMatrix<T> &A = m_originalMatrix; // source A_k
        const CSRMatrix<T> &B = m_targetMatrix;   // target A_0
        assert(A.m_rows == B.m_rows && A.m_cols == B.m_cols &&
               "Source and target must have matching dimensions to merge.");

        const std::vector<T> dA = diagonal<diagonalType::forScaling>(A);
        const std::vector<T> dB = diagonal<diagonalType::forScaling>(B);

        m_H.m_rows = A.m_rows;
        m_H.m_cols = A.m_cols;
        m_H.m_row_pointers.assign(m_H.m_rows + 1, 0);

        const csr::offset_t *ap = A.m_row_pointers.data();
        const csr::index_t *ac = A.m_col_indices.data();
        const csr::offset_t *bp = B.m_row_pointers.data();
        const csr::index_t *bc = B.m_col_indices.data();

        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_H.m_rows), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                m_H.m_row_pointers[i + 1] = sam::mergeRowsCount(ac + ap[i], ac + ap[i + 1],
                                                                bc + bp[i], bc + bp[i + 1]);
            }
        });

        m_H.m_nnz = m_H.scanRowSize();
        // X3: no fill value - the merge pass below writes every element, and the serial
        // zero-fill this replaces was 60-68% of the whole phase. It also lets each page be
        // first-touched by the thread that fills it. See csrTypes.hpp.
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

    // ================ Sparsity Pattern Computation ================
    void computeSimplePattern() {
        const candidateView c = prepareCandidate();

        m_pattern = std::make_unique<CSRMatrix<int>>();
        m_pattern->m_rows = c.rows;
        m_pattern->m_cols = c.cols;
        m_pattern->m_nnz = c.row_pointers[c.rows];
        m_pattern->m_row_pointers.assign(c.row_pointers, c.row_pointers + c.rows + 1);
        m_pattern->m_col_indices.assign(c.col_indices, c.col_indices + m_pattern->m_nnz);
        m_pattern->m_vals.assign(m_pattern->m_nnz, 1);

        // sam::extend_pattern(*m_pattern, m_level);
    }

    template<typename Func>
    void buildPattern(const candidateView &c, Func filter) {
        m_pattern = std::make_unique<CSRMatrix<int>>();
        m_pattern->m_rows = c.rows;
        m_pattern->m_cols = c.cols;
        m_pattern->m_row_pointers.resize(m_pattern->m_rows + 1, 0);

        // Count nnz per row
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_pattern->m_rows), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                m_pattern->m_row_pointers[i + 1] = filter(i, nullptr); // no write operation
            }
        });

        m_pattern->m_nnz = m_pattern->scanRowSize();
        m_pattern->m_col_indices.resize(m_pattern->m_nnz); // filled below, every element
        m_pattern->m_vals.resize(m_pattern->m_nnz, 1);

        // Fill the column indices
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_pattern->m_rows), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                csr::index_t *dest = m_pattern->m_col_indices.data() + m_pattern->m_row_pointers[i];
                filter(i, dest);
                std::sort(dest, m_pattern->m_col_indices.data() + m_pattern->m_row_pointers[i + 1]);
            }
        });

        const auto t_filtered = std::chrono::steady_clock::now();
        sam::extend_pattern(*m_pattern, m_level);
        m_phase_times.extend_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_filtered).count();
    }

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
                    if (dest) dest[count] = colIdx;
                    count++;
                }
            }
            if (!diagonal_found) {
                if (dest) dest[count] = i; // Ensure diagonal is present
                count++;
            }
            return count;
        };

        buildPattern(c, filter);
    }

    void computeColumnThresholdPattern(const double tau) {
        const candidateView c = prepareCandidate();

        auto filter = [&](size_t i, csr::index_t* dest) {
            const size_t rowStart = c.row_pointers[i];
            const size_t rowEnd = c.row_pointers[i + 1];

            // Find max absolute value in the row
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
                    if (dest) dest[count] = colIdx;
                    count++;
                }
            }
            if (!diagonal_found) {
                if (dest) dest[count] = i; // Ensure diagonal is present
                count++;
            }
            return count;
        };

        buildPattern(c, filter);
    }

    void computeFixedNNZPattern(const size_t lfil) {
        const candidateView c = prepareCandidate();

        tbb::enumerable_thread_specific<std::vector<std::pair<T, size_t>>> local_entries;
        auto filter = [&](size_t i, csr::index_t *dest) {
            const size_t rowStart = c.row_pointers[i];
            const size_t rowEnd = c.row_pointers[i + 1];
            const size_t nnz = rowEnd - rowStart;

            size_t count = std::min(lfil, nnz);

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

    void computeCombinedPattern(const double global_thresh, const double column_thresh) {
        const candidateView c = prepareCandidate();

        auto filter = [&](size_t i, csr::index_t* dest) {
            const size_t rowStart = c.row_pointers[i];
            const size_t rowEnd = c.row_pointers[i + 1];

            // Find max absolute value in the row
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
                    if (dest) dest[count] = colIdx;
                    count++;
                }
            }
            if (!diagonal_found) {
                if (dest) dest[count] = i; // Ensure diagonal is present
                count++;
            }
            return count;
        };

        buildPattern(c, filter);
    }

    // =============== Helper Functions ================
    // Symmetric diagonal scaling, D^-1/2 A D^-1/2, applied in place to `values`.
    // diagonal[idx] is a random memory access.
    void diagonalScaling(const csr::vec<csr::offset_t> &rowPointers, const csr::vec<csr::index_t> &colIndices,
                         csr::vec<T> &values, const std::vector<T> &diagonal) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, rowPointers.size() - 1), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const size_t rowStart = rowPointers[i];
                const size_t rowEnd = rowPointers[i + 1];
                for (size_t j = rowStart; j < rowEnd; ++j) {
                    values[j] *= diagonal[i] * diagonal[colIndices[j]];
                }
            }
        });
    }
};

template <typename X, typename PatternType>
bool operator==(const SparsityPattern<X, PatternType> &lhs, const SparsityPattern<X, PatternType> &rhs) {
    return ((*(lhs.m_originalMatrix) == *(rhs.m_originalMatrix)) &&
            ((lhs.m_pattern == nullptr && rhs.m_pattern == nullptr) ||
             (lhs.m_pattern != nullptr && rhs.m_pattern != nullptr && *(lhs.m_pattern) == *(rhs.m_pattern))));
}

template <typename X, typename PatternType>
bool operator!=(const SparsityPattern<X, PatternType> &lhs, const SparsityPattern<X, PatternType> &rhs) {
    return !(lhs == rhs);
}

template <typename X, typename Type>
std::ostream &operator<<(std::ostream &os, const SparsityPattern<X, Type> &p) {
    os << "Sparsity Pattern: " << std::endl;
    os << *(p.m_pattern);
    return os;
}