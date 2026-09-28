#pragma once

#include "sam_hpc/core/spgemm.hpp"

/**
 * @todo: functions to implement:
 * spectral_radius(), inverse()
 */

// ============================= Vector Operations =======================================
/// @brief Computes the inner product of two vectors a = (u^T)v using Kahan summation.
/// @param vec1 The first vector (u).
/// @param vec2 The second vector (v).
/// @return The inner product of vec1 and vec2.
template<typename T>
T innerProduct(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    assert((vec1.size() == vec2.size()) && "Vectors must be of same size");
    const size_t n = vec1.size();
    
    struct KahanState {
        T sum = T(0);
        T correction = T(0);
    };

    KahanState final_state = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), KahanState(),
        [&](const tbb::blocked_range<size_t>& r, KahanState state) -> KahanState {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                T d = (vec1[i] * vec2[i]) - state.correction;
                T t = state.sum + d;
                state.correction = (t - state.sum) - d;
                state.sum = t;
            }
            return state;
        },
        [](KahanState a, KahanState b) -> KahanState {
            T d = b.sum - a.correction;
            T t = a.sum + d;
            a.correction = (t - a.sum) - d;
            a.sum = t;
            a.correction += b.correction;
            return a;
        }
    );
    
    return final_state.sum;
}

template <typename T>
T norm(const std::vector<T>& x) {
    return std::abs(sqrt(innerProduct(x, x)));
}

/// @brief Compute axpby operation between two vectors y = ax + by 
/// @param a Scalar 'a'
/// @param x Vector 'x'
/// @param b Scalar 'b'
/// @param y Vector 'y' (input and output)
template<typename T>
void axpby(T a, const std::vector<T>& x, T b, std::vector<T>& y) {
    assert((x.size() == y.size()) && "Vectors must be of same size");

    if (b != T(0)) {
        std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), y.begin(),
                       [&](const auto xi, const auto yi) {
                           return a * xi + b * yi;
                       });
    } else {
        std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(),
                       [&](const auto xi) { 
                            return a * xi; 
                        });
    }
}

/// @brief Compute axpbypcz operation z = ax + by + cz
/// @param a Scalar 'a'
/// @param x Vector 'x'
/// @param b Scalar 'b'
/// @param y Vector 'y'
/// @param c Scalar 'c'
/// @param z Vector 'z' (input and output)
template<typename T>
void axpbypcz(T a, const std::vector<T>& x, T b, const std::vector<T>& y, T c, std::vector<T>& z) {
    assert((x.size() == y.size()) && "Vectors  x and y must be of same size");
    assert((x.size() == z.size()) && "Vectors  x and z must be of same size");
    const size_t n = x.size();

    // TODO: change approach for three vectors, use launch threads
    if (c != T(0)) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));
        std::transform(std::execution::par_unseq, indices.begin(), indices.end(), z.begin(), [&](size_t i)
                       { return a * x[i] + b * y[i] + c * z[i]; });
    } else {
        std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), z.begin(), [&](const auto xi, const auto yi)
                       { return a * xi + b * yi; });
    }
}

/// @brief Computes the linear combination of vectors y = alpha*y_old + c0*v0 + c1*v1 + ... + c(n-1)*v(n-1)
/// @param n Number of elements.
/// @param c Vector of coefficients c_k.
/// @param v Vector of vectors v_k.
/// @param alpha Scalar multiplier for the initial vector y.
/// @param y Input vector y_old, output vector y_new.
template <typename T>
void linComb(size_t n, const std::vector<T>& c, const std::vector<std::vector<T>>& v, const T& alpha, std::vector<T>& y) {
    // First term and scaling of y: y_new = alpha*y_old + c[0]*v[0]
    axpby(c[0], v[0], alpha, y);
    size_t i = 1;
    for (; i + 1 < n; i += 2) {
        // Process pairs of terms: y_new = y_current + c[i]*v[i] + c[i+1]*v[i+1]
        axpbypcz(c[i], v[i], c[i + 1], v[i + 1], T(1), y);
    }

    // Process any remaining single term: y_new = y_current + c[i]*v[i]
    for (; i < n; ++i) {
        axpby(c[i], v[i], T(1), y);
    }
}

/**
 * @brief Performs a fused element-wise vector multiplication and addition.
 *
 * This function computes the generalized vector operation:
 * z[i] = alpha * x[i] * y[i] + beta * z[i], for each element `i` in the vectors.
 *
 * @param[in] alpha The scalar multiplier for the element-wise product of `x` and `y`.
 * @param[in] x     The first input vector.
 * @param[in] y     The second input vector.
 * @param[in] beta  The scalar multiplier for the `z` vector. If `beta` is zero,
 * the initial values in `z` are ignored.
 * @param[in,out] z The vector used for input (when beta != 0) and to store the final result.
 */
template<typename T>
void vmul(T alpha, const std::vector<T>& x, const std::vector<T>& y, T beta, std::vector<T>& z) {
    const size_t n = x.size();

    // TODO: change approach for three vectors, use launch threads
    if (beta != T(0)) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));
        std::transform(std::execution::par_unseq, indices.begin(), indices.end(), z.begin(), [&](size_t i)
                       { return alpha * x[i] * y[i] + beta * z[i]; });
    } else {
        std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), z.begin(), [&](const auto xi, const auto yi)
                       { return alpha * xi * yi; });
    }
}

// ============================= Matrix Operations =======================================
/// @brief Computes the sparse matrix vector product y = alpha * Ax + beta * y
/// @param alpha Scalar multiplier for Ax.
/// @param A The input CSRMatrix.
/// @param x The input vector.
/// @param beta Scalar multiplier for y_old.
/// @param y The input/output vector. y is overwritten.
template <typename T>
void spmv(T alpha, const CSRMatrix<T>& A, const std::vector<T>& x, T beta, std::vector<T>& y) {
    assert((A.m_cols == x.size() && A.m_rows == y.size()) && "Dimensions not comaptible for SpMV");
    if (beta != T(0)) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                T sum = T(0);
                for (auto a = A.rowBegin(i); a; ++a) {
                    sum += a.value() * x[a.col()];
                }
                y[i] = alpha * sum + beta * y[i];
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                T sum = T(0);
                for (auto a = A.rowBegin(i); a; ++a) {
                    sum += a.value() * x[a.col()]; // random access into vector x
                }
                y[i] = alpha * sum;
            }
        });
    }
}

/// @brief Computes the residual r = rhs - Ax.
/// @param rhs The right-hand side vector.
/// @param A The input CSRMatrix.
/// @param x The vector to be multiplied by A.
/// @param res The output vector where the residual will be stored.
template <typename T>
void residual(const std::vector<T>& rhs, const CSRMatrix<T>& A, const std::vector<T>& x, std::vector<T>& res) {
    assert(A.m_cols == x.size() && "Matrix A's column count must match vector x's size for residual calculation.");
    assert(A.m_rows == rhs.size() && "Matrix A's row count must match vector rhs's size for residual calculation.");
    assert(A.m_rows == res.size() && "Matrix A's row count must match vector res's size for residual calculation.");

    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            T sum = T(0);
            for (auto a = A.rowBegin(i); a; ++a) {
                sum += a.value() * x[a.col()];
            }
            res[i] = rhs[i] - sum;
        }
    });
}

/// @brief Computes the product of two CSRMatrices C = A * B.
/// Dispatches to different SpGEMM algorithms based on the number of threads.
/// @param A The left-hand side input CSRMatrix.
/// @param B The right-hand side input CSRMatrix.
/// @param sort Optional: whether to sort the rows of the resulting matrix C.
/// @return A shared_ptr to the new CSRMatrix C, the result of A * B.
template <typename T>
std::shared_ptr<CSRMatrix<T>> matrix_product(const CSRMatrix<T> &A, const CSRMatrix<T> &B, bool sort = false) {
    assert(A.m_cols == B.m_rows && "Matrix dimensions incompatible for multiplication");
    std::shared_ptr<CSRMatrix<T>> C = std::make_shared<CSRMatrix<T>>();

    if (num_threads > 16) {
        spgemm_rmerge(A, B, *C);
    }
    else {
        spgemm_saad(A, B, *C, sort);
    }

    return C;
}

/// @brief Computes the product of two CSRMatrices C = A * B.
/// Dispatches to different SpGEMM algorithms based on the number of threads.
/// @param A The left-hand side input CSRMatrix.
/// @param B The right-hand side input CSRMatrix.
/// @param C The resultant output CSRMatrix.
/// @param sort Optional: whether to sort the rows of the resulting matrix C.
/// @return The CSRMatrix C contains the result of A * B.
template <typename T>
void matrix_product(const CSRMatrix<T> &A, const CSRMatrix<T> &B, CSRMatrix<T>& C, bool sort = false) {
    assert((A.m_rows == B.m_cols) && "Matrix dimensions incompatible for multiplication");
    if (!C.isEmpty())
        C.clear();

    if (num_threads > 16) {
        spgemm_rmerge(A, B, C);
    }
    else {
        spgemm_saad(A, B, C, sort);
    }
}


/// @brief Computes the sum of two scaled CSRMatrices C = alpha*A + beta*B.
/// @param alpha Scalar multiplier for matrix A.
/// @param A The first input CSRMatrix.
/// @param beta Scalar multiplier for matrix B.
/// @param B The second input CSRMatrix.
/// @param sort Optional: whether to sort the rows of the resulting matrix C.
/// @return A shared_ptr to the new CSRMatrix C.
template <typename T>
std::shared_ptr<CSRMatrix<T>> matrix_sum(T alpha, const CSRMatrix<T> &A, T beta, const CSRMatrix<T> &B, bool sort = false) {
    assert((A.m_rows == B.m_rows && A.m_cols == B.m_cols) && "Matrix dimensions incompatible for addition.");

    std::shared_ptr<CSRMatrix<T>> C = std::make_shared<CSRMatrix<T>>();

    C->m_rows = A.m_rows;
    C->m_cols = A.m_cols;
    C->m_row_pointers.resize(C->m_rows + 1, 0);

    // Phase 1: Compute non-zero count per row in the result matrix C
    // C->m_row_pointers will store counts in C->m_row_pointers[i+1] for row i
    tbb::enumerable_thread_specific<std::vector<int>> local_markers_1(C->m_cols, -1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
        std::vector<int>& marker = local_markers_1.local();
        for (size_t i = r.begin(); i < r.end(); ++i) {
            size_t C_cols = 0;

            for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
                const size_t colIdx = A.m_col_indices[j];
                if (marker[colIdx] != static_cast<int>(i)) {
                    marker[colIdx] = static_cast<int>(i);
                    C_cols++;
                }
            }

            for (size_t j = B.m_row_pointers[i], e = B.m_row_pointers[i + 1]; j < e; ++j) {
                const size_t colIdx = B.m_col_indices[j];
                if (marker[colIdx] != static_cast<int>(i)) {
                    marker[colIdx] = static_cast<int>(i);
                    C_cols++;
                }
            }

            C->m_row_pointers[i + 1] = C_cols;
        }
    });

    C->m_nnz = C->scanRowSize();

    if (C->m_nnz > 0) {
        C->m_col_indices.resize(C->m_nnz, 0);
        C->m_vals.resize(C->m_nnz, T(0));
    } else {
        return C;
    }

    // Phase 2: Compute the column indices and values of the result matrix C
    tbb::enumerable_thread_specific<std::vector<int>> local_markers_2(C->m_cols, -1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
        std::vector<int>& marker = local_markers_2.local();
        for (size_t i = r.begin(); i < r.end(); ++i) {
            const size_t rowBeg = C->m_row_pointers[i];
            size_t rowEnd = rowBeg;

            for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
                size_t colIdx = A.m_col_indices[j];
                T value = alpha * A.m_vals[j];

                if (marker[colIdx] < static_cast<int>(rowBeg)) {
                    marker[colIdx] = static_cast<int>(rowEnd);
                    C->m_col_indices[rowEnd] = colIdx;
                    C->m_vals[rowEnd] = value;
                    rowEnd++;
                } else {
                    C->m_vals[marker[colIdx]] += value;
                }
            }

            for (size_t j = B.m_row_pointers[i], e = B.m_row_pointers[i + 1]; j < e; ++j) {
                size_t colIdx = B.m_col_indices[j];
                T value = beta * B.m_vals[j];

                if (marker[colIdx] < static_cast<int>(rowBeg)) {
                    marker[colIdx] = static_cast<int>(rowEnd);
                    C->m_col_indices[rowEnd] = colIdx;
                    C->m_vals[rowEnd] = value;
                    rowEnd++;
                } else {
                    C->m_vals[marker[colIdx]] += value;
                }
            }

            // Clean up the marker entries used for the current row.
            for (size_t k = rowBeg; k < rowEnd; ++k) {
                marker[C->m_col_indices[k]] = -1;
            }

            if (sort) {
                sortRow(C->m_col_indices.data() + rowBeg, C->m_vals.data() + rowBeg, static_cast<int>(rowEnd - rowBeg));
            }
        }
    });

    return C;
}

/// @brief Computes the difference of two scaled CSRMatrices C = alpha*A - beta*B.
/// @param alpha Scalar multiplier for matrix A.
/// @param A The first input CSRMatrix.
/// @param beta Scalar multiplier for matrix B.
/// @param B The second input CSRMatrix.
/// @param sort Optional: whether to sort the rows of the resulting matrix C.
/// @return A shared_ptr to the new CSRMatrix C.
template <typename T>
std::shared_ptr<CSRMatrix<T>> matrix_subtraction(T alpha, const CSRMatrix<T> &A, T beta, const CSRMatrix<T> &B, bool sort = false) {
    assert((A.m_rows == B.m_rows && A.m_cols == B.m_cols) && "Matrix dimensions incompatible for addition.");

    std::shared_ptr<CSRMatrix<T>> C = std::make_shared<CSRMatrix<T>>();

    C->m_rows = A.m_rows;
    C->m_cols = A.m_cols;
    C->m_row_pointers.resize(C->m_rows + 1, 0);

    tbb::enumerable_thread_specific<std::vector<int>> local_markers_1(C->m_cols, -1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
        std::vector<int>& marker = local_markers_1.local();
        for (size_t i = r.begin(); i < r.end(); ++i) {
            size_t C_cols = 0;

            for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
                const size_t colIdx = A.m_col_indices[j];
                if (marker[colIdx] != static_cast<int>(i)) {
                    marker[colIdx] = static_cast<int>(i);
                    C_cols++;
                }
            }

            for (size_t j = B.m_row_pointers[i], e = B.m_row_pointers[i + 1]; j < e; ++j) {
                const size_t colIdx = B.m_col_indices[j];
                if (marker[colIdx] != static_cast<int>(i)) {
                    marker[colIdx] = static_cast<int>(i);
                    C_cols++;
                }
            }

            C->m_row_pointers[i + 1] = C_cols;
        }
    });

    C->m_nnz = C->scanRowSize();

    if (C->m_nnz > 0) {
        C->m_col_indices.resize(C->m_nnz, 0);
        C->m_vals.resize(C->m_nnz, T(0));
    } else {
        return C;
    }

    tbb::enumerable_thread_specific<std::vector<int>> local_markers_2(C->m_cols, -1);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows), [&](const tbb::blocked_range<size_t>& r) {
        std::vector<int>& marker = local_markers_2.local();
        for (size_t i = r.begin(); i < r.end(); ++i) {
            const size_t rowBeg = C->m_row_pointers[i];
            size_t rowEnd = rowBeg;

            for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
                size_t colIdx = A.m_col_indices[j];
                T value = alpha * A.m_vals[j];

                if (marker[colIdx] < static_cast<int>(rowBeg)) {
                    marker[colIdx] = static_cast<int>(rowEnd);
                    C->m_col_indices[rowEnd] = colIdx;
                    C->m_vals[rowEnd] = value;
                    rowEnd++;
                } else {
                    C->m_vals[marker[colIdx]] += value;
                }
            }

            for (size_t j = B.m_row_pointers[i], e = B.m_row_pointers[i + 1]; j < e; ++j) {
                size_t colIdx = B.m_col_indices[j];
                T value = beta * B.m_vals[j];

                if (marker[colIdx] < static_cast<int>(rowBeg)) {
                    marker[colIdx] = static_cast<int>(rowEnd);
                    C->m_col_indices[rowEnd] = colIdx;
                    C->m_vals[rowEnd] = (-value);
                    rowEnd++;
                } else {
                    C->m_vals[marker[colIdx]] -= value;
                }
            }

            // Clean up the marker entries used for the current row.
            for (size_t k = rowBeg; k < rowEnd; ++k) {
                marker[C->m_col_indices[k]] = -1;
            }

            if (sort) {
                sortRow(C->m_col_indices.data() + rowBeg, C->m_vals.data() + rowBeg, static_cast<int>(rowEnd - rowBeg));
            }
        }
    });

    return C;
}