#pragma once

/// BLAS-1 style vector kernels and sparse matrix operations. The building blocks of
/// GMRES, preconditioner applications and AMG setup.

#include "sam_hpc/core/spgemm.hpp"

/// Computes the inner product of two vectors a = (u^T)v.
template<typename T>
T innerProduct(const std::vector<T>& vec1, const std::vector<T>& vec2) {
  assert((vec1.size() == vec2.size()) && "Vectors must be of same size");
  const size_t n = vec1.size();

  // Using tbb::parallel_deterministic_reduce to proudce deterministic output for
  // reduction of a vector of doubles. Minimum chunk size of reduce_grain prevents
  // smaller chunks and increased overhead.
  return tbb::parallel_deterministic_reduce(
    tbb::blocked_range<size_t>(0, n, reduce_grain), T(0),
    [&](const tbb::blocked_range<size_t>& r, T init) -> T {
      T sum = T(0);
      for (size_t i = r.begin(); i < r.end(); ++i) {
        sum += vec1[i] * vec2[i];
      }
      return init + sum;
    },
    [](T a, T b) -> T { return a + b; }
  );
}

/// Euclidean norm ||x||_2 = sqrt(x^T x), via innerProduct.
template <typename T>
T norm(const std::vector<T>& x) {
  return sqrt(innerProduct(x, x));
}

/// Compute axpby operation between two vectors y = ax + by.
template<typename T>
void axpby(T a, const std::vector<T>& x, T b, std::vector<T>& y) {
  assert((x.size() == y.size()) && "Vectors must be of same size");

  if (b != T(0)) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size(), vector_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          y[i] = a * x[i] + b * y[i];
        }
      }
    );
  } else {
    // Compute y = ax, if b == 0
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size(), vector_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          y[i] = a * x[i];
        }
      }
    );
  }
}

/// Compute axpbypcz operation z = ax + by + cz
template<typename T>
void axpbypcz(T a, const std::vector<T>& x, T b, const std::vector<T>& y, T c, std::vector<T>& z) {
  assert((x.size() == y.size()) && "Vectors  x and y must be of same size");
  assert((x.size() == z.size()) && "Vectors  x and z must be of same size");
  const size_t n = x.size();

  if (c != T(0)) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, vector_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          z[i] = a * x[i] + b * y[i] + c * z[i];
        }
      }
    );
  } else {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, vector_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          z[i] = a * x[i] + b * y[i];
        }
      }
    );
  }
}

/// Computes the linear combination of vectors y = alpha*y_old + c0*v0 + c1*v1 + ... + c(n-1)*v(n-1)
/// n is the total number of terms (vectors) to combine and must be >= 1. c and v are the
/// vector of coefficients and vectors. c and v may be longer but only the first n are used.
template <typename T>
void linComb(size_t n, const std::vector<T>& c, const std::vector<std::vector<T>>& v,
             const T& alpha, std::vector<T>& y)
{
  // Handle first term, y = alpha*y + c[0]*v[0]
  axpby(c[0], v[0], alpha, y);
  
  size_t i = 1;
  for (; i + 1 < n; i += 2) {
    // Process terms in pairs, y = y + c[i]*v[i] + c[i+1]*v[i+1].
    // Underlying operation uses TBB.
    axpbypcz(c[i], v[i], c[i + 1], v[i + 1], T(1), y);
  }

  // Process the remaining term, y = y + c[n-1]*v[n-1]
  for (; i < n; ++i) {
    axpby(c[i], v[i], T(1), y);
  }
}

/// Performs a fused element-wise vector multiplication and addition, z[i] = alpha * x[i] * y[i] + beta * z[i].
template<typename T>
void vmul(T alpha, const std::vector<T>& x, const std::vector<T>& y, T beta, std::vector<T>& z) {
  const size_t n = x.size();

  if (beta != T(0)) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, vector_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          z[i] = alpha * x[i] * y[i] + beta * z[i];
        }
      }
    );
  } else {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, vector_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          z[i] = alpha * x[i] * y[i];
        }
      }
    );
  }
}

// ============================= Matrix Operations =======================================
/// Computes the sparse matrix vector product y = alpha * Ax + beta * y
/// Each row is a dot product, computed in parallel.
template <typename T>
void spmv(T alpha, const csr::CSRMatrix<T>& A, const std::vector<T>& x, T beta, std::vector<T>& y) {
  assert((A.m_cols == x.size() && A.m_rows == y.size()) && "Dimensions not comaptible for SpMV");
  if (beta != T(0)) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows, row_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          T sum = T(0);
          // Compute the inner product between row i and vector x
          for (auto a = A.rowBegin(i); a; ++a) {
            // Calculate the summation with the value from the corresponding index in x
            sum += a.value() * x[a.col()];
          }
          y[i] = alpha * sum + beta * y[i];
        }
      }
    );
  } else {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows, row_grain),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          T sum = T(0);
          for (auto a = A.rowBegin(i); a; ++a) {
            sum += a.value() * x[a.col()];
          }
          y[i] = alpha * sum;
        }
      }
    );
  }
}

/// Computes the residual r = rhs - Ax.
template <typename T>
void residual(const std::vector<T>& rhs, const csr::CSRMatrix<T>& A,
              const std::vector<T>& x, std::vector<T>& res)
{
  assert(A.m_cols == x.size() && "Matrix A's column count must match vector x's size for residual calculation.");
  assert(A.m_rows == rhs.size() && "Matrix A's row count must match vector rhs's size for residual calculation.");
  assert(A.m_rows == res.size() && "Matrix A's row count must match vector res's size for residual calculation.");

  // Compute spmv and subtract from rhs in a single pass.
  tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows, row_grain),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        T sum = T(0);
        for (auto a = A.rowBegin(i); a; ++a) {
          sum += a.value() * x[a.col()];
        }
        res[i] = rhs[i] - sum;
      }
    }
  );
}

/// Computes the product of two CSR matrices C = A * B.
template <typename T>
csr::CSRMatrix<T> matrix_product(const csr::CSRMatrix<T> &A, const csr::CSRMatrix<T> &B, 
                                bool sort = false)
{
  assert(A.m_cols == B.m_rows && "Matrix dimensions incompatible for multiplication");
  csr::CSRMatrix<T> C;
  
  if (spgemm_acc == spgemm_accumulator::hash) {
    spgemm_hash(A, B, C, sort);
  } else {
    spgemm_saad(A, B, C, sort);
  }

  return C;
}


namespace csr::detail {

/// Merging two sorted rows a and b. When Fill is false the corresponding column indices
/// values are not written and only the width of the row is returned. If Fill is true the
/// column indices and values are updated according to the merge and the row width is returned.
template <bool Fill, typename T>
size_t scaled_row_union(T alpha, const index_t* ac_start, const index_t* ac_end, const T* av,
                        T beta, const index_t* bc_start, const index_t* bc_end, const T* bv,
                        index_t* cols, T* vals)
{
  size_t width = 0;
  while (ac_start != ac_end && bc_start != bc_end) {
    // Row a is smaller
    if (*ac_start < *bc_start) {
      if constexpr (Fill) { 
        cols[width] = *ac_start;
        vals[width] = alpha * *av; 
      }
      ++ac_start;
      ++av;
    } else if (*bc_start < *ac_start) {
      // Row b is smaller
      if constexpr (Fill) { 
        cols[width] = *bc_start; 
        vals[width] = beta * *bv; 
      }
      ++bc_start;
      ++bv;
    } else {
      // The indices are the same, so the value is added.
      if constexpr (Fill) { 
        cols[width] = *ac_start; 
        vals[width] = alpha * *av + beta * *bv; 
      }
      ++ac_start; 
      ++av; 
      ++bc_start; 
      ++bv;
    }
    ++width;
  }
  // Handle the remaining entries in either of the rows
  for (; ac_start != ac_end; ++ac_start, ++av, ++width) {
    if constexpr (Fill) {
      cols[width] = *ac_start; 
      vals[width] = alpha * *av; 
    }
  }
  for (; bc_start != bc_end; ++bc_start, ++bv, ++width) {
    if constexpr (Fill) { 
      cols[width] = *bc_start; 
      vals[width] = beta * *bv; 
    }
  }
  return width;
}

} // namespace csr::detail

/// Computes the sum of two scaled CSR matrices C = alpha*A + beta*B.
template <typename T>
csr::CSRMatrix<T> matrix_sum(T alpha, const csr::CSRMatrix<T> &A, T beta, const csr::CSRMatrix<T> &B) {
  assert((A.m_rows == B.m_rows && A.m_cols == B.m_cols) && "Matrix dimensions incompatible for addition.");
  assert(A.hasSortedRows() && B.hasSortedRows() && "matrix_sum needs sorted rows in column indices order.");

  const auto &ap = A.m_row_pointers, &bp = B.m_row_pointers;
  const auto &ac = A.m_col_indices, &bc = B.m_col_indices;
  const auto &av = A.m_vals, &bv = B.m_vals;

  csr::CSRMatrix<T> C;
  C.m_rows = A.m_rows;
  C.m_cols = A.m_cols;
  C.m_row_pointers.resize(C.m_rows + 1, 0);

  // Count non-zeros per row
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows, row_grain),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Compute the merged row i from row i of A and B
        C.m_row_pointers[i + 1] = csr::detail::scaled_row_union<false, T>(
          alpha, ac.data() + ap[i], ac.data() + ap[i + 1], av.data() + ap[i],
          beta, bc.data() + bp[i], bc.data() + bp[i + 1], bv.data() + bp[i], 
          nullptr, nullptr
        );
      }
    }
  );

  // Scan non-zero count into cumulative row pointers
  C.m_nnz = C.scanRowSize();
  C.m_col_indices.resize(C.m_nnz);
  C.m_vals.resize(C.m_nnz);

  // Fill the col_indices and vals array of C
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows, row_grain),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        const size_t off = C.m_row_pointers[i];
        csr::detail::scaled_row_union<true, T>(
          alpha, ac.data() + ap[i], ac.data() + ap[i + 1], av.data() + ap[i],
          beta, bc.data() + bp[i], bc.data() + bp[i + 1], bv.data() + bp[i],
          C.m_col_indices.data() + off, C.m_vals.data() + off
        );
      }
    }
  );

  return C;
}

/// Computes the difference of two scaled CSR matrices C = alpha*A - beta*B.
template <typename T>
csr::CSRMatrix<T> matrix_subtraction(T alpha, const csr::CSRMatrix<T> &A, T beta, const csr::CSRMatrix<T> &B) {
  return matrix_sum(alpha, A, -beta, B);
}