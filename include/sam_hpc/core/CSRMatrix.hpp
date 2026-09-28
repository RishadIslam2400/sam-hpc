#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <numeric>
#include <algorithm>
#include <memory>
#include <cmath>
#include <execution>

#include "sam_hpc/core/csrTypes.hpp"
#include "sam_hpc/util/parallel.hpp"

template <typename T>
  requires std::is_arithmetic_v<T>
class CSRMatrix {
public:
  size_t m_rows, m_cols, m_nnz;
  csr::vec<T> m_vals;
  csr::vec<csr::offset_t> m_row_pointers;
  csr::vec<csr::index_t> m_col_indices; // Column indices are 32 bits

  CSRMatrix() : m_rows(0), m_cols(0), m_nnz(0) {
    m_row_pointers.push_back(0);
  }

  CSRMatrix(const size_t m, const size_t n, const std::vector<T> &vals,
            const std::vector<size_t> &row_pointers, const std::vector<size_t> &col_indices)
    : m_rows(m), m_cols(n)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector can not be empty");
    assert((row_pointers.size() == m_rows + 1)
            && "Rows pointers array does not match matrix row dimension.");

    m_nnz = row_pointers.back();

    assert((col_indices.size() == m_nnz)
            && "Column indices array does not match nonzero count.");
    assert((vals.size() == m_nnz) && "Values array does not match nonzero count.");

    construct(vals, row_pointers, col_indices);
  }

  CSRMatrix(const size_t m, const size_t n, const size_t nnz, const std::vector<T> &vals,
            const std::vector<size_t> &row_pointers, const std::vector<size_t> &col_indices)
    : m_rows(m), m_cols(n), m_nnz(nnz)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector can not be empty");
    assert((row_pointers.size() == m_rows + 1)
            && "Rows pointers array does not match matrix row dimension.");
    assert((col_indices.size() == row_pointers.back())
            && "Column indices array does not match nonzero count.");
    assert((vals.size() == col_indices.size()) && "Values array does not match nonzero count.");
    assert((m_nnz == row_pointers.back()) && "Nonzero count does not match rows pointers.");
    assert((vals.size() == m_nnz) && "Values array does not match input nnz.");

    construct(vals, row_pointers, col_indices);
  }

  CSRMatrix(const std::vector<T> &vals, const std::vector<size_t> &row_pointers,
            const std::vector<size_t> &col_indices)
  {
    assert(!row_pointers.empty() && "Row pointers vector cannot be empty.");
    m_rows = row_pointers.size() - 1;
    m_nnz = row_pointers.back();

    assert(vals.size() == col_indices.size()
           && "Values array size must match column indices array size.");
    assert(col_indices.size() == m_nnz
           && "Column indices array size must match row_pointers.back().");

    if (col_indices.empty()) {
      m_cols = 0;
    } else {
      m_cols = *std::max_element(col_indices.begin(), col_indices.end()) + 1;
    }

    construct(vals, row_pointers, col_indices);
  }

  CSRMatrix(const size_t m, const size_t n, const T *vals, const size_t *row_pointers,
            const size_t *col_indices) : m_rows(m), m_cols(n)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert(((row_pointers != nullptr) && (col_indices != nullptr) && (vals != nullptr))
           && "Input arrays can not be null Pointers.");

    m_nnz = row_pointers[m_rows];

    csr::assign_converted(m_row_pointers, row_pointers, row_pointers + m_rows + 1);
    csr::assign_converted(m_col_indices, col_indices, col_indices + m_nnz);
    m_vals.assign(vals, vals + m_nnz);
  }

  CSRMatrix(const size_t m, const size_t n, const size_t nnz, const T *vals,
            const size_t *row_pointers, const size_t *col_indices)
    : m_rows(m), m_cols(n), m_nnz(nnz)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert(((row_pointers != nullptr) && (col_indices != nullptr) && (vals != nullptr))
           && "Input arrays can not be null Pointers.");
    assert((m_nnz == row_pointers[m_rows]) && "Nonzero count does not match rows pointers.");

    csr::assign_converted(m_row_pointers, row_pointers, row_pointers + m_rows + 1);
    csr::assign_converted(m_col_indices, col_indices, col_indices + m_nnz);
    m_vals.assign(vals, vals + m_nnz);
  }

  CSRMatrix(const std::vector<std::vector<T>> &matrix)
    : m_rows(matrix.size()), m_cols(matrix.empty() ? 0 : matrix[0].size())
  {
    if (m_rows == 0) {
      m_nnz = 0;
      m_row_pointers.push_back(0);
      return;
    }

    m_row_pointers.resize(m_rows + 1, 0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        int row_width = 0;
        for (size_t j = 0; j < m_cols; ++j) {
          if (matrix[i][j] != T(0)) {
            ++row_width;
          }
        }
        m_row_pointers[i + 1] = row_width;
      }
    });

    m_nnz = scanRowSize();
    m_vals.resize(m_nnz);
    m_col_indices.resize(m_nnz);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t row_head = m_row_pointers[i];
        for (size_t j = 0; j < m_cols; ++j) {
          if (matrix[i][j] != T(0)) {
            m_vals[row_head] = matrix[i][j];
            m_col_indices[row_head] = j;
            ++row_head;
          }
        }
      }
    });
  }

  // copy constructor
  CSRMatrix(const CSRMatrix<T> &other) {
    deepCopy(other);
  }

  CSRMatrix<T> &operator=(const CSRMatrix<T> &other) {
    if (this != &other) {
      deepCopy(other);
    }

    return *this;
  }

  // move constructor
  CSRMatrix(CSRMatrix<T> &&other) { 
    shallowCopy(std::move(other));
  }

  CSRMatrix<T> &operator=(CSRMatrix<T> &&other) {
    if (this != &other) {
      clear();
      shallowCopy(std::move(other));
    }

    return *this;
  }

  CSRMatrix(const size_t m, const size_t n, std::vector<T> &&vals,
            std::vector<size_t> &&row_pointers, std::vector<size_t> &&col_indices)
    : m_rows(m), m_cols(n)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector cannot be empty.");
    assert((row_pointers.size() == m_rows + 1)
            && "Rows pointers array does not match matrix row dimension.");

    m_nnz = row_pointers.back();

    assert((col_indices.size() == m_nnz)
            && "Column indices array does not match nonzero count.");
    assert((vals.size() == m_nnz) && "Values array does not match nonzero count.");

    // Cannot move the buffers with csr::vec.
    construct(vals, row_pointers, col_indices);
    release(vals);
    release(row_pointers);
    release(col_indices);
  }

  CSRMatrix(const size_t m, const size_t n, const size_t nnz, std::vector<T> &&vals,
            std::vector<size_t> &&row_pointers, std::vector<size_t> &&col_indices)
    : m_rows(m), m_cols(n), m_nnz(nnz)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector cannot be empty.");
    assert((row_pointers.size() == m_rows + 1)
            && "Rows pointers array does not match matrix row dimension.");
    assert((col_indices.size() == row_pointers.back())
            && "Column indices array does not match nonzero count.");
    assert((vals.size() == col_indices.size())
            && "Values array does not match nonzero count.");
    assert((m_nnz == row_pointers.back()) && "Nonzero count does not match rows pointers.");
    assert((vals.size() == m_nnz) && "Values array does not match input nnz.");

    // Cannot move the buffers with csr::vec.
    construct(vals, row_pointers, col_indices);
    release(vals);
    release(row_pointers);
    release(col_indices);
  }

  CSRMatrix(std::vector<T> &&vals, std::vector<size_t> &&row_pointers,
            std::vector<size_t> &&col_indices)
  {
    assert((!row_pointers.empty()) && "Row pointers vector cannot be empty.");
    assert((vals.size() == col_indices.size())
            && "Values array size must match column indices array size.");
    assert((col_indices.size() == row_pointers.back())
            && "Column indices array size must match row_pointers.back().");

    m_rows = row_pointers.size() - 1;
    if (col_indices.empty()) {
      m_cols = 0;
    } else {
      m_cols = *std::max_element(col_indices.begin(), col_indices.end()) + 1;
    }
    m_nnz = row_pointers.back();

    // Cannot move the buffers with csr::vec.
    construct(vals, row_pointers, col_indices);
    release(vals);
    release(row_pointers);
    release(col_indices);
  }

  inline bool isEmpty() const {
    return m_nnz == 0;
  }

  class row_iterator {
  public:
    row_iterator(const csr::index_t *col, const csr::index_t *end, const T *val)
        : m_col(col), m_end(end), m_val(val) {}

    row_iterator &operator++() {
      ++m_col;
      ++m_val;
      return *this;
    }

    operator bool() const { return m_col < m_end; }
    size_t col() const { return *m_col; }
    T value() const { return *m_val; }

  private:
    const csr::index_t *m_col;
    const csr::index_t *m_end;
    const T *m_val;
  };
  row_iterator rowBegin(size_t row) const {
    assert((row < m_rows) && "Row index out of bounds.");
    const size_t start = m_row_pointers[row];
    const size_t end = m_row_pointers[row + 1];
    return row_iterator(m_col_indices.data() + start, m_col_indices.data() + end, m_vals.data() + start);
  }

  T get(size_t row, size_t col) const {
    assert((row < m_rows) && "Row index out of bounds.");
    assert((col < m_cols) && "Column index out of bounds.");

    int current_col{-1};
    for (size_t pos = m_row_pointers[row]; pos < m_row_pointers[row + 1]; ++pos) {
      current_col = static_cast<int>(m_col_indices[pos]);

      if (current_col == static_cast<int>(col)) {
        return m_vals[pos];
      } else if (current_col > static_cast<int>(col)) {
        break;
      }
    }

    return T(0);
  }

  CSRMatrix<T> &set(T val, size_t row, size_t col) {
    assert((row < m_rows) && "Row index out of bounds.");
    assert((col < m_cols) && "Column index out of bounds.");

    size_t pos = m_row_pointers[row];
    int current_col{-1};

    for (; pos < m_row_pointers[row + 1]; ++pos) {
      current_col = static_cast<int>(m_col_indices[pos]);

      if (current_col >= static_cast<int>(col)) {
        break;
      }
    }

    if (current_col != static_cast<int>(col)) {
      if (!(val == T(0))) {
        insert(pos, row, col, val);
      }
    } else if (val == T(0)) {
        remove(pos, row);
    } else {
        m_vals[pos] = val;
    }

    return *this;
  }

  /// Turns per-row nonzero counts into row pointers, in place.
  size_t scanRowSize() {
    if (m_row_pointers.empty())
      return 0;
    std::partial_sum(m_row_pointers.begin(), m_row_pointers.end(), m_row_pointers.begin());
    return m_row_pointers[m_rows];
  }

  size_t bytes() const {
    return sizeof(size_t) * (m_rows + 1) + sizeof(size_t) * m_nnz + sizeof(T) * m_nnz;
  }

  template <typename X>
  friend bool operator==(const CSRMatrix<X> &lhs, const CSRMatrix<X> &rhs);

  template <typename X>
  friend bool operator!=(const CSRMatrix<X> &lhs, const CSRMatrix<X> &rhs);

  template <typename X>
  friend std::ostream &operator<<(std::ostream &os, const CSRMatrix<X> &matrix);

  template <typename X>
  friend void swap(CSRMatrix<X> &lhs, CSRMatrix<X> &rhs);

private:
  /// Empties a vector the caller handed us as an rvalue, freeing its buffer.
  template <typename V>
  static void release(V &v) {
    V().swap(v);
  }

  /// Boundary between the std::vector<size_t> the public constructors.
  void construct(const std::vector<T> &vals, const std::vector<size_t> &row_pointers,
                 const std::vector<size_t> &col_indices)
  {
    assert((col_indices.empty() ||
            *std::max_element(col_indices.begin(), col_indices.end()) <= csr::max_index) &&
            "Column index exceeds csr::index_t; widen csr::index_t in csrTypes.hpp.");
    csr::assign_converted(m_row_pointers, row_pointers);
    csr::assign_converted(m_col_indices, col_indices);
    m_vals.assign(vals.begin(), vals.end());
  }

  void clear() {
    m_row_pointers.clear();
    m_col_indices.clear();
    m_vals.clear();
    m_rows = 0;
    m_cols = 0;
    m_nnz = 0;
    m_row_pointers.push_back(0);
  }

  void deepCopy(const CSRMatrix<T> &other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_nnz = other.m_nnz;
    m_row_pointers = other.m_row_pointers;
    m_col_indices = other.m_col_indices;
    m_vals = other.m_vals;
  }
  void shallowCopy(CSRMatrix<T> &&other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_nnz = other.m_nnz;
    m_row_pointers = std::move(other.m_row_pointers);
    m_vals = std::move(other.m_vals);
    m_col_indices = std::move(other.m_col_indices);

    other.m_rows = 0;
    other.m_cols = 0;
    other.m_nnz = 0;
    other.m_row_pointers.clear();
    other.m_row_pointers.push_back(0);
    other.m_col_indices.clear();
    other.m_vals.clear();
  }

  void insert(size_t index, size_t row, size_t col, T val) {
    m_vals.insert(m_vals.begin() + index, val);
    m_col_indices.insert(m_col_indices.begin() + index, col);
    m_nnz += 1;

    for (size_t i = row; i < m_rows; ++i) {
      m_row_pointers[i + 1] += 1;
    }
  }

  void remove(size_t index, size_t row) {
    assert((!isEmpty()) && "Cannot remove from an empty matrix.");
    m_vals.erase(m_vals.begin() + index);
    m_col_indices.erase(m_col_indices.begin() + index);
    m_nnz -= 1;

    for (size_t i = row; i < m_rows; ++i) {
      m_row_pointers[i + 1] -= 1;
    }
  }
};

/// @brief Sort a single row of the matrix columnwise.
/// Simultaneously sorts col_indices and permutes vals accordingly.
/// @param col_indices Pointer to the start of column indices for the row.
/// @param vals Pointer to the start of values for the row.
/// @param size Number of non-zeros in the row.
template <typename T>
void sortRow(csr::index_t* col_indices, T* vals, int size) {
  // insertion sort
  for (int j = 1; j < size; ++j) {
    size_t col_idx = col_indices[j];
    T val = vals[j];

    int i = j - 1;

    while (i >= 0 && col_indices[i] > col_idx) {
      col_indices[i + 1] = col_indices[i];
      vals[i + 1] = vals[i];
      i--;
    }

    col_indices[i + 1] = col_idx;
    vals[i + 1] = val;
  }
}

/// @brief Sorts all rows of the CSRMatrix column-wise.
/// This ensures that for each row, the column indices are in ascending order.
template <typename T>
void sortRows(CSRMatrix<T>& A) {
  if (A.isEmpty())
    return;

  tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows),
  [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      size_t row_start = A.m_row_pointers[i];
      size_t row_end = A.m_row_pointers[i + 1];

      sortRow(A.m_col_indices.data() + row_start,  A.m_vals.data() + row_start,
              static_cast<int>(row_end - row_start));
    }
  });
}

/// @brief Computes the transpose of a CSRMatrix.
/// @param A The input CSRMatrix.
/// @return A shared_ptr to the new transposed CSRMatrix.
template <typename T>
std::shared_ptr<CSRMatrix<T>> transpose(const CSRMatrix<T>& A) {
  if (A.isEmpty() && A.m_rows == 0 && A.m_cols == 0)
      return std::make_shared<CSRMatrix<T>>();
  
  const size_t original_rows = A.m_rows;
  const size_t original_cols = A.m_cols;
  const size_t nnz = A.m_nnz;

  auto transposedMatrix = std::make_shared<CSRMatrix<T>>();
  transposedMatrix->m_rows = original_cols;
  transposedMatrix->m_cols = original_rows;
  transposedMatrix->m_nnz = nnz;
  transposedMatrix->m_row_pointers.resize(transposedMatrix->m_rows + 1, 0);
  if (nnz > 0) {
    transposedMatrix->m_col_indices.resize(nnz, 0);
    transposedMatrix->m_vals.resize(nnz, T(0));
  } else {
    return transposedMatrix;
  }

  // Step 1: Count number of non-zero elements in each column of A
  // These counts become the row counts for the transposed matrix.
  for (size_t j = 0; j < nnz; ++j) {
    ++(transposedMatrix->m_row_pointers[A.m_col_indices[j] + 1]);
  }

  transposedMatrix->scanRowSize();

  for (size_t i = 0; i < original_rows; ++i) {
    const size_t start = A.m_row_pointers[i];
    const size_t end = A.m_row_pointers[i + 1];
    for (size_t j = start; j < end; ++j) {
      size_t head = transposedMatrix->m_row_pointers[A.m_col_indices[j]]++;
      transposedMatrix->m_col_indices[head] = i;
      transposedMatrix->m_vals[head] = A.m_vals[j];
    }
  }

  std::rotate(transposedMatrix->m_row_pointers.begin(), transposedMatrix->m_row_pointers.begin() + original_cols, transposedMatrix->m_row_pointers.begin() + original_cols + 1);
  transposedMatrix->m_row_pointers[0] = 0;

  return transposedMatrix;
}

/// @brief Scales all non-zero values of a CSRMatrix by a given factor.
/// @param A The CSRMatrix to scale (modified in-place).
/// @param factor The scaling factor.
template <typename T>
void scale(CSRMatrix<T>& A, T factor) {
  if (A.isEmpty()) {
    return;
  }

  std::transform(std::execution::par_unseq, A.m_vals.begin(), A.m_vals.end(), A.m_vals.begin(), 
                  [factor](T val) { return val * factor; });
}

enum class diagonalType {
  simple,
  inverse,
  forScaling
};

/// @brief Extracts the diagonal of a CSRMatrix using compile-time dispatch.
/// @tparam type The type of diagonal to extract (must be a compile-time constant).
/// @tparam T The numeric type of the vector elements.
/// @param A The input CSRMatrix.
/// @return A std::vector<T> containing the diagonal elements.
template <diagonalType type, typename T>
std::vector<T> diagonal(const CSRMatrix<T>& A) {
  const size_t m = A.m_rows;
  
  if (m == 0) {
    return {};
  }

  std::vector<T> diag(m);
  if constexpr (type == diagonalType::simple) {
    std::fill(diag.begin(), diag.end(), T(0));
  } else {
    std::fill(diag.begin(), diag.end(), T(1));
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, m),[&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      for (auto a = A.rowBegin(i); a; ++a) {
      if (a.col() == i) {
          T val = a.value();
          
          if constexpr (type == diagonalType::simple) {
            diag[i] = val;
          } 
          else if constexpr (type == diagonalType::inverse) {
            if (val != T(0)) {
              diag[i] = T(1) / val;
            }
          }
          else if constexpr (type == diagonalType::forScaling) {
            if (val != T(0)) {
              diag[i] = T(1) / std::sqrt(std::abs(val));
            }
          }
          
          break; // Found diagonal, no need to search further in this row.
        } else if (a.col() > i) {
          // Assumes sorted columns, so we've passed the diagonal's position.
          break;
        }
      }
    }
  });

  return diag;
}

template <typename X>
bool operator==(const CSRMatrix<X> &lhs, const CSRMatrix<X> &rhs) {
  if ((lhs.m_rows != rhs.m_rows) || (lhs.m_cols != rhs.m_cols) || (lhs.m_nnz != rhs.m_nnz)) {
    return false;
  }

  return (lhs.m_row_pointers == rhs.m_row_pointers) &&
         (lhs.m_col_indices == rhs.m_col_indices) &&
         (lhs.m_vals == rhs.m_vals);
}

template <typename X>
bool operator!=(const CSRMatrix<X> &lhs, const CSRMatrix<X> &rhs) {
  return !(lhs == rhs);
}

template <typename X>
std::ostream &operator<<(std::ostream &os, const CSRMatrix<X> &matrix) {
  if (matrix.isEmpty()) {
    os << "Matrix is empty.";
  } else {
    os << "Matrix (" << matrix.m_rows << "x" << matrix.m_cols
       << "), NNZ: " << matrix.m_nnz << "\n";
    for (size_t i = 0; i < matrix.m_rows; ++i) {
      for (size_t j = 0; j < matrix.m_cols; ++j) {
        X val = matrix.get(i, j);
        if (val != X()) {
          os << "(" << i << ", " << j << "): ";
          os << val << "\n";
        }
      }
    }
  }

  return os;
}

template <typename X>
void swap(CSRMatrix<X> &lhs, CSRMatrix<X> &rhs) {
  std::swap(lhs.m_rows, rhs.m_rows);
  std::swap(lhs.m_cols, rhs.m_cols);
  std::swap(lhs.m_nnz, rhs.m_nnz);
  lhs.m_row_pointers.swap(rhs.m_row_pointers);
  lhs.m_col_indices.swap(rhs.m_col_indices);
  lhs.m_vals.swap(rhs.m_vals);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << vec[i];
  }
  os << "]";
  return os;
}

inline bool operator==(const std::vector<double> &lhs, const std::vector<double> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (std::abs(lhs[i] - rhs[i]) > 1e-6)
      return false;
  }
  return true;
}