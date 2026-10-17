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

#include "sam_hpc/core/csrTypes.hpp"
#include "sam_hpc/util/parallel.hpp"

namespace csr {

namespace detail {

/// Sorts a single row in column indices order. Simultaneously sorts col_indices and
/// permutes vals accordingly.
template <typename T>
void sortRow(index_t* col_indices, T* vals, int size) {
  // insertion sort suitable for smaller number of non-zeros per row
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

} // namespace detail

/// A generic CSR implementation for sparse matrix operations.
///
/// Row i entries are in [m_row_pointers[i], m_row_pointers[i + 1]) of m_col_indices
/// and m_vals. m_row_pointers has m_rows + 1 entries from 0 to m_nnz. m_col_indices and
/// m_vals have m_nnz entries. The column indices in a row are assumed to be sorted at 
/// all times. 
template <typename T>
  requires std::is_arithmetic_v<T>
class CSRMatrix {
public:
  // Member variables are exposed for optimizing the matrix matrix multiplication
  size_t m_rows, m_cols, m_nnz;
  csr::vec<T> m_vals;
  csr::vec<csr::offset_t> m_row_pointers;
  csr::vec<csr::index_t> m_col_indices;

  /// Empty matrix constructor
  CSRMatrix() : m_rows(0), m_cols(0), m_nnz(0) {
    m_row_pointers.push_back(0);
  }

  /// Constructor for m x n matrix with input vectors for the csr matrix arrays.
  CSRMatrix(const size_t m, const size_t n, const std::vector<T> &vals,
            const std::vector<size_t> &row_pointers, const std::vector<size_t> &col_indices)
    : m_rows(m), m_cols(n)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector can not be empty");
    assert((row_pointers.size() == m_rows + 1) && "Rows pointers array does not match matrix row dimension.");

    // Take nnz from last row_pointers entry
    m_nnz = row_pointers.back();

    assert((col_indices.size() == m_nnz) && "Column indices array does not match nonzero count.");
    assert((vals.size() == m_nnz) && "Values array does not match nonzero count.");

    construct(vals, row_pointers, col_indices);
  }

  /// Constructor for m x n matrix with input vectors for the csr matrix arrays and
  /// nnz value passed explicitly.
  CSRMatrix(const size_t m, const size_t n, const size_t nnz, const std::vector<T> &vals,
            const std::vector<size_t> &row_pointers, const std::vector<size_t> &col_indices)
    : m_rows(m), m_cols(n), m_nnz(nnz)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector can not be empty");
    assert((row_pointers.size() == m_rows + 1) && "Rows pointers array does not match matrix row dimension.");
    assert((col_indices.size() == row_pointers.back()) && "Column indices array does not match nonzero count.");
    assert((vals.size() == col_indices.size()) && "Values array does not match nonzero count.");
    assert((m_nnz == row_pointers.back()) && "Nonzero count does not match rows pointers.");
    assert((vals.size() == m_nnz) && "Values array does not match input nnz.");

    construct(vals, row_pointers, col_indices);
  }

  /// Constructor without explicit matrix dimensions as the input.
  CSRMatrix(const std::vector<T> &vals, const std::vector<size_t> &row_pointers,
            const std::vector<size_t> &col_indices)
  {
    assert(!row_pointers.empty() && "Row pointers vector cannot be empty.");
    
    // Infer the rows and column from row pointers array
    m_rows = row_pointers.size() - 1;
    m_nnz = row_pointers.back();

    assert(vals.size() == col_indices.size() && "Values array size must match column indices array size.");
    assert(col_indices.size() == m_nnz && "Column indices array size must match row_pointers.back().");

    if (col_indices.empty()) {
      m_cols = 0;
    } else {
      m_cols = *std::max_element(col_indices.begin(), col_indices.end()) + 1;
    }

    construct(vals, row_pointers, col_indices);
  }

  /// Constructor csr matrix arrys from raw array pointers.
  CSRMatrix(const size_t m, const size_t n, const T *vals, const size_t *row_pointers,
            const size_t *col_indices) : m_rows(m), m_cols(n)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert(((row_pointers != nullptr) && (col_indices != nullptr) && (vals != nullptr)) && "Input arrays can not be null Pointers.");

    m_nnz = row_pointers[m_rows];
    csr::assign_converted(m_row_pointers, row_pointers, row_pointers + m_rows + 1);
    csr::assign_converted(m_col_indices, col_indices, col_indices + m_nnz);
    m_vals.assign(vals, vals + m_nnz);
  }

  /// Constructor csr matrix arrys from raw array pointers and explicit nnz value.
  CSRMatrix(const size_t m, const size_t n, const size_t nnz, const T *vals,
            const size_t *row_pointers, const size_t *col_indices)
    : m_rows(m), m_cols(n), m_nnz(nnz)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert(((row_pointers != nullptr) && (col_indices != nullptr) && (vals != nullptr)) && "Input arrays can not be null Pointers.");
    assert((m_nnz == row_pointers[m_rows]) && "Nonzero count does not match rows pointers.");

    csr::assign_converted(m_row_pointers, row_pointers, row_pointers + m_rows + 1);
    csr::assign_converted(m_col_indices, col_indices, col_indices + m_nnz);
    m_vals.assign(vals, vals + m_nnz);
  }

  /// Construct a CSR matrix from a row-major dense matrix.
  CSRMatrix(const std::vector<std::vector<T>> &matrix)
    : m_rows(matrix.size()), m_cols(matrix.empty() ? 0 : matrix[0].size())
  {
    if (m_rows == 0) {
      m_nnz = 0;
      m_row_pointers.push_back(0);
      return;
    }

    m_row_pointers.resize(m_rows + 1, 0);

    // Calculate row width in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        int row_width = 0;
        for (size_t j = 0; j < m_cols; ++j) {
          if (matrix[i][j] != T(0)) {
            ++row_width;
          }
        }

        // All row width are added independtly in each slot
        m_row_pointers[i + 1] = row_width;
      }
    });

    // Partial sum over row widths and compute nnz to allocate col_indices and vals array
    m_nnz = scanRowSize();
    m_vals.resize(m_nnz);
    m_col_indices.resize(m_nnz);

    // Fill up row chunks in parallel
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

  /// Copy constructor
  CSRMatrix(const CSRMatrix<T> &other) {
    deepCopy(other);
  }

  /// Copy assignment constructor
  CSRMatrix<T> &operator=(const CSRMatrix<T> &other) {
    if (this != &other) {
      deepCopy(other);
    }

    return *this;
  }

  /// Move constructor
  CSRMatrix(CSRMatrix<T> &&other) { 
    shallowCopy(std::move(other));
  }

  /// Move assignment constructor
  CSRMatrix<T> &operator=(CSRMatrix<T> &&other) {
    if (this != &other) {
      clear();
      shallowCopy(std::move(other));
    }

    return *this;
  }

  /// Constructor for m x n matrix with input vectors passed as a reference.
  /// The vector are in std::vector form and the CSR arrays are in csr::vec form. So
  /// they cannot interchange pointers to move the content. The input vectors are freed
  /// afterwards.
  CSRMatrix(const size_t m, const size_t n, std::vector<T> &&vals,
            std::vector<size_t> &&row_pointers, std::vector<size_t> &&col_indices)
    : m_rows(m), m_cols(n)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector cannot be empty.");
    assert((row_pointers.size() == m_rows + 1) && "Rows pointers array does not match matrix row dimension.");

    m_nnz = row_pointers.back();

    assert((col_indices.size() == m_nnz) && "Column indices array does not match nonzero count.");
    assert((vals.size() == m_nnz) && "Values array does not match nonzero count.");

    construct(vals, row_pointers, col_indices);
    release(vals);
    release(row_pointers);
    release(col_indices);
  }

  /// Constructor for m x n matrix with input vectors passed as a reference and an explicit
  /// nnz value. The vector are in std::vector form and the CSR arrays are in csr::vec form.
  /// So they cannot interchange pointers to move the content. The input vectors are freed
  /// afterwards.
  CSRMatrix(const size_t m, const size_t n, const size_t nnz, std::vector<T> &&vals,
            std::vector<size_t> &&row_pointers, std::vector<size_t> &&col_indices)
    : m_rows(m), m_cols(n), m_nnz(nnz)
  {
    assert((m_rows >= 0 && m_cols >= 0) && "Matrix dimensions cannot be negative.");
    assert((!row_pointers.empty()) && "Input row pointers vector cannot be empty.");
    assert((row_pointers.size() == m_rows + 1) && "Rows pointers array does not match matrix row dimension.");
    assert((col_indices.size() == row_pointers.back()) && "Column indices array does not match nonzero count.");
    assert((vals.size() == col_indices.size()) && "Values array does not match nonzero count.");
    assert((m_nnz == row_pointers.back()) && "Nonzero count does not match rows pointers.");
    assert((vals.size() == m_nnz) && "Values array does not match input nnz.");

    construct(vals, row_pointers, col_indices);
    release(vals);
    release(row_pointers);
    release(col_indices);
  }

  /// Constructor with input vectors passed as a reference and without explicit matrix
  /// dimensions as the input.
  CSRMatrix(std::vector<T> &&vals, std::vector<size_t> &&row_pointers, std::vector<size_t> &&col_indices) {
    assert((!row_pointers.empty()) && "Row pointers vector cannot be empty.");
    assert((vals.size() == col_indices.size()) && "Values array size must match column indices array size.");
    assert((col_indices.size() == row_pointers.back()) && "Column indices array size must match row_pointers.back().");

    m_rows = row_pointers.size() - 1;
    if (col_indices.empty()) {
      m_cols = 0;
    } else {
      m_cols = *std::max_element(col_indices.begin(), col_indices.end()) + 1;
    }
    m_nnz = row_pointers.back();

    construct(vals, row_pointers, col_indices);
    release(vals);
    release(row_pointers);
    release(col_indices);
  }

  /// True when the matrix contains no non-zero entries.
  inline bool isEmpty() const {
    return m_nnz == 0;
  }

  /// Returns true when every row is strictly sorted in column indices order.
  bool hasSortedRows() const {
    for (size_t i = 0; i < m_rows; ++i) {
      for (size_t j = m_row_pointers[i] + 1; j < m_row_pointers[i + 1]; ++j) {
        if (m_col_indices[j - 1] >= m_col_indices[j])
          return false;
      }
    }
    return true;
  }

  /// Forward iterator over a single a row of the matrix. It iterates over the
  /// col_indices array and the vals array simultaneously.
  class row_iterator {
  public:
    row_iterator(const csr::index_t *start, const csr::index_t *end, const T *val)
      : m_start(start), m_end(end), m_val(val) {}

    row_iterator &operator++() {
      ++m_start;
      ++m_val;
      return *this;
    }

    operator bool() const { return m_start < m_end; }
    size_t col() const { return *m_start; }
    T value() const { return *m_val; }

  private:
    const csr::index_t *m_start;
    const csr::index_t *m_end;
    const T *m_val;
  };

  /// Return a row_iterator over a single row.
  row_iterator rowBegin(size_t row) const {
    assert((row < m_rows) && "Row index out of bounds.");
    const size_t start = m_row_pointers[row];
    const size_t end = m_row_pointers[row + 1];
    return row_iterator(m_col_indices.data() + start, m_col_indices.data() + end, m_vals.data() + start);
  }

  /// Return a value from a given row, col. Return 0 if there is no non zero entry.
  T get(size_t row, size_t col) const {
    assert((row < m_rows) && "Row index out of bounds.");
    assert((col < m_cols) && "Column index out of bounds.");

    int current_col{-1};
    for (size_t pos = m_row_pointers[row]; pos < m_row_pointers[row + 1]; ++pos) {
      current_col = static_cast<int>(m_col_indices[pos]);

      // Return the corresponding value if the column index matches. Otherwise if the
      // column index crosses the input col then exit the search, since the rows are
      // column index sorted.
      if (current_col == static_cast<int>(col)) {
        return m_vals[pos];
      } else if (current_col > static_cast<int>(col)) {
        break;
      }
    }

    return T(0);
  }

  /// Sets a value to a given row-coulmn index. If there exists a non-zero entry in that
  /// index the value is updated. Otherwise, a new non zero entry is added to the 
  /// csr matrix and the arrays are updated accordingly.
  CSRMatrix<T> &set(T val, size_t row, size_t col) {
    assert((row < m_rows) && "Row index out of bounds.");
    assert((col < m_cols) && "Column index out of bounds.");

    size_t pos = m_row_pointers[row];
    int current_col{-1};

    // Return the corresponding value if the column index matches. Otherwise if the
    // column index crosses the input col then exit the search, since the rows are
    // column index sorted.
    for (; pos < m_row_pointers[row + 1]; ++pos) {
      current_col = static_cast<int>(m_col_indices[pos]);
      if (current_col >= static_cast<int>(col)) {
        break;
      }
    }

    // The current column index is less than or equal to col. 
    if (current_col != static_cast<int>(col)) {
      // If it is less than a new non-zero entry will be added if the input val is non-zero.
      if (!(val == T(0))) {
        insert(pos, row, col, val);
      }
    } else if (val == T(0)) {
      // Otherwise, the indices match and if the input value is 0, an existing non
      // zero will be removed.
      remove(pos, row);
    } else {
      // Otherwise, update the existing non-zero entry.
      m_vals[pos] = val;
    }

    return *this;
  }

  /// Converts the per row non-zero entries to cumulative sum and creates the
  /// row_pointers array in place. Returns the total non-zeros in the matrix (nnz) from
  /// the last value in the row_pointers array.
  size_t scanRowSize() {
    if (m_row_pointers.empty())
      return 0;
    std::partial_sum(m_row_pointers.begin(), m_row_pointers.end(), m_row_pointers.begin());
    return m_row_pointers[m_rows];
  }

  /// Sorts all rows in column indices order.
  void sortRows() {
    if (__glibc_unlikely(isEmpty()))
      return;

    // Sort row chunks in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t row_start = m_row_pointers[i];
        size_t row_end = m_row_pointers[i + 1];

        detail::sortRow(m_col_indices.data() + row_start, m_vals.data() + row_start,
                        static_cast<int>(row_end - row_start));
      }
    });
  }

  /// Scales all non-zero values by a given factor.
  void scale(T factor) {
    if (__glibc_unlikely(isEmpty())) {
      return;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_nnz, vector_grain),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        m_vals[i] *= factor;
      }
    });
  }

  /// Memory footprint of the three arrays in bytes.
  size_t bytes() const {
    return sizeof(size_t) * (m_rows + 1) + sizeof(size_t) * m_nnz + sizeof(T) * m_nnz;
  }

  /// Every member should be the same to be equal
  friend bool operator==(const CSRMatrix &lhs, const CSRMatrix &rhs) = default;

  /// Cout operator overloading for the CSR matrix. Explicit zeros are not printed.
  friend std::ostream &operator<<(std::ostream &os, const CSRMatrix &matrix) {
    if (__glibc_unlikely(matrix.isEmpty())) {
      os << "Matrix is empty.";
    } else {
      os << "Matrix (" << matrix.m_rows << "x" << matrix.m_cols
         << "), NNZ: " << matrix.m_nnz << "\n";
      for (size_t i = 0; i < matrix.m_rows; ++i) {
        for (size_t j = 0; j < matrix.m_cols; ++j) {
          T val = matrix.get(i, j);
          if (val != T()) {
            os << "(" << i << ", " << j << "): ";
            os << val << "\n";
          }
        }
      }
    }

    return os;
  }

  /// Swap the members of two CSR matrices.
  friend void swap(CSRMatrix &lhs, CSRMatrix &rhs) {
    std::swap(lhs.m_rows, rhs.m_rows);
    std::swap(lhs.m_cols, rhs.m_cols);
    std::swap(lhs.m_nnz, rhs.m_nnz);
    lhs.m_row_pointers.swap(rhs.m_row_pointers);
    lhs.m_col_indices.swap(rhs.m_col_indices);
    lhs.m_vals.swap(rhs.m_vals);
  }

private:
  /// Swap a given vector passed as a reference with an empty vector.
  template <typename V>
  static void release(V &v) {
    V().swap(v);
  }

  /// Construct the csr::vec arrays from the std::vec arrays.
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

  /// Resets a matrix as a 0x0 matrix. The capacity is still maintained.
  void clear() {
    m_row_pointers.clear();
    m_col_indices.clear();
    m_vals.clear();
    m_rows = 0;
    m_cols = 0;
    m_nnz = 0;
    m_row_pointers.push_back(0);
  }

  /// Copy all the contents from a given CSR matrix.
  void deepCopy(const CSRMatrix<T> &other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_nnz = other.m_nnz;
    m_row_pointers = other.m_row_pointers;
    m_col_indices = other.m_col_indices;
    m_vals = other.m_vals;
  }

  /// Move the content from another CSR matrix to the current matrix and reset the other
  /// matrix to 0x0.
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

  /// Insert a new non zero entry in a given index. index indicates the exact location in
  /// the col_indices array to insert the new entry.
  void insert(size_t index, size_t row, size_t col, T val) {
    // Insert in the speficied index and shift every entry to the right afterwards.
    m_vals.insert(m_vals.begin() + index, val);
    m_col_indices.insert(m_col_indices.begin() + index, col);
    m_nnz += 1;

    // Shift the row pointers to the right from the given `row` to the last row.
    for (size_t i = row; i < m_rows; ++i) {
      m_row_pointers[i + 1] += 1;
    }
  }

  /// Remove an existing non zero entry a given index. index indicates the exact location
  /// in the col_indices array for the removal.
  void remove(size_t index, size_t row) {
    assert((!isEmpty()) && "Cannot remove from an empty matrix.");

    // Remove from the speficied index and shift every entry to the left afterwards.
    m_vals.erase(m_vals.begin() + index);
    m_col_indices.erase(m_col_indices.begin() + index);
    m_nnz -= 1;

    // Shift the row pointers to the left from the given `row` to the last row.
    for (size_t i = row; i < m_rows; ++i) {
      m_row_pointers[i + 1] -= 1;
    }
  }
};

/// Computes the transpose of a given CSRMatrix. Serial and O(nnz).
template <typename T>
CSRMatrix<T> transpose(const CSRMatrix<T>& A) {
  if (__glibc_unlikely(A.isEmpty() && A.m_rows == 0 && A.m_cols == 0))
    return CSRMatrix<T>();
  
  const size_t original_rows = A.m_rows;
  const size_t original_cols = A.m_cols;
  const size_t nnz = A.m_nnz;

  // Initialize and allocate memory for the transposed matrix
  CSRMatrix<T> transposedMatrix;
  transposedMatrix.m_rows = original_cols;
  transposedMatrix.m_cols = original_rows;
  transposedMatrix.m_nnz = nnz;
  transposedMatrix.m_row_pointers.resize(transposedMatrix.m_rows + 1, 0);
  if (__glibc_likely(nnz > 0)) {
    transposedMatrix.m_col_indices.resize(nnz, 0);
    transposedMatrix.m_vals.resize(nnz, T(0));
  } else {
    return transposedMatrix;
  }

  // Count number of non-zero elements in each column of A. These counts become the row
  // counts for the transposed matrix.
  for (size_t j = 0; j < nnz; ++j) {
    ++(transposedMatrix.m_row_pointers[A.m_col_indices[j] + 1]);
  }

  // Create the row pointers array for the transposed matrix.
  transposedMatrix.scanRowSize();

  // Scatter the column indices and values from the orginal matrix to the transposed matrix.
  // Head moves forward in each individual row for each new entry. By the time the operation
  // ends the each row's starting point indicates the starting index of the next row.
  for (size_t i = 0; i < original_rows; ++i) {
    const size_t start = A.m_row_pointers[i];
    const size_t end = A.m_row_pointers[i + 1];

    for (size_t j = start; j < end; ++j) {
      size_t head = transposedMatrix.m_row_pointers[A.m_col_indices[j]]++;
      transposedMatrix.m_col_indices[head] = i;
      transposedMatrix.m_vals[head] = A.m_vals[j];
    }
  }

  // Every row has shifted to right, perform a rotation operation to fix.
  std::rotate(transposedMatrix.m_row_pointers.begin(), transposedMatrix.m_row_pointers.begin() + original_cols, transposedMatrix.m_row_pointers.begin() + original_cols + 1);
  transposedMatrix.m_row_pointers[0] = 0;

  return transposedMatrix;
}

/// Return type of a row (a_ii) during a diagonal extraction from the matrix.
///   simple      a_ii                  (0 when a_ii is not stored)
///   inverse     1 / a_ii              (1 when a_ii is missing or zero)
///   forScaling  1 / sqrt(|a_ii|)      (1 when a_ii is missing or zero).
enum class diagonalType {
  simple,
  inverse,
  forScaling
};

/// Extracts the diagonal of a CSRMatrix based on a template parameter. The CSR matrix
/// is assumed to be sorted in column indices order. Returns a vector of diagonal entries
/// from each row.
template <diagonalType type, typename T>
std::vector<T> diagonal(const CSRMatrix<T>& A) {
  const size_t m = A.m_rows;
  
  if (__glibc_unlikely(m == 0)) {
    return {};
  }

  // Initialize the diagonal vector with the default value.
  std::vector<T> diag(m);
  if constexpr (type == diagonalType::simple) {
    std::fill(diag.begin(), diag.end(), T(0));
  } else {
    std::fill(diag.begin(), diag.end(), T(1));
  }

  // Scan the row chunks in parallel
  tbb::parallel_for(tbb::blocked_range<size_t>(0, m),[&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) { // Iterate of all the rows in a chunk
      for (auto a = A.rowBegin(i); a; ++a) { // Iterate over all the columns in a row
        if (a.col() == i) {
          // The diagonal entry exists, since row == col
          T val = a.value();
          
          // Fill the diagonal vector based on the diagonal type
          if constexpr (type == diagonalType::simple) {
            diag[i] = val;
          } else if constexpr (type == diagonalType::inverse) {
            if (val != T(0)) {
              diag[i] = T(1) / val;
            }
          } else if constexpr (type == diagonalType::forScaling) {
            if (val != T(0)) {
              diag[i] = T(1) / std::sqrt(std::abs(val));
            }
          }
          
          // Once the diagonal entry is found, exit the current row
          break;
        } else if (a.col() > i) {
          // If the column index exceeds the row index, then the diagonal entry does not exist.
          break;
        }
      }
    }
  });

  return diag;
}

} // namespace csr
