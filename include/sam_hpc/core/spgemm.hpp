#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"

#include <atomic>

//=============================== saad spgemm =======================================
/// @brief Computes the product of two CSRMatrices C = A * B using Saad's algorithm.
template <typename T>
void spgemm_saad(const CSRMatrix<T> &A, const CSRMatrix<T> &B, CSRMatrix<T> &C, bool sort = true) {
  C.m_rows = A.m_rows;
  C.m_cols = B.m_cols;
  C.m_row_pointers.resize(C.m_rows + 1, 0);

  // Phase 1: Symbolic phase to count nnz per row of C
  tbb::enumerable_thread_specific<std::vector<int>> local_markers_1(B.m_cols, -1);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows), [&](const tbb::blocked_range<size_t> &r) {
    std::vector<int>& marker = local_markers_1.local();
    for (size_t ia = r.begin(); ia < r.end(); ++ia) {
      size_t count = 0;
      const size_t rowBegA = A.m_row_pointers[ia];
      const size_t rowEndA = A.m_row_pointers[ia + 1];
      for (size_t ja = rowBegA; ja < rowEndA; ++ja) {
        const size_t colIdxA = A.m_col_indices[ja];

        const size_t rowBegB = B.m_row_pointers[colIdxA];
        const size_t rowEndB = B.m_row_pointers[colIdxA + 1];
        for (size_t jb = rowBegB; jb < rowEndB; ++jb) {
          const size_t colIdxB = B.m_col_indices[jb];
          
          if (marker[colIdxB] != static_cast<int>(ia)) {
              marker[colIdxB] = static_cast<int>(ia);
              ++count;
          }
        }
      }
      C.m_row_pointers[ia + 1] = count;
    }
  });

  // Initialized but unallocated
  C.m_nnz = C.scanRowSize();
  C.m_col_indices.resize(C.m_nnz);
  C.m_vals.resize(C.m_nnz);

  // Phase 2: Numeric phase to compute values of C
  local_markers_1.clear();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows), [&](const tbb::blocked_range<size_t>& r) {
    std::vector<int>& marker = local_markers_1.local();
    for (size_t ia = r.begin(); ia < r.end(); ++ia) {
      const size_t rowBeg = C.m_row_pointers[ia];
      size_t rowEnd = rowBeg;

      // This nested loop performs the numeric multiplication C(ia,:) = A(ia,:) * B
      for (size_t ja = A.m_row_pointers[ia], ea = A.m_row_pointers[ia + 1]; ja < ea; ++ja) {
        size_t colIdxA = A.m_col_indices[ja];
        T valA = A.m_vals[ja];

        for (size_t jb = B.m_row_pointers[colIdxA], eb = B.m_row_pointers[colIdxA + 1]; jb < eb; ++jb) {
          size_t colIdxB = B.m_col_indices[jb];
          T valB = B.m_vals[jb];
          
          if (marker[colIdxB] < static_cast<int>(rowBeg)) {
            marker[colIdxB] = static_cast<int>(rowEnd);
            C.m_col_indices[rowEnd] = colIdxB;
            C.m_vals[rowEnd] = valA * valB;
            ++rowEnd;
          } else {
            C.m_vals[marker[colIdxB]] += valA * valB;
          }
        }
      }
      
      for (size_t k = rowBeg; k < rowEnd; ++k) {
        marker[C.m_col_indices[k]] = -1;
      }

      if (sort) {
        sortRow(C.m_col_indices.data() + rowBeg, C.m_vals.data() + rowBeg, static_cast<int>(rowEnd - rowBeg));
      }
    }
  });
}

//=============================== row merge spgemm =======================================
/// @brief Merges two sorted lists of column indices.
/// If needOut is true, writes the merged list to col3 and returns a pointer to the end
/// of the written sequence.
/// If needOut is false, col3 is not written to, but the function returns a pointer
/// indicating where col3 would have ended, allowing the caller to calculate the count
/// of unique elements.
template <bool needOut>
csr::index_t* mergeRows(const csr::index_t *col1, const csr::index_t *col1_end,
                        const csr::index_t *col2, const csr::index_t *col2_end,
                        csr::index_t *col3)
{
  while (col1 != col1_end && col2 != col2_end) {
    const csr::index_t c1 = *col1;
    const csr::index_t c2 = *col2;

    if (c1 < c2) {
      if (needOut)
        *col3 = c1;
      ++col1;
    } else if (c1 == c2) {
      if (needOut)
        *col3 = c1;
      ++col1;
      ++col2;
    } else {
      if (needOut)
        *col3 = c2;
      ++col2;
    }
    ++col3;
  }

  if (needOut) {
    if (col1 < col1_end) {
      return std::copy(col1, col1_end, col3);
    } else if (col2 < col2_end) {
      return std::copy(col2, col2_end, col3);
    } else {
      return col3;
    }
  } else {
    return col3 + (col1_end - col1) + (col2_end - col2);
  }
}

/// @brief Merges two scaled rows (col_indices, vals) into a third row.
/// Output: col3, val3. Returns a pointer to the end of the written col3 sequence.
template <typename T>
csr::index_t* mergeRows(const T &alpha1, const csr::index_t *col1,
                        const csr::index_t *col1_end, const T *val1,
                        const T &alpha2, const csr::index_t *col2,
                        const csr::index_t *col2_end, const T *val2,
                        csr::index_t *col3, T *val3)
{
  while (col1 != col1_end && col2 != col2_end) {
    const csr::index_t c1 = *col1;
    const csr::index_t c2 = *col2;

    if (c1 < c2) {
      ++col1;
      *col3 = c1;
      *val3 = alpha1 * (*val1++);
    } else if (c1 == c2) {
      ++col1;
      ++col2;

      *col3 = c1;
      *val3 = alpha1 * (*val1++) + alpha2 * (*val2++);
    } else {
      ++col2;
      *col3 = c2;
      *val3 = alpha2 * (*val2++);
    }

    ++col3;
    ++val3;
  }

  while (col1 < col1_end) {
    *col3++ = *col1++;
    *val3++ = alpha1 * (*val1++);
  }

  while (col2 < col2_end) {
    *col3++ = *col2++;
    *val3++ = alpha2 * (*val2++);
  }

  return col3;
}

/// @brief Helper for SpGEMM: determines the width (number of unique column indices) 
/// of a row in C that results from multiplying a row of A with matrix B. This is used in
/// the symbolic phase.
/// @param acol Pointer to column indices of the current row in A.
/// @param acol_end Pointer to the end of column indices for the current row in A.
/// @param bptr Row pointers of matrix B.
/// @param bcol Column indices of matrix B.
/// @param tmp_col1 Temporary buffer for merging.
/// @param tmp_col2 Temporary buffer for merging.
/// @param tmp_col3 Temporary buffer for merging.
/// @return Number of non-zeros (unique column indices) in the resulting row of C.
inline size_t prodRowWidth(const csr::index_t *acol, const csr::index_t *acol_end,
                           const csr::offset_t *bptr, const csr::index_t *bcol,
                           csr::index_t *tmp_col1, csr::index_t *tmp_col2,
                           csr::index_t *tmp_col3)
{
  const size_t nrows = acol_end - acol;

  // No rows merge, nothing to do
  if (nrows == 0)
    return 0;

  // Single row, just copy it to output
  if (nrows == 1)
    return bptr[*acol + 1] - bptr[*acol];

  // Two rows, merge them
  if (nrows == 2) {
    const csr::index_t row1 = acol[0];
    const csr::index_t row2 = acol[1];

    return mergeRows<false>(bcol + bptr[row1], bcol + bptr[row1 + 1], bcol + bptr[row2],
                            bcol + bptr[row2 + 1], tmp_col1) - tmp_col1;
  }

  /**
   * Generic case (more than two rows).
   *
   * Merge rows by pairs, then merge the results together. When merging two rows,
   * the result is always wider (or equal). Merging by pairs allows to work with short
   * rows as often as possible.
   */
  // merge first two rows
  csr::index_t r1 = *acol++;
  csr::index_t r2 = *acol++;
  size_t ncols1 = mergeRows<true>(bcol + bptr[r1], bcol + bptr[r1 + 1], bcol + bptr[r2],
                                  bcol + bptr[r2 + 1], tmp_col1) - tmp_col1;

  // Go by pairs
  while (acol + 1 < acol_end) {
    r1 = *acol++;
    r2 = *acol++;
    size_t ncols2 = mergeRows<true>(bcol + bptr[r1], bcol + bptr[r1 + 1], bcol + bptr[r2],
                                    bcol + bptr[r2 + 1], tmp_col2) - tmp_col2;

    if (acol == acol_end) {
      return mergeRows<false>(tmp_col1, tmp_col1 + ncols1, tmp_col2, tmp_col2 + ncols2,
                              tmp_col3) - tmp_col3;
    } else {
      ncols1 = mergeRows<true>(tmp_col1, tmp_col1 + ncols1, tmp_col2, tmp_col2 + ncols2,
                               tmp_col3) - tmp_col3;
      std::swap(tmp_col1, tmp_col3);
    }
  }

  // Merge the tail
  const csr::index_t tail = *acol;
  return mergeRows<false>(tmp_col1, tmp_col1 + ncols1, bcol + bptr[tail],
                          bcol + bptr[tail + 1], tmp_col2) - tmp_col2;
}

/// @brief Helper for SpGEMM: computes one row of C = A * B.
/// Merges scaled rows of B based on non-zeros in a row of A.
/// @param acol Pointer to column indices of the current row in A.
/// @param acol_end Pointer to the end of column indices for the current row in A.
/// @param aval Pointer to values of the current row in A.
/// @param bptr Row pointers of matrix B.
/// @param bcol Column indices of matrix B.
/// @param bval Values of matrix B.
/// @param out_col Output buffer for column indices of the resulting C row.
/// @param out_val Output buffer for values of the resulting C row.
/// @param tmp_col2 Temporary buffer for column indices.
/// @param tmp_val2 Temporary buffer for values.
/// @param tmp_col3 Temporary buffer for column indices.
/// @param tmp_val3 Temporary buffer for values.
template <typename T>
void prodRow(const csr::index_t *acol, const csr::index_t *acol_end, const T *aval,
             const csr::offset_t *bptr, const csr::index_t *bcol, const T *bval,
             csr::index_t *out_col, T *out_val, csr::index_t *tmp_col2, T *tmp_val2,
             csr::index_t *tmp_col3, T *tmp_val3)
{
  const size_t nrows = acol_end - acol;

  // No rows to merge, nothing to do
  if (nrows == 0)
    return;

  // Single row, just copy it to output
  if (nrows == 1) {
    const csr::index_t idx = *acol;
    const T val = *aval;

    const csr::index_t* browStart = bcol + bptr[idx];
    const csr::index_t* browEnd = bcol + bptr[idx + 1];
    const T* browStartVal = bval + bptr[idx];

    while (browStart != browEnd) {
      *out_col++ = *browStart++;
      *out_val++ = val * (*browStartVal++);
    }

    return;
  }

  // Two rows, merge them
  if (nrows == 2) {
    const csr::index_t row_colind1 = acol[0];
    const csr::index_t row_colind2 = acol[1];
    const T row_val1 = aval[0];
    const T row_val2 = aval[1];
    mergeRows(row_val1, bcol + bptr[row_colind1], bcol + bptr[row_colind1 + 1],
              bval + bptr[row_colind1], row_val2, bcol + bptr[row_colind2],
              bcol + bptr[row_colind2 + 1], bval + bptr[row_colind2], out_col, out_val);

    return;
  }

  /**
   * Generic case (more than two rows).
   *
   * Merge rows by pairs, then merge the results together. When merging two rows, the result is always wider (or equal).
   * Merging by pairs allows to work with short rows as often as possible.
   */
  // Merge first two rows
  size_t acol1 = *acol++;
  size_t acol2 = *acol++;
  T aval1 = *aval++;
  T aval2 = *aval++;
  csr::index_t *tmp_col1 = out_col;
  T *tmp_val1 = out_val;

  size_t c_numcol1 = mergeRows(aval1, bcol + bptr[acol1], bcol + bptr[acol1 + 1],
                               bval + bptr[acol1], aval2, bcol + bptr[acol2],
                               bcol + bptr[acol2 + 1], bval + bptr[acol2], tmp_col1,
                               tmp_val1) - tmp_col1;

  // Go by pairs
  while (acol + 1 < acol_end) {
    acol1 = *acol++;
    acol2 = *acol++;
    aval1 = *aval++;
    aval2 = *aval++;
  
    size_t c_numcol2 = mergeRows(aval1, bcol + bptr[acol1], bcol + bptr[acol1 + 1],
                                 bval + bptr[acol1], aval2, bcol + bptr[acol2],
                                 bcol + bptr[acol2 + 1], bval + bptr[acol2], 
                                 tmp_col2, tmp_val2) - tmp_col2;

    c_numcol1 = mergeRows(T(1), tmp_col1, tmp_col1 + c_numcol1, tmp_val1,
                          T(1), tmp_col2, tmp_col2 + c_numcol2, tmp_val2,
                          tmp_col3, tmp_val3) - tmp_col3;

    std::swap(tmp_col3, tmp_col1);
    std::swap(tmp_val3, tmp_val1);
  }

  // Merge the tail
  if (acol < acol_end) {
    acol2 = *acol++;
    aval2 = *aval++;

    c_numcol1 = mergeRows(T(1), tmp_col1, tmp_col1 + c_numcol1, tmp_val1, aval2,
                          bcol + bptr[acol2], bcol + bptr[acol2 + 1], bval + bptr[acol2],
                          tmp_col3, tmp_val3) - tmp_col3;

    std::swap(tmp_col3, tmp_col1);
    std::swap(tmp_val3, tmp_val1);
  }

  if (tmp_col1 != out_col) {
    std::copy(tmp_col1, tmp_col1 + c_numcol1, out_col);
    std::copy(tmp_val1, tmp_val1 + c_numcol1, out_val);
  }
}

/// @brief Computes the product of two CSRMatrices C = A * B using a row-merging strategy.
template <typename T>
void spgemm_rmerge(const CSRMatrix<T> &A, const CSRMatrix<T> &B, CSRMatrix<T> &C) {
  C.m_rows = A.m_rows;
  C.m_cols = B.m_cols;
  C.m_row_pointers.resize(C.m_rows + 1, 0);

  size_t maxRowWidth = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, A.m_rows), size_t(0),
    [&](const tbb::blocked_range<size_t>& r, size_t local_max) -> size_t {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t rowWidth = 0;

        for (size_t j = A.m_row_pointers[i]; j < A.m_row_pointers[i + 1]; ++j) {
          size_t colIdx = A.m_col_indices[j];
          rowWidth += B.m_row_pointers[colIdx + 1] - B.m_row_pointers[colIdx];
        }

        local_max = std::max(local_max, rowWidth);
      }
      return local_max;
    },
    [](size_t a, size_t b) -> size_t {
      return std::max(a, b);
    }
  );

  // Temporary row of C for each thread.
  struct spgemm_internal_storage {
    std::vector<csr::index_t> tempCol;
    std::vector<T> tempVal;

    spgemm_internal_storage(size_t maxWidth) {
        tempCol.resize(3 * maxWidth);
        tempVal.resize(2 * maxWidth);
    }
  };

  tbb::enumerable_thread_specific<spgemm_internal_storage> local_storages(maxRowWidth);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows),
  [&](const tbb::blocked_range<size_t>& r) {
    spgemm_internal_storage& ls = local_storages.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      const size_t rowStart = A.m_row_pointers[i];
      const size_t rowEnd = A.m_row_pointers[i + 1];
      C.m_row_pointers[i + 1] = prodRowWidth(A.m_col_indices.data() + rowStart,
                                             A.m_col_indices.data() + rowEnd,
                                             B.m_row_pointers.data(), B.m_col_indices.data(),
                                             ls.tempCol.data(), ls.tempCol.data() + maxRowWidth,
                                             ls.tempCol.data() + 2 * maxRowWidth);
    }
  });

  C.m_nnz = C.scanRowSize();
  C.m_col_indices.resize(C.m_nnz);
  C.m_vals.resize(C.m_nnz);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows),
    [&](const tbb::blocked_range<size_t>& r) {
      spgemm_internal_storage& ls = local_storages.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        const size_t rowStart = A.m_row_pointers[i];
        const size_t rowEnd = A.m_row_pointers[i + 1];
        prodRow(A.m_col_indices.data() + rowStart, A.m_col_indices.data() + rowEnd,
                A.m_vals.data() + rowStart, B.m_row_pointers.data(),
                B.m_col_indices.data(), B.m_vals.data(),
                C.m_col_indices.data() + C.m_row_pointers[i],
                C.m_vals.data() + C.m_row_pointers[i], ls.tempCol.data(),
                ls.tempVal.data(), ls.tempCol.data() + maxRowWidth,
                ls.tempVal.data() + maxRowWidth);
      }
    }
  );
}