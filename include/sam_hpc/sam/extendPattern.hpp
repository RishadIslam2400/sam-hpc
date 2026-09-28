#pragma once

#include <atomic>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/util/parallel.hpp"

namespace sam {

/// Symbolic product out = A * B of two sparsity patterns, marker based.
inline void saad_pattern_extension(const CSRMatrix<int>& A, const CSRMatrix<int>& B,
                                   CSRMatrix<int>& out)
{
  out.m_rows = A.m_rows;
  out.m_cols = B.m_cols;
  out.m_row_pointers.assign(A.m_rows + 1, 0);

  // Count the number of nonzero elements in each row of out
  tbb::enumerable_thread_specific<std::vector<int>> local_markers_1(out.m_cols, -1);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, A.m_rows),
  [&](const tbb::blocked_range<size_t>& r) {
    std::vector<int>& marker = local_markers_1.local();
    for (size_t ia = r.begin(); ia < r.end(); ++ia) {
      int count = 0;
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
      out.m_row_pointers[ia + 1] = count;
    }
  });

  out.m_nnz = out.scanRowSize();
  out.m_col_indices.resize(out.m_nnz);
  out.m_vals.resize(out.m_nnz, 1);

  // Compute the column indices
  local_markers_1.clear();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, out.m_rows),
  [&](const tbb::blocked_range<size_t>& r) {
    std::vector<int>& marker = local_markers_1.local();
    for (size_t ia = r.begin(); ia < r.end(); ++ia) {
      const size_t rowBeg = out.m_row_pointers[ia];
      size_t rowEnd = rowBeg;

      for (size_t ja = A.m_row_pointers[ia], ea = A.m_row_pointers[ia + 1]; ja < ea; ++ja) {
        size_t colIdxA = A.m_col_indices[ja];

        for (size_t jb = B.m_row_pointers[colIdxA], eb = B.m_row_pointers[colIdxA + 1]; jb < eb; ++jb) {
          size_t colIdxB = B.m_col_indices[jb];
          if (marker[colIdxB] < static_cast<int>(rowBeg)) {
            marker[colIdxB] = static_cast<int>(rowEnd);
            out.m_col_indices[rowEnd] = colIdxB;
            ++rowEnd;
          }
        }
      }

      for (size_t k = rowBeg; k < rowEnd; ++k) {
        marker[out.m_col_indices[k]] = -1;
      }

      std::sort(out.m_col_indices.begin() + rowBeg, out.m_col_indices.begin() + rowEnd);
    }
  });
}

template <bool needOut>
csr::index_t* mergeRows(const csr::index_t* row1, const csr::index_t* row1_end,
                        const csr::index_t* row2, const csr::index_t* row2_end,
                        csr::index_t* result)
{
  while (row1 != row1_end && row2 != row2_end) {
    const csr::index_t r1 = *row1;
    const csr::index_t r2 = *row2;

    if (r1 < r2) {
      if constexpr (needOut)
        *result = r1;
      ++row1;
    } else if (r1 == r2) {
      if constexpr (needOut)
        *result = r1;
      ++row1;
      ++row2;
    } else {
      if constexpr (needOut)
        *result = r2;
      ++row2;
    }
    ++result;
  }

  if constexpr (needOut) {
    if (row1 < row1_end) {
      return std::copy(row1, row1_end, result);
    } else if (row2 < row2_end) {
      return std::copy(row2, row2_end, result);
    } else {
      return result;
    }
  } else {
    return result + (row1_end - row1) + (row2_end - row2);
  }
}

/// Size of the union of two column-sorted rows.
inline size_t mergeRowsCount(const csr::index_t* row1, const csr::index_t* row1_end,
                             const csr::index_t* row2, const csr::index_t* row2_end)
{
  size_t count = 0;

  while (row1 != row1_end && row2 != row2_end) {
    const csr::index_t r1 = *row1;
    const csr::index_t r2 = *row2;

    if (r1 < r2) {
      ++row1;
    } else if (r1 == r2) {
      ++row1;
      ++row2;
    } else {
      ++row2;
    }
    ++count;
  }

  return count + (row1_end - row1) + (row2_end - row2);
}

inline size_t prodRowWidth(const csr::index_t* arow, const csr::index_t* arow_end,
                           const csr::offset_t* bptr, const csr::index_t* bcol,
                           csr::index_t* tmp_row1, csr::index_t* tmp_row2,
                           csr::index_t* tmp_row3)
{
  const size_t nrows = arow_end - arow;

  // No rows merge, nothing to do
  if (nrows == 0)
    return 0;

  // Single row, just copy it to output
  if (nrows == 1)
    return bptr[*arow + 1] - bptr[*arow];

  // Two rows, merge them
  if (nrows == 2) {
    const csr::index_t row1 = arow[0];
    const csr::index_t row2 = arow[1];
    return mergeRows<false>(bcol + bptr[row1], bcol + bptr[row1 + 1],
                            bcol + bptr[row2], bcol + bptr[row2 + 1],
                            tmp_row1) - tmp_row1;
  }

  /**
   * Generic case (more than two rows).
   *
   * Merge rows by pairs, then merge the results together. When merging two
   * rows, the result is always wider (or equal). Merging by pairs allows to
   * work with short rows as often as possible.
   */
  // merge first two rows
  csr::index_t r1 = *arow++;
  csr::index_t r2 = *arow++;
  size_t ncols1 = mergeRows<true>(bcol + bptr[r1], bcol + bptr[r1 + 1],
                                  bcol + bptr[r2], bcol + bptr[r2 + 1],
                                  tmp_row1) - tmp_row1;

  // Go by pairs
  while (arow + 1 < arow_end) {
    r1 = *arow++;
    r2 = *arow++;
    size_t ncols2 = mergeRows<true>(bcol + bptr[r1], bcol + bptr[r1 + 1],
                                    bcol + bptr[r2], bcol + bptr[r2 + 1],
                                    tmp_row2) - tmp_row2;
    if (arow == arow_end) {
      return mergeRows<false>(tmp_row1, tmp_row1 + ncols1, tmp_row2,
                              tmp_row2 + ncols2, tmp_row3) - tmp_row3;
    } else {
      ncols1 = mergeRows<true>(tmp_row1, tmp_row1 + ncols1, tmp_row2,
                               tmp_row2 + ncols2, tmp_row3) - tmp_row3;
      std::swap(tmp_row1, tmp_row3);
    }
  }

  // Merge the tail
  const csr::index_t tail = *arow++;
  return mergeRows<false>(tmp_row1, tmp_row1 + ncols1, bcol + bptr[tail],
                          bcol + bptr[tail + 1], tmp_row2) - tmp_row2;
}

inline void prodRow(const csr::index_t* arow, const csr::index_t* arow_end,
                    const csr::offset_t* bptr, const csr::index_t* bcol,
                    csr::index_t* out_row, csr::index_t* tmp_row2,
                    csr::index_t* tmp_row3)
{
  const size_t nrows = arow_end - arow;

  // No rows to merge, nothing to do
  if (nrows == 0)
    return;

  // Single row, just copy it to output
  if (nrows == 1) {
    const csr::index_t idx = *arow;
    const csr::index_t* browStart = bcol + bptr[idx];
    const csr::index_t* browEnd = bcol + bptr[idx + 1];

    while (browStart != browEnd) {
      *out_row++ = *browStart++;
    }

    return;
  }

  // Two rows, merge them
  if (nrows == 2) {
    const csr::index_t row_colind1 = arow[0];
    const csr::index_t row_colind2 = arow[1];
    mergeRows<true>(bcol + bptr[row_colind1], bcol + bptr[row_colind1 + 1],
                    bcol + bptr[row_colind2], bcol + bptr[row_colind2 + 1], out_row);

    return;
  }

  /**
   * Generic case (more than two rows).
   *
   * Merge rows by pairs, then merge the results together. When merging two
   * rows, the result is always wider (or equal). Merging by pairs allows to
   * work with short rows as often as possible.
   */
  // Merge first two rows
  csr::index_t r1 = *arow++;
  csr::index_t r2 = *arow++;
  csr::index_t* tmp_row1 = out_row;

  size_t c_numcol1 = mergeRows<true>(bcol + bptr[r1], bcol + bptr[r1 + 1],
                                     bcol + bptr[r2], bcol + bptr[r2 + 1],
                                     tmp_row1) - tmp_row1;

  // Go by pairs
  while (arow + 1 < arow_end) {
    r1 = *arow++;
    r2 = *arow++;
    size_t c_numcol2 = mergeRows<true>(bcol + bptr[r1], bcol + bptr[r1 + 1],
                        bcol + bptr[r2], bcol + bptr[r2 + 1],
                        tmp_row2) - tmp_row2;
    c_numcol1 = mergeRows<true>(tmp_row1, tmp_row1 + c_numcol1,
                                tmp_row2, tmp_row2 + c_numcol2,
                                tmp_row3) - tmp_row3;

    std::swap(tmp_row3, tmp_row1);
  }

  // Merge the tail
  if (arow < arow_end) {
    r2 = *arow++;
    c_numcol1 = mergeRows<true>(tmp_row1, tmp_row1 + c_numcol1,
                                bcol + bptr[r2], bcol + bptr[r2 + 1],
                                tmp_row3) - tmp_row3;

    std::swap(tmp_row3, tmp_row1);
  }

  if (tmp_row1 != out_row) {
    std::copy(tmp_row1, tmp_row1 + c_numcol1, out_row);
  }
}

/// Symbolic product out = A * B of two sparsity patterns, sorted-merge based.
/// Produces each row already sorted, so unlike saad_pattern_extension it needs
/// no sort.
inline void rmerge_pattern_extension(const CSRMatrix<int>& A, const CSRMatrix<int>& B,
                                     CSRMatrix<int>& out)
{
  out.m_rows = A.m_rows;
  out.m_cols = B.m_cols;
  out.m_row_pointers.assign(out.m_rows + 1, 0);

  size_t maxRowWidth = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, A.m_rows),
    size_t(0),
    [&](const tbb::blocked_range<size_t>& r, size_t localMax) -> size_t {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t rowWidth = 0;
        for (size_t j = A.m_row_pointers[i]; j < A.m_row_pointers[i + 1]; ++j) {
          size_t colIdx = A.m_col_indices[j];
          rowWidth += B.m_row_pointers[colIdx + 1] - B.m_row_pointers[colIdx];
        }
        localMax = std::max(localMax, rowWidth);
      }
      return localMax;
    },
    [](size_t a, size_t b) { return std::max(a, b); }
  );

  // Temporary row of C for each thread.
  struct internal_storage {
    std::vector<csr::index_t> tempCol;
    internal_storage(size_t maxWidth) { tempCol.resize(3 * maxWidth); }
  };

  tbb::enumerable_thread_specific<internal_storage> local_storages(maxRowWidth);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, out.m_rows),
  [&](const tbb::blocked_range<size_t>& r) {
    internal_storage& ls = local_storages.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      const size_t rowStart = A.m_row_pointers[i];
      const size_t rowEnd = A.m_row_pointers[i + 1];
      out.m_row_pointers[i + 1] = prodRowWidth(A.m_col_indices.data() + rowStart,
                                               A.m_col_indices.data() + rowEnd,
                                               B.m_row_pointers.data(), B.m_col_indices.data(),
                                               ls.tempCol.data(), ls.tempCol.data() + maxRowWidth,
                                               ls.tempCol.data() + 2 * maxRowWidth);
    }
  });

  out.m_nnz = out.scanRowSize();
  out.m_col_indices.resize(out.m_nnz);
  out.m_vals.resize(out.m_nnz, 1);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, out.m_rows),
  [&](const tbb::blocked_range<size_t>& r) {
    internal_storage& ls = local_storages.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      const size_t rowStart = A.m_row_pointers[i];
      const size_t rowEnd = A.m_row_pointers[i + 1];
      prodRow(A.m_col_indices.data() + rowStart, A.m_col_indices.data() + rowEnd,
             B.m_row_pointers.data(), B.m_col_indices.data(),
             out.m_col_indices.data() + out.m_row_pointers[i],
             ls.tempCol.data(), ls.tempCol.data() + maxRowWidth);
    }
  });
}

/// One symbolic product, dispatched to whichever kernel suits the thread count.
inline void pattern_product(const CSRMatrix<int>& A, const CSRMatrix<int>& B,
                            CSRMatrix<int>& out)
{
  if (num_threads < 16) {
    saad_pattern_extension(A, B, out);
  } else {
    rmerge_pattern_extension(A, B, out);
  }
}

/// Replaces S with S^level, the level-th pattern extension.
inline void extend_pattern(CSRMatrix<int>& S, const int level) {
  assert(level >= 1 && "Pattern extension level must be at least 1.");

  // Level 1 is the pattern itself, so there is nothing to extend.
  if (level <= 1)
    return;

  // S is only written by the final swap, so it can serve as the right operand
  // throughout and never needs to be copied.
  CSRMatrix<int> acc;
  pattern_product(S, S, acc);  // S^2

  for (int l = 3; l <= level; ++l) {
    CSRMatrix<int> next;
    pattern_product(acc, S, next);  // S^l = S^(l-1) * S
    swap(acc, next);
  }

  swap(S, acc);
}
}  // namespace sam