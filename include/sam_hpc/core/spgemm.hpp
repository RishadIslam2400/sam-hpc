#pragma once

#include <limits>

#include "sam_hpc/core/CSRMatrix.hpp"

/// Computes `C = A * B` with Saad's dense marker accumulator (Gustavson's algorithm).
/// Entries contributing to the same output position are summed. If `sort == true`, each
/// output row is sorted by column index; otherwise, columns retain their insertion order.
template <typename T>
void spgemm_saad(const csr::CSRMatrix<T> &A, const csr::CSRMatrix<T> &B,
                 csr::CSRMatrix<T> &C, bool sort = true)
{
  C.m_rows = A.m_rows;
  C.m_cols = B.m_cols;
  C.m_row_pointers.resize(C.m_rows + 1, 0);

  // Symbolic phase: count unique columns in each output row.
  tbb::enumerable_thread_specific<std::vector<int>> markerByColumn(B.m_cols, -1);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows), [&](const tbb::blocked_range<size_t> &rowRange) {
    std::vector<int>& rowMarkers = markerByColumn.local();
    for (size_t outputRow = rowRange.begin(); outputRow < rowRange.end(); ++outputRow) {
      size_t outputColumnCount = 0;
      const size_t aRowBegin = A.m_row_pointers[outputRow];
      const size_t aRowEnd = A.m_row_pointers[outputRow + 1];

      // Each non zero entry in the row of A, finds a corresponding row in B
      for (size_t aEntry = aRowBegin; aEntry < aRowEnd; ++aEntry) {
        const size_t sharedColumn = A.m_col_indices[aEntry];
        const size_t bRowBegin = B.m_row_pointers[sharedColumn];
        const size_t bRowEnd = B.m_row_pointers[sharedColumn + 1];

        // For each non-zero column in B, add it to the count if not already marked
        for (size_t bEntry = bRowBegin; bEntry < bRowEnd; ++bEntry) {
          const size_t outputColumn = B.m_col_indices[bEntry];
          
          if (rowMarkers[outputColumn] != static_cast<int>(outputRow)) {
            rowMarkers[outputColumn] = static_cast<int>(outputRow);
            ++outputColumnCount;
          }
        }
      }
      C.m_row_pointers[outputRow + 1] = outputColumnCount;
    }
  });

  // Allocate the column and value array.
  C.m_nnz = C.scanRowSize();
  C.m_col_indices.resize(C.m_nnz);
  C.m_vals.resize(C.m_nnz);

  // Numeric phase: compute the value for the output matrix
  markerByColumn.clear();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows), [&](const tbb::blocked_range<size_t>& rowRange) {
    std::vector<int>& rowMarkers = markerByColumn.local();
    for (size_t outputRow = rowRange.begin(); outputRow < rowRange.end(); ++outputRow) {
      const size_t outputRowBegin = C.m_row_pointers[outputRow];
      size_t outputRowEnd = outputRowBegin;

      // Accumulate the products contributing to the output row.
      for (size_t aEntry = A.m_row_pointers[outputRow],
                    aRowEnd = A.m_row_pointers[outputRow + 1];
           aEntry < aRowEnd; ++aEntry)
      {
        const size_t sharedColumn = A.m_col_indices[aEntry];
        const T aValue = A.m_vals[aEntry];

        for (size_t bEntry = B.m_row_pointers[sharedColumn], bRowEnd = B.m_row_pointers[sharedColumn + 1];
             bEntry < bRowEnd; ++bEntry)
        {
          const size_t outputColumn = B.m_col_indices[bEntry];
          const T bValue = B.m_vals[bEntry];
          
          // Markers below the row start belong to earlier output rows.
          if (rowMarkers[outputColumn] < static_cast<int>(outputRowBegin)) {
            rowMarkers[outputColumn] = static_cast<int>(outputRowEnd);
            C.m_col_indices[outputRowEnd] = outputColumn;
            C.m_vals[outputRowEnd] = aValue * bValue;
            ++outputRowEnd;
          } else {
            C.m_vals[rowMarkers[outputColumn]] += aValue * bValue;
          }
        }
      }
      
      // Clear only the markers used by this row.
      for (size_t outputEntry = outputRowBegin; outputEntry < outputRowEnd; ++outputEntry) {
        rowMarkers[C.m_col_indices[outputEntry]] = -1;
      }

      if (sort) {
        csr::detail::sortRow(C.m_col_indices.data() + outputRowBegin,
                             C.m_vals.data() + outputRowBegin,
                             static_cast<int>(outputRowEnd - outputRowBegin));
      }
    }
  });
}

/// Computes `C = A * B` with a thread-local hash accumulator.
template <typename T>
void spgemm_hash(const csr::CSRMatrix<T> &A, const csr::CSRMatrix<T> &B, csr::CSRMatrix<T> &C, bool sort = true) {
  constexpr csr::index_t emptySlot = std::numeric_limits<csr::index_t>::max();

  C.m_rows = A.m_rows;
  C.m_cols = B.m_cols;
  C.m_row_pointers.resize(C.m_rows + 1, 0);

  // Upper bound estimate the hash table size for a single output row
  const auto tableBitsForRow = [&](size_t outputRow) {
    size_t productCount = 0;
    const size_t aRowEnd = A.m_row_pointers[outputRow + 1];

    // For a row i of A, every non zero entry A(i, k) causes row k of B to be traversed.
    for (size_t aEntry = A.m_row_pointers[outputRow]; aEntry < aRowEnd; ++aEntry) {
      const size_t sharedIndex = A.m_col_indices[aEntry];

      // Count the number of entres visited in row k of B
      productCount += B.m_row_pointers[sharedIndex + 1] - B.m_row_pointers[sharedIndex];
    }
    
    // Select table capacity as powers of 2
    uint32_t bits = 1;
    while ((size_t(1) << bits) < 2 * std::max<size_t>(productCount, 1))
      ++bits;
    return bits;
  };

  // Thread local hash table storage
  struct table {
    std::vector<csr::index_t> keys;  // Output column stored in each slot.
    std::vector<uint32_t> offsets;   // Output-row offset associated with each slot.
  };
  tbb::enumerable_thread_specific<table> threadTables;

  // Symbolic phase: count unique columns in each output row. 
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows), [&](const tbb::blocked_range<size_t> &rowRange) {
    table& threadTable = threadTables.local();
    for (size_t outputRow = rowRange.begin(); outputRow < rowRange.end(); ++outputRow) {
      const uint32_t tableBits = tableBitsForRow(outputRow);
      const uint32_t tableSize = uint32_t(1) << tableBits;
      const uint32_t tableMask = tableSize - 1;
      const uint32_t hashShift = 32 - tableBits;
      if (threadTable.keys.size() < tableSize)
        threadTable.keys.resize(tableSize);
      std::fill_n(threadTable.keys.begin(), tableSize, emptySlot);

      size_t outputColumnCount = 0;
      const size_t aRowEnd = A.m_row_pointers[outputRow + 1];
      for (size_t aEntry = A.m_row_pointers[outputRow]; aEntry < aRowEnd; ++aEntry) {
        const size_t sharedIndex = A.m_col_indices[aEntry];
        const size_t bRowEnd = B.m_row_pointers[sharedIndex + 1];
        
        for (size_t bEntry = B.m_row_pointers[sharedIndex]; bEntry < bRowEnd; ++bEntry) {
          const csr::index_t outputColumn = B.m_col_indices[bEntry];
          uint32_t hashSlot =
            static_cast<uint32_t>((static_cast<uint64_t>(outputColumn) * 2654435769u) & 0xffffffffu)
            >> hashShift;
          
          while (threadTable.keys[hashSlot] != emptySlot && threadTable.keys[hashSlot] != outputColumn) {
            hashSlot = (hashSlot + 1) & tableMask;
          }
          if (threadTable.keys[hashSlot] == emptySlot) {
            threadTable.keys[hashSlot] = outputColumn;
            ++outputColumnCount;
          }
        }
      }
      C.m_row_pointers[outputRow + 1] = outputColumnCount;
    }
  });

  // Allocate the column and value array.
  C.m_nnz = C.scanRowSize();
  C.m_col_indices.resize(C.m_nnz);
  C.m_vals.resize(C.m_nnz);

  // Numeric phase: compute the value for the output matrix
  tbb::parallel_for(tbb::blocked_range<size_t>(0, C.m_rows), [&](const tbb::blocked_range<size_t>& rowRange) {
    table& threadTable = threadTables.local();
    for (size_t outputRow = rowRange.begin(); outputRow < rowRange.end(); ++outputRow) {
      // Allocate hash table
      const uint32_t tableBits = tableBitsForRow(outputRow);
      const uint32_t tableSize = uint32_t(1) << tableBits;
      const uint32_t tableMask = tableSize - 1;
      const uint32_t hashShift = 32 - tableBits;

      if (threadTable.keys.size() < tableSize)
        threadTable.keys.resize(tableSize);
      if (threadTable.offsets.size() < tableSize)
        threadTable.offsets.resize(tableSize);
      std::fill_n(threadTable.keys.begin(), tableSize, emptySlot);

      const size_t outputRowBegin = C.m_row_pointers[outputRow];
      csr::index_t* outputColumns = C.m_col_indices.data() + outputRowBegin;
      T* outputValues = C.m_vals.data() + outputRowBegin;
      uint32_t outputWidth = 0;

      const size_t aRowEnd = A.m_row_pointers[outputRow + 1];
      for (size_t aEntry = A.m_row_pointers[outputRow]; aEntry < aRowEnd; ++aEntry) {
        const size_t sharedIndex = A.m_col_indices[aEntry];
        const T aValue = A.m_vals[aEntry];
        const size_t bRowEnd = B.m_row_pointers[sharedIndex + 1];
        for (size_t bEntry = B.m_row_pointers[sharedIndex]; bEntry < bRowEnd; ++bEntry) {
          const csr::index_t outputColumn = B.m_col_indices[bEntry];
          uint32_t hashSlot =
            static_cast<uint32_t>((static_cast<uint64_t>(outputColumn) * 2654435769u) & 0xffffffffu)
            >> hashShift;
          
          while (threadTable.keys[hashSlot] != emptySlot && threadTable.keys[hashSlot] != outputColumn) {
            hashSlot = (hashSlot + 1) & tableMask;
          }
          
          if (threadTable.keys[hashSlot] == emptySlot) {
            threadTable.keys[hashSlot] = outputColumn;
            threadTable.offsets[hashSlot] = outputWidth;
            outputColumns[outputWidth] = outputColumn;
            outputValues[outputWidth] = aValue * B.m_vals[bEntry];
            ++outputWidth;
          } else {
            outputValues[threadTable.offsets[hashSlot]] += aValue * B.m_vals[bEntry];
          }
        }
      }

      if (sort) {
        csr::detail::sortRow(outputColumns, outputValues, static_cast<int>(outputWidth));
      }
    }
  });
}