#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/util/parallel.hpp"

namespace sam {

namespace detail {

/// Set of column indices in [0, ncols) with one generation stamp per column. An index is
/// present when its stamp equals `cur`; advancing `cur` empties the set in O(1).
struct dense_index_set {
  std::vector<uint32_t> marker;
  uint32_t cur = 0;
  std::vector<csr::index_t> members;

  explicit dense_index_set(size_t ncols = 0) : marker(ncols, 0) {}

  /// Clears the set in constant time by advancing its generation.
  void reset(size_t /*bound*/ = 0) {
    members.clear();
    if (__glibc_unlikely(++cur == 0)) {
      std::fill(marker.begin(), marker.end(), 0u);
      cur = 1;
    }
  }

  /// Inserts a column index and reports whether it was not already present.
  bool insert(csr::index_t c) {
    if (marker[c] == cur)
      return false;
    marker[c] = cur;
    members.push_back(c);
    return true;
  }
};

/// Set of column indices backed by an open-addressing table with linear probing.
struct hash_index_set {
  static constexpr csr::index_t empty = std::numeric_limits<csr::index_t>::max();

  std::vector<csr::index_t> table;
  std::vector<uint32_t> used;
  std::vector<csr::index_t> members;
  uint32_t size = 0, mask = 0, shift = 32;

  explicit hash_index_set(size_t /*ncols*/ = 0) {}

  /// Clears the occupied slots and sizes the table for the expected number of members.
  void reset(size_t bound) {
    for (uint32_t s : used)
      table[s] = empty;
    used.clear();
    members.clear();
    resize_table(bound);
  }

  /// Inserts a column index and reports whether it was not already present.
  bool insert(csr::index_t c) {
    uint32_t h = slot_of(c);
    while (table[h] != empty) {
      if (table[h] == c)
        return false;
      h = (h + 1) & mask;
    }
    table[h] = c;
    used.push_back(h);
    members.push_back(c);
    if (2 * members.size() > size)
      grow();
    return true;
  }

 private:
  uint32_t slot_of(csr::index_t c) const {
    return static_cast<uint32_t>((static_cast<uint64_t>(c) * 2654435769u) & 0xffffffffu) >> shift;
  }

  void resize_table(size_t bound) {
    uint32_t bits = 1;
    while ((size_t(1) << bits) < 2 * std::max<size_t>(bound, 1))
      ++bits;
    size = uint32_t(1) << bits;
    mask = size - 1;
    shift = 32 - bits;
    if (table.size() < size)
      table.resize(size, empty);
  }

  /// Doubles the table and rehashes the existing members.
  void grow() {
    for (uint32_t s : used)
      table[s] = empty;
    used.clear();
    resize_table(members.size());
    for (const csr::index_t c : members) {
      uint32_t h = slot_of(c);
      while (table[h] != empty)
        h = (h + 1) & mask;
      table[h] = c;
      used.push_back(h);
    }
  }
};

} // namespace detail

/// Union of two sorted, duplicate-free index rows, as one step of a two-way merge.
/// With needOut = true the union is written to `result`; 
/// with needOut = false nothing is written and only the width is computed.
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

/// Returns the size of the union of two sorted, duplicate-free rows.
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

/// Whether every row of the square pattern S stores its diagonal entry.
inline bool has_full_diagonal(const csr::CSRMatrix<int>& S) {
  return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, S.m_rows), true,
    [&](const tbb::blocked_range<size_t>& r, bool ok) {
      for (size_t i = r.begin(); ok && i < r.end(); ++i) {
        const auto* b = S.m_col_indices.data() + S.m_row_pointers[i];
        const auto* e = S.m_col_indices.data() + S.m_row_pointers[i + 1];
        ok = std::find(b, e, static_cast<csr::index_t>(i)) != e;
      }
      return ok;
    },
    [](bool a, bool b) { return a && b; });
}

/// Row i of power of sparsity pattern matrix S^level.
///
/// @param frontier  has_full_diagonal(S).
/// @param walk      Temp array for the exact-length path; untouched on the frontier path.
template <class Accumulator>
void pattern_power_row(const csr::CSRMatrix<int>& S, size_t i, int level, bool frontier,
                       Accumulator& acc, std::vector<csr::index_t>& walk)
{
  const auto& row_pointers = S.m_row_pointers;
  const auto& col_indices = S.m_col_indices;
  const auto row_width = [&](const csr::index_t* beg, const csr::index_t* end) {
    size_t w = 0;
    for (; beg != end; ++beg)
      w += row_pointers[*beg + 1] - row_pointers[*beg];
    return w;
  };
  const csr::index_t* row_beg = col_indices.data() + row_pointers[i];
  const csr::index_t* row_end = col_indices.data() + row_pointers[i + 1];

  // Frontier expansion path
  if (frontier || level <= 1) {
    // For level <= 1, accumulator size estimate is number of entries in the current row
    // For higher powers, the estimates is the sum of the width of the rows referenced
    // by the current row's columns
    acc.reset(level <= 1 ? static_cast<size_t>(row_end - row_beg) : row_width(row_beg, row_end));
    for (const csr::index_t* col = row_beg; col != row_end; ++col)
      acc.insert(*col);

    size_t lo = 0;
    for (int l = 2; l <= level; ++l) {
      // hi captures the number of members discovered so far
      const size_t hi = acc.members.size();
      if (lo == hi)
        break;
      for (size_t m = lo; m < hi; ++m) {
        const csr::index_t k = acc.members[m];
        if (k == i)
          continue;
        // Add all the unique column indices reachable from k in one hop
        for (size_t p = row_pointers[k]; p < row_pointers[k + 1]; ++p)
          acc.insert(col_indices[p]);
      }
      lo = hi;
    }
    return;
  }

  // Not all rows have diagonal, complete walk path
  walk.assign(row_beg, row_end);
  for (int l = 2; l <= level; ++l) {
    acc.reset(row_width(walk.data(), walk.data() + walk.size()));
    for (const csr::index_t k : walk) {
      for (size_t p = row_pointers[k]; p < row_pointers[k + 1]; ++p)
        acc.insert(col_indices[p]);
    }
    if (l < level)
      walk.assign(acc.members.begin(), acc.members.end());
  }
}

/// Per-thread output for one-pass row construction.
struct row_buffer {
  struct piece { size_t row_begin, row_end, offset; };
  std::vector<csr::index_t> buf;
  std::vector<piece> pieces;
};

/// Assembles per-thread row buffers after `out`'s row pointers have been scanned.
template <class Locals, class Get>
void assemble_rows(Locals& locals, Get&& get_buffer, csr::CSRMatrix<int>& out) {
  out.m_col_indices.resize(out.m_nnz);
  out.m_vals.resize(out.m_nnz);

  std::vector<std::pair<const row_buffer*, row_buffer::piece>> all;
  for (auto& l : locals) {
    const row_buffer& rb = get_buffer(l);
    for (const auto& p : rb.pieces)
      all.push_back({&rb, p});
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(0, all.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t t = r.begin(); t < r.end(); ++t) {
      const auto& [rb, p] = all[t];
      const size_t dst = out.m_row_pointers[p.row_begin];
      const size_t len = out.m_row_pointers[p.row_end] - dst;
      std::copy_n(rb->buf.begin() + p.offset, len, out.m_col_indices.begin() + dst);
      std::fill_n(out.m_vals.begin() + dst, len, 1);
    }
  });
}

/// Computes `out = S^level` in one pass using the accumulator.
template <class Accumulator>
void powers_of_pattern(const csr::CSRMatrix<int>& S, const int level, csr::CSRMatrix<int>& out) {
  assert(S.m_rows == S.m_cols && "Pattern extension needs a square pattern.");
  const size_t n = S.m_rows;
  const bool frontier = has_full_diagonal(S);

  out.m_rows = n;
  out.m_cols = S.m_cols;
  out.m_row_pointers.assign(n + 1, 0);

  struct local_t {
    Accumulator acc;
    std::vector<csr::index_t> walk;
    row_buffer rows;
  };
  tbb::enumerable_thread_specific<local_t> locals([&] { return local_t{Accumulator(S.m_cols), {}, {}}; });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
    local_t& L = locals.local();
    L.rows.pieces.push_back({r.begin(), r.end(), L.rows.buf.size()});
    for (size_t i = r.begin(); i < r.end(); ++i) {
      pattern_power_row(S, i, level, frontier, L.acc, L.walk);
      auto& m = L.acc.members;
      std::sort(m.begin(), m.end());
      L.rows.buf.insert(L.rows.buf.end(), m.begin(), m.end());
      out.m_row_pointers[i + 1] = m.size();
    }
  });

  out.m_nnz = out.scanRowSize();
  assemble_rows(locals, [](const local_t& l) -> const row_buffer& { return l.rows; }, out);
}

/// Replaces `S` with its level-th pattern power.
///
/// @param S      Square pattern, rows sorted. Replaced in place; its old storage is freed.
/// @param level  >= 1. Level 1 leaves S unchanged.
inline void extend_pattern(csr::CSRMatrix<int>& S, const int level) {
  assert(level >= 1 && "Pattern extension level must be at least 1.");
  if (level <= 1)
    return;

  csr::CSRMatrix<int> out;

  if (spgemm_acc == spgemm_accumulator::hash) {
    powers_of_pattern<detail::hash_index_set>(S, level, out);
  } else {
    powers_of_pattern<detail::dense_index_set>(S, level, out);
  }
  
  swap(S, out);
}

}  // namespace sam
