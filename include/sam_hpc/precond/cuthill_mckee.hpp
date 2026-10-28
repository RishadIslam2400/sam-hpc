#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"

/// Options for cuthill_mckee.
struct cuthill_mckee_params {
  /// Reverse the final ordering to obtain reverse Cuthill-McKee (RCM).
  bool reverse_permutation = false;

  /// Traverse the pattern of A + A^T instead of A. Required for structurally nonsymmetric
  /// matrices; otherwise nodes reachable only through column entries are treated as
  /// separate components.
  bool symmetrize = false;

  /// Start each component at a pseudo-peripheral node (George-Liu) instead of its
  /// lowest-numbered unnumbered node. Most effective with `symmetrize` or a structurally
  /// symmetric matrix.
  bool pseudo_peripheral_start = false;
};

/// Cuthill-McKee bandwidth-reducing ordering computed from the sparsity pattern of A.
/// Nodes are numbered level by level in breadth-first order. Within a level, nodes are
/// expanded in order of increasing degree, and each newly discovered neighbor receives the
/// next number.
///
/// @tparam T        Matrix value type; only the sparsity pattern is used.
/// @tparam reverse  If true, expand each level in decreasing degree order. This is
///                  independent of cuthill_mckee_params::reverse_permutation.
template <typename T, bool reverse = false>
struct cuthill_mckee {
  /// Computes the ordering with default options.
  static void get(const csr::CSRMatrix<T>& A, std::vector<ptrdiff_t>& perm) {
    get(A, perm, cuthill_mckee_params());
  }

  /// Computes the ordering.
  /// @param A    Square matrix whose sparsity pattern defines the graph.
  /// @param perm Output permutation of size n, with `perm[new] = old`.
  /// @param prm  Ordering options.
  /// @throws std::invalid_argument If A is not square.
  /// @throws std::out_of_range     If a column index lies outside [0, n).
  static void get(const csr::CSRMatrix<T>& A, std::vector<ptrdiff_t>& perm,
                  const cuthill_mckee_params& prm)
  {
    if (A.m_rows != A.m_cols)
      throw std::invalid_argument("cuthill_mckee: matrix must be square");

    const ptrdiff_t n = static_cast<ptrdiff_t>(A.m_rows);
    perm.resize(n);
    if (n == 0)
      return;

    validate_columns(A, n);

    const adjacency g = prm.symmetrize ? adjacency::symmetrized(A, n) : adjacency::of(A);

    // Node degrees (self-loops included) and their maximum, which sizes the buckets.
    std::vector<ptrdiff_t> degree(n);
    const ptrdiff_t maxDegree = tbb::parallel_reduce(
      tbb::blocked_range<ptrdiff_t>(0, n), ptrdiff_t(0),
      [&](const tbb::blocked_range<ptrdiff_t>& r, ptrdiff_t local_max) -> ptrdiff_t {
        for (ptrdiff_t i = r.begin(); i < r.end(); ++i) {
          degree[i] = g.end(i) - g.begin(i);
          local_max = std::max(local_max, degree[i]);
        }
        return local_max;
      },
      [](ptrdiff_t a, ptrdiff_t b) -> ptrdiff_t { return std::max(a, b); }
    );

    // 0 marks an unnumbered node; otherwise the level in which the node was numbered.
    // Levels are numbered consecutively across components.
    std::vector<ptrdiff_t> levelSet(n, 0);

    // Successor links of the degree-bucket lists. A node is linked once, when discovered.
    std::vector<ptrdiff_t> nextSameDegree(n, -1);

    // Bucket heads, indexed by degree, for the current level and for the level being
    // built. Outside the traversal loop, firstWithDegree is populated only in
    // [0, maxDegreeInCurrentLevelSet] and nFirstWithDegree is all -1. Clearing only the
    // used range keeps the total clearing cost at O(n + nnz).
    std::vector<ptrdiff_t> firstWithDegree(maxDegree + 1, -1);
    std::vector<ptrdiff_t> nFirstWithDegree(maxDegree + 1, -1);

    // BFS scratch, needed only for the pseudo-peripheral search.
    bfs_workspace ws(prm.pseudo_peripheral_start ? n : 0);

    ptrdiff_t next = 0;  // nodes numbered so far
    ptrdiff_t scan = 0;  // all nodes below scan are numbered
    ptrdiff_t currentLevelSet = 0;
    ptrdiff_t maxDegreeInCurrentLevelSet = 0;

    // Numbers the root of a new component as the sole node of a new level. `scan` only
    // advances, so locating all roots costs O(n) in total.
    auto start_component = [&]() {
      while (scan < n && levelSet[scan] != 0)
        ++scan;
      assert(scan < n && "cuthill_mckee: no unnumbered node left for a new component");

      const ptrdiff_t root = prm.pseudo_peripheral_start
                             ? pseudo_peripheral_node(g, scan, degree, levelSet, ws)
                             : scan;

      ++currentLevelSet;
      levelSet[root] = currentLevelSet;
      perm[next++] = root;
      maxDegreeInCurrentLevelSet = degree[root];
      firstWithDegree[maxDegreeInCurrentLevelSet] = root;
    };

    start_component();

    // Breadth-first traversal; each iteration expands one level.
    while (next < n) {
      ptrdiff_t nMDICLS = 0;  // maximum degree in the next level
      bool empty = true;      // no unnumbered neighbor found in this level

      // Visit the buckets in increasing degree order (decreasing when `reverse`).
      const ptrdiff_t firstVal  = reverse ? maxDegreeInCurrentLevelSet : 0;
      const ptrdiff_t finalVal  = reverse ? -1 : maxDegreeInCurrentLevelSet + 1;
      const ptrdiff_t increment = reverse ? -1 : 1;

      for (ptrdiff_t soughtDegree = firstVal; soughtDegree != finalVal; soughtDegree += increment) {
        // Buckets are LIFO lists terminated by -1, so nodes of equal degree are
        // expanded in reverse discovery order.
        for (ptrdiff_t node = firstWithDegree[soughtDegree]; node >= 0; node = nextSameDegree[node]) {
          // Number each unnumbered neighbor and add it to the next level's bucket.
          for (ptrdiff_t k = g.begin(node), e = g.end(node); k < e; ++k) {
            const ptrdiff_t c = g.col_at(k);
            if (levelSet[c] == 0) {
              levelSet[c] = currentLevelSet + 1;
              perm[next++] = c;
              empty = false;
              nextSameDegree[c] = nFirstWithDegree[degree[c]];
              nFirstWithDegree[degree[c]] = c;
              nMDICLS = std::max(nMDICLS, degree[c]);
            }
          }
        }
      }

      std::fill(firstWithDegree.begin(), firstWithDegree.begin() + maxDegreeInCurrentLevelSet + 1, ptrdiff_t(-1));

      if (empty) {
        // Component exhausted. No node was discovered, so nFirstWithDegree is still all -1.
        start_component();
      } else {
        // The next level becomes current; the cleared array is reused for the level after.
        std::swap(firstWithDegree, nFirstWithDegree);
        ++currentLevelSet;
        maxDegreeInCurrentLevelSet = nMDICLS;
      }
    }

    if (prm.reverse_permutation)
      std::reverse(perm.begin(), perm.end());
  }

private:
  /// Read-only adjacency view of A or A + A^T.
  struct adjacency {
    const csr::CSRMatrix<T>* A = nullptr;  // borrowed pattern; null when owned
    std::vector<ptrdiff_t> ptr;            // owned row pointers
    std::vector<ptrdiff_t> col;            // owned column indices

    ptrdiff_t begin(ptrdiff_t i) const {
      return A ? static_cast<ptrdiff_t>(A->m_row_pointers[i]) : ptr[i];
    }
    ptrdiff_t end(ptrdiff_t i) const { return begin(i + 1); }
    ptrdiff_t col_at(ptrdiff_t k) const {
      return A ? static_cast<ptrdiff_t>(A->m_col_indices[k]) : col[k];
    }

    static adjacency of(const csr::CSRMatrix<T>& A) {
      adjacency g;
      g.A = &A;
      return g;
    }

    /// Builds the pattern of A + A^T with duplicates removed; diagonal entries appear once.
    static adjacency symmetrized(const csr::CSRMatrix<T>& A, ptrdiff_t n) {
      const adjacency a = of(A);

      // Row lengths of A + A^T, including duplicates.
      std::vector<ptrdiff_t> cnt(n + 1, 0);
      for (ptrdiff_t i = 0; i < n; ++i) {
        for (ptrdiff_t k = a.begin(i), e = a.end(i); k < e; ++k) {
          const ptrdiff_t c = a.col_at(k);
          ++cnt[i + 1];
          if (c != i)
            ++cnt[c + 1];
        }
      }
      std::partial_sum(cnt.begin(), cnt.end(), cnt.begin());

      // Scatter each entry (i, c) and its transpose (c, i); the diagonal is added once.
      std::vector<ptrdiff_t> pos(cnt.begin(), cnt.end() - 1);
      std::vector<ptrdiff_t> tmp(cnt[n]);
      for (ptrdiff_t i = 0; i < n; ++i) {
        for (ptrdiff_t k = a.begin(i), e = a.end(i); k < e; ++k) {
          const ptrdiff_t c = a.col_at(k);
          tmp[pos[i]++] = c;
          if (c != i)
            tmp[pos[c]++] = i;
        }
      }

      // Deduplicate each row, keeping the first occurrence of each column.
      adjacency g;
      g.ptr.assign(n + 1, 0);
      g.col.reserve(tmp.size());
      std::vector<ptrdiff_t> marker(n, -1);
      for (ptrdiff_t i = 0; i < n; ++i) {
        for (ptrdiff_t k = cnt[i]; k < cnt[i + 1]; ++k) {
          const ptrdiff_t c = tmp[k];
          if (marker[c] != i) {
            marker[c] = i;
            g.col.push_back(c);
          }
        }
        g.ptr[i + 1] = static_cast<ptrdiff_t>(g.col.size());
      }
      return g;
    }
  };

  /// Scratch space for the George-Liu breadth-first searches. Each search uses a new
  /// stamp value to mark visited nodes, so `stamp` is never reset.
  struct bfs_workspace {
    std::vector<ptrdiff_t> stamp;
    std::vector<ptrdiff_t> queue;
    ptrdiff_t counter = 0;  // stamp of the most recent search

    explicit bfs_workspace(ptrdiff_t n) : stamp(n, 0) {}
  };

  /// Throws std::out_of_range if any column index lies outside [0, n).
  static void validate_columns(const csr::CSRMatrix<T>& A, ptrdiff_t n) {
    const ptrdiff_t nnz = static_cast<ptrdiff_t>(A.m_row_pointers[n]);
    const bool bad = tbb::parallel_reduce(
      tbb::blocked_range<ptrdiff_t>(0, nnz), false,
      [&](const tbb::blocked_range<ptrdiff_t>& r, bool found) -> bool {
        // Skip the scan if an earlier subrange already found an invalid index.
        if (found)
          return true;
        for (ptrdiff_t k = r.begin(); k < r.end(); ++k) {
          const ptrdiff_t c = static_cast<ptrdiff_t>(A.m_col_indices[k]);
          if (c < 0 || c >= n)
            return true;
        }
        return false;
      },
      [](bool a, bool b) -> bool { return a || b; });

    if (bad)
      throw std::out_of_range("cuthill_mckee: column index out of range");
  }

  /// Breadth-first search from `root` over unnumbered nodes.
  /// On return, ws.queue holds the visit order and `last_begin` is the queue offset of
  /// the deepest level. Returns the number of levels (eccentricity of root + 1).
  static ptrdiff_t level_structure(const adjacency& g, ptrdiff_t root,
                                   const std::vector<ptrdiff_t>& levelSet,
                                   bfs_workspace& ws, ptrdiff_t& last_begin)
  {
    const ptrdiff_t s = ++ws.counter;
    ws.queue.clear();
    ws.queue.push_back(root);
    ws.stamp[root] = s;

    ptrdiff_t levels = 0;
    ptrdiff_t begin  = 0;
    // Each iteration expands one level, stored in ws.queue[begin, end).
    while (begin < static_cast<ptrdiff_t>(ws.queue.size())) {
      const ptrdiff_t end = static_cast<ptrdiff_t>(ws.queue.size());
      last_begin = begin;
      ++levels;
      for (ptrdiff_t q = begin; q < end; ++q) {
        const ptrdiff_t node = ws.queue[q];
        for (ptrdiff_t k = g.begin(node), e = g.end(node); k < e; ++k) {
          const ptrdiff_t c = g.col_at(k);
          if (levelSet[c] == 0 && ws.stamp[c] != s) {
            ws.stamp[c] = s;
            ws.queue.push_back(c);
          }
        }
      }
      begin = end;
    }
    return levels;
  }

  /// George-Liu pseudo-peripheral node search. Starting from `start`, repeatedly moves
  /// to a minimum-degree node of the deepest level while doing so strictly increases
  /// the depth of the level structure. At most `max_sweeps` moves are made.
  static ptrdiff_t pseudo_peripheral_node(const adjacency& g, ptrdiff_t start,
                                          const std::vector<ptrdiff_t>& degree,
                                          const std::vector<ptrdiff_t>& levelSet,
                                          bfs_workspace& ws)
  {
    constexpr int max_sweeps = 8;

    ptrdiff_t root       = start;
    ptrdiff_t last_begin = 0;
    ptrdiff_t depth      = level_structure(g, root, levelSet, ws, last_begin);

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
      // Minimum-degree node of the deepest level.
      ptrdiff_t candidate = ws.queue[last_begin];
      for (ptrdiff_t q = last_begin + 1; q < static_cast<ptrdiff_t>(ws.queue.size()); ++q) {
        if (degree[ws.queue[q]] < degree[candidate])
          candidate = ws.queue[q];
      }

      ptrdiff_t candidate_last = 0;
      const ptrdiff_t candidate_depth =
        level_structure(g, candidate, levelSet, ws, candidate_last);
      if (candidate_depth <= depth)
        break;

      // Accept the candidate. ws.queue now holds its level structure, which the next
      // sweep searches.
      root       = candidate;
      depth      = candidate_depth;
      last_begin = candidate_last;
    }
    return root;
  }
};