#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/util/parallel.hpp"

#include <barrier>

/// Gauss-Seidel relaxation with serial and level-scheduled parallel sweeps.
/// Forward and backward sweeps update x in place. The parallel schedule is equivalent to
/// the serial sweep for structurally symmetric sparsity patterns; nonsymmetric patterns
/// can introduce races or different dependency ordering.
///
/// The parallel schedule is built for the global `num_threads` value at construction.
/// That value must remain unchanged for the smoother's lifetime. Rows without a stored
/// diagonal use one as the diagonal value.
template <typename T>
struct gauss_seidel {
  /// Relaxation parameters.
  struct params {
    /// Force serial sweeps.
    bool serial;

    params() : serial(false) { }
    params(bool serial) : serial(serial) { }
  } prm;

  /// Resolved execution mode, fixed at construction.
  bool is_serial;

  /// Constructs the smoother and, for parallel execution, builds both sweep schedules.
  /// @param A System matrix used to construct the schedules.
  /// @param prm Relaxation parameters.
  gauss_seidel(const csr::CSRMatrix<T>& A, const params& prm)
    : is_serial(prm.serial || num_threads < 4)
  {
    if (!is_serial) {
      forward = std::make_unique<parallel_sweep<true>>(A);
      backward = std::make_unique<parallel_sweep<false>>(A);
    }
  }

  /// Applies one forward sweep in place. `tmp` is unused.
  void apply_pre(const csr::CSRMatrix<T>& A, const std::vector<T>& rhs,
                 std::vector<T>& x, std::vector<T>& tmp) const
  {
    if (is_serial)
      serial_sweep(A, rhs, x, true);
    else
      forward->sweep(rhs, x);
  }

  /// Applies one backward sweep in place. `tmp` is unused.
  void apply_post(const csr::CSRMatrix<T>& A, const std::vector<T>& rhs,
                  std::vector<T>& x, std::vector<T>& tmp) const
  {
    if (is_serial)
      serial_sweep(A, rhs, x, false);
    else
      backward->sweep(rhs, x);
  }

  /// Applies a forward sweep followed by a backward sweep. `tmp` is unused.
  void apply(const csr::CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x,
             std::vector<T>& tmp) const
  {
    if (is_serial) {
      serial_sweep(A, rhs, x, true);
      serial_sweep(A, rhs, x, false);
    }
    else {
      forward->sweep(rhs, x);
      backward->sweep(rhs, x);
    }
  }

private:
  /// Performs one in-place serial sweep. Rows need not be sorted.
  /// @param forward Selects increasing or decreasing row order.
  static void serial_sweep(const csr::CSRMatrix<T>& A, const std::vector<T>& rhs,
                           std::vector<T>& x, bool forward)
  {
    const size_t n = A.m_rows;
    
    const ptrdiff_t beg = forward ? 0 : n - 1;
    const ptrdiff_t end = forward ? n : -1;
    const ptrdiff_t inc = forward ? 1 : -1;

    for (ptrdiff_t i = beg; i != end; i += inc) {
      T D = T(1);
      T X = rhs[i];

      for (auto a = A.rowBegin(i); a; ++a) {
        size_t c = a.col();
        T v = a.value();

        if (c == i)
          D = v;
        else
          X -= v * x[c];
      }

      x[i] = (T(1) / D) * X;
    }
  }

  /// Level schedule and thread-local matrix storage for one sweep direction.
  /// @tparam forward  true: dependencies are the strictly lower entries (c < i) and
  ///                  rows are leveled in increasing order; false: strictly upper
  ///                  entries, decreasing order.
  template <bool forward>
  struct parallel_sweep {
    /// Range of local rows assigned to one thread for one level.
    struct task {
      ptrdiff_t beg, end;
      task(ptrdiff_t beg, ptrdiff_t end) : beg(beg), end(end) { }
    };

    std::vector<std::vector<task>> tasks;
    std::vector<std::vector<size_t>> row_ptrs;
    std::vector<std::vector<size_t>> col_indices;
    std::vector<std::vector<T>> vals;
    std::vector<std::vector<size_t>> ord;

    /// Builds the dependency levels, work schedule, and thread-local matrix copies.
    /// @param A System matrix to schedule.
    parallel_sweep(const csr::CSRMatrix<T> &A)
      : tasks(num_threads), row_ptrs(num_threads), col_indices(num_threads), vals(num_threads), ord(num_threads)
    {
      const size_t n = A.m_rows;
      size_t nlev = 0;

      // Assign each row a level after all dependencies required by this sweep.
      std::vector<size_t> level(n, 0);

      ptrdiff_t beg = forward ? 0 : n - 1;
      ptrdiff_t end = forward ? n : -1;
      ptrdiff_t inc = forward ? 1 : -1;

      for (ptrdiff_t i = beg; i != end; i += inc) {
        size_t l = level[i];

        for (auto a = A.rowBegin(i); a; ++a) {
          size_t c = a.col();

          if (forward) {
            if (c >= i)
              continue;
          } else {
            if (c <= i)
              continue;
          }

          l = std::max(l, level[c] + 1);
        }

        level[i] = l;
        nlev = std::max(nlev, l + 1);
      }

      // Group rows by level.
      std::vector<size_t> order(n, 0);
      std::vector<size_t> start(nlev + 1, 0);

      for (size_t i = 0; i < n; ++i) {
        ++start[level[i] + 1];
      }

      std::partial_sum(start.begin(), start.end(), start.begin());

      for (size_t i = 0; i < n; ++i) {
        order[start[level[i]]++] = i; 
      }

      // Restore level starts after using them as insertion cursors.
      std::rotate(start.begin(), start.end() - 1, start.end());
      start[0] = 0;

      // Split each level into one task per thread and count local storage.
      std::vector<size_t> thread_rows(num_threads, 0);
      std::vector<size_t> thread_cols(num_threads, 0);

      // Every thread receives one task per level so all threads reach each barrier.
      auto organizeTasks = [&](int thread_id) {
        tasks[thread_id].reserve(nlev);

        for (size_t lev = 0; lev < nlev; ++lev) {
          size_t lev_size = start[lev + 1] - start[lev];
          size_t chunk_size = (lev_size + num_threads - 1) / num_threads;

          ptrdiff_t beg = std::min(static_cast<size_t>(thread_id) * chunk_size, lev_size);
          ptrdiff_t end = std::min(beg + chunk_size, lev_size);

          beg += start[lev];
          end += start[lev];

          tasks[thread_id].emplace_back(beg, end);

          thread_rows[thread_id] += end - beg;
          for (ptrdiff_t i = beg; i < end; ++i) {
            size_t j = order[i];
            thread_cols[thread_id] += (A.m_row_pointers[j + 1] - A.m_row_pointers[j]);
          }
        }
      };

      launchThreadsWithID(organizeTasks);

      // Copy assigned rows into thread-local CSR storage.
      auto reorderMatrix = [&](int thread_id) {
        col_indices[thread_id].reserve(thread_cols[thread_id]);
        vals[thread_id].reserve(thread_cols[thread_id]);
        ord[thread_id].reserve(thread_rows[thread_id]);
        row_ptrs[thread_id].reserve(thread_rows[thread_id] + 1);
        row_ptrs[thread_id].push_back(0);

        for (task &t : tasks[thread_id]) {
          ptrdiff_t loc_beg = row_ptrs[thread_id].size() - 1;
          ptrdiff_t loc_end = loc_beg;

          for (ptrdiff_t r = t.beg; r < t.end; ++r, ++loc_end) {
            size_t i = order[r];
            ord[thread_id].push_back(i);

            for (auto a = A.rowBegin(i); a; ++a) {
              col_indices[thread_id].push_back(a.col());
              vals[thread_id].push_back(a.value());
            }

            row_ptrs[thread_id].push_back(col_indices[thread_id].size());
          }

          t.beg = loc_beg;
          t.end = loc_end;
        }
      };

      launchThreadsWithID(reorderMatrix);
    }

    /// Executes one scheduled sweep. A barrier separates consecutive dependency levels.
    void sweep(const std::vector<T>& rhs, std::vector<T>& x) {
      std::barrier sync_point(num_threads);
      auto compute = [&](int thread_id) {
        for (const task& t : tasks[thread_id]) {
          for (ptrdiff_t r = t.beg; r < t.end; ++r) {
            size_t i = ord[thread_id][r];
            size_t beg = row_ptrs[thread_id][r];
            size_t end = row_ptrs[thread_id][r + 1];

            T D = T(1);
            T X = rhs[i];
            for (size_t j = beg; j < end; ++j) {
              size_t c = col_indices[thread_id][j];
              T v = vals[thread_id][j];

              if (c == i)
                D = v;
              else
                X -= v * x[c];
            }

            x[i] = (T(1) / D) * X;
          }

          sync_point.arrive_and_wait();
        }
      };

      launchThreadsWithID(compute);
    }
  };

  std::unique_ptr<parallel_sweep<true>> forward;
  std::unique_ptr<parallel_sweep<false>> backward;
};