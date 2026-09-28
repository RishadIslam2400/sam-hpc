#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/util/parallel.hpp"

#include <barrier>

/// Gauss-Seidel relaxation.
template <typename T>
struct gauss_seidel {
    /// Relaxation parameters
    struct params {
        /// Use serial version of the algorithm
        bool serial;

        params() : serial(false) { }
        params(bool serial) : serial(serial) { }
    } prm;

    bool is_serial;

    /// Construct smoother for the system matrix
    /**
     * \param A   the system matrix
     * \param prm Relaxation parameters
     */
    gauss_seidel(const CSRMatrix<T>& A, const params& prm) : is_serial(prm.serial || num_threads < 4) {
        if (!is_serial) {
            forward = std::make_unique<parallel_sweep<true>>(A);
            backward = std::make_unique<parallel_sweep<false>>(A);
        }
    }

    void apply_pre(const CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
        if (is_serial)
            serial_sweep(A, rhs, x, true);
        else
            forward->sweep(rhs, x);
    }

    void apply_post(const CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
        if (is_serial)
            serial_sweep(A, rhs, x, false);
        else
            backward->sweep(rhs, x);
    }

    void apply(const CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
        if (is_serial) {
            serial_sweep(A, rhs, x, true);
            serial_sweep(A, rhs, x, false);
        }
        else {
            forward->sweep(rhs, x);
            backward->sweep(rhs, x);
        }
    }

    // todo: implement bytes method

private:
    /// @brief Performs a single, in-place, serial Gauss-Seidel relxation sweep
    /// @param A The input system matrix in CSR format
    /// @param rhs The right hand side vector
    /// @param x The solution vector which is updated in place.
    /// @param forward Determines sweep directtion
    static void serial_sweep(const CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, bool forward) {
        const size_t n = A.m_rows;
        
        // Set up loop parameters for forward and backward sweep
        const ptrdiff_t beg = forward ? 0 : n - 1;
        const ptrdiff_t end = forward ? n : -1;
        const ptrdiff_t inc = forward ? 1 : -1;

        for (ptrdiff_t i = beg; i != end; i += inc) {
            T D = T(1);
            T X = rhs[i];

            // Summation,  X = b_i - sum(A_ij - x_j) for j != i
            for (auto a = A.rowBegin(i); a; ++a) {
                size_t c = a.col();
                T v = a.value();

                if (c == i)
                    D = v;
                else
                    X -= v * x[c];
            }

            // Update solution vector
            x[i] = (T(1) / D) * X;
        }
    }

    template <bool forward>
    struct parallel_sweep {
        /// @brief Defines a range of work for each thread to process
        struct task {
            ptrdiff_t beg, end;
            task(ptrdiff_t beg, ptrdiff_t end) : beg(beg), end(end) { }
        };

        // thread specific storage
        std::vector<std::vector<task>> tasks;         // task schedule for each thread.
        std::vector<std::vector<size_t>> row_ptrs;    // thread local row pointers
        std::vector<std::vector<size_t>> col_indices; // thread local column indices
        std::vector<std::vector<T>> vals;             // thread local values
        std::vector<std::vector<size_t>> ord;         // thread local mapping to original row indices

        /// @brief Consturctor builds the parallel execution scheme.
        /// @param A The system matrix to be analyzed
        parallel_sweep(const CSRMatrix<T> &A)
            : tasks(num_threads), row_ptrs(num_threads), col_indices(num_threads),
              vals(num_threads), ord(num_threads)
        {
            const size_t n = A.m_rows;
            size_t nlev = 0;

            // Phase 1: Split rows into levels based on data dependencies (Topological sort).
            // level[i] stores the dependency level for row i.
            std::vector<size_t> level(n, 0);

            ptrdiff_t beg = forward ? 0 : n - 1;
            ptrdiff_t end = forward ? n : -1;
            ptrdiff_t inc = forward ? 1 : -1;

            for (ptrdiff_t i = beg; i != end; i += inc) {
                size_t l = level[i];

                for (auto a = A.rowBegin(i); a; ++a) {
                    size_t c = a.col();

                    if (forward) {
                        if (c >= i) continue;
                    } else {
                        if (c <= i) continue;
                    }

                    l = std::max(l, level[c] + 1);
                }

                level[i] = l;
                nlev = std::max(nlev, l + 1);
            }

            // Phase 2: Reorder matrix rows into level sorted groups
            // order[k] stores the original row index for the k-th row in the sorted ordering.
            std::vector<size_t> order(n, 0);
            // start[k] stores the starting index in `order` for rows of level k.
            std::vector<size_t> start(nlev + 1, 0);

            for (size_t i = 0; i < n; ++i) {
                ++start[level[i] + 1];
            }

            std::partial_sum(start.begin(), start.end(), start.begin());

            for (size_t i = 0; i < n; ++i) {
                order[start[level[i]]++] = i; 
            }

            std::rotate(start.begin(), start.end() - 1, start.end());
            start[0] = 0;

            // Phase 3: Organize level sorted matrix rows into tasks for each thread
            // Keep track of total number of rows and non-zeros for each thread
            std::vector<size_t> thread_rows(num_threads, 0);
            std::vector<size_t> thread_cols(num_threads, 0);

            auto organizeTasks = [&](int thread_id) {
                tasks[thread_id].reserve(nlev);

                for (size_t lev = 0; lev < nlev; ++lev) {
                    // Split each level into tasks
                    size_t lev_size = start[lev + 1] - start[lev];
                    size_t chunk_size = (lev_size + num_threads - 1) / num_threads;

                    ptrdiff_t beg = std::min(static_cast<size_t>(thread_id) * chunk_size, lev_size);
                    ptrdiff_t end = std::min(beg + chunk_size, lev_size);

                    beg += start[lev];
                    end += start[lev];

                    tasks[thread_id].emplace_back(beg, end);

                    // count rows and non-zeros in current task
                    thread_rows[thread_id] += end - beg;
                    for (ptrdiff_t i = beg; i < end; ++i) {
                        size_t j = order[i];
                        thread_cols[thread_id] += (A.m_row_pointers[j + 1] - A.m_row_pointers[j]);
                    }
                }
            };

            launchThreadsWithID(organizeTasks);

            // 4. Reorganize matrix data for to thread local storage
            auto reorderMatrix = [&](int thread_id) {
                col_indices[thread_id].reserve(thread_cols[thread_id]);
                vals[thread_id].reserve(thread_cols[thread_id]);
                ord[thread_id].reserve(thread_rows[thread_id]);
                row_ptrs[thread_id].reserve(thread_rows[thread_id] + 1);
                row_ptrs[thread_id].push_back(0);

                // Iterate through this thread's assigned tasks (one per level).
                for (task &t : tasks[thread_id]) {
                    ptrdiff_t loc_beg = row_ptrs[thread_id].size() - 1;
                    ptrdiff_t loc_end = loc_beg;

                    // Copy data for each row in the task from the global matrix to local storage.
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

                    // each task corresponds to a level, so we need
                    // to synchronize across threads at this point:
                    sync_point.arrive_and_wait();
                }
            };

            launchThreadsWithID(compute);
        }
    };

    std::unique_ptr<parallel_sweep<true>> forward;
    std::unique_ptr<parallel_sweep<false>> backward;
};