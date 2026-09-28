#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

template <typename T>
class ilu_solve_iterative {
public:
    struct params {
        /// Number of Jacobi iterations
        unsigned iters;

        /// Damping factor
        T damping;

        params() : iters(2), damping(0.72) {}
    } prm;

    ilu_solve_iterative(std::shared_ptr<CSRMatrix<T>> L, std::shared_ptr<CSRMatrix<T>> U, std::shared_ptr<std::vector<T>> D, const params &prm = params())
        : prm(prm), L(L), U(U), D(D), y0(L->m_rows), y1(L->m_rows) {}

    // A standard ILU solve computes (LU)^-1 x by first solving Ly = x and then Uz = y
    void solve(std::vector<T>& x) {
        // Approximating Ly0 = x
        axpby(prm.damping, x, 0.0, y0); // initial guess for the Jacobi iterative solver
        for (unsigned i = 0; i < prm.iters; ++i) {
            residual(x, *L, y0, y1); // y1 = x - Ly0

            // Update the solution y0
            // weighted average of old y0 and the residual
            axpby(prm.damping, y1, (1 - prm.damping), y0);
        }

        // Approximating Uz = y0
        // x (in place of z) is the initial guess
        vmul(prm.damping, *D, y0, 0.0, x); // x[i] = omega * D[i] * y0[i]
        for (unsigned i = 0; i < prm.iters; ++i) {
            residual(y0, *U, x, y1); // y1 = y0 - Ux
            vmul(prm.damping, *D, y1, (1 - prm.damping), x); // Final solution stored in x
        }
    }

    // TODO: bytes method

private:
    std::shared_ptr<CSRMatrix<T>> L;
    std::shared_ptr<CSRMatrix<T>> U;
    std::shared_ptr<std::vector<T>> D;
    std::vector<T> y0, y1;
};

template <typename T>
class ilu_solve_direct {
public:
    struct params {
        /// Use serial version of the algorithm
        bool serial;

        params() : serial(num_threads < 4) {}
    } prm;

    ilu_solve_direct(std::shared_ptr<CSRMatrix<T>> L,
                     std::shared_ptr<CSRMatrix<T>> U,
                     std::shared_ptr<std::vector<T>> D,
                     const params &prm = params()) : prm(prm)
    {
        if (prm.serial)
            serial_init(L, U, D);
        else
            parallel_init(L, U, D);
    }

    void solve(std::vector<T> &x) {
        if (prm.serial)
            serial_solve(x);
        else
            parallel_solve(x);
    }

    // TODO: bytes method

private:
    std::shared_ptr<CSRMatrix<T>> L;
    std::shared_ptr<CSRMatrix<T>> U;
    std::shared_ptr<std::vector<T>> D;
    
    void serial_init(std::shared_ptr<CSRMatrix<T>> L, std::shared_ptr<CSRMatrix<T>> U, std::shared_ptr<std::vector<T>> D) {
        this->L = L;
        this->U = U;
        this->D = D;
    }

    void serial_solve(std::vector<T>& x) {
        const size_t n = L->m_rows;

        const CSRMatrix<T> &L = *(this->L);
        const CSRMatrix<T> &U = *(this->U);
        const std::vector<T> &D = *(this->D);

        for(size_t i = 0; i < n; i++) {
            for(size_t j = L.m_row_pointers[i], e = L.m_row_pointers[i + 1]; j < e; ++j)
                x[i] -= L.m_vals[j] * x[L.m_col_indices[j]];
        }

        for (size_t i = n; i-- > 0;) {
            for (size_t j = U.m_row_pointers[i], e = U.m_row_pointers[i + 1]; j < e; ++j) {
                x[i] -= U.m_vals[j] * x[U.m_col_indices[j]];
            }
            x[i] = D[i] * x[i];
        }
    }
        // TBB solver for sparse triangular systems.
        // The solver uses level scheduling approach.
        // Each level (a set of matrix rows that can be computed independently)
        // is split into tasks, a task per thread, and the matrix data is
        // distributed across threads to improve cache and NUMA locality.
    template <bool lower>
    struct sptr_solve {
        // a task is a set of rows that can be computed independently by a single thread.
        struct task {
            ptrdiff_t beg, end;
            task(ptrdiff_t beg, ptrdiff_t end) : beg(beg), end(end) {}
        };

        // thread-specific storage
        std::vector<std::vector<task>> tasks;
        std::vector<std::vector<size_t>> ptr;
        std::vector<std::vector<size_t>> col;
        std::vector<std::vector<T>> val;
        std::vector<std::vector<size_t>> ord; // rows ordered by levels
        std::vector<std::vector<T>> D;
        size_t nlev;

        sptr_solve(const CSRMatrix<T>& A, const T *_D = 0) 
            : tasks(num_threads), ptr(num_threads), col(num_threads), val(num_threads), ord(num_threads), nlev(0)
        {
            size_t n = A.m_rows;

            std::vector<size_t> level(n, 0);
            std::vector<size_t> order(n, 0);

            // 1. split rows into levels.
            ptrdiff_t beg = lower ? 0 : n - 1;
            ptrdiff_t end = lower ? n : -1;
            ptrdiff_t inc = lower ? 1 : -1;

            for (ptrdiff_t i = beg; i != end; i += inc) {
                size_t l = level[i];

                for (size_t j = A.m_row_pointers[i]; j < A.m_row_pointers[i + 1]; ++j) {
                    l = std::max(l, level[A.m_col_indices[j]] + 1);
                }

                level[i] = l;
                nlev = std::max(nlev, l + 1);
            }

            // 2. reorder matrix rows.
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

            // 3. Organize matrix rows into tasks.
            //    Each level is split into num_threads tasks.
            std::vector<size_t> thread_rows(num_threads, 0);
            std::vector<size_t> thread_cols(num_threads, 0);

            tbb::parallel_for(0, num_threads, [&](int tid) {
                tasks[tid].reserve(nlev);

                for (size_t lev = 0; lev < nlev; ++lev) {
                    // split each level into tasks.
                    size_t lev_size = start[lev + 1] - start[lev];
                    size_t chunk_size = (lev_size + num_threads - 1) / num_threads;

                    ptrdiff_t beg = std::min(tid * chunk_size, lev_size);
                    ptrdiff_t end = std::min(beg + chunk_size, lev_size);

                    beg += start[lev];
                    end += start[lev];

                    tasks[tid].push_back(task(beg, end));

                    // count rows and nonzeros in the current task
                    thread_rows[tid] += end - beg;
                    for(ptrdiff_t i = beg; i < end; ++i) {
                        size_t j = order[i];
                        thread_cols[tid] += A.m_row_pointers[j + 1] - A.m_row_pointers[j];
                    }
                }
            });

            // 4. reorganize matrix data for better cache and NUMA locality.
            if (!lower) D.resize(num_threads);
            tbb::parallel_for(0, num_threads, [&](int tid) {
                col[tid].reserve(thread_cols[tid]);
                val[tid].reserve(thread_cols[tid]);
                ord[tid].reserve(thread_rows[tid]);
                ptr[tid].reserve(thread_rows[tid] + 1);
                ptr[tid].push_back(0);

                if (!lower) D[tid].reserve(thread_rows[tid]);

                for (task& t : tasks[tid]) {
                    ptrdiff_t loc_beg = ptr[tid].size() - 1;
                    ptrdiff_t loc_end = loc_beg;

                    for (ptrdiff_t r = t.beg; r < t.end; ++r, ++loc_end) {
                        size_t i = order[r];
                        if(!lower)
                            D[tid].push_back(_D[i]);
                        ord[tid].push_back(i);

                        for (size_t j = A.m_row_pointers[i]; j < A.m_row_pointers[i + 1]; ++j) {
                            col[tid].push_back(A.m_col_indices[j]);
                            val[tid].push_back(A.m_vals[j]);
                        }
                        ptr[tid].push_back(col[tid].size());
                    }

                    t.beg = loc_beg;
                    t.end = loc_end;
                }
            });
        }

        void solve(std::vector<T>& x) const {
            // Loop through levels sequentially and parallelize the work within each level.
            for (size_t lev = 0; lev < nlev; ++lev) {
                tbb::parallel_for(0, num_threads, [&](int tid) {
                    const task &t = tasks[tid][lev];

                    for (ptrdiff_t r = t.beg; r < t.end; ++r) {
                        ptrdiff_t i = ord[tid][r];
                        ptrdiff_t beg = ptr[tid][r];
                        ptrdiff_t end = ptr[tid][r + 1];

                        T X = T(0);
                        for (ptrdiff_t j = beg; j < end; ++j) {
                            X += val[tid][j] * x[col[tid][j]];
                        }

                        if (lower) {
                            x[i] -= X;
                        } else {
                            x[i] = D[tid][r] * (x[i] - X);
                        }
                    }
                });
            }  // Implicit barrier: all threads finish this level before the next loop iteration.
        }

        // TODO: bytes method
    };

    std::shared_ptr<sptr_solve<true>> lower;
    std::shared_ptr<sptr_solve<false>> upper;

    void parallel_init(std::shared_ptr<CSRMatrix<T>> L, std::shared_ptr<CSRMatrix<T>> U, std::shared_ptr<std::vector<T>> D) {
            lower = std::make_shared<sptr_solve<true >>(*L, D->data());
            upper = std::make_shared<sptr_solve<false>>(*U, D->data());
    }

    void parallel_solve(std::vector<T> &x) {
        lower->solve(x);
        upper->solve(x);
    }
};