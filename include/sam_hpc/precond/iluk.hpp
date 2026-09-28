#pragma once

#include "sam_hpc/precond/ilu_solve.hpp"

#include <queue>


/// ILU(k) smoother.
/// Implements an Incomplete LU factorization with level-of-fill-in.
/// This factorization is used as a smoother or preconditioner for iterative solvers.
template <typename T>
struct iluk {
    /// Relaxation parameters.
    struct params {
        /// The allowed level of fill-in. k=0 means no new non-zeros are created.
        int k;

        /// Damping factor.
        T damping;

        /// Parameters for sparse triangular system solver
        typename ilu_solve_iterative<T>::params solve;
        // typename ilu_solve_direct<T>::params solve;

        params() : k(1), damping(1) {}
    } prm;

public:

    /// @brief Performs the ILU(k) factorization on the input matrix A.
    /// @param A The input sparse matrix in CSR format.
    /// @param prm The parameters for the factorization.
    iluk(const CSRMatrix<T> &A, const params &prm = params()) : prm(prm) {
        // Get matrix dimensions and number of non-zeros.
        const size_t n = A.m_rows;
        size_t Anz = A.m_nnz;

        // Storage for the L (Lower triangular) factor in CSR format
        std::vector<size_t> Lptr; 
        Lptr.reserve(n + 1); 
        Lptr.push_back(0);
        std::vector<size_t> Lcol; 
        Lcol.reserve(Anz / 3); // Heuristic pre-allocation
        std::vector<T> Lval; 
        Lval.reserve(Anz / 3);

        // Storage for the U (Upper triangular) factor in CSR format
        std::vector<size_t> Uptr; 
        Uptr.reserve(n + 1);
        Uptr.push_back(0);
        std::vector<size_t> Ucol;
        Ucol.reserve(Anz / 3);
        std::vector<T> Uval; 
        Uval.reserve(Anz / 3);

        // Storage for the level of each non-zero in the U factor
        // This is crucial for calculating the level of new fill-in.
        std::vector<int> Ulev; 
        Ulev.reserve(Anz / 3);

        // Storage for the inverse of the diagonal elements
        std::shared_ptr<std::vector<T>> D = std::make_shared<std::vector<T>>(n);

        // A temporary workspace to build each row of the factors.
        sparse_vector w(n, prm.k);
        for(size_t i = 0; i < n; ++i) {
            std::cout << "Starting row " << i << std::endl;
            // Clear the workspace for the current row i.
            w.reset(i);

            // Copy the non-zeros from row i of the original matrix A into the workspace.
            // These initial entries are all assigned a level of 0.
            for(auto a = A.rowBegin(i); a; ++a) {
                w.add(a.col(), a.value(), 0);
            }

            // Gaussian elimination for the current row
            // It processes elements in the lower part of the row to update the upper part.
            while(!w.q.empty()) {
                // Get the unprocessed non-zero L[i, k] with the smallest column index `k`.
                nonzero &a = w.next_nonzero();
                // Scale the element: L[i, k] = L[i, k] / U[K, kl].
                a.val = a.val * (*D)[a.col];

                // For each non-zero U[k, j] in the previously computed row `k` of U...
                for(size_t j = Uptr[a.col], e = Uptr[a.col + 1]; j < e; ++j) {
                    // Calculate the level of the potential new fill-in element.
                    int lev = std::max(a.lev, Ulev[j]) + 1;
                    // Update the workspace: w_j = w_j - L[i, k] * U[k, j].
                    // The add() method enforces the dropping rule: if lev > k, the new
                    // element is discarded, making the factorization "incomplete".
                    w.add(Ucol[j], -a.val * Uval[j], lev);
                }
            }

            // Sort the computed non-zeros for the current row by column index.
            w.sort();

            // Partition the completed row from the workspace into L, D, and U factors
            for(const nonzero &e : w.nz) {
                if (e.col < i) {
                    // Elements left of the diagonal belong to the L factor.
                    Lcol.push_back(e.col);
                    Lval.push_back(e.val);
                } else if (e.col == i) {
                    // Diagonal element
                    const T tolerance = T(1e-10);
                    T pivot = e.val;

                    if (std::abs(pivot) < tolerance) {
                        pivot = (pivot >= 0) ? tolerance : -tolerance;
                    }

                    (*D)[i] = 1 / pivot;
                } else {
                    // Elements right of the diagonal belong to the U factor.
                    Ucol.push_back(e.col);
                    Uval.push_back(e.val);
                    Ulev.push_back(e.lev); // Store its level for future calculations.
                }
            }

            // Finalize row i for the L and U factors in CSR format.
            Lptr.push_back(Lcol.size());
            Uptr.push_back(Ucol.size());
        }

        ilu = std::make_shared<ilu_solve_iterative<T>>(
                std::make_shared<CSRMatrix<T>>(n, n, Lval, Lptr, Lcol),
                std::make_shared<CSRMatrix<T>>(n, n, Uval, Uptr, Ucol),
                D, prm.solve);

        /* ilu = std::make_shared<ilu_solve_direct<T>>(
                std::make_shared<CSRMatrix<T>>(n, n, Lval, Lptr, Lcol),
                std::make_shared<CSRMatrix<T>>(n, n, Uval, Uptr, Ucol),
                D, prm.solve); */
    }

    void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
        x = rhs;
        ilu->solve(x);
    }

    void apply_pre(const CSRMatrix<T> &A, const std::vector<T> &rhs,
                   std::vector<T> &x, std::vector<T> &tmp) const
    {
        residual(rhs, A, x, tmp);
        ilu->solve(tmp);
        axpby(prm.damping, tmp, T(1), x);
    }

    void apply_post(const CSRMatrix<T> &A, const std::vector<T> &rhs,
                   std::vector<T> &x, std::vector<T> &tmp) const
    {
        residual(rhs, A, x, tmp);
        ilu->solve(tmp);
        axpby(prm.damping, tmp, T(1), x);
    }

    // TODO: bytes method

private:
    std::shared_ptr<ilu_solve_iterative<T>> ilu;
    //std::shared_ptr<ilu_solve_direct<T>> ilu;

    /// @brief A helper struct to hold a non-zero matrix element's properties.
    struct nonzero {
        ptrdiff_t col; /// Column index of the element.
        T val;         /// Numerical value of the element.
        int lev;       /// Fill-in level of the element.

        nonzero() : col(-1) {}

        nonzero(ptrdiff_t col, const T &val, int lev)
            : col(col), val(val), lev(lev) {}

        /// Defines ordering based on column index, required for std::sort.
        friend bool operator<(const nonzero &a, const nonzero &b) {
            return a.col < b.col;
        }
    };

    /// @brief An efficient data structure for managing the non-zeros of a single row during factorization.
    struct sparse_vector {
        /// A custom comparator (functor) for the priority queue.
        /// It creates a min-priority queue based on the column index of non-zeros.
        struct comp_indices {
            const std::deque<nonzero> &nz;

            comp_indices(const std::deque<nonzero> &nz) : nz(nz) {}

            /// Compares indices a and b by looking up their corresponding column indices in nz.
            bool operator()(int a, int b) const {
                return nz[a].col > nz[b].col;
            }
        };

        /// Alias for our specialized priority queue. It stores indices into the `nz` deque.
        typedef std::priority_queue<int, std::vector<int>, comp_indices> priority_queue;

        int lfil;                   /// The maximum allowed level of fill-in (k).
        std::deque<nonzero> nz;     /// Stores the actual non-zero elements.
        std::vector<ptrdiff_t> idx; /// A lookup map: idx[col] -> position in nz. For O(1) access.
        priority_queue q;           /// A min-priority queue of indices into `nz` for the lower part.
        ptrdiff_t dia;              /// Column index of the diagonal for the current row.

        /// @brief Constructor for the sparse_vector.
        /// @param n The total size (number of columns) of the matrix.
        /// @param lfil The level-of-fill-in parameter.
        sparse_vector(size_t n, int lfil)
            : lfil(lfil), idx(n, -1), q(comp_indices(nz)), dia(0) {}

        /// @brief Adds or updates a value in the sparse vector.
        void add(ptrdiff_t col, const T &val, int lev) {
            // If no element exists at this column yet..
            if (idx[col] < 0) {
                // ...and its level is within the allowed limit (the dropping rule)...
                if (lev <= lfil) {
                    // ...then add it.
                    int p = nz.size();
                    idx[col] = p; // Update the lookup map.
                    nz.push_back(nonzero(col, val, lev));
                    // If it's in the lower part, add its index to the priority queue.
                    if (col < dia) q.push(p);
                }
            } else {
                // If an element already exists, just update its value and level.
                nonzero &a = nz[idx[col]];
                a.val += val;
                a.lev = std::min(a.lev, lev); // Level of an entry is the minimum of all paths.
            }
        }

        typename std::deque<nonzero>::iterator begin() {
            return nz.begin();
        }

        typename std::deque<nonzero>::iterator end() {
            return nz.end();
        }

         /// @brief Gets the unprocessed element from the lower part with the smallest column index.
        nonzero& next_nonzero() {
            int p = q.top(); // Get index from top of min-priority queue.
            q.pop();
            return nz[p];
        }

        /// @brief Sorts the non-zero elements by column index.
        void sort() {
            std::sort(nz.begin(), nz.end());
        }

        /// @brief Clears the workspace for processing a new row.
        /// @param d The new diagonal index.
        void reset(ptrdiff_t d) {
            // Invalidate all entries in the lookup map.
            for(const nonzero &e : nz) idx[e.col] = -1;
            nz.clear(); // Clear the storage.
            dia = d;    // Set the diagonal for the new row.
        }
    };
};