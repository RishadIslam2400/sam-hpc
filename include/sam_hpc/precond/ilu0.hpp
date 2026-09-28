#pragma once

#include "sam_hpc/precond/ilu_solve.hpp"

/// ILU(0) smoother.
/// This class computes an Incomplete LU factorization with zero fill-in for a given
/// sparse matrix A. The resulting factors L and U are used as a preconditioner
/// to accelerate iterative solvers. The preconditioner M is an approximation of A, where M = LU.
template <typename T>
struct ilu0 {
    /// Relaxation parameters.
    struct params {
        /// Damping factor.
        T damping;

        /// Parameters for sparse triangular system solver
        // typename ilu_solve_iterative<T>::params solve;
        typename ilu_solve_direct<T>::params solve;
        
        params() : damping(1) {}
    } prm;

    /// Constructor that performs the ILU(0) factorization.
    /// @param A   The input sparse matrix in CSR format to be factorized.
    /// @param prm The parameters for the factorization and solver.
    ilu0(const CSRMatrix<T> &A, const params &prm = params()) : prm(prm) {
        const size_t n = A.m_rows;

        // Step 1: Count the number of non-zeros for L and U factors ---
        // For ILU(0), the sparsity pattern of L and U is the same as the
        // strictly lower and upper parts of A.
        size_t Lnz = 0, Unz = 0;
        for(size_t i = 0; i < n; ++i) {
            size_t row_beg = A.m_row_pointers[i];
            size_t row_end = A.m_row_pointers[i + 1];

            for(size_t j = row_beg; j < row_end; ++j) {
                size_t c = A.m_col_indices[j];
                if (c < i)      // Element is in the strictly lower part.
                    ++Lnz;
                else if (c > i) // Element is in the strictly upper part.
                    ++Unz;
            }
        }

        // Step 2: Allocate memory for the L, U, and D factors ---
        std::shared_ptr<CSRMatrix<T>> L = std::make_shared<CSRMatrix<T>>();
        std::shared_ptr<CSRMatrix<T>> U = std::make_shared<CSRMatrix<T>>();

        // Allocate L matrix structure.
        L->m_rows = n;
        L->m_cols = n;
        L->m_nnz = Lnz;
        L->m_row_pointers.resize(L->m_rows + 1);
        L->m_col_indices.resize(L->m_nnz);
        L->m_vals.resize(L->m_nnz, T(0));

        // Allocate U matrix structure.
        U->m_rows = n;
        U->m_cols = n;
        U->m_nnz = Unz;
        U->m_row_pointers.resize(U->m_rows + 1);
        U->m_col_indices.resize(U->m_nnz);
        U->m_vals.resize(U->m_nnz, T(0));

        size_t Lhead = 0; // Current position in L's data arrays.
        size_t Uhead = 0; // Current position in U's data arrays.

        // D stores the inverse diagonal of U faster solves.
        std::shared_ptr<std::vector<T>> D = std::make_shared<std::vector<T>>(n, T(0));

        // Step 3: Perform the factorization row by row
        // 'work' is a sparse accumulator. It holds pointers to the elements of the
        // current row being factorized, allowing for efficient updates.
        std::vector<T*> work(n, NULL);

        for(size_t i = 0; i < n; ++i) {
            size_t row_beg = A.m_row_pointers[i];
            size_t row_end = A.m_row_pointers[i + 1];

            // Scatter Phase
            // Copy the structure and values of row 'i' from A into L, U, D
            // and populate the 'work' vector with pointers to these values.
            for(size_t j = row_beg; j < row_end; ++j) {
                size_t  c = A.m_col_indices[j];
                T v = A.m_vals[j];

                if (c < i) { // Lower triangular part
                    L->m_col_indices[Lhead] = c;
                    L->m_vals[Lhead] = v;
                    work[c] = L->m_vals.data() + Lhead;
                    ++Lhead;
                } else if (c == i) { // Diagonal part
                    (*D)[i] = v + 1e-10;
                    work[c] = &(*D)[i];
                } else { // Upper triangular part
                    U->m_col_indices[Uhead] = c;
                    U->m_vals[Uhead] = v;
                    work[c] = U->m_vals.data() + Uhead;
                    ++Uhead;
                }
            }

            // Set the row pointers for the current row.
            L->m_row_pointers[i + 1] = Lhead;
            U->m_row_pointers[i + 1] = Uhead;

            // Elimination Phase
            // Iterate through the columns of the lower part of row 'i' in A.
            for(size_t j = row_beg; j < row_end; ++j) {
                size_t c = A.m_col_indices[j];

                // Stop when we reach the diagonal.
                // Elimination is only done using previous rows (c < i).
                if (c >= i) {
                    assert(c == i && "No diagonal value in system matrix");
                    // assert((*D)[i] == T(0) && "Zero pivot in ILU");
                    if ((*D)[i] == T(0))
                        (*D)[i] = T(1e-10);

                    // Invert the diagonal element. This replaces division with multiplication
                    // during the solve phase, which is faster.
                    (*D)[i] = 1 / (*D)[i];
                    break;
                }

                // Compute the multiplier L[i, c] = A[i, c] / U[c, c] = A[i, c] * (1/U[c, c]).
                // (*D)[c] already holds the inverse of the diagonal of U from a previous iteration.
                T tl = (*work[c]) * (*D)[c];
                *work[c] = tl; // Store the final value in the L factor.

                // Perform the linear combination: R_i = R_i - L[i, c] * R_c(U).
                // This updates the values in the current row 'i' using the already computed row 'c' of U.
                // Update is only done in the upper triangular and diagonal part
                for(size_t k = U->m_row_pointers[c]; k < U->m_row_pointers[c + 1]; ++k) {
                    // Find the element in the current row 'i' that corresponds to the column of U[c, k].
                    T *w = work[U->m_col_indices[k]];

                    // If 'w' is not null, an element exists at this position in the original
                    // sparsity pattern. Update it. If 'w' is null, this would be fill-in,
                    // which ILU(0) ignores.
                    if (w) *w -= tl * U->m_vals[k];
                }
            }

            // Cleanup Phase
            // Reset the pointers in the work vector for the current row's sparsity pattern.
            for(size_t j = row_beg; j < row_end; ++j)
                work[A.m_col_indices[j]] = NULL;
        }

        // Compact the L matrix in-place
        size_t new_Lnz = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t row_start = new_Lnz; // Record the start of the new compacted row

            // Iterate through the uncompacted row using the old pointers
            for (size_t j = L->m_row_pointers[i]; j < L->m_row_pointers[i + 1]; ++j) {
                T v = L->m_vals[j];
                // If the value is non-zero, copy it to the new position
                if (v != T(0)) {
                    L->m_col_indices[new_Lnz] = L->m_col_indices[j];
                    L->m_vals[new_Lnz] = v;
                    ++new_Lnz;
                }
            }
            // Update the row pointer to point to the start of the new compacted row
            L->m_row_pointers[i] = row_start;
        }
        // Set the final pointer and the new non-zero count
        L->m_row_pointers[n] = new_Lnz;
        L->m_nnz = new_Lnz;
        // Trim excess capacity from vectors to save memory
        L->m_col_indices.resize(L->m_nnz);
        L->m_vals.resize(L->m_nnz);

        // Compact the U matrix in-place (same logic)
        size_t new_Unz = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t row_start = new_Unz;

            for (size_t j = U->m_row_pointers[i]; j < U->m_row_pointers[i + 1]; ++j) {
                T v = U->m_vals[j];
                if (v != T(0)) {
                    U->m_col_indices[new_Unz] = U->m_col_indices[j];
                    U->m_vals[new_Unz] = v;
                    ++new_Unz;
                }
            }
            U->m_row_pointers[i] = row_start;
        }
        U->m_row_pointers[n] = new_Unz;
        U->m_nnz = new_Unz;
        U->m_col_indices.resize(U->m_nnz);
        U->m_vals.resize(U->m_nnz);

        // ilu = std::make_shared<ilu_solve_iterative<T>>(L, U, D, prm.solve);
        ilu = std::make_shared<ilu_solve_direct<T>>(L, U, D, prm.solve);
    }

     /// Applies the preconditioner as a pre-smoother in an iterative solver.
    /// This computes x = x + w * M^-1 * (b - Ax).
    void apply_pre(const CSRMatrix<T> &A, const std::vector<T> &rhs,
                   std::vector<T> &x, std::vector<T> &tmp) const
    {
        // Compute residual r = b - Ax, store in tmp.
        residual(rhs, A, x, tmp);
        // Solve Mz = r. The solution z = M^-1*r is stored in tmp.
        ilu->solve(tmp);
        // Update solution x = x + damping * z.
        axpby(prm.damping, tmp, T(1), x);
    }

    /// Applies the preconditioner as a post-smoother. For a stationary
    /// preconditioner like ILU(0), this is identical to the pre-smoother.
    void apply_post(const CSRMatrix<T> &A, const std::vector<T> &rhs,
                    std::vector<T> &x, std::vector<T> &tmp) const
    {
        residual(rhs, A, x, tmp);
        ilu->solve(tmp);
        axpby(prm.damping, tmp, T(1), x);
    }

    /// Applies the preconditioner as a standalone solver.
    /// This computes x = M^-1 * b.
    void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
        x = rhs;
        ilu->solve(x);
    }

    // Todo: bytes method

private:
    /// The solver object that performs the forward/backward substitutions.
    // std::shared_ptr<ilu_solve_iterative<T>> ilu;
    std::shared_ptr<ilu_solve_direct<T>> ilu;
};