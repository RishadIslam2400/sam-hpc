#pragma once

#include "sam_hpc/precond/ilu_solve.hpp"

/// Incomplete LU factorization with zero fill-in. The factors satisfy 
/// `A ~= L U`: L is strictly lower with an implicit unit diagonal, U is strictly upper,
/// and D stores the inverse diagonal of U. Updates outside A's sparsity pattern are
/// discarded. Triangular solves use `ilu_solve_direct`.
///
/// A must be square, have sorted column indices in every row, and contain a stored
/// diagonal entry in every row. The factorization performs no pivoting.
template <typename T>
struct ilu0 {
  /// Relaxation parameters.
  struct params {
    /// Smoothing weight used by apply_pre and apply_post.
    T damping;

    /// Parameters for the triangular solver.
    typename ilu_solve_direct<T>::params solve;
    
    params() : damping(1) {}
  } prm;

  /// Constructs and factors the ILU(0) preconditioner.
  /// @param A Matrix to factorize. It is not retained for residual computations.
  /// @param prm Factorization and triangular-solver parameters.
  ilu0(const csr::CSRMatrix<T> &A, const params &prm = params()) : prm(prm) {
    const size_t n = A.m_rows;

    // Allocate factor storage using the strict lower and upper patterns of A.
    size_t Lnz = 0, Unz = 0;
    for(size_t i = 0; i < n; ++i) {
      size_t row_beg = A.m_row_pointers[i];
      size_t row_end = A.m_row_pointers[i + 1];

      for(size_t j = row_beg; j < row_end; ++j) {
        size_t c = A.m_col_indices[j];
        if (c < i)
          ++Lnz;
        else if (c > i)
          ++Unz;
      }
    }

    std::shared_ptr<csr::CSRMatrix<T>> L = std::make_shared<csr::CSRMatrix<T>>();
    std::shared_ptr<csr::CSRMatrix<T>> U = std::make_shared<csr::CSRMatrix<T>>();

    L->m_rows = n;
    L->m_cols = n;
    L->m_nnz = Lnz;
    L->m_row_pointers.resize(L->m_rows + 1);
    L->m_col_indices.resize(L->m_nnz);
    L->m_vals.resize(L->m_nnz, T(0));

    U->m_rows = n;
    U->m_cols = n;
    U->m_nnz = Unz;
    U->m_row_pointers.resize(U->m_rows + 1);
    U->m_col_indices.resize(U->m_nnz);
    U->m_vals.resize(U->m_nnz, T(0));

    size_t Lhead = 0;
    size_t Uhead = 0;

    // D stores the inverse diagonal of U.
    std::shared_ptr<std::vector<T>> D = std::make_shared<std::vector<T>>(n, T(0));

    // Pointers into the current row provide sparse accumulator entries without
    // allocating a separate work row.
    std::vector<T*> work(n, NULL);

    for(size_t i = 0; i < n; ++i) {
      size_t row_beg = A.m_row_pointers[i];
      size_t row_end = A.m_row_pointers[i + 1];

      // Scatter the row into factor storage and index its entries through work.
      for(size_t j = row_beg; j < row_end; ++j) {
        size_t  c = A.m_col_indices[j];
        T v = A.m_vals[j];

        if (c < i) {
          L->m_col_indices[Lhead] = c;
          L->m_vals[Lhead] = v;
          work[c] = L->m_vals.data() + Lhead;
          ++Lhead;
        } else if (c == i) {
          (*D)[i] = v + 1e-10;
          work[c] = &(*D)[i];
        } else {
          U->m_col_indices[Uhead] = c;
          U->m_vals[Uhead] = v;
          work[c] = U->m_vals.data() + Uhead;
          ++Uhead;
        }
      }

      L->m_row_pointers[i + 1] = Lhead;
      U->m_row_pointers[i + 1] = Uhead;

      // Eliminate with previously factored lower entries.
      for(size_t j = row_beg; j < row_end; ++j) {
        size_t c = A.m_col_indices[j];

        if (c >= i) {
          assert(c == i && "No diagonal value in system matrix");
          if ((*D)[i] == T(0))
            (*D)[i] = T(1e-10);

          (*D)[i] = 1 / (*D)[i];
          break;
        }

        T tl = (*work[c]) * (*D)[c];
        *work[c] = tl;

        // Update only existing upper and diagonal entries; missing entries are
        // fill-in and are ignored by ILU(0).
        for(size_t k = U->m_row_pointers[c]; k < U->m_row_pointers[c + 1]; ++k) {
          T *w = work[U->m_col_indices[k]];
          if (w)
            *w -= tl * U->m_vals[k];
        }
      }

      // Clear accumulator entries before processing the next row.
      for(size_t j = row_beg; j < row_end; ++j)
        work[A.m_col_indices[j]] = NULL;
    }

    // Remove entries that are exactly zero after elimination.
    size_t new_Lnz = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t row_start = new_Lnz;

      for (size_t j = L->m_row_pointers[i]; j < L->m_row_pointers[i + 1]; ++j) {
        T v = L->m_vals[j];
        if (v != T(0)) {
          L->m_col_indices[new_Lnz] = L->m_col_indices[j];
          L->m_vals[new_Lnz] = v;
          ++new_Lnz;
        }
      }
      L->m_row_pointers[i] = row_start;
    }
    L->m_row_pointers[n] = new_Lnz;
    L->m_nnz = new_Lnz;
    L->m_col_indices.resize(L->m_nnz);
    L->m_vals.resize(L->m_nnz);

    // Compact U using the same in-place pass.
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

    ilu = std::make_shared<ilu_solve_direct<T>>(L, U, D, prm.solve);
  }

  /// Applies one damped ILU correction as a pre-smoother.
  /// `x <- x + damping * M^{-1} * (rhs - A x)`.
  void apply_pre(const csr::CSRMatrix<T> &A, const std::vector<T> &rhs,
                 std::vector<T> &x, std::vector<T> &tmp) const
  {
    residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    axpby(prm.damping, tmp, T(1), x);
  }

  /// Applies one damped ILU correction as a post-smoother.
  void apply_post(const csr::CSRMatrix<T> &A, const std::vector<T> &rhs,
                  std::vector<T> &x, std::vector<T> &tmp) const
  {
    residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    axpby(prm.damping, tmp, T(1), x);
  }

  /// Applies the undamped ILU preconditioner: `x = M^{-1} * rhs`.
  /// The output is resized to match rhs. Calls on distinct output vectors are safe.
  void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
    x = rhs;
    ilu->solve(x);
  }

private:
  /// The solver object that performs the forward/backward substitutions.
  std::shared_ptr<ilu_solve_direct<T>> ilu;
};