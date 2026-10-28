#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/cuthill_mckee.hpp"

/// Direct solver based on skyline LU factorization.
/// @tparam ordering of matrix
template <typename T, class ordering = cuthill_mckee<T, false>>
class skyline_lu {
public:
  /// Coarse-grid size at which AMG may switch to this direct solver.
  static size_t coarse_enough() {
    return 3000;
  }

  /// Reorders, builds the skyline envelope, and factors A.
  /// @param A Square matrix. Explicit zeros are not stored in the envelope.
  skyline_lu(const csr::CSRMatrix<T> &A) : n(A.m_rows), perm(n), ptr(n + 1, 0), D(n, T(0)), y(n) {
    ordering::get(A, perm);

    // Map original indices to their positions in the permuted matrix.
    std::vector<int> invperm(n);
    for(int i = 0; i < n; ++i) 
      invperm[perm[i]] = i;

    // Determine the maximum distance from the diagonal required by each envelope
    // slot. The same width is used for row i of L and column i of U.
    for(int i = 0; i < n; ++i) {
      for(auto a = A.rowBegin(i); a; ++a) {
        int j = a.col();
        T v = a.value();
        int newi = invperm[i];
        int newj = invperm[j];

        if (v != T(0)) {
          if (newi > newj) {
            // lower triangular entry
            if (ptr[newi] < newi - newj) 
              ptr[newi] = newi - newj;
          } else if (newi < newj) {
            // upper triangular portion
            if (ptr[newj] < newj - newi) 
              ptr[newj] = newj - newi;
          }
        }
      }
    }

    // Convert widths to offsets. ptr[i] is the start of slot i and ptr[n] is the
    // total number of stored off-diagonal entries.
    {
      int last = 0;
      for(int i = 1; i <= n; ++i) {
        int tmp = ptr[i];
        ptr[i] = ptr[i - 1] + last;
        last = tmp;
      }
    }

    L.resize(ptr.back(), T(0));
    U.resize(ptr.back(), T(0));

    // Copy the permuted matrix into its skyline slots.
    for(int i = 0; i < n; ++i) {
      for(auto a = A.rowBegin(i); a; ++a) {
        int  j = a.col();
        T v = a.value();

        int newi = invperm[i];
        int newj = invperm[j];

        if (v != T(0)) {
          if (newi < newj) {
            U[ptr[newj + 1] + newi - newj] = v;
          } else if (newi == newj) {
            D[newi] = v;
          } else {
            L[ptr[newi + 1] + newj - newi] = v;
          }
        }
      }
    }

    factorize();
  }

  /// Solves A x = rhs in the original numbering.
  /// @param rhs  Right-hand side, length n.
  /// @param x    Output, must already have length n; fully overwritten. rhs is consumed
  ///             into the internal scratch before x is written, so x may alias rhs.
  /// Not thread-safe: the scratch vector y is a mutable member shared by all callers.
  void operator()(const std::vector<T> &rhs, std::vector<T> &x) const {
    for(int i = 0; i < n; ++i) {
      T sum;
      sum = rhs[perm[i]];
      for(int k = ptr[i], j = i - ptr[i + 1] + k; k < ptr[i + 1]; ++k, ++j)
        sum -= L[k] * y[j];

      y[i] = D[i] * sum;
    }

    // U is stored by columns, so each completed y[j] updates the rows above it.
    for(int j = n - 1; j >= 0; --j) {
      for(int k = ptr[j], i = j - ptr[j + 1] + k; k < ptr[j + 1]; ++k, ++i)
        y[i] -= U[k] * y[j];

    }

    for(int i = 0; i < n; ++i) 
      x[perm[i]] = y[i];
  }

private:
  // All storage uses permuted indices. L and U share envelope slots; U's diagonal is
  // implicit and D stores 1 / L(i, i).
  int n;
  std::vector<ptrdiff_t> perm;
  std::vector<int> ptr;
  std::vector<T> L;
  std::vector<T> U;
  std::vector<T> D;

  mutable std::vector<T> y;

  // In-place Crout factorization. On entry, L, U, and D contain the permuted matrix;
  // on exit they contain the factors described above.
  void factorize() {
    assert(D[0] != T(0) && "Zero diagonal in skyline_lu");
    D[0] = 1 / D[0];

    for(int k = 0; k < n - 1; ++k) {
      // Handle row 0 of the new column of U
      if (ptr[k + 1] + k + 1 == ptr[k + 2]) {
        U[ptr[k + 1]] = D[0] * U[ptr[k + 1]];
      }

      // Compute column k + 1 of U
      int indexEntry = ptr[k + 1];
      int iBeginCol  = k + 1 - ptr[k + 2] + ptr[k + 1];
      for(int i = iBeginCol; i <= k; ++indexEntry, ++i) {
        if (i == 0) 
          continue;

        T sum = U[indexEntry];

        int jBeginRow  = i - ptr[i + 1] + ptr[i];
        int jBeginMult = std::max(iBeginCol, jBeginRow);

        int indexL = ptr[i] + jBeginMult - jBeginRow;
        int indexU = ptr[k + 1] + jBeginMult - iBeginCol;
        for(int j = jBeginMult; j < i; ++j, ++indexL, ++indexU)
          sum -= L[indexL] * U[indexU];

        U[indexEntry] = D[i] * sum;
      }

      indexEntry = ptr[k + 1];
      int jBeginRow = k + 1 - ptr[k + 2] + ptr[k + 1];
      for(int i = iBeginCol; i <= k; ++indexEntry, ++i) {
        if (i == 0) 
          continue;

        T sum = L[indexEntry];

        int jBeginCol  = i - ptr[i + 1] + ptr[i];
        int jBeginMult = std::max(jBeginCol, jBeginRow);

        int indexL = ptr[k + 1] + jBeginMult - jBeginRow;
        int indexU = ptr[i] + jBeginMult - jBeginCol;

        for(int j = jBeginMult; j < i; ++j, ++indexL, ++indexU)
          sum -= L[indexL] * U[indexU];

        L[indexEntry] = sum;
      }

      T sum = D[k + 1];
      for(int j = ptr[k + 1]; j < ptr[k + 2]; ++j)
        sum -= L[j] * U[j];

      assert(sum != T(0) && "Zero sum in skyline_lu factorization");

      D[k + 1] = 1 / sum;
    }
  }
};