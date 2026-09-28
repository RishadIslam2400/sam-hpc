#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/cuthill_mckee.hpp"

/// Direct solver that uses skyline LU factorization.
template <typename T, class ordering = cuthill_mckee<T, false>>
class skyline_lu {
public:
    static size_t coarse_enough() {
        return 3000;
    }

    skyline_lu(const CSRMatrix<T> &A) : n(A.m_rows), perm(n), ptr(n + 1, 0), D(n, T(0)), y(n) {
        // Find the permutation for the ordering.
        ordering::get(A, perm);

        // Get inverse permutation to map from original index to permuted index
        std::vector<int> invperm(n);
        for(int i = 0; i < n; ++i) 
            invperm[perm[i]] = i;

        /* Let us find how large the rows of L and the columns of U should
         * be. Provisionally, we will store in ptr[i] the minimum required
         * height of column i over the diagonal, and length of row i below
         * the diagonal. The value(i,j) in the reordered matrix will be
         * the same as the value(perm[i],perm[j]) in the original matrix;
         * or, the value(i,j) in the original matrix will be the same as
         * value(invperm[i],invperm[j]) in the reordered matrix.
         */

        // Traverse the matrix finding nonzero elements
        for(int i = 0; i < n; ++i) {
            for(auto a = A.rowBegin(i); a; ++a) {
                int j = a.col();
                T v = a.value();

                // Find the index at the permuted matrix
                int newi = invperm[i]; // row index
                int newj = invperm[j]; // col index

                if (v != T(0)) {
                    // If the element is in the lower triangular part
                    if (newi > newj) {
                        // row newi needs length at least newi - newj
                        if (ptr[newi] < newi - newj) 
                            ptr[newi] = newi - newj;
                    } else if (newi < newj) {
                        // column newj needs height at least newj - newi
                        if (ptr[newj] < newj - newi) 
                            ptr[newj] = newj - newi;
                    }
                }
            }
        }

        // Transform ptr so that it doesn't contain the required lengths
        // and heights, but the indexes to the entries
        {
            int last = 0;
            // perform a cumulative sum
            for(int i = 1; i <= n; ++i) {
                int tmp = ptr[i];
                ptr[i] = ptr[i - 1] + last;
                last = tmp;
            }
        }

        // Allocate variables for skyline format entries
        // ptr[n] contains the total number of off-diagonal elements
        L.resize(ptr.back(), T(0));
        U.resize(ptr.back(), T(0));

        // And finally traverse again the CSR matrix, copying its entries
        // into the correct places in the skyline format
        for(int i = 0; i < n; ++i) {
            for(auto a = A.rowBegin(i); a; ++a) {
                int  j = a.col();
                T v = a.value();

                int newi = invperm[i];
                int newj = invperm[j];

                if (v != T(0)) {
                    // upper triangular
                    if (newi < newj) {
                        U[ptr[newj + 1] + newi - newj] = v;
                    } else if (newi == newj) { /* diagonal */
                        D[newi] = v;
                    } else /* newi > newj */ {
                        L[ptr[newi + 1] + newj - newi] = v;
                    }
                }
            }
        }

        factorize();
    }

    void operator()(const std::vector<T> &rhs, std::vector<T> &x) const {
        // y = L^-1 * perm[rhs] ;
        // y = U^-1 * y ;
        // x = invperm[y];

        for(int i = 0; i < n; ++i) {
            T sum;
            sum = rhs[perm[i]];
            for(int k = ptr[i], j = i - ptr[i + 1] + k; k < ptr[i + 1]; ++k, ++j)
                sum -= L[k] * y[j];

            y[i] = D[i] * sum;
        }

        for(int j = n - 1; j >= 0; --j) {
            for(int k = ptr[j], i = j - ptr[j + 1] + k; k < ptr[j + 1]; ++k, ++i)
                y[i] -= U[k] * y[j];

        }

        for(int i = 0; i < n; ++i) 
            x[perm[i]] = y[i];
    }

    // TODO: implement bytes method
    /* size_t bytes() const {

    } */
private:
    int n;
    std::vector<ptrdiff_t> perm;
    std::vector<int> ptr;
    std::vector<T> L;
    std::vector<T> U;
    std::vector<T> D;

    mutable std::vector<T> y;

    /*
     * Perform and in-place LU factorization of a skyline matrix by Crout's
     * algorithm. The diagonal of U contains the 1's.
     * The equivalent MATLAB code for a full matrix would be:
     * for k=1:n-1
     *   A(1,k+1)=A(1,k+1)/A(1,1);
     *   for i=2:k
     *     sum=A(i,k+1);
     *       for j=1:i-1
     *         sum=sum-A(i,j)*A(j,k+1);
     *       end;
     *       A(i,k+1)=sum/A(i,i);
     *   end
     *   for i=2:k
     *     sum=A(k+1,i);
     *     for j=1:i-1
     *       sum=sum-A(j,i)*A(k+1,j);
     *     end;
     *     A(k+1,i)=sum;
     *   end
     *   sum=A(k+1,k+1);
     *   for i=1:k
     *     sum=sum-A(k+1,i)*A(i,k+1);
     *   end
     *   A(k+1,k+1)=sum;
     * end
     */
    void factorize() {
        assert(D[0] != T(0) && "Zero diagonal in skyline_lu");
        D[0] = 1 / D[0];

        // In each iteration k, the algorithm computes the final values for:
        // 1. Column k+1 of the U matrix.
        // 2. Row k+1 of the L matrix.
        // 3. The diagonal element L(k+1, k+1).
        for(int k = 0; k < n - 1; ++k) {
            // check whether A(1, k+1) lies within the skyline structure
            if (ptr[k + 1] + k + 1 == ptr[k + 2]) {
                U[ptr[k + 1]] = D[0] * U[ptr[k + 1]];
            }

            // Compute column k + 1 of U
            int indexEntry = ptr[k + 1];
            int iBeginCol  = k + 1 - ptr[k + 2] + ptr[k + 1];
            for(int i = iBeginCol; i <= k; ++indexEntry, ++i) {
                if (i == 0) 
                    continue;

                T sum = U[indexEntry]; // this is element U(i, k + 1)

                // Multiply row i of L and Column k + 1 of U
                int jBeginRow  = i - ptr[i + 1] + ptr[i];
                int jBeginMult = std::max(iBeginCol, jBeginRow);

                int indexL = ptr[i] + jBeginMult - jBeginRow;
                int indexU = ptr[k + 1] + jBeginMult - iBeginCol;
                for(int j = jBeginMult; j < i; ++j, ++indexL, ++indexU)
                    sum -= L[indexL] * U[indexU];

                U[indexEntry] = D[i] * sum;
            }

            // Compute row k + 1 of L
            indexEntry = ptr[k + 1];
            int jBeginRow = k + 1 - ptr[k + 2] + ptr[k + 1];
            for(int i = iBeginCol; i <= k; ++indexEntry, ++i) {
                if (i == 0) 
                    continue;

                T sum = L[indexEntry]; // this is the element L(k + 1, i)

                // Multiply row k+1 of L and column i of U
                int jBeginCol  = i - ptr[i + 1] + ptr[i];
                int jBeginMult = std::max(jBeginCol, jBeginRow);

                int indexL = ptr[k + 1] + jBeginMult - jBeginRow;
                int indexU = ptr[i] + jBeginMult - jBeginCol;

                for(int j = jBeginMult; j < i; ++j, ++indexL, ++indexU)
                    sum -= L[indexL] * U[indexU];

                L[indexEntry] = sum;
            }

            // Find element in diagonal
            T sum = D[k + 1];
            for(int j = ptr[k + 1]; j < ptr[k + 2]; ++j)
                sum -= L[j] * U[j];

            assert(sum != T(0) && "Zero sum in skyline_lu factorization");

            D[k + 1] = 1 / sum;
        }
    }
};