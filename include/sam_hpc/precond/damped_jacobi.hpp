#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

template <typename T>
struct damped_jacobi {
    /// Relaxation parameters
    struct params {
        /// Damping factor
        double damping;

        params(double damping = 0.72) : damping(damping) { }
    } prm;

    std::vector<T> dia;

    /// Construct smoother for the system matrix
    /**
     * \param A   the system matrix
     * \param prm Relaxation parameters
     */
    damped_jacobi(const CSRMatrix<T> &A, const params &prm = params()) : prm(prm), dia(diagonal<diagonalType::inverse>(A)) { }

    /// Apply pre-relaxation
    /**
     * \param A   System matrix.
     * \param rhs Right-hand side.
     * \param x   Solution vector.
     * \param tmp Scratch vector.
     * \param prm Relaxation parameters.
     */
    void apply_pre(const CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
        residual(rhs, A, x, tmp);
        vmul(prm.damping, dia, tmp, T(1), x);
    }

    /// Apply post-relaxation
    /**
     * \param A   System matrix.
     * \param rhs Right-hand side.
     * \param x   Solution vector.
     * \param tmp Scratch vector.
     * \param prm Relaxation parameters.
     */
    void apply_post(const CSRMatrix<T>& A, const std::vector<T>& rhs, std::vector<T>& x, std::vector<T>& tmp) const {
        residual(rhs, A, x, tmp);
        vmul(prm.damping, dia, tmp, T(1), x);
    }

    void apply(const CSRMatrix<T>&, const std::vector<T>& rhs, std::vector<T>& x) const {
        vmul(T(1), dia, rhs, T(0), x);
    }
};
