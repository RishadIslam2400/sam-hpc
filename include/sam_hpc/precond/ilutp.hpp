#pragma once

#include <eigen3/Eigen/Sparse>
#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

template<typename T>
struct ilutp {
    struct params {
        /// The drop tolerance. Entries with magnitude smaller than this tolerance
        /// relative to the row's norm are dropped. A typical value is 1e-4.
        T droptol;

        /// The fill factor. Controls the amount of memory allocated for the factors.
        /// A value of 1 means no extra fill-in is allocated.
        int fill_factor;

        /// Damping factor.
        T damping;

        params() : droptol(1e-4), fill_factor(10), damping(1.0) {}
    } prm;

private:
    Eigen::IncompleteLUT<T> m_ilu;
    size_t m_matrix_size;

public:
    ilutp(const CSRMatrix<T> &A, const params &prm) : prm(prm), m_matrix_size(A.m_rows) {
        std::vector<int> row_ptr(A.m_row_pointers.begin(), A.m_row_pointers.end());
        std::vector<int> col_ind(A.m_col_indices.begin(), A.m_col_indices.end());

        Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor>> eigen_A(
            static_cast<int>(A.m_rows),
            static_cast<int>(A.m_cols),
            static_cast<int>(A.m_nnz),
            row_ptr.data(),
            col_ind.data(),
            const_cast<T*>(A.m_vals.data())
        );

        m_ilu.setDroptol(prm.droptol);
        m_ilu.setFillfactor(prm.fill_factor);
        m_ilu.analyzePattern(eigen_A);
        m_ilu.factorize(eigen_A);

        if(m_ilu.info() != Eigen::Success) {
            throw std::runtime_error("Eigen ILUTP factorization failed.");
        }
    }

    void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_rhs(rhs.data(), m_matrix_size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_x(x.data(), m_matrix_size);

        eigen_x = m_ilu.solve(eigen_rhs);

        if (m_ilu.info() != Eigen::Success) {
             throw std::runtime_error("Eigen ILU solve failed.");
        }
    }

    void apply_pre(const CSRMatrix<T> &A, const std::vector<T> &rhs,
                   std::vector<T> &x, std::vector<T> &tmp) const
    {
        residual(rhs, A, x, tmp);
        apply(tmp, tmp);
        axpby(prm.damping, tmp, T(1), x);
    }

    void apply_post(const CSRMatrix<T> &A, const std::vector<T> &rhs,
                    std::vector<T> &x, std::vector<T> &tmp) const
    {
        residual(rhs, A, x, tmp);
        apply(tmp, tmp);
        axpby(prm.damping, tmp, T(1), x);
    }
};