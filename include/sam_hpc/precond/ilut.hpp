#pragma once

#include <eigen3/Eigen/Sparse>
#include <tbb/enumerable_thread_specific.h>
#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/precond/ilu_solve.hpp"

namespace ilut_detail {

/// Eigen::IncompleteLUT keeps its factor and permutations protected; this exposes them
/// read-only, plus a way to free the factor once it has been copied out.
template <typename T>
struct IncompleteLUTAccess : Eigen::IncompleteLUT<T> {
  const typename Eigen::IncompleteLUT<T>::FactorType& factor() const { return this->m_lu; }
  const auto& perm() const { return this->m_P.indices(); }
  const auto& perm_inv() const { return this->m_Pinv.indices(); }
  void release_factor() { this->m_lu = typename Eigen::IncompleteLUT<T>::FactorType(); }
};

} // namespace ilut_detail

/// Eigen IncompleteLUT exposed through the common preconditioner interface.
template<typename T>
struct ilut {
  struct params {
    /// Drop tolerance. U uses a row-scaled threshold; L uses an absolute threshold.
    T droptol;

    /// Maximum fill factor used by Eigen to limit entries retained per row.
    int fill_factor;

    /// Smoothing weight used by apply_pre and apply_post.
    T damping;

    /// Use Eigen's serial solve instead of extracted CSR factors.
    bool eigen_solve;

    /// Parameters for the extracted triangular solver.
    typename ilu_solve_direct<T>::params solve;

    params() : droptol(1e-3), fill_factor(3), damping(1.0), eigen_solve(false) {
      solve.serial = true;
    }
  } prm;

private:
  ilut_detail::IncompleteLUTAccess<T> m_ilu;
  size_t m_matrix_size;

  // Permutations used to gather into and scatter out of Eigen's factor ordering.
  std::vector<size_t> m_perm;
  std::vector<size_t> m_perm_inv;
  std::shared_ptr<ilu_solve_direct<T>> m_solver;
  mutable tbb::enumerable_thread_specific<std::vector<T>> m_work;

  /// Converts Eigen's factor into the CSR convention used by ilu_solve_direct.
  void extract_factor() {
    const auto& lu = m_ilu.factor();
    const size_t n = m_matrix_size;
    std::vector<size_t> lp{0}, lc, up{0}, uc;
    std::vector<T> lv, uv;
    auto D = std::make_shared<std::vector<T>>(n, T(1));
    lc.reserve(lu.nonZeros() / 2); lv.reserve(lu.nonZeros() / 2);
    uc.reserve(lu.nonZeros() / 2); uv.reserve(lu.nonZeros() / 2);

    for (size_t i = 0; i < n; ++i) {
      bool has_diag = false;
      for (typename Eigen::IncompleteLUT<T>::FactorType::InnerIterator it(lu, i); it; ++it) {
        const size_t c = static_cast<size_t>(it.index());
        if (c < i) {
          lc.push_back(c);
          lv.push_back(it.value());
        } else if (c > i) {
          uc.push_back(c);
          uv.push_back(it.value());
        } else {
          (*D)[i] = T(1) / it.value();
          has_diag = true;
        }
      }
      assert(has_diag && "Eigen ILUT factor row without a diagonal entry.");
      (void)has_diag;
      lp.push_back(lc.size());
      up.push_back(uc.size());
    }

    auto L = std::make_shared<csr::CSRMatrix<T>>(n, n, std::move(lv), std::move(lp), std::move(lc));
    auto U = std::make_shared<csr::CSRMatrix<T>>(n, n, std::move(uv), std::move(up), std::move(uc));
    L->sortRows();
    U->sortRows();

    const auto& p = m_ilu.perm();
    const auto& pinv = m_ilu.perm_inv();
    m_perm.assign(p.data(), p.data() + n);
    m_perm_inv.assign(pinv.data(), pinv.data() + n);

    m_solver = std::make_shared<ilu_solve_direct<T>>(L, U, D, prm.solve);
  }

public:
  /// Factors A. Throws std::runtime_error if Eigen reports failure.
  /// @param A Square matrix with sorted CSR rows. Dimensions and nonzero count must fit
  ///        in Eigen's `int` index type. A is viewed only during construction.
  ilut(const csr::CSRMatrix<T> &A, const params &prm) : prm(prm), m_matrix_size(A.m_rows) {
    std::vector<int> row_ptr(A.m_row_pointers.begin(), A.m_row_pointers.end());
    std::vector<int> col_ind(A.m_col_indices.begin(), A.m_col_indices.end());

    Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor>> eigen_A(
      static_cast<int>(A.m_rows), static_cast<int>(A.m_cols), static_cast<int>(A.m_nnz),
      row_ptr.data(), col_ind.data(), const_cast<T*>(A.m_vals.data()));

    m_ilu.setDroptol(prm.droptol);
    m_ilu.setFillfactor(prm.fill_factor);
    m_ilu.analyzePattern(eigen_A);
    m_ilu.factorize(eigen_A);

    if(m_ilu.info() != Eigen::Success) {
      throw std::runtime_error("Eigen ILUT factorization failed.");
    }

    if (!prm.eigen_solve) {
      extract_factor();
      m_ilu.release_factor();
    }
  }

  /// Applies the incomplete-factor solve: `x = M^{-1} * rhs`.
  /// rhs and x may alias. The output must already have the system length.
  void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
    if (!prm.eigen_solve) {
      const size_t n = m_matrix_size;
      std::vector<T>& y = m_work.local();
      y.resize(n);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n, vector_grain),
        [&](const tbb::blocked_range<size_t>& r) {
          for (size_t j = r.begin(); j < r.end(); ++j)
            y[j] = rhs[m_perm[j]];
        }
      );
      m_solver->solve(y);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n, vector_grain),
        [&](const tbb::blocked_range<size_t>& r) {
          for (size_t j = r.begin(); j < r.end(); ++j)
            x[j] = y[m_perm_inv[j]];
        }
      );
      return;
    }

    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_rhs(rhs.data(), m_matrix_size);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> eigen_x(x.data(), m_matrix_size);

    eigen_x = m_ilu.solve(eigen_rhs);

    if (m_ilu.info() != Eigen::Success) {
      throw std::runtime_error("Eigen ILU solve failed.");
    }
  }

  /// Applies one damped ILUT correction as a pre-smoother.
  void apply_pre(const csr::CSRMatrix<T> &A, const std::vector<T> &rhs,
                 std::vector<T> &x, std::vector<T> &tmp) const
  {
    residual(rhs, A, x, tmp);
    apply(tmp, tmp);
    axpby(prm.damping, tmp, T(1), x);
  }

  void apply_post(const csr::CSRMatrix<T> &A, const std::vector<T> &rhs,
                  std::vector<T> &x, std::vector<T> &tmp) const
  {
    residual(rhs, A, x, tmp);
    apply(tmp, tmp);
    axpby(prm.damping, tmp, T(1), x);
  }
};