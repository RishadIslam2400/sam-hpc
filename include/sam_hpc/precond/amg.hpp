#pragma once

#include <list>

#include "sam_hpc/precond/skyline_lu.hpp"
#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

/// Algebraic multigrid hierarchy for a square CSR system matrix.
/// \tparam T Value type of matrices and vectors.
/// \tparam Coarsening Coarsening policy used to build transfer operators.
/// \tparam Relax Relaxation policy used on non-direct levels.
template <typename T, template <class> class Coarsening, template <class> class Relax>
class amg {
public:
  typedef Coarsening<T> coarsening_type;
  typedef Relax<T> relax_type;

  /// AMG and component-specific parameters.
  struct params {
    typedef typename coarsening_type::params coarsening_params;
    typedef typename relax_type::params relax_params;

    coarsening_params coarsening;
    relax_params relax;

    /// Stop coarsening when the next level has at most this many rows.
    unsigned coarse_enough;

    /// Use a direct solver when coarsening reaches `coarse_enough`.
    bool direct_coarse;

    /// Maximum hierarchy depth.
    unsigned max_levels;

    /// Number of pre-smoothing steps per cycle.
    unsigned npre;

    /// Number of post-smoothing steps per cycle.
    unsigned npost;

    /// Number of recursive visits at each non-coarsest level.
    unsigned ncycle;

    /// Number of cycles performed by apply(); zero makes apply() the identity.
    unsigned pre_cycles;

    /// Retain transfer operators so rebuild() can reuse the hierarchy structure.
    bool allow_rebuild;

    params() : coarse_enough(skyline_lu<T>::coarse_enough()), direct_coarse(true),
               max_levels(std::numeric_limits<unsigned>::max()), npre(1), npost(1),
               ncycle(1), pre_cycles(1), allow_rebuild(true) { }
  } prm;

  /// Builds the hierarchy from a copied, row-sorted matrix.
  /// @param M Square system matrix in CSR format.
  /// @param p AMG parameters.
  amg(const csr::CSRMatrix<T>& M, const params& p = params()) : prm(p) {
    auto A = std::make_shared<csr::CSRMatrix<T>>(M);
    A->sortRows();
    do_init(A);
  }

  /// Builds the hierarchy without copying the input matrix. The matrix is retained as
  /// the finest level and must outlive any external mutation that would invalidate the
  /// hierarchy. Its rows must already be sorted.
  /// @param A Shared pointer to the square system matrix.
  /// @param p AMG parameters.
  amg(std::shared_ptr<csr::CSRMatrix<T>> A, const params &p = params()) : prm(p) {
    do_init(A);
  }

  /// Rebuilds the numerical hierarchy using reused transfer operators.
  /// This requires `allow_rebuild`. Coarse-grid structure is retained, while smoothers,
  /// direct factors, and Galerkin operators are recomputed. M is copied and sorted.
  void rebuild(const csr::CSRMatrix<T>& M) {
    auto A = std::make_shared<csr::CSRMatrix<T>>(M);
    A->sortRows();
    rebuild(A);
  }

  /// Rebuilds from a shared matrix using the existing transfer operators.
  /// The matrix must have the original dimensions and sorted rows. Only its numerical
  /// values may differ from the original setup.
  void rebuild(std::shared_ptr<csr::CSRMatrix<T>> A) {
    assert(prm.allow_rebuild && "allow_rebuild is not set");
    assert((A->m_rows == system_matrix().m_rows && A->m_rows == A->m_cols) && "Matrix dimensions differ from original ones!");

    coarsening_type C(prm.coarsening);
    for (auto & level : levels) {
      A = level.rebuild(A, C, prm);
    }
  }

  /// Performs one multigrid cycle from the current solution estimate.
  /// @param rhs Right-hand side vector.
  /// @param x Solution vector, updated in place and not resized.
  void cycle(const std::vector<T> &rhs, std::vector<T> &x) const {
    cycle(levels.begin(), rhs, x);
  }

  /// Applies the configured number of cycles from a zero initial guess.
  /// @param rhs Right-hand side vector.
  /// @param x Output vector; it must have the system dimension when preconditioning is
  ///          enabled and is assigned rhs when `pre_cycles` is zero.
  void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
    if (prm.pre_cycles) {
      std::fill(x.begin(), x.end(), T(0));
      for (unsigned i = 0; i < prm.pre_cycles; ++i)
        cycle(rhs, x);
    } else {
      x = rhs;
    }
  }

  /// Returns the system matrix from the finest level.
  std::shared_ptr<csr::CSRMatrix<T>> system_matrix_ptr() const {
    return levels.front().A;
  }

  const csr::CSRMatrix<T>& system_matrix() const {
    return *system_matrix_ptr();
  }

private:
  /// State and operators associated with one hierarchy level.
  struct level {
    size_t m_rows, m_nonzeros;

    // f and u hold coarse-grid residuals and corrections during a cycle.
    mutable std::vector<T> f;
    mutable std::vector<T> u;
    mutable std::vector<T> t;

    std::shared_ptr<csr::CSRMatrix<T>> A;
    std::shared_ptr<csr::CSRMatrix<T>> P;
    std::shared_ptr<csr::CSRMatrix<T>> R;

    // Aliases retained for numerical rebuilds when enabled.
    std::shared_ptr<csr::CSRMatrix<T>> bP;
    std::shared_ptr<csr::CSRMatrix<T>> bR;

    std::shared_ptr<skyline_lu<T>> solve;
    std::shared_ptr<relax_type> relax;

    level() : m_rows(0), m_nonzeros(0) {}
    /// Constructs a level solved by relaxation.
    level(std::shared_ptr<csr::CSRMatrix<T>>& A, params& prm) 
      : m_rows(A->m_rows), m_nonzeros(A->m_nnz), f(m_rows, T(0)), u(m_rows, T(0)), t(m_rows, T(0))
    {
      this->A = A;
      relax = std::make_shared<relax_type>(*A, prm.relax);
    }

    /// Builds transfer operators and returns the next Galerkin operator.
    std::shared_ptr<csr::CSRMatrix<T>> step_down(std::shared_ptr<csr::CSRMatrix<T>> A,
                                                 coarsening_type& C, bool allow_rebuild)
    {
      std::shared_ptr<csr::CSRMatrix<T>> P, R;

      try {
        std::tie(P, R) = C.transfer_operators(*A);
      } catch (const std::runtime_error& e) {
        // A null result indicates that the coarsening policy could not produce another
        // level; in that case this level remains the coarsest relaxed level.
        return std::shared_ptr<csr::CSRMatrix<T>>();
      }

      P->sortRows();
      R->sortRows();

      if (allow_rebuild) {
        bP = P;
        bR = R;
      }

      this->P = P;
      this->R = R;

      A = C.coarse_operator(*A, *P, *R);
      A->sortRows();

      return A;
    }

    /// Makes this level the direct-solve coarsest level.
    void create_coarse(std::shared_ptr<csr::CSRMatrix<T>> A, bool single_level) {
      m_rows = A->m_rows;
      m_nonzeros = A->m_nnz;

      u.resize(m_rows, T(0));
      f.resize(m_rows, T(0));

      solve = std::make_shared<skyline_lu<T>>(*A);
      if (single_level) {
        this->A = A;
      }
    }

    /// Rebuilds this level's numerical data and returns the next coarse matrix.
    std::shared_ptr<csr::CSRMatrix<T>> rebuild(std::shared_ptr<csr::CSRMatrix<T>> A,
                                               const coarsening_type& C, const params& prm)
    {
      if (this->A) {
        this->A = A;
      }

      if (relax) {
        relax = std::make_shared<relax_type>(*A, prm.relax);
      }

      if (solve) {
        solve = std::make_shared<skyline_lu<T>>(*A);
      }

      if (bP && bR) {
        A = C.coarse_operator(*A, *bP, *bR);
        A->sortRows();
      }

      return A;
    }

    size_t rows() const {
      return m_rows;
    }

    size_t nonzeros() const {
      return m_nonzeros;
    }
  };

  typedef typename std::list<level>::const_iterator level_iterator;
  std::list<level> levels;

  /// Builds the hierarchy from finest to coarsest.
  void do_init(std::shared_ptr<csr::CSRMatrix<T>> A) {
    assert(A->m_rows == A->m_cols && "Matrix should be square!");
    bool direct_coarse_solve = true;

    coarsening_type C(prm.coarsening);

    while(A->m_rows > prm.coarse_enough) {
      levels.push_back(level(A, prm));

      if (levels.size() >= prm.max_levels)
        break;

      A = levels.back().step_down(A, C, prm.allow_rebuild);
      if (!A) {
        direct_coarse_solve = false;
        break;
      }
    }

    if (!A || A->m_rows > prm.coarse_enough) {
      direct_coarse_solve = false;
    }

    if (direct_coarse_solve) {
      if (prm.direct_coarse) {
        level l;
        l.create_coarse(A, levels.empty());
        levels.push_back(l);
      } else {
        levels.push_back(level(A, prm));
      }
    }
  }

  /// Recursively applies one multigrid cycle at level `lvl`.
  void cycle(level_iterator lvl, const std::vector<T>& rhs, std::vector<T>& x) const {
    level_iterator nxt = lvl, end = levels.end();
    ++nxt;

    if (nxt == end) {
      if (lvl->solve) {
        (*lvl->solve)(rhs, x);
      } else {
        for(size_t i = 0; i < prm.npre;  ++i)
          lvl->relax->apply_pre(*lvl->A, rhs, x, lvl->t);
        for(size_t i = 0; i < prm.npost; ++i)
          lvl->relax->apply_post(*lvl->A, rhs, x, lvl->t);
      }
    } else {
      for (size_t j = 0; j < prm.ncycle; ++j) {
        for(size_t i = 0; i < prm.npre; ++i)
          lvl->relax->apply_pre(*lvl->A, rhs, x, lvl->t);

        residual(rhs, *lvl->A, x, lvl->t);
        spmv(T(1), *lvl->R, lvl->t, T(0), nxt->f);

        // Solve the coarse correction from zero before prolongating it.
        std::fill(nxt->u.begin(), nxt->u.end(), T(0));
        cycle(nxt, nxt->f, nxt->u);

        spmv(T(1), *lvl->P, nxt->u, T(1), x);

        for(size_t i = 0; i < prm.npost; ++i)
          lvl->relax->apply_post(*lvl->A, rhs, x, lvl->t);
      }
    }
  }

  template <class B, template <class> class C, template <class> class R>
  friend std::ostream& operator<<(std::ostream &os, const amg<B, C, R> &a);
};

/// Writes an AMG hierarchy summary to an output stream.
/// This is currently a no-op.
template <class B, template <class> class C, template <class> class R>
std::ostream& operator<<(std::ostream &os, const amg<B, C, R> &a) {
  (void)a;
  return os;
}