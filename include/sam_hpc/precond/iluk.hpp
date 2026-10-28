#pragma once

#include "sam_hpc/precond/ilu_solve.hpp"

#include <queue>

/// Incomplete LU factorization with level-based fill control.
///
/// Original entries have level zero. Eliminating through entries with levels `a` and `b`
/// assigns a new entry level `max(a, b) + 1`. Existing entries keep the minimum level
/// found, while new entries above `k` are discarded.
///
/// L is strictly lower with an implicit unit diagonal, U is strictly upper, and D stores
/// the inverse pivot. Small pivots are replaced by a signed `1e-10` value; no pivoting is
/// performed.
template <typename T>
struct iluk {
  /// Relaxation parameters.
  struct params {
    /// Maximum allowed fill level.
    int k;

    /// Smoothing weight used by apply_pre and apply_post.
    T damping;

    /// Parameters for the iterative triangular solver.
    typename ilu_solve_iterative<T>::params solve;

    params() : k(1), damping(1) {}
  } prm;

public:
  /// Constructs and factors the ILU(k) preconditioner.
  /// @param A Square input matrix. It is not retained.
  /// @param prm Factorization and solver parameters.
  iluk(const csr::CSRMatrix<T> &A, const params &prm = params()) : prm(prm) {
    const size_t n = A.m_rows;
    size_t Anz = A.m_nnz;

    // Reserve storage for the factor patterns.
    std::vector<size_t> Lptr; 
    Lptr.reserve(n + 1); 
    Lptr.push_back(0);
    std::vector<size_t> Lcol; 
    Lcol.reserve(Anz / 3);
    std::vector<T> Lval; 
    Lval.reserve(Anz / 3);

    std::vector<size_t> Uptr; 
    Uptr.reserve(n + 1);
    Uptr.push_back(0);
    std::vector<size_t> Ucol;
    Ucol.reserve(Anz / 3);
    std::vector<T> Uval; 
    Uval.reserve(Anz / 3);

    // Levels of U entries are needed when generating fill.
    std::vector<int> Ulev; 
    Ulev.reserve(Anz / 3);

    // A missing diagonal remains zero.
    std::shared_ptr<std::vector<T>> D = std::make_shared<std::vector<T>>(n);

    // Reused workspace for one row at a time.
    sparse_vector w(n, prm.k);
    for(size_t i = 0; i < n; ++i) {
      w.reset(i);

      // Original entries start at level zero.
      for(auto a = A.rowBegin(i); a; ++a) {
        w.add(a.col(), a.value(), 0);
      }

      // Eliminate lower entries in increasing column order.
      while(!w.q.empty()) {
        nonzero &a = w.next_nonzero();
        a.val = a.val * (*D)[a.col];

        for(size_t j = Uptr[a.col], e = Uptr[a.col + 1]; j < e; ++j) {
          int lev = std::max(a.lev, Ulev[j]) + 1;
          w.add(Ucol[j], -a.val * Uval[j], lev);
        }
      }

      w.sort();

      // Store the completed row in L, D, and U.
      for(const nonzero &e : w.nz) {
        if (e.col < i) {
          Lcol.push_back(e.col);
          Lval.push_back(e.val);
        } else if (e.col == i) {
          const T tolerance = T(1e-10);
          T pivot = e.val;

          if (std::abs(pivot) < tolerance) {
            pivot = (pivot >= 0) ? tolerance : -tolerance;
          }

          (*D)[i] = 1 / pivot;
        } else {
          Ucol.push_back(e.col);
          Uval.push_back(e.val);
          Ulev.push_back(e.lev);
        }
      }

      Lptr.push_back(Lcol.size());
      Uptr.push_back(Ucol.size());
    }

    ilu = std::make_shared<ilu_solve_iterative<T>>(
            std::make_shared<csr::CSRMatrix<T>>(n, n, Lval, Lptr, Lcol),
            std::make_shared<csr::CSRMatrix<T>>(n, n, Uval, Uptr, Ucol),
            D, prm.solve
          );

  }

  /// Applies the approximate ILU(k) preconditioner: `x = M^{-1} * rhs`.
  ///
  /// The output is resized to match rhs. The iterative solver is not thread-safe.
  void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
      x = rhs;
      ilu->solve(x);
  }

  /// Applies one damped ILU(k) correction as a pre-smoother.
  void apply_pre(const csr::CSRMatrix<T> &A, const std::vector<T> &rhs,
                  std::vector<T> &x, std::vector<T> &tmp) const
  {
      residual(rhs, A, x, tmp);
      ilu->solve(tmp);
      axpby(prm.damping, tmp, T(1), x);
  }

  void apply_post(const csr::CSRMatrix<T> &A, const std::vector<T> &rhs,
                  std::vector<T> &x, std::vector<T> &tmp) const
  {
      residual(rhs, A, x, tmp);
      ilu->solve(tmp);
      axpby(prm.damping, tmp, T(1), x);
  }

private:
  std::shared_ptr<ilu_solve_iterative<T>> ilu;

  /// Workspace entry containing a value and its fill level.
  struct nonzero {
    ptrdiff_t col;
    T val;
    int lev;

    nonzero() : col(-1) {}

    nonzero(ptrdiff_t col, const T &val, int lev) : col(col), val(val), lev(lev) {}

    friend bool operator<(const nonzero &a, const nonzero &b) {
      return a.col < b.col;
    }
  };

  /// Sparse row workspace with constant-time column lookup and a min-heap of lower
  /// entries awaiting elimination.
  struct sparse_vector {
    struct comp_indices {
      const std::deque<nonzero> &nz;

      comp_indices(const std::deque<nonzero> &nz) : nz(nz) {}

      bool operator()(int a, int b) const {
        return nz[a].col > nz[b].col;
      }
    };

    typedef std::priority_queue<int, std::vector<int>, comp_indices> priority_queue;

    int lfil;
    std::deque<nonzero> nz;
    std::vector<ptrdiff_t> idx;
    priority_queue q;
    ptrdiff_t dia;

    /// @param n Number of matrix columns.
    /// @param lfil Maximum fill level.
    sparse_vector(size_t n, int lfil) : lfil(lfil), idx(n, -1), q(comp_indices(nz)), dia(0) {}

    /// Adds or updates an entry. The level limit applies only to new entries.
    void add(ptrdiff_t col, const T &val, int lev) {
      if (idx[col] < 0) {
        if (lev <= lfil) {
          int p = nz.size();
          idx[col] = p;
          nz.push_back(nonzero(col, val, lev));
          if (col < dia) q.push(p);
        }
      } else {
        nonzero &a = nz[idx[col]];
        a.val += val;
        a.lev = std::min(a.lev, lev);
      }
    }

    typename std::deque<nonzero>::iterator begin() {
      return nz.begin();
    }

    typename std::deque<nonzero>::iterator end() {
      return nz.end();
    }

    /// Returns the next lower entry in column order.
    nonzero& next_nonzero() {
      int p = q.top();
      q.pop();
      return nz[p];
    }

    void sort() {
      std::sort(nz.begin(), nz.end());
    }

    /// Clears the workspace for a new row.
    /// @param d Diagonal column index.
    void reset(ptrdiff_t d) {
      for(const nonzero &e : nz)
        idx[e.col] = -1;
      nz.clear();
      dia = d;
    }
  };
};