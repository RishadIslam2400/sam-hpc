#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/coarsening/galerkin.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/core/utilities.hpp"

/// Classical Ruge-Stuben coarsening with direct interpolation. The coarse grid is a
/// subset of the fine variables. C-points define the coarse grid, while F-points are
/// interpolated from their strong C-point dependencies. The resulting operators satisfy
/// `R = P^T` and `A_c = R A P`.
template <typename T>
struct ruge_stuben {
  /// Parameters controlling strength detection and interpolation truncation.
  struct params {
    /// Relative magnitude threshold for a strong off-diagonal connection.
    float eps_strong;

    /// Whether to drop small interpolation weights and rescale the survivors.
    bool do_trunc;

    /// Relative threshold used when truncation is enabled.
    float eps_trunc;

    params() : eps_strong(0.4f), do_trunc(true), eps_trunc(0.2f) {}
  } prm;

  ruge_stuben(const params& prm = params()) : prm(prm) {}

  /// Splits the variables into C/F points and builds direct interpolation.
  ///
  /// \param A Square system matrix. Its rows need not be sorted.
  /// \returns `(P, R)` where `P` is the prolongation operator and `R = P^T`.
  /// \throws std::runtime_error If no C-point is selected.
  std::tuple<std::shared_ptr<csr::CSRMatrix<T>>, std::shared_ptr<csr::CSRMatrix<T>>>
  transfer_operators(const csr::CSRMatrix<T>& A) const {
    const size_t n = A.m_rows;
    static const T epsilon = eps<T>(1);

    std::vector<char> cf(n, 'U');
    csr::CSRMatrix<char> S;
    connect(A, prm.eps_strong, S, cf);
    cfsplit(A, S, cf);

    // Number C-points in fine-grid order so coarse columns are deterministic.
    size_t nc = 0;
    std::vector<size_t> cidx(n);
    for (size_t i = 0; i < n; ++i) {
      if (cf[i] == 'C')
        cidx[i] = nc++;
    }

    if (!nc)
      throw std::runtime_error("The level is empty.");

    std::shared_ptr<csr::CSRMatrix<T>> P = std::make_shared<csr::CSRMatrix<T>>();
    P->m_rows = n;
    P->m_cols = nc;
    P->m_row_pointers.resize(n + 1, 0);

    std::vector<T> Amin, Amax;
    if (prm.do_trunc) {
      Amin.resize(n);
      Amax.resize(n);
    }

    // Count non-zero entries of each row of P.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // A C-point's row in P holds exactly one entry.
        if (cf[i] == 'C') {
          ++P->m_row_pointers[i + 1];
          continue;
        }

        size_t nnz = 0;
        if (prm.do_trunc) {
          T amin = T(0);
          T amax = T(0);
          for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
            if (!S.m_vals[j] || cf[A.m_col_indices[j]] != 'C')
              continue;

            // Collect most negative and positive entry from the row
            amin = std::min(amin, A.m_vals[j]);
            amax = std::max(amax, A.m_vals[j]);
          }

          // Calculate the truncation threshold for the current row.
          Amin[i] = (amin *= prm.eps_trunc);
          Amax[i] = (amax *= prm.eps_trunc);

          for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
            if (!S.m_vals[j] || cf[A.m_col_indices[j]] != 'C')
              continue;

            // Count the non-zero entries that survive the truncation
            if (A.m_vals[j] < amin || amax < A.m_vals[j]) {
              nnz++;
            }
          }
        } else {
          // Without truncation, count every non-zero entries that are C-points
          for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
            if (S.m_vals[j] && cf[A.m_col_indices[j]] == 'C') {
              nnz++;
            }
          }
        }
        P->m_row_pointers[i + 1] = nnz;
      }
    });

    P->m_nnz = P->scanRowSize();
    P->m_col_indices.resize(P->m_nnz, 0);
    P->m_vals.resize(P->m_nnz, T(0));

    // Compute the weights of the interpolation operator P
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t row_head = P->m_row_pointers[i];

        // For C-points, write the single entry in the corresponding column 
        if (cf[i] == 'C') {
          P->m_col_indices[row_head] = cidx[i];
          P->m_vals[row_head] = T(1);
          continue;
        }

        // For F-points, the value is consturcted using the weighted average of its
        // strong C-point neighbors.
        T dia = T(0);
        T a_num = T(0), a_den = T(0);
        T b_num = T(0), b_den = T(0);
        T d_neg = T(0), d_pos = T(0);  

        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          size_t c = A.m_col_indices[j];
          T v = A.m_vals[j];

          // Extract the diagonal
          if (c == i) {
            dia = v;
            continue;
          }

          if (v < T(0)) {
            // Sum all the negative entries
            a_num += v;
            if (S.m_vals[j] && cf[c] == 'C') {
              // Sum the negative entries that are strong and C-point
              a_den += v;
              // Mark the entry for truncation if the truncation condition holds
              if (prm.do_trunc && v >= Amin[i])
                d_neg += v;
            }
          } else {
            // Sum all the positive entries
            b_num += v;
            if (S.m_vals[j] && cf[c] == 'C') {
              // Sum the positive entries that are strong and C-point
              b_den += v;
              // Mark the entry for truncation if the truncation condition holds
              if (prm.do_trunc && v <= Amax[i])
                d_pos += v;
            }
          }
        }

        T cf_neg = 1;
        T cf_pos = 1;

        // Compute the rescaling ratio after truncation
        if (prm.do_trunc) {
          if (std::abs(a_den - d_neg) > epsilon) {
            cf_neg = std::abs(a_den) / std::abs(a_den - d_neg);
          }

          if (std::abs(b_den - d_pos) > epsilon) {
            cf_pos = std::abs(b_den) / std::abs(b_den - d_pos);
          }
        }

        // If there are positive off-diagonals but none of them is an interpolatory
        // (strong, C) coupling, lump all of them onto the diagonal.
        if (T(0) < b_num && std::abs(b_den) < epsilon) {
          dia += b_num;
        }

        // Form sign-specific interpolation factors, guarding singular scales.
        const bool has_dia = std::abs(dia) > epsilon;
        T alpha = (has_dia && std::abs(a_den) > epsilon) ? -cf_neg * a_num / (dia * a_den) : 0;
        T beta = (has_dia && std::abs(b_den) > epsilon) ? -cf_pos * b_num / (dia * b_den) : 0;

        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          size_t c = A.m_col_indices[j];
          T v = A.m_vals[j];

          if (!S.m_vals[j] || cf[c] != 'C')
            continue;
          if (prm.do_trunc && Amin[i] <= v && v <= Amax[i])
            continue;

          P->m_col_indices[row_head] = cidx[c];
          P->m_vals[row_head] = (v < T(0) ? alpha : beta) * v;
          ++row_head;
        }
      }
    });

    return std::make_tuple(P, std::make_shared<csr::CSRMatrix<T>>(transpose(*P)));
  }

  std::shared_ptr<csr::CSRMatrix<T>> coarse_operator(const csr::CSRMatrix<T>& A,
                                                     const csr::CSRMatrix<T>& P,
                                                     const csr::CSRMatrix<T>& R) const
  {
    return galerkin(A, P, R);
  }

 private:
  /// Builds the strength relation and initializes the C/F state. On return, `S.m_vals`
  /// is aligned with the input sparsity pattern. `S.m_row_pointers` and `S.m_col_indices`
  /// stores the transpose of relation. Variables with no significant off-diagonal
  /// connection are marked as F-points.
  static void connect(const csr::CSRMatrix<T>& A, float eps_strong,
                      csr::CSRMatrix<char>& S, std::vector<char>& cf)
  {
    const size_t n = A.m_rows;
    const size_t nnz = A.m_nnz;
    const T epsilon = eps<T>(1);

    S.m_rows = S.m_cols = n;
    S.m_row_pointers.resize(n + 1, 0);
    S.m_vals.resize(nnz, 0);

    // Detect strong off-diagonal connections independently in each row.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Find the largest off-diagonal magnitude in the row
        T a_max_abs = T(0);
        for (auto a = A.rowBegin(i); a; ++a) {
          if (a.col() != i)
            a_max_abs = std::max(a_max_abs, std::abs(a.value()));
        }

        // If the maximum magnitude in the row is less than machine epsilon, then mark as
        // F point and continue
        if (a_max_abs < epsilon) {
          cf[i] = 'F';
          continue;
        }

        // Fix the strength threshold for the row
        a_max_abs *= eps_strong;
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          // Variable i strongly depends on variable j when abs(aij) >= eps * abs(amax)
          S.m_vals[j] = (A.m_col_indices[j] != i && std::abs(A.m_vals[j]) >= a_max_abs);
        }
      }
    });

    // Build row counts for the transposed strength relation to find the variables that 
    // depend on row i.
    for (size_t i = 0; i < nnz; ++i) {
      if (S.m_vals[i])
        ++(S.m_row_pointers[A.m_col_indices[i] + 1]);
    }
    S.scanRowSize();
    S.m_col_indices.resize(S.m_row_pointers[n]);

    // Scatter each strong connection into its transposed row.
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
        if (S.m_vals[j])
          S.m_col_indices[S.m_row_pointers[A.m_col_indices[j]]++] = i;
      }
    }

    // Restore row pointers after using them as insertion cursors.
    std::rotate(S.m_row_pointers.data(), S.m_row_pointers.data() + n, S.m_row_pointers.data() + n + 1);
    S.m_row_pointers[0] = 0;
  }

  /// Splits variables into C-points and F-points using the strength relation.
  /// @param A The system matrix.
  /// @param S The transposed strength matrix.
  /// @param cf [in,out] C/F state.
  ///
  /// On entry, `cf` contains `U` or `F`; on exit it contains only `C` and `F`.
  /// Ruge–Stüben heuristics:
  /// 1. Every F-point should strongly depend on at least one C-point.
  /// 2. The C-points should be roughly an independent set in the strength graph,
  ///    so the coarse grid is as small as possible.
  static void cfsplit(const csr::CSRMatrix<T>& A, const csr::CSRMatrix<char>& S,
                      std::vector<char>& cf)
  {
    const size_t n = A.m_rows;
    std::vector<size_t> lambda(n);

    size_t max_lambda = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t temp = 0;
      // Measure how useful variable i would be as a C-point:
      // lambda_i = (number of undecided dependents of i) + 2 * (number of F dependents of i)
      for (size_t j = S.m_row_pointers[i], e = S.m_row_pointers[i + 1]; j < e; ++j) {
        temp += (cf[S.m_col_indices[j]] == 'U' ? 1 : 2);
      }
      lambda[i] = temp;

      // Degree is the number of dependents of i. A dependent contributes at most 2, so
      // lambda_i can never exceed 2 * degree.
      const size_t deg = S.m_row_pointers[i + 1] - S.m_row_pointers[i];
      max_lambda = std::max(max_lambda, 2 * deg);
    }

    // Bucket queue by priority.
    const size_t nb = max_lambda + 1;
    std::vector<size_t> ptr(nb + 1, 0); // starting position of a bucket
    std::vector<size_t> cnt(nb, 0);     // number of unprocessed variables in each bucket
    std::vector<size_t> i2n(n);         // extract the variable at position i
    std::vector<size_t> n2i(n);         // extract the position of variable n

    // Counting sort and prefix sum to sort buckets in increasing size
    for (size_t i = 0; i < n; ++i) {
      ++ptr[lambda[i] + 1];
    }
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

    // Place the variables in their respective buckets
    for (size_t i = 0; i < n; ++i) {
      size_t lam = lambda[i];
      size_t idx = ptr[lam] + cnt[lam]++;
      i2n[idx] = i;
      n2i[i] = idx;
    }

    // Select candidates in decreasing priority order.
    for (size_t top = n; top-- > 0;) {
      size_t i = i2n[top];
      size_t lam = lambda[i];

      // With zero as top priority, no remaining dependency distinguishes the nodes.
      if (lam == 0) {
        std::replace(cf.begin(), cf.end(), 'U', 'C');
        break;
      }

      // Decrease the count for the bucket and skip if any F point reaches top of the queue
      --cnt[lam];
      if (cf[i] == 'F')
        continue;

      // Convert the highest priority undecided variable to a new C-point
      cf[i] = 'C';

      // Dependents of a new C-point become F-points.
      for (size_t j = S.m_row_pointers[i], e = S.m_row_pointers[i + 1]; j < e; ++j) {
        size_t c = S.m_col_indices[j];
        if (cf[c] != 'U')
          continue;
        cf[c] = 'F';

        // Newly created F-points increase the priority of their undecided strong influencers.
        for (size_t aj = A.m_row_pointers[c], ae = A.m_row_pointers[c + 1]; aj < ae; ++aj) {
          if (!S.m_vals[aj])
            continue;

          size_t ac = A.m_col_indices[aj];
          size_t lam_a = lambda[ac];
          if (cf[ac] != 'U' || lam_a + 1 >= nb)
            continue;

          // Move the variable ac from bucket lam_a to bucket lam_a + 1
          size_t old_pos = n2i[ac];
          size_t new_pos = ptr[lam_a] + cnt[lam_a] - 1; // last unprocessed slot of bucket lam_a

          // Swap place with whatever variable occupies the last slot
          n2i[i2n[old_pos]] = new_pos;
          n2i[i2n[new_pos]] = old_pos;
          std::swap(i2n[old_pos], i2n[new_pos]);

          --cnt[lam_a];
          ++cnt[lam_a + 1];
          ptr[lam_a + 1] = ptr[lam_a] + cnt[lam_a];
          lambda[ac] = lam_a + 1;
        }
      }

      // The selected C-point no longer contributes to its undecided neighbors' priorities.
      for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; j++) {
        if (!S.m_vals[j])
          continue;

        size_t c = A.m_col_indices[j];
        size_t lam = lambda[c];

        if (cf[c] != 'U' || lam == 0)
          continue;

        size_t old_pos = n2i[c];
        size_t new_pos = ptr[lam];

        n2i[i2n[old_pos]] = new_pos;
        n2i[i2n[new_pos]] = old_pos;

        std::swap(i2n[old_pos], i2n[new_pos]);

        --cnt[lam];
        ++cnt[lam - 1];
        ++ptr[lam];
        lambda[c] = lam - 1;
      }
    }
  }
};