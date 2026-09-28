#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/coarsening/galerkin.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/core/utilities.hpp"

/// Classic Ruge-Stuben coarsening with direct interpolation
template <typename T>
struct ruge_stuben {
  /// Coarsening parameters
  struct params {
    /// Parameter eps_strong defines strong couplings.
    /**
     * Given a threshold value 0 < eps_strong <= 1, the variable i strongly
     * depends on the variable j if, -A(i, j) >= eps_strong * max_{k != i}(-A(i,
     * k)). In practice, a value of eps_strong = 0.25 is usually taken.
     */
    float eps_strong;

    /// Truncate prolongation operator?
    /**
     * Interpolation operators, and, hence coarse operators may increase
     * substabtially towards coarser levels. Without truncation, this may
     * become too costly. Truncation ignores all interpolatory connections
     * which are smaller (in absolute value) than the largest one by a
     * factor of eps_trunc. The remaining weights are rescaled
     * so that the total sum remains unchanged. In practice, a value of
     * eps_trunc = 0.2 is usually taken.
     */
    bool do_trunc;

    /// Truncation parameter
    float eps_trunc;

    params() : eps_strong(0.4f), do_trunc(true), eps_trunc(0.2f) {}
  } prm;

  ruge_stuben(const params& prm = params()) : prm(prm) {}

  std::tuple<std::shared_ptr<CSRMatrix<T>>, std::shared_ptr<CSRMatrix<T>>>
  transfer_operators(const CSRMatrix<T>& A) const {
    const size_t n = A.m_rows;
    static const T epsilon = eps<T>(1);

    std::vector<char> cf(n, 'U');
    CSRMatrix<char> S;

    // call the splitting algorithm
    connect(A, prm.eps_strong, S, cf);
    cfsplit(A, S, cf);

    // Interpolation
    size_t nc = 0;
    std::vector<size_t> cidx(n);
    for (size_t i = 0; i < n; ++i) {
      if (cf[i] == 'C') cidx[i] = nc++;
    }

    if (!nc)
      throw std::runtime_error("The level is empty.");

    std::shared_ptr<CSRMatrix<T>> P = std::make_shared<CSRMatrix<T>>();
    P->m_rows = n;
    P->m_cols = nc;
    P->m_row_pointers.resize(n + 1, 0);

    std::vector<T> Amin, Amax;

    if (prm.do_trunc) {
      Amin.resize(n);
      Amax.resize(n);
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // For C-points, row P[i] has exactly one non-zero (the identity).
        if (cf[i] == 'C') {
          ++P->m_row_pointers[i + 1];
          continue;
        }

        // For F-points, its value is iterpolated from strongly connected coarse neighbors.
        size_t nnz = 0;
        if (prm.do_trunc) {
          T amin = T(0);
          T amax = T(0);

          // Check the neighbors of the current node for nodes that strongly influnces it
          for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
            // The neighbor should have a strong influnce on i and should be a coarse point
            if (!S.m_vals[j] || cf[A.m_col_indices[j]] != 'C')
              continue;

            // Find the maximum and minimum value among the strongly connected neighbors
            amin = std::min(amin, A.m_vals[j]);
            amax = std::max(amax, A.m_vals[j]);
          }

          // Calculate truncation threshold for node i and store
          Amin[i] = (amin *= prm.eps_trunc);
          Amax[i] = (amax *= prm.eps_trunc);

          // Iterate again to count neighbors that satisfy the truncated criteria
          for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
            if (!S.m_vals[j] || cf[A.m_col_indices[j]] != 'C')
              continue;

            // Fliter out values outside the range of amin and amax.
            if (A.m_vals[j] < amin || amax < A.m_vals[j]) {
              nnz++;
            }
          }
        } else {
          // Without truncation consider all strongly connected neighbors
          // for interpolation
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

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r){
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t row_head = P->m_row_pointers[i];

        // Identity mapping for coarse nodes
        if (cf[i] == 'C') {
          P->m_col_indices[row_head] = cidx[i];
          P->m_vals[row_head] = T(1);
          continue;
        }

        // Handling F points
        // stores the values of the diagonal elements and weakly connected neighbors
        // (F points that are not strongly connected to node i)
        T dia = T(0);

        // a_num is the sum of all negative off-diagonal elements, a_den is the sum of
        // all the negative off diagonals that connect to strong, coarse neighbors
        T a_num = T(0), a_den = T(0);

        // b_num is the sum of all positive off-diagonal elements, b_den is the sum of
        // all the positive off diagonals that connect to strong, coarse neighbors
        T b_num = T(0), b_den = T(0);
        
        // sum of the connections that will be dropped
        T d_neg = T(0), d_pos = T(0);  

        // Iterate over all the nonzeros in the the current row i
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          size_t c = A.m_col_indices[j];
          T v = A.m_vals[j];

          // store the diagonal element and move on to the next nonzero entry
          if (c == i) {
            dia = v;
            continue;
          }

          if (v < T(0)) {
            a_num += v;
            if (S.m_vals[j] && cf[c] == 'C') {
              a_den += v;
              if (prm.do_trunc && v > Amin[i]) d_neg += v;
            }
          } else {
            b_num += v;
            if (S.m_vals[j] && cf[c] == 'C') {
              b_den += v;
              if (prm.do_trunc && v < Amax[i]) d_pos += v;
            }
          }
        }

        // correction factors used for truncation
        T cf_neg = 1;
        T cf_pos = 1;

        if (prm.do_trunc) {
          // check if the sum of the negative strong connections that are
          // being kept is significant
          if (std::abs(a_den - d_neg) > epsilon) {
            // ratio of the total negative strong connections and the strong
            // connections being kept
            cf_neg = std::abs(a_den) / std::abs(a_den - d_neg);
          }

          if (std::abs(b_den - d_pos) > epsilon) {
            cf_pos = std::abs(b_den) / std::abs(b_den - d_pos);
          }
        }

        // positive off diagonal elements that are very small are included
        // in the digonal sum
        if (T(0) < b_num && std::abs(b_den) < epsilon) {
          dia += b_num;
        }

        // final interpoaltion coefficients: -(sum of all connections) /
        // (diagonal * sum of all strong connections)
        T alpha = std::abs(a_den) > epsilon
                  ? -cf_neg * std::abs(a_num) / (std::abs(dia) * std::abs(a_den))
                  : 0;
        T beta = std::abs(b_den) > epsilon
                 ? -cf_pos * std::abs(b_num) / (std::abs(dia) * std::abs(b_den))
                 : 0;

        // build the P matrix
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          size_t c = A.m_col_indices[j];
          T v = A.m_vals[j];

          // filter out neighbors that are not strong connection and dropped by truncation
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

    return std::make_tuple(P, transpose(*P));
  }

  std::shared_ptr<CSRMatrix<T>> coarse_operator(const CSRMatrix<T>& A,
                                                const CSRMatrix<T>& P,
                                                const CSRMatrix<T>& R) const
  {
    return galerkin(A, P, R);
  }

 private:
  /// @brief Determines strong connections in a matrix for AMG coarsening (Ruge-Stüben).
  ///
  /// @details Analyzes the system matrix `A` to build a boolean "strength of
  /// connection" matrix `S` based on the fact that if i strongly depends on j,
  /// then j strongly influnces i. It also identifies nodes with no strong
  /// off-diagonal influence and pre-classifies them as 'F' (Fine) points.
  ///
  /// On return:
  /// - `S.m_vals` contains the boolean flags (1 for strong, 0 for weak) for S,
  ///   sharing the sparsity pattern of A.
  /// - `S.m_row_pointers` and `S.m_col_indices` are recomputed to hold the
  ///   structure of the transpose of the strength matrix, S^T.
  ///
  /// @param[in] A The input system matrix in CSR format. Assumed to be an M-matrix
  ///              where off-diagonal entries are non-positive.
  /// @param[in] eps_strong The strength threshold.
  /// @param[out] S The output strength matrix.
  /// @param[out] cf The Coarse/Fine splitting vector.
  static void connect(const CSRMatrix<T>& A, float eps_strong,
                      CSRMatrix<char>& S, std::vector<char>& cf)
  {
    const size_t n = A.m_rows;
    const size_t nnz = A.m_nnz;
    const T epsilon = eps<T>(1);

    S.m_rows = S.m_cols = n;
    S.m_row_pointers.resize(n + 1, T(0));
    S.m_vals.resize(nnz, T(0));

    // Phase 1: Filter connections in parallel to build the boolean strength matrix S
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Find the most negative off-diagonal value in this row.
        // This represents the node j that has the strongest influence on node i.
        T a_max_abs = T(0);

        // Iterate through the row to find the largest negative number in the row.
        for (auto a = A.rowBegin(i); a; ++a) {
          if (a.col() != i)
            a_max_abs = std::max(a_max_abs, std::abs(a.value()));
        }

        // If there are no negative off-diagonals, this point is not strongly coupled to
        // any other node and cannot be used for interpolation.
        // Mark it as a Fine point and continue.
        if (a_max_abs < epsilon) {
          cf[i] = 'F';
          continue;
        }

        // Calculate the row-specific strength threshold.
        a_max_abs *= eps_strong;

        // Iterate through the row again and mark all connections that are stronger than
        // the threshold. This identifies the set of points that strongly influnence i
        // (i strongly depends on) and can be used to interpolate i.
        for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
          S.m_vals[j] = (A.m_col_indices[j] != i && std::abs(A.m_vals[j]) >= a_max_abs);
        }
      }
    });

    // Phase 2: Compute the transpose S^T in-place.
    // S^T stores the information if node i strongly influnces node j
    // (i.e. if node j is strongly dependent on node i)

    // 2a. Build a histogram of column counts of S.
    for (size_t i = 0; i < nnz; ++i) {
      if (S.m_vals[i]) ++(S.m_row_pointers[A.m_col_indices[i] + 1]);
    }

    // 2b. Convert counts to offsets via prefix sum.
    S.scanRowSize();
    S.m_col_indices.resize(S.m_row_pointers[n]);

    // 2d. Place the transposed elements into their final positions.
    // For each strong connection S(i,c), we create the entry S^T(c,i).
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = A.m_row_pointers[i], e = A.m_row_pointers[i + 1]; j < e; ++j) {
        if (S.m_vals[j])
          S.m_col_indices[S.m_row_pointers[A.m_col_indices[j]]++] = i;
      }
    }

    // 2e. Finalize the row pointers. The previous step left them pointing to
    // the end of each row's data. This trick restores them to point to the beginning.
    std::rotate(S.m_row_pointers.data(), S.m_row_pointers.data() + n,
                S.m_row_pointers.data() + n + 1);
    S.m_row_pointers[0] = 0;
  }

  /// @brief Splits the variables into Coarse (C) and Fine (F) sets based on
  /// strength of connection.
  /// @details The two heuristic for C/F splitting:
  /// 1. For each F-point i, every point j in S[i] that strongly influences i
  /// either should be in the coarse interpolatory set C_i or should strongly
  /// depend on at least one point in C_i.
  /// 2. The set of coarse points C should be a maximal subset of all points
  /// with the property that no C-point strongly depends on another C-point
  /// @param A The system matrix, used to determine strong connections from a C-point.
  /// @param S The transposed strength matrix S^T. Used to find nodes that
  /// strongly influence a given node.
  /// @param cf [in,out] The coarse/fine splitting vector.
  static void cfsplit(const CSRMatrix<T>& A, const CSRMatrix<char>& S,
                      std::vector<char>& cf)
  {
    // Phase 1: Initialize Lambda (Priority) Values
    // lambda[i] holds a weighted count of nodes that are strongly influenced by node i.
    const size_t n = A.m_rows;
    std::vector<size_t> lambda(n);

    for (size_t i = 0; i < n; ++i) {
      size_t temp = 0;
      // Iterate through row i of S^T to count all nodes that strongly depends on i.
      for (size_t j = S.m_row_pointers[i], e = S.m_row_pointers[i + 1]; j < e; ++j) {
        temp += (cf[S.m_col_indices[j]] == 'U' ? 1 : 2);
      }
      lambda[i] = temp;
    }

    // Phase 2: Bucket Sort nodes based on their lambda value
    // `ptr[k]` will be the starting index in the sorted `i2n` array for nodes
    // with lambda value k.
    std::vector<size_t> ptr(n + 1, 0);
    // `cnt[k]` is a running counter for the number of nodes placed in the
    // bucket for lambda value k.
    std::vector<size_t> cnt(n, 0);
    // `i2n[k]` stores the original node index `i` at the k-th position in the sorted order.
    std::vector<size_t> i2n(n);
    // `n2i[i]` stores the sorted position `k` for the original node `i`.
    std::vector<size_t> n2i(n);

    // Create a histogram of lambda values.
    for (size_t i = 0; i < n; ++i) {
      ++ptr[lambda[i] + 1];
    }

    // Convert counts to starting offsets via prefix sum.
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

    // Place each node i into its sorted bucket.
    for (size_t i = 0; i < n; ++i) {
      size_t lam = lambda[i];
      // Calculate the unique position for this node in the sorted array i2n.
      // ptr[lam] is the start of the bucket, cnt[lam] is the local offset.
      // cnt[lam]++ increments the offset for the next node with the same lambda.
      size_t idx = ptr[lam] + cnt[lam]++;
      i2n[idx] = i;  // Store original node `i` in its new sorted position `idx`.
      n2i[i] = idx;  // Record the new sorted position `idx` for node `i`.
    }

    // Process variables by decreasing lambda value. i2n is sorted in ascending order.
    // 1. The vaiable with maximum value of lambda becomes next C-variable.
    // 2. The node selected as a C-variable should appear in the interpolation
    // formula for the nodes that strongly depend on this point (the
    // correspnding row in S^T). So they should become F-variables.
    // 3. Keep lambda values in sync.
    for (size_t top = n; top-- > 0;) {
      // Pick the current highest-priority node from the sorted list.
      size_t i = i2n[top];
      size_t lam = lambda[i];

      // If the highest priority remaining is 0, it means all remaining 'U'
      // nodes are not strongly influenced by any other 'U' nodes. We can safely
      // make them all C-points to ensure the coarse grid covers the entire domain.
      if (lam == 0) {
        std::replace(cf.begin(), cf.end(), 'U', 'C');
        break;
      }

      // Remove the variable from its group.
      --cnt[lam];

      // If this node has already been made an F-point by a previously chosen
      // C-point, there is nothing to do, so we skip it.
      if (cf[i] == 'F')
        continue;

      // The current node `i` is selected to be a Coarse point.
      cf[i] = 'C';

      // Mark neighbors of `i` in S^T as F-points.
      // When `i` becomes a C-point, any undecided neighbor `c` that is
      // strongly dependent on  `i` must become an F-point.
      for (size_t j = S.m_row_pointers[i], e = S.m_row_pointers[i + 1]; j < e; ++j) {
        size_t c = S.m_col_indices[j];

        if (cf[c] != 'U')
          continue;

        // This dependent neighbor `c` is now designated as an F-point.
        cf[c] = 'F';

        // The other nodes that strongly influence these new F-points are
        // potential C-points. For each new F-point j in S[i]^T, increment the
        // lambda value of each undecided point k that strongly influences j.
        // That is each undecided member k of S[j] (non-zero columns of jth row
        // of S) Increase lambdas of the newly created F's neighbours.
        for (size_t aj = A.m_row_pointers[c], ae = A.m_row_pointers[c + 1]; aj < ae; ++aj) {
          if (!S.m_vals[aj])
            continue;

          size_t ac = A.m_col_indices[aj];  // node k that strongly influences j
          size_t lam_a = lambda[ac];

          // Only update undecided neighbors.
          if (cf[ac] != 'U' || lam_a + 1 >= n)
            continue;

          size_t old_pos = n2i[ac]; // get the old index in sorted array
          size_t new_pos = ptr[lam_a] + cnt[lam_a] - 1; // Index of the last position in the current bucket

          // old_pos = n2i[ac] => i2n[old_pos] == ac
          n2i[i2n[old_pos]] = new_pos;  // sorted index for node `ac` is updated to new_pos
          n2i[i2n[new_pos]] = old_pos;  // sorted index for the node at new_pos is updated to old_pos

          // Swap the node `ac` with the node at the end of the bucket.
          // After this, `ac` is at `new_pos`, and the other node is at `old_pos`.
          std::swap(i2n[old_pos], i2n[new_pos]);

          --cnt[lam_a];      // The bucket for `lam_a` now contains one less element.
          ++cnt[lam_a + 1];  // The bucket for `lam_a + 1` now contains one more element.
          
          // The starting pointer for the next bucket must be adjusted, as the bucket for `lam_a` has just shrunk.
          ptr[lam_a + 1] = ptr[lam_a] + cnt[lam_a];

          // Finally, update the actual lambda value for node `ac`.
          lambda[ac] = lam_a + 1;
        }
      }

      // Decrease lambdas of the newly create C's neighbours in S.
      // Since the neighbors in C no longer needs to influence C their lambnda
      // value is decreased This maintains H-2
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