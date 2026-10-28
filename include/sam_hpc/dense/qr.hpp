#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

/// Storage order used by the dense QR convenience overloads.
enum storage_order { row_major, col_major };

/// In-place Householder QR factorization and rank-aware least-squares solver. The caller
/// owns the matrix buffer. Factorization overwrites it with the triangular factor and
/// Householder vectors. The standard solve uses column pivoting and truncates at
/// numerical rank. `try_solve_unpivoted()` provides a fast path for full-rank systems.
/// compute() and factorize() stay unpivoted, since R() and Q() expose those factors.
template <typename T>
class QR {
 public:
  QR() : m(0), n(0), row_stride(0), col_stride(0), r(NULL) {}

  /// Sets the relative threshold used to determine numerical rank.
  /// Nonpositive values select a dimension-scaled machine-epsilon threshold.
  void set_rank_tolerance(T tol) { rank_tol = tol; }

  /// Returns the numerical rank computed by the last solve().
  int rank() const { return rank_; }

  /// Computes an in-place Householder factorization without column pivoting.
  /// The upper portion stores R; the lower portion stores the reflector
  /// vectors and their coefficients are retained internally.
  void compute(int rows, int cols, int row_stride, int col_stride, T* A) {
    const int m = rows;
    const int n = cols;
    const int k = std::min(m, n);
    if (k <= 0)
      return;

    r = A;
    tau.resize(k);
    perm.resize(n);
    for (int j = 0; j < n; ++j)
      perm[j] = j;

    for (int i = 0, ii = 0; i < k; ++i, ii += row_stride + col_stride) {
      // Build a Householder reflector that zeros entries below the diagonal.
      tau[i] = gen_reflector(m - i, A[ii], A + ii + row_stride, row_stride);

      if (i + 1 < n) {
        // Apply the reflector to the remaining columns.
        apply_reflector(m - i, n - i - 1, A + ii, row_stride, tau[i],
                        A + ii + col_stride, row_stride, col_stride);
      }
    }
  }

  /// Computes a factorization from a row- or column-major buffer.
  void compute(int rows, int cols, T* A, storage_order order = row_major) {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    compute(rows, cols, row_stride, col_stride, A);
  }

  /// Computes a column-pivoted factorization: `A P = Q R`.
  /// The permutation is stored in `perm`. Trailing column norms are updated
  /// after each reflector and recomputed when necessary.
  void compute_pivoted(int rows, int cols, int row_stride, int col_stride, T* A) {
    const int m = rows;
    const int n = cols;
    const int k = std::min(m, n);
    if (k <= 0)
      return;

    r = A;
    tau.resize(k);
    perm.resize(n);
    vn1.resize(n);  // Current trailing-column norm estimates.
    vn2.resize(n);  // Reference norms for recomputation checks.
    for (int j = 0, ja = 0; j < n; ++j, ja += col_stride) {
      perm[j] = j;
      vn1[j] = vn2[j] = column_norm(m, A + ja, row_stride);
    }
    const T tol3z = std::sqrt(std::numeric_limits<T>::epsilon());

    for (int i = 0, ii = 0; i < k; ++i, ii += row_stride + col_stride) {
      // Swap the column of largest remaining norm into position i.
      int p = i;
      for (int j = i + 1; j < n; ++j) {
        if (vn1[j] > vn1[p])
          p = j;
      }
      if (p != i) {
        T* ci = A + i * col_stride;
        T* cp = A + p * col_stride;
        for (int l = 0, la = 0; l < m; ++l, la += row_stride)
          std::swap(ci[la], cp[la]);
        std::swap(perm[i], perm[p]);
        vn1[p] = vn1[i];
        vn2[p] = vn2[i];
      }

      tau[i] = gen_reflector(m - i, A[ii], A + ii + row_stride, row_stride);

      if (i + 1 < n) {
        apply_reflector(m - i, n - i - 1, A + ii, row_stride, tau[i],
                        A + ii + col_stride, row_stride, col_stride);
      }

      // Update the norms of the unreduced columns.
      for (int j = i + 1, ij = ii + col_stride; j < n; ++j, ij += col_stride) {
        if (vn1[j] == T(0))
          continue;
        T t = std::abs(A[ij]) / vn1[j];
        t = std::max(T(0), T(1) - t * t);
        T t2 = t * sqr(vn1[j] / vn2[j]);
        if (t2 <= tol3z) {
          vn1[j] = vn2[j] = (i + 1 < m) ? column_norm(m - i - 1, A + ij + row_stride, row_stride)
                                        : T(0);
        } else {
          vn1[j] *= std::sqrt(t);
        }
      }
    }
  }

  /// Factors A and explicitly forms the first `n` columns of Q.
  /// Requires `rows >= cols` and strides describing a dense `rows` by `cols` layout.
  void factorize(int rows, int cols, int row_stride, int col_stride, T* A) {
    compute(rows, cols, row_stride, col_stride, A);
    m = rows;
    n = cols;
    int k = std::min(m, n);
    this->row_stride = row_stride;
    this->col_stride = col_stride;
    q.resize(m * n);

    // Initialize any columns beyond the reflector count.
    for (int i = 0, ia = 0; i < m; ++i, ia += row_stride) {
      for (int j = k, ja = k * col_stride; j < n; ++j, ja += col_stride) {
        q[ia + ja] = (i == j ? T(1) : T(0));
      }
    }

    for (int i = k - 1, ic = i * col_stride, ii = i * (row_stride + col_stride);
         i >= 0; --i, ic -= col_stride, ii -= row_stride + col_stride)
    {
      // Apply the reflector to the columns already formed.
      if (i < n - 1) {
        apply_reflector(m - i, n - i - 1, r + ii, row_stride, tau[i],
                        &q[ii + col_stride], row_stride, col_stride);
      }

      // Store the reflector as the current column of Q.
      for (int j = 0, jr = 0; j < i; ++j, jr += row_stride) {
        q[jr + ic] = T(0);
      }

      q[ii] = T(1) - tau[i];

      for (int j = i + 1, jr = j * row_stride; j < m; ++j, jr += row_stride) {
        q[jr + ic] = -tau[i] * r[jr + ic];
      }
    }
  }

  /// Factors a row- or column-major buffer and forms Q explicitly.
  void factorize(int rows, int cols, T* A, storage_order order = row_major) {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    factorize(rows, cols, row_stride, col_stride, A);
  }

  /// Returns an element of R; entries below the diagonal are zero.
  T R(int i, int j) const {
    if (j < i) return T(0);
    return r[i * row_stride + j * col_stride];
  }

  /// Returns an element of the explicit Q formed by factorize().
  T Q(int i, int j) const { return q[i * row_stride + j * col_stride]; }

  /// Solves a least-squares or minimum-norm problem using pivoted QR.
  /// The triangular solve is truncated at the detected numerical rank.
  /// Set `computed` to reuse a compatible previous factorization.
  void solve(int rows, int cols, int row_stride, int col_stride, T* A,
             const T* b, T* x, bool computed = false)
  {
    if (!computed) {
      if (rows >= cols)
        compute_pivoted(rows, cols, row_stride, col_stride, A);
      else
        compute_pivoted(cols, rows, col_stride, row_stride, A);
    }
    rank_ = numerical_rank(std::min(rows, cols), std::max(rows, cols), row_stride + col_stride);
    finish_solve(rows, cols, row_stride, col_stride, b, x);
  }

  /// Attempts an unpivoted solve when all diagonal entries meet the rank threshold.
  /// Returns false and leaves `x` unchanged if the factorization is rank-deficient.
  /// The caller may then rebuild `A` and use solve().
  bool try_solve_unpivoted(int rows, int cols, int row_stride, int col_stride, T* A,
                           const T* b, T* x)
  {
    if (rows >= cols)
      compute(rows, cols, row_stride, col_stride, A);
    else
      compute(cols, rows, col_stride, row_stride, A);
    const int k = std::min(rows, cols);
    if (!full_rank(k, std::max(rows, cols), row_stride + col_stride))
      return false;
    rank_ = k;
    finish_solve(rows, cols, row_stride, col_stride, b, x);
    return true;
  }

  /// try_solve_unpivoted() for a system stored in row- or column-major order.
  bool try_solve_unpivoted(int rows, int cols, T* A, const T* b, T* x,
                           storage_order order = row_major)
  {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    return try_solve_unpivoted(rows, cols, row_stride, col_stride, A, b, x);
  }

  /// Solves a system stored in row- or column-major order.
  void solve(int rows, int cols, T* A, const T* b, T* x,
             storage_order order = row_major, bool computed = false)
  {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    solve(rows, cols, row_stride, col_stride, A, b, x, computed);
  }

  /// Returns the size of the internal scratch buffers in bytes.
  size_t bytes() {
    return sizeof(T) * (tau.size() + f.size() + q.size() + vn1.size() + vn2.size()) + sizeof(int) * perm.size();
  }

  /// Returns the squared 2-norm of the residual from the last solve().
  T residual_sq() const { return resid_sq; }

 private:
  static T sqr(T x) { return x * x; }
  int m, n, row_stride, col_stride;  // Dimensions and strides used by Q() and R().
  T resid_sq = T(0);
  T rank_tol = T(0);           // <= 0: max(m, n) * eps
  int rank_ = 0;               // Numerical rank from the last solve.

  T* r;                        // Caller-owned factored matrix.
  std::vector<T> tau, f;       // Reflector coefficients and solve workspace.
  std::vector<T> q;            // Explicit Q, using the matrix strides.
  std::vector<int> perm;       // Column permutation from the last pivoted factorization.
  std::vector<T> vn1, vn2;     // Working and reference column norms.

  /// Applies the stored factorization, truncated at `rank_`, to solve for `x`.
  void finish_solve(int rows, int cols, int row_stride, int col_stride, const T* b, T* x) {
    f.resize(rows);
    const int dstride = row_stride + col_stride;  // Stride between diagonal entries.

    if (rows >= cols) {
      // Tall: solve R x = Q^T b.
      std::copy(b, b + rows, f.begin());

      // Apply the reflectors to compute Q^T b.
      for (int i = 0, ii = 0; i < cols; ++i, ii += dstride) {
        apply_reflector(rows - i, 1, r + ii, row_stride, tau[i], &f[i], 1, 1);
      }

      // Components beyond the numerical rank form the residual.
      resid_sq = T(0);
      for (int i = rank_; i < rows; ++i) {
        resid_sq += sqr(f[i]);
      }

      // Solve the leading triangular block by back substitution.
      for (int i = rank_, ia = (rank_ - 1) * col_stride; i-- > 0; ia -= col_stride) {
        f[i] = (1 / r[i * dstride]) * f[i];
        for (int j = 0, ja = 0; j < i; ++j, ja += row_stride) {
          f[j] -= r[ia + ja] * f[i];
        }
      }

      // Undo the permutation and zero the dropped coefficients.
      for (int i = 0; i < cols; ++i)
        x[perm[i]] = (i < rank_ ? f[i] : T(0));
    } else {
      // Wide: solve R^T z = P^T b, then recover x = Q z.
      for (int i = 0; i < rows; ++i)
        f[i] = b[perm[i]];

      // Solve the leading triangular system R^T z = P^T b.
      for (int i = 0, ia = 0; i < rank_; ++i, ia += col_stride) {
        f[i] = (1 / r[i * dstride]) * f[i];
        for (int j = i + 1, ja = j * row_stride; j < rows; ++j, ja += row_stride)
          f[j] -= r[ia + ja] * f[i];
      }

      resid_sq = T(0);
      for (int i = rank_; i < rows; ++i) {
        resid_sq += sqr(f[i]);
        f[i] = T(0);
      }

      std::copy(f.begin(), f.end(), x);
      std::fill(x + rows, x + cols, T(0));

      // Recover x = Q z.
      for (int i = rows; i-- > 0;) {
        int ii = i * dstride;
        apply_reflector(cols - i, 1, r + ii, col_stride, tau[i], x + i, 1, 1);
      }
    }
  }

  T rank_threshold(int dim) const {
    return rank_tol > T(0) ? rank_tol : dim * std::numeric_limits<T>::epsilon();
  }

  /// True if no diagonal entry of an unpivoted R falls below the threshold times max |R(j, j)|.
  bool full_rank(int k, int dim, int dstride) const {
    T amax = T(0);
    for (int i = 0; i < k; ++i)
      amax = std::max(amax, std::abs(r[i * dstride]));
    if (amax == T(0))
      return false;
    const T thresh = rank_threshold(dim) * amax;
    for (int i = 0; i < k; ++i)
      if (!(std::abs(r[i * dstride]) > thresh))
        return false;
    return true;
  }

  static T column_norm(int len, const T* x, int stride) {
    T s = T(0);
    for (int i = 0, ii = 0; i < len; ++i, ii += stride)
      s += sqr(x[ii]);
    return std::sqrt(s);
  }

  /// Returns the number of diagonal entries above the numerical-rank threshold.
  int numerical_rank(int k, int dim, int dstride) const {
    if (k <= 0 || r[0] == T(0))
      return 0;
    const T thresh = rank_threshold(dim) * std::abs(r[0]);
    int rk = 1;
    while (rk < k && std::abs(r[rk * dstride]) > thresh)
      ++rk;
    return rk;
  }

  /// Builds a Householder reflector that maps `(alpha, x)` to `(beta, 0)`.
  /// `alpha` and `x` are overwritten with `beta` and the stored reflector
  /// components. The returned coefficient defines `H = I - tau * v * v^T`.
  static T gen_reflector(int order, T& alpha, T* x, int stride) {
    // Construct H = I - tau * v * v^T and store v below the diagonal.
    T tau = T(0);
    if (order <= 1)
      return tau;

    const int n = order - 1;

    // Compute the squared norm of the vector below the diagonal.
    T xnorm2 = 0;
    for (int i = 0, ii = 0; i < n; ++i, ii += stride) {
      xnorm2 += sqr(x[ii]);
    }

    if (xnorm2 == T(0))
      return tau;

    // Choose beta to avoid cancellation in alpha - beta.
    T beta = -sqrt(sqr(alpha) + xnorm2);
    if (alpha < 0) {
      beta = -beta;
    }

    tau = T(1) - (1 / beta) * alpha;

    // Store the reflector vector below the diagonal.
    alpha = 1 / (alpha - beta);
    for (int i = 0, ii = 0; i < n; ++i, ii += stride) {
      x[ii] = alpha * x[ii];
    }
    alpha = beta;
    return tau;
  }

  /// Applies `H = I - tau * v * v^T` to the left of a dense matrix.
  /// The matrix `C` is updated in place using the supplied storage strides.
  static void apply_reflector(int m, int n, const T* v, int v_stride, T tau,
                              T* C, int row_stride, int col_stride)
  {
    if (tau == T(0))
      return;

    // Apply H * C = C - v * tau * (v^T * C) one column at a time.
    for (int i = 0, ia = 0; i < n; ++i, ia += col_stride) {
      T s = C[ia];
      for (int j = 1, jv = v_stride, ja = row_stride; j < m; ++j, jv += v_stride, ja += row_stride) {
        s += C[ja + ia] * v[jv];
      }

      s *= tau;
      C[ia] -= s;
      for (int j = 1, jv = v_stride, ja = row_stride; j < m; ++j, jv += v_stride, ja += row_stride) {
        C[ja + ia] -= v[jv] * s;
      }
    }
  }
};