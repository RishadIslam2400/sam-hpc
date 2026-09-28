#pragma once

#include <cmath>
#include <vector>

enum storage_order { row_major, col_major };

/// In-place QR factorization
template <typename T>
class QR {
 public:
  QR() : m(0), n(0), row_stride(0), col_stride(0), r(NULL) {}

  void compute(int rows, int cols, int row_stride, int col_stride, T* A) {
    /*
     *  Ported from ZGEQR2
     *  ==================
     *
     *  Computes a QR factorization of an matrix A: A = Q * R.
     *
     *  Arguments
     *  =========
     *
     *  rows    The number of rows of the matrix A.
     *  cols    The number of columns of the matrix A.
     *
     *  A       On entry, the rows by cols matrix A.
     *          On exit, the elements on and above the diagonal of the
     *          array contain the min(m,n) by n upper trapezoidal
     *          matrix R (R is upper triangular if m >= n); the
     *          elements below the diagonal, with the array TAU,
     *          represent the unitary matrix Q as a product of
     *          elementary reflectors (see Further Details).
     *
     *  Further Details
     *  ===============
     *
     *  The matrix Q is represented as a product of elementary reflectors
     *
     *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
     *
     *  Each H(i) has the form
     *
     *     H(i) = I - tau * v * v'
     *
     *  where tau is a T scalar, and v is a T vector
     *  with v[0:i) = 0 and v[i] = 1; v[i:m) is stored on exit in
     *  A[i+1:m)[i], and tau in tau[i].
     *  ==============================================================
     */
    const int m = rows;
    const int n = cols;
    const int k = std::min(m, n);
    if (k <= 0)
      return;

    r = A;
    tau.resize(k);
    for (int i = 0, ii = 0; i < k; ++i, ii += row_stride + col_stride) {
      // Generate elementary reflector H(i) to annihilate A[i + 1 : m)[i]
      // subvector size = m - i (from diagonal to last row)
      // alpha is the diagonal element
      // A + ii + row_stride = element below the digonal
      tau[i] = gen_reflector(m - i, A[ii], A + ii + row_stride, row_stride);

      // Check if there is any column to the right
      if (i + 1 < n) {
        // Apply H(i)' to A[i:m)[i + 1 : n) from left
        apply_reflector(m - i, n - i - 1, A + ii, row_stride, tau[i],
                        A + ii + col_stride, row_stride, col_stride);
      }
    }
  }

  void compute(int rows, int cols, T* A, storage_order order = row_major) {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    compute(rows, cols, row_stride, col_stride, A);
  }

  // Computes Q explicitly
  void factorize(int rows, int cols, int row_stride, int col_stride, T* A) {
    /*
     *  Ported from ZUNG2R
     *  ==================
     *
     *  Generates an m by n matrix Q with orthonormal columns, which is
     *  defined as the first n columns of a product of k elementary
     *  reflectors of order m
     *
     *        Q  =  H(1) H(2) . . . H(k)
     *
     *  as returned by compute() [ZGEQR2].
     *
     *  ==============================================================
     */
    compute(rows, cols, row_stride, col_stride, A);
    m = rows;
    n = cols;
    int k = std::min(m, n);
    this->row_stride = row_stride;
    this->col_stride = col_stride;
    q.resize(m * n);

    // Initialize columns k+1:n to zero.
    for (int i = 0, ia = 0; i < m; ++i, ia += row_stride) {
      for (int j = k, ja = k * col_stride; j < n; ++j, ja += col_stride) {
        q[ia + ja] = (i == j ? T(1) : T(0));
      }
    }

    // i is the index of the current reflector
    // ic is the memory offset of ith column
    // ii is the diagonal memory offset, A[i][i]
    for (int i = k - 1, ic = i * col_stride, ii = i * (row_stride + col_stride);
         i >= 0; --i, ic -= col_stride, ii -= row_stride + col_stride) {
      // Apply H(i) to Q[i:m)[i+1:n) from the left
      if (i < n - 1) {
        apply_reflector(m - i, n - i - 1, r + ii, row_stride, tau[i],
                        &q[ii + col_stride], row_stride, col_stride);
      }

      // Copy i-th reflector (including zeros and unit diagonal)
      // to the column of Q to be processed next set the elements above the diagonal to zero
      for (int j = 0, jr = 0; j < i; ++j, jr += row_stride) {
        q[jr + ic] = T(0);
      }

      q[ii] = T(1) - tau[i];  // set the diagonal element

      for (int j = i + 1, jr = j * row_stride; j < m; ++j, jr += row_stride) {
        q[jr + ic] = -tau[i] * r[jr + ic];
      }
    }
  }

  void factorize(int rows, int cols, T* A, storage_order order = row_major) {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    factorize(rows, cols, row_stride, col_stride, A);
  }

  // Returns elements of the matrix R.
  T R(int i, int j) const {
    if (j < i) return T(0);
    return r[i * row_stride + j * col_stride];
  }

  // Returns elements of the matrix Q.
  T Q(int i, int j) const { return q[i * row_stride + j * col_stride]; }

  // Solves the system QRx = f
  void solve(int rows, int cols, int row_stride, int col_stride, T* A,
             const T* b, T* x, bool computed = false)
  {
    // Fill up member variable rhs vector f with b
    f.resize(rows);
    std::copy(b, b + rows, f.begin());

    if (rows >= cols) {
      // We are solving overdetermined (tall) system Ax = f by writing the
      // matrix as A = QR and solving for x as x = R^-1 Q^-1 f = R^-1 Q^T f.
      if (!computed)
        compute(rows, cols, row_stride, col_stride, A);

      // Compute y = Q^T b
      // Q^T = H(k - 1) * ... * H(1) * H(0)
      for (int i = 0, ii = 0; i < cols; ++i, ii += row_stride + col_stride) {
        // Apply H(i) to the vector f and modify
        // rows - i = size of the reflector applied to a single column f
        // r + ii points to the starting of the reflector
        // &f[i] is the pointer to the subvector where H(i) will be applied to
        apply_reflector(rows - i, 1, r + ii, row_stride, tau[i], &f[i], 1, 1);
      }

      // Rows cols..rows of Q^T b cannot be reduced by any choice of x, so their
      // norm is exactly the least squares residual. Read it before back substitution.
      resid_sq = T(0);
      for (int i = cols; i < rows; ++i) {
        resid_sq += sqr(f[i]);
      }

      std::copy(f.begin(), f.begin() + cols, x);

      // Solve Rx = y via back substitution
      for (int i = cols, ia = (cols - 1) * col_stride; i-- > 0; ia -= col_stride) {
        T rii = r[i * (row_stride + col_stride)];
        if (rii == T(0))
          continue;
        x[i] = (1 / rii) * x[i];

        for (int j = 0, ja = 0; j < i; ++j, ja += row_stride) {
          x[j] -= r[ia + ja] * x[i];
        }
      }
    } else {
      // We are solving underdetermined (wide) system Ax = f by writing the
      // matrix A^T as A^T = QR and solving for x as x = Q^-T R^-T f = Q R^-T f
      // Underdetermined system has infinite solutions. Find the one with min norm
      
      resid_sq = T(0);
      if (!computed) {
        // Compute QR factorization of A^T by passing col and row stride swapped
        compute(cols, rows, col_stride, row_stride, A);
      }

      // If A^T = QR, then R^T Q^T x = b
      // Solve for R^T z = b such that Q^T x = z, R^T is a lower triangular matrix.
      // Use forward substitution
      for (int i = 0, ia = 0; i < rows; ++i, ia += col_stride) {
        T rii = r[i * (row_stride + col_stride)];
        if (rii == T(0))
          continue;
        f[i] = (1 / rii) * f[i];

        for (int j = i + 1, ja = j * row_stride; j < rows; ++j, ja += row_stride)
          f[j] -= r[ia + ja] * f[i];
      }

      std::copy(f.begin(), f.end(), x);
      std::fill(x + rows, x + cols, T(0));  // fill rest of the x vector with 0

      // Solve for Qx = z or x = Qz
      // Apply the reflectors in reverse order to vector z
      for (int i = rows; i-- > 0;) {
        int ii = i * (col_stride + row_stride);
        apply_reflector(cols - i, 1, r + ii, col_stride, tau[i], x + i, 1, 1);
      }
    }
  }

  void solve(int rows, int cols, T* A, const T* b, T* x,
             storage_order order = row_major, bool computed = false)
  {
    int row_stride = (order == row_major ? cols : 1);
    int col_stride = (order == row_major ? 1 : rows);
    solve(rows, cols, row_stride, col_stride, A, b, x, computed);
  }

  size_t bytes() { return sizeof(T) * (tau.size() + f.size() + q.size()); }

  /// Squared 2-norm of the least squares residual from the last solve(), i.e.
  /// ||Ax - b||_2^2. For an overdetermined system this is the tail of Q^T b.
  /// Underdetermined systems are solved exactly, so the residual is zero.
  T residual_sq() const { return resid_sq; }

 private:
  static T sqr(T x) { return x * x; }
  int m, n, row_stride, col_stride;
  T resid_sq = T(0);

  T* r;
  std::vector<T> tau, f;
  std::vector<T> q;

  static T gen_reflector(int order, T& alpha, T* x, int stride) {
    /*
     *  Ported from ZLARFG
     *  ==================
     *
     *  Generates a type t elementary reflector H of order n, such that
     *
     *        H' * ( alpha ) = ( beta ),   H' * H = I.
     *             (   x   )   (   0  )
     *
     *  where alpha and beta are scalars, with beta real, and x is an
     *  (n-1)-element type T vector. H is represented in the form
     *
     *        H = I - tau * ( 1 ) * ( 1 v' ) ,
     *                      ( v )
     *
     *  where tau is a type T scalar and v is a type T
     *  (n-1)-element vector. Note that H is not hermitian.
     *
     *  If the elements of x are all zero and alpha is real,
     *  then tau = 0 and H is taken to be the unit matrix.
     *
     *  Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
     *
     *  Arguments
     *  =========
     *
     *  order   The order of the elementary reflector.
     *
     *  alpha   On entry, the value alpha.
     *          On exit, it is overwritten with the value beta.
     *
     *  x       dimension (1+(order-2)*abs(stride))
     *          On entry, the vector x.
     *          On exit, it is overwritten with the vector v.
     *
     *  stride  The increment between elements of x.
     *
     *  Returns the value tau.
     *
     *  ==============================================================
     */
    T tau = T(0);  // Scaling factor for H = I - tau * vv^T
    if (order <= 1)
      return tau;  // If the vector dimension less than 2 Housholder reflection is not applicable
    int n = order - 1;

    // Calculate the Euclidean norm of the subvector x
    T xnorm2 = 0;
    for (int i = 0, ii = 0; i < n; ++i, ii += stride) {
      xnorm2 += sqr(x[ii]);
    }

    if (xnorm2 == T(0))
      return tau;  // if the subvector already 0, nothing to do

    // beta = -sign(alpha)sqrt(alpha^2 + xnorm2)
    T beta = -sqrt(sqr(alpha) + xnorm2);
    if (alpha < 0) {
      beta = -beta;
    }

    // tau = 1 - alpha / beta
    tau = T(1) - (1 / beta) * alpha;

    // compute vector v by scaling the original vector x
    // v_i = x_i / (alpha - beta)
    alpha = 1 / (alpha - beta);
    for (int i = 0, ii = 0; i < n; ++i, ii += stride) {
      x[ii] = alpha * x[ii];
    }
    alpha = beta;
    return tau;
  }

  static void apply_reflector(int m, int n, const T* v, int v_stride, T tau,
                              T* C, int row_stride, int col_stride)
  {
    /*
     *  Ported from ZLARF
     *  =================
     *
     *  Applies an elementary reflector H to an m-by-n matrix C from
     *  the left. H is represented in the form
     *
     *        H = I - v * tau * v'
     *
     *  where tau is a type T scalar and v is a T vector.
     *
     *  If tau = 0, then H is taken to be the unit matrix.
     *
     *  To apply H' (the conjugate transpose of H), supply adjoint(tau)
     *  instead of tau.
     *
     *  Arguments
     *  =========
     *
     *  m          The number of rows of the matrix C.
     *
     *  n          The number of columns of the matrix C.
     *
     *  v          The vector v in the representation of H.
     *             v is not used if tau = 0.
     *             The value of v[0] is ignored and assumed to be 1.
     *
     *  v_stride   The increment between elements of v.
     *
     *  tau        The value tau in the representation of H.
     *
     *  C          On entry, the m-by-n matrix C.
     *             On exit, C is overwritten by the matrix H * C.
     *
     *  row_stride The increment between the rows of C.
     *  col_stride The increment between the columns of C.
     *
     *  ==============================================================
     */
    if (tau == T(0))
      return;

    // H * C = (I - tau * v * v^T) * C = C - v * tau * (v^T * C)
    // Iterate over the columns of C, ia is the memory offset to the start of
    // the i-th column
    for (int i = 0, ia = 0; i < n; ++i, ia += col_stride) {
      //  s = v^T * C_col
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