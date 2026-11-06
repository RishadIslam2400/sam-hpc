#include "testlib.hpp"
#include "helpers.hpp"
#include "sam_hpc/dense/qr.hpp"

/// Verify the defining identities Q^T Q = I and QR = A for a thin factorization.
void test_factorization() {
  std::cout << "qr factorization..." << std::flush;

  const int m = 3, n = 2;

  std::vector<double> A = {
    1.0, 1.0, 1.0,
    -1.0, 0.0, 1.0
  };
  std::vector<double> A_copy = A;

  QR<double> qr;

  // Columns are (1, 1, 1) and (-1, 0, 1).
  qr.factorize(m, n, A_copy.data(), storage_order::col_major);

  std::vector<double> Q(m * n);
  std::vector<double> R(n * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Q[i * n + j] = qr.Q(i, j);
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      R[i * n + j] = qr.R(i, j);
    }
  }

  std::vector<double> Qt(n * m);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Qt[j * m + i] = Q[i * n + j];
    }
  }

  std::vector<double> QtQ = multiplyMatrices<double>(n, m, Qt, m, n, Q);
  std::vector<double> I(n * n, 0.0);
  for(int i = 0; i < n; ++i) I[i * n + i] = 1.0;
  assertEquals(I, QtQ, "Matrix Q is not orthogonal!");

  // Q and R are stored row-major here, while the input factorization is column-major.
  std::vector<double> A_row_major(m * n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A_row_major[i * n + j] = A[j * m + i];
    }
  }
  std::vector<double> QR_reconstructed = multiplyMatrices<double>(m, n, Q, n, n, R);
  assertEquals(A_row_major, QR_reconstructed, "QR factorization incorrect!");


  std::cout << "OK" << std::endl;
}

/// Verify an exact least-squares solution for an overdetermined system.
void test_solve_overdetermined() {
  std::cout << "overdetermined system (least squares)..." << std::flush;

  const int m = 3, n = 2;
  std::vector<double> A = {
    1.0, -1.0,
    1.0,  0.0,
    1.0,  1.0
  };
  std::vector<double> b = {1.0, 2.0, 3.0};
  std::vector<double> x(n);
  
  std::vector<double> expected_x = {2.0, 1.0};

  QR<double> qr;
  qr.solve(m, n, A.data(), b.data(), x.data());

  assertEquals(expected_x, x, "Solution incorrect!");
  std::cout << "OK" << std::endl;
}

/// Verify the minimum-norm solution for an underdetermined system.
void test_solve_underdetermined() {
  std::cout << "underdetermined system (minimum norm)..." << std::flush;

  const int m = 2, n = 3;
  std::vector<double> A = {
    1.0, 1.0, 1.0,
    0.0, 1.0, 2.0
  };
  std::vector<double> b = {6.0, 8.0};
  std::vector<double> x(n);

  std::vector<double> expected_x = {1.0, 2.0, 3.0};

  QR<double> qr;
  qr.solve(m, n, A.data(), b.data(), x.data());

  assertEquals(expected_x, x, "Solution is incorrect!");
  
  std::cout << "OK" << std::endl;
}

/// Verify residual handling when a square system has an exactly zero pivot.
void test_solve_rank_deficient() {
  std::cout << "rank deficient square system..." << std::flush;

  const int m = 3, n = 3;
  std::vector<double> A = {
    1.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 0.0, 0.0
  };
  std::vector<double> b = {1.0, 2.0, 5.0};
  std::vector<double> x(n);

  QR<double> qr;
  qr.solve(m, n, A.data(), b.data(), x.data());

  const std::vector<double> Ax = {x[0] + x[2], x[1] + x[2], 0.0};
  double resid_sq = 0.0;
  for (int i = 0; i < m; ++i) resid_sq += (Ax[i] - b[i]) * (Ax[i] - b[i]);

  assertEquals(25.0, resid_sq, "Returned x is not a least squares solution!");
  assertEquals(resid_sq, qr.residual_sq(), "Reported residual does not match the returned x!");
  std::cout << "OK" << std::endl;
}

/// Verify that the rank test rejects a tiny pivot and keeps the solution bounded.
void test_solve_nearly_rank_deficient() {
  std::cout << "nearly rank deficient tall system..." << std::flush;

  const int m = 4, n = 3;
  const double d = 1e-17;
  std::vector<double> A = {
    1.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 0.0, d,
    0.0, 0.0, 0.0
  };
  const std::vector<double> A0 = A;
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> x(n);

  QR<double> qr;
  qr.solve(m, n, A.data(), b.data(), x.data());

  double resid_sq = 0.0, xmax = 0.0;
  for (int i = 0; i < m; ++i) {
    double ax = 0.0;
    for (int j = 0; j < n; ++j)
      ax += A0[i * n + j] * x[j];
    resid_sq += (ax - b[i]) * (ax - b[i]);
  }
  for (double v : x)
    xmax = std::max(xmax, std::abs(v));

  assertTrueMsg(qr.rank() == 2, "Rank test did not drop the dependent column!");
  assertTrueMsg(xmax < 10.0, "Solution blew up on a tiny pivot!");
  assertEquals(25.0, resid_sq, "Returned x is not a least squares solution!");
  assertEquals(resid_sq, qr.residual_sq(), "Reported residual does not match the returned x!");
  std::cout << "OK" << std::endl;
}

/// Verify rank handling for a wide, inconsistent system with a tiny pivot.
void test_solve_nearly_rank_deficient_wide() {
  std::cout << "nearly rank deficient wide system..." << std::flush;

  const int m = 3, n = 4;
  const double d = 1e-17;
  std::vector<double> A = {
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 2.0, d
  };
  const std::vector<double> A0 = A;
  std::vector<double> b = {1.0, 2.0, 5.0};
  std::vector<double> x(n);

  QR<double> qr;
  qr.solve(m, n, A.data(), b.data(), x.data());

  double resid_sq = 0.0, xmax = 0.0;
  for (int i = 0; i < m; ++i) {
    double ax = 0.0;
    for (int j = 0; j < n; ++j)
      ax += A0[i * n + j] * x[j];
    resid_sq += (ax - b[i]) * (ax - b[i]);
  }
  for (double v : x)
    xmax = std::max(xmax, std::abs(v));

  assertTrueMsg(qr.rank() == 2, "Rank test did not drop the dependent equation!");
  assertTrueMsg(xmax < 10.0, "Solution blew up on a tiny pivot!");
  assertTrueMsg(resid_sq > 0.5, "An inconsistent system cannot be solved exactly!");
  assertEquals(resid_sq, qr.residual_sq(), "Reported residual does not match the returned x!");
  std::cout << "OK" << std::endl;
}

/// Verify that column pivoting is undone before returning the solution.
void test_solve_pivot_order() {
  std::cout << "pivoted solve returns x in input order..." << std::flush;

  const int m = 4, n = 3;
  std::vector<double> A = {
    1.0, 0.0, 10.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 10.0,
    0.0, 0.0, 0.0
  };
  std::vector<double> b = {31.0, 2.0, 30.0, 7.0};
  std::vector<double> x(n);

  QR<double> qr;
  qr.solve(m, n, A.data(), b.data(), x.data());

  assertEquals(std::vector<double>{1.0, 2.0, 3.0}, x, "Permutation not undone!");
  assertEquals(49.0, qr.residual_sq(), "Residual incorrect!");
  std::cout << "OK" << std::endl;
}

/// Verify that the unpivoted fast path rejects tiny pivots and solves a well-conditioned
/// full-rank system correctly.
void test_try_solve_unpivoted() {
  std::cout << "unpivoted fast path refuses tiny pivots..." << std::flush;

  std::vector<double> A = {
    1.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 0.0, 1e-17,
    0.0, 0.0, 0.0
  };
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> x = {-7.0, -7.0, -7.0};

  QR<double> qr;
  assertTrueMsg(!qr.try_solve_unpivoted(4, 3, A.data(), b.data(), x.data()), "Fast path accepted a tiny pivot!");
  assertEquals(std::vector<double>{-7.0, -7.0, -7.0}, x, "A refused solve wrote to x!");

  std::vector<double> A2 = {
    1.0, -1.0,
    1.0,  0.0,
    1.0,  1.0
  };
  std::vector<double> b2 = {1.0, 2.0, 3.0};
  std::vector<double> x2(2);
  assertTrueMsg(qr.try_solve_unpivoted(3, 2, A2.data(), b2.data(), x2.data()), "Fast path refused a well-conditioned system!");
  assertEquals(std::vector<double>{2.0, 1.0}, x2, "Fast path solution incorrect!");
  std::cout << "OK" << std::endl;
}