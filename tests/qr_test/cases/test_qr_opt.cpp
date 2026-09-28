#include "testlib.hpp"
#include "helpers.hpp"
#include "sam_hpc/dense/qr.hpp"

// Test the explicit QR factorization (factorize method)
void test_factorization() {
    std::cout << "qr factorization..." << std::flush;

    const int m = 3, n = 2;

    std::vector<double> A = {
        1.0, 1.0, 1.0,
        -1.0, 0.0, 1.0
    };
    std::vector<double> A_copy = A;

    QR<double> qr;

    // A is stored column-major: columns (1, 1, 1) and (-1, 0, 1)
    qr.factorize(m, n, A_copy.data(), storage_order::col_major);

    // Extract Q and R matrices
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

    // Test 1: Q should be orthogonal, so Q^T * Q = I
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

    // Test 2: Q * R should reconstruct the original matrix A. Q and R were extracted
    // row-major above, so compare against A in row-major order, not its col-major buffer.
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

// Test solving an overdetermined system (least squares)
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
    
    // The known least-squares solution is x = {2, 1}
    std::vector<double> expected_x = {2.0, 1.0};

    QR<double> qr;
    qr.solve(m, n, A.data(), b.data(), x.data());

    assertEquals(expected_x, x, "Solution incorrect!");
    std::cout << "OK" << std::endl;
}

// Test solving an underdetermined system (minimum norm)
void test_solve_underdetermined() {
    std::cout << "underdetermined system (minimum norm)..." << std::flush;

    const int m = 2, n = 3;
    std::vector<double> A = {
        1.0, 1.0, 1.0,
        0.0, 1.0, 2.0
    };
    std::vector<double> b = {6.0, 8.0};
    std::vector<double> x(n);

    // The known minimum-norm solution is x = {1, 2, 3}
    std::vector<double> expected_x = {1.0, 2.0, 3.0};

    QR<double> qr;
    qr.solve(m, n, A.data(), b.data(), x.data());

    // Test 1: The solution should be correct
    assertEquals(expected_x, x, "Solution is incorrect!");
    
    std::cout << "OK" << std::endl;
}
// A square system whose R has an exactly zero pivot. The third column is the sum of the
// first two, so x_2 is free; the solver must set it to 0 and report the part of b that no
// x can reach as residual, rather than leaving (Q^T b)_2 in x_2.
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

    // Ax = b is solved in the first two rows; the third row is unreachable.
    const std::vector<double> Ax = {x[0] + x[2], x[1] + x[2], 0.0};
    double resid_sq = 0.0;
    for (int i = 0; i < m; ++i) resid_sq += (Ax[i] - b[i]) * (Ax[i] - b[i]);

    assertEquals(25.0, resid_sq, "Returned x is not a least squares solution!");
    assertEquals(resid_sq, qr.residual_sq(), "Reported residual does not match the returned x!");
    std::cout << "OK" << std::endl;
}
