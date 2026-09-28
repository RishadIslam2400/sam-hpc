#include "testlib.hpp"
#include "helpers.hpp"
#include "qr.hpp"

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