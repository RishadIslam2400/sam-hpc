#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

// Test case to verify the constructor and the inverse diagonal calculation
void test_jacobi_constructor() {
    std::cout << "constuctor... " << std::flush;
    // A = [[4, -1, -1],
    //      [-1, 4, -1],
    //      [-1, -1, 4]]
    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);

    typename damped_jacobi<double>::params prm;
    damped_jacobi<double> smoother(A, prm);

    // The diagonal of A is [4, 4, 4].
    // The inverse diagonal should be [0.25, 0.25, 0.25].
    std::vector<double> inverseDiag = {0.25, 0.25, 0.25};
    assertEquals(inverseDiag, smoother.dia, "Diagonal inverse not correct!");

    std::cout<< "OK" << std::endl;
}

// Test case for pre-smoothing
void test_jacobi_presmoothing() {
    std::cout << "presmoothing... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp(3);
    typename damped_jacobi<double>::params prm(0.5);
    damped_jacobi<double> smoother(A, prm);

    // Initial x = [0, 0, 0]
    // Initial residual r = rhs - A * x = rhs = [2, 2, 2]
    // Update: x_new = x_old + damping * D^-1 * r
    // x_new = [0, 0, 0] + 0.5 * [0.25, 0.25, 0.25] * [2, 2, 2];
    // x_new = 0.5 * [0.5, 0.5, 0.5]
    // x_new = [0.25, 0.25, 0.25]
    smoother.apply_pre(A, rhs, x, tmp);

    std::vector<double> x_new = {0.25, 0.25, 0.25};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}

// Test case for post-smoothing
void test_jacobi_postsmoothing() {
    std::cout << "postsmoothing... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp(3);
    typename damped_jacobi<double>::params prm(0.5);
    damped_jacobi<double> smoother(A, prm);

    // Initial x = [0, 0, 0]
    // Initial residual r = rhs - A * x = rhs = [2, 2, 2]
    // Update: x_new = x_old + damping * D^-1 * r
    // x_new = [0, 0, 0] + 0.5 * [0.25, 0.25, 0.25] * [2, 2, 2];
    // x_new = 0.5 * [0.5, 0.5, 0.5]
    // x_new = [0.25, 0.25, 0.25]
    smoother.apply_post(A, rhs, x, tmp);

    std::vector<double> x_new = {0.25, 0.25, 0.25};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}

// Test case for direct solver (without damping factor)
void test_direct_jacobi() {
    std::cout << "direct solver... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp(3);
    typename damped_jacobi<double>::params prm(0.5);
    damped_jacobi<double> smoother(A, prm);

    // apply should compute x = D^-1 * rhs
    // x = [0.25, 0.25, 0.25] * [2, 2, 2]
    // x = [0.5, 0.5, 0.5]
    smoother.apply(A, rhs, x);

    std::vector<double> x_new = {0.5, 0.5, 0.5};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}