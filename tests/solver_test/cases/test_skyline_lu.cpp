#include "sam_hpc/precond/skyline_lu.hpp"
#include "testlib.hpp"

void test_skyline_lu_solver() {
    std::cout << "skyline lu..." << std::flush;

    // 1. Setup the problem
    // A = [[ 2, -1,  0],
    //      [-1,  3, -1],
    //      [ 0, -1,  2]]
    size_t n = 3;
    std::vector<size_t> ptr = {0, 2, 5, 7};
    std::vector<size_t> col = {0, 1, 0, 1, 2, 1, 2};
    std::vector<double> val = {2, -1, -1, 3, -1, -1, 2};
    CSRMatrix<double> A(n, n, val, ptr, col);

    // Let the exact solution be x = [1, 2, 3]^T.
    // Then the right-hand side b = A*x is:
    // b = [2*1 - 1*2, -1*1 + 3*2 - 1*3, -1*2 + 2*3]^T = [0, 2, 4]^T
    std::vector<double> rhs = {0.0, 2.0, 4.0};
    std::vector<double> expected_x = {1.0, 2.0, 3.0};
    std::vector<double> x(n);

    // 2. Create the solver and solve the system.
    // The template parameter is the mock ordering class.
    skyline_lu<double, cuthill_mckee<double, false>> solver(A);
    solver(rhs, x);

    // 3. Assert the results
    assertEquals(expected_x, x);

    std::cout << "OK" << std::endl;
}