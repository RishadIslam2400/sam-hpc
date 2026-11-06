#include "sam_hpc/precond/skyline_lu.hpp"
#include "testlib.hpp"

void test_skyline_lu_solver() {
	std::cout << "skyline lu..." << std::flush;

	// Set up the tridiagonal system.
	size_t n = 3;
	std::vector<size_t> ptr = {0, 2, 5, 7};
	std::vector<size_t> col = {0, 1, 0, 1, 2, 1, 2};
	std::vector<double> val = {2, -1, -1, 3, -1, -1, 2};
	csr::CSRMatrix<double> A(n, n, val, ptr, col);

	// Use x = [1, 2, 3] with the corresponding right-hand side b = [0, 2, 4].
	std::vector<double> rhs = {0.0, 2.0, 4.0};
	std::vector<double> expected_x = {1.0, 2.0, 3.0};
	std::vector<double> x(n);

	// Use forward Cuthill-McKee ordering for fill reduction.
	skyline_lu<double, cuthill_mckee<double, false>> solver(A);
	solver(rhs, x);

	assertEquals(expected_x, x);

	std::cout << "OK" << std::endl;
}