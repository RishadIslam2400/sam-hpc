#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/amg.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"
#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "helpers.hpp"
#include "testlib.hpp"

#include <cmath>

// GMRES with a SAM map solves N A x = N b for a left map and A N u = b for a right map.
// The fixture uses a shifted target so the map is not the identity. The exact solution is
// known: x = 1 and b = A x.
// Compute the true relative residual independently of GMRES bookkeeping.
double sam_true_rel_residual(const csr::CSRMatrix<double>& A, const std::vector<double>& b,
                             const std::vector<double>& x)
{
	std::vector<double> t(b.size(), 0.0);
	spmv(1.0, A, x, 0.0, t);
	double num = 0.0, den = 0.0;
	for (size_t i = 0; i < b.size(); ++i) {
		const double d = b[i] - t[i];
		num += d * d;
		den += b[i] * b[i];
	}
	return std::sqrt(num) / std::sqrt(den);
}

// Build a nonsymmetric source, shifted target, and selected map.
void sam_gmres_fixture(csr::CSRMatrix<double>& source, csr::CSRMatrix<double>& target,
                       csr::CSRMatrix<double>& map, mapSide side = mapSide::left)
{
	source = create_laplacian(5);
	source.sortRows();
	for (size_t i = 0; i < source.m_rows; ++i)
		for (size_t p = source.m_row_pointers[i]; p < source.m_row_pointers[i + 1]; ++p)
			if (source.m_col_indices[p] == i + 1)
				source.m_vals[p] = -0.25;

	// Shift the diagonal so the target differs from the source.
	target = source;
	for (size_t i = 0; i < target.m_rows; ++i) {
		for (size_t p = target.m_row_pointers[i]; p < target.m_row_pointers[i + 1]; ++p) {
			if (target.m_col_indices[p] == i)
				target.m_vals[p] += 2.0;
		}
	}

	SparsityPattern<double, SimplePattern> pattern(source, target, SimplePattern());
	pattern.computePattern();
	SparseApproximateMap<double, SimplePattern>::computeMapSided(side, target, source, *pattern.getPattern(), map);
}

// Verify that either map side solves the original system A x = b.
void test_gmres_sam_solves_the_original_system(precondSide side, const char* side_name,
                                               mapSide ms, const char* ms_name)
{
	std::cout << "GMRES with " << ms_name << " SAM map solves A x = b (" << side_name
						<< ")..." << std::flush;

	csr::CSRMatrix<double> source, target, map;
	sam_gmres_fixture(source, target, map, ms);
	const int N = source.m_rows;

	std::vector<double> x_exact(N, 1.0);
	std::vector<double> b(N);
	spmv(1.0, source, x_exact, 0.0, b);

	amg<double, ruge_stuben, damped_jacobi> P0(target);

	GMRES<double>::params prm;
	prm.pside = side;
	prm.tol = 1e-12;
	prm.maxIter = 500;
	GMRES<double> solver(N, prm);

	std::vector<double> x(N, 0.0);
	auto [iters, error] = ms == mapSide::left ? solver.solve(source, P0, map, b, x)
																						: solver.solveRightMap(source, P0, map, b, x);

	// The map must not change the requested solution.
	assertEquals(x_exact, x, "SAM map solve did not solve A x = b!");

	// Check the true residual independently.
	const double truth = sam_true_rel_residual(source, b, x);
	assertEquals(0.0, truth, "SAM map solve did not converge in ||b - A x|| / ||b||!", 1e-8);

	// The reported residual must use the original system.
	assertEquals(truth, error,
							 "SAM map solve reported a residual that is not ||b - A x|| / ||b||!", 1e-9);

	// Ensure convergence occurred before maxIter.
	assertTrueMsg(iters < prm.maxIter,
							  "SAM map solve hit maxIter - it did not converge, it just ran out!");

	std::cout << "OK (" << iters << " iters)" << std::endl;
}

void test_gmres_sam_right() {
  test_gmres_sam_solves_the_original_system(precondSide::right, "right", mapSide::left, "left");
}
void test_gmres_sam_left() {
  test_gmres_sam_solves_the_original_system(precondSide::left, "left", mapSide::left, "left");
}
void test_gmres_sam_rightmap_right() {
  test_gmres_sam_solves_the_original_system(precondSide::right, "right", mapSide::right, "right");
}
void test_gmres_sam_rightmap_left() {
  test_gmres_sam_solves_the_original_system(precondSide::left, "left", mapSide::right, "right");
}

// Left and right map solves must agree on x for the same nonsymmetric system.
void test_gmres_sam_both_sides_agree() {
	std::cout << "Left-map and right-map solves agree on x..." << std::flush;

	csr::CSRMatrix<double> source, target, mapL, mapR;
	sam_gmres_fixture(source, target, mapL, mapSide::left);
	sam_gmres_fixture(source, target, mapR, mapSide::right);
	const int N = source.m_rows;

	std::vector<double> x_exact(N, 1.0);
	std::vector<double> b(N);
	spmv(1.0, source, x_exact, 0.0, b);

	amg<double, ruge_stuben, damped_jacobi> P0(target);
	GMRES<double>::params prm;
	prm.pside = precondSide::right;
	prm.tol = 1e-12;
	prm.maxIter = 500;

	std::vector<double> xL(N, 0.0), xR(N, 0.0);
	GMRES<double> sL(N, prm), sR(N, prm);
	auto [itL, errL] = sL.solve(source, P0, mapL, b, xL);
	auto [itR, errR] = sR.solveRightMap(source, P0, mapR, b, xR);

	assertEquals(xL, xR, "Left-map and right-map solves disagree on x!");
	assertTrueMsg(errL < 1e-8 && errR < 1e-8, "One of the two map solves did not converge!");

	std::cout << "OK (left " << itL << " iters, right " << itR << " iters)" << std::endl;
}

// A zero right-hand side has x = 0 as its exact solution.
void test_gmres_sam_zero_rhs() {
	std::cout << "GMRES with SAM map on a zero rhs..." << std::flush;

	csr::CSRMatrix<double> source, target, map;
	sam_gmres_fixture(source, target, map);
	const int N = source.m_rows;

	amg<double, ruge_stuben, damped_jacobi> P0(target);
	GMRES<double> solver(N);
	std::vector<double> b(N, 0.0);
	std::vector<double> x(N, 0.0);

	auto [iters, error] = solver.solve(source, P0, map, b, x);
	assertEquals<size_t>(0, iters, "Zero rhs should take no iterations!");
	std::cout << "OK" << std::endl;
}

// Force multiple restart cycles for both map overloads.
void test_gmres_sam_restart() {
	std::cout << "SAM map solves restart correctly..." << std::flush;

	for (mapSide ms : {mapSide::left, mapSide::right}) {
		csr::CSRMatrix<double> source, target, map;
		sam_gmres_fixture(source, target, map, ms);
		const int N = source.m_rows;

		std::vector<double> x_exact(N, 1.0);
		std::vector<double> b(N);
		spmv(1.0, source, x_exact, 0.0, b);

		amg<double, ruge_stuben, damped_jacobi> P0(target);

		GMRES<double>::params prm;
		prm.pside = precondSide::right;
		prm.M = 3;
		prm.maxIter = 2000;
		prm.tol = 1e-12;

		std::vector<double> x(N, 0.0);
		GMRES<double> solver(N, prm);
		auto [iters, error] = ms == mapSide::left
																? solver.solve(source, P0, map, b, x)
																: solver.solveRightMap(source, P0, map, b, x);

		assertTrueMsg(iters > prm.M, "GMRES(3) never restarted - the test proves nothing!");
		assertTrueMsg(iters < prm.maxIter, "Restarted map solve hit maxIter!");
		assertEquals(x_exact, x, "Restarted map solve did not reach the right solution!");
	}
	std::cout << "OK" << std::endl;
}

// With ns_search, a zero rhs must produce a finite residual from a nonzero initial guess.
void test_gmres_sam_ns_search_zero_rhs() {
	std::cout << "GMRES ns_search on a zero rhs returns a finite residual..." << std::flush;

	csr::CSRMatrix<double> source, target, map;
	sam_gmres_fixture(source, target, map);
	const int N = source.m_rows;
	amg<double, ruge_stuben, damped_jacobi> P0(target);
	const std::vector<double> b(N, 0.0);

	for (int overload = 0; overload < 3; ++overload) {
		GMRES<double>::params prm;
		prm.ns_search = true;
		GMRES<double> solver(N, prm);
		std::vector<double> x(N, 1.0);
		auto [iters, error] = overload == 0 ? solver.solve(source, P0, b, x)
												: overload == 1 ? solver.solve(source, P0, map, b, x)
																				: solver.solveRightMap(source, P0, map, b, x);
		assertTrueMsg(std::isfinite(error), "ns_search residual is not finite!");
		assertTrueMsg(error < 1e-7, "ns_search did not converge!");
		assertTrueMsg(iters < prm.maxIter, "ns_search ran to maxIter!");
	}
	std::cout << "OK" << std::endl;
}
