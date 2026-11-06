#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/amg.hpp"
#include "helpers.hpp"
#include "testlib.hpp"
#include "dummy_precond.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"

// The identity preconditioner provides the unpreconditioned baseline.
void test_gmres_no_precond() {
	std::cout << "GMRES without preconditioner..." << std::flush;
	csr::CSRMatrix<double> A = create_laplacian(5);
	const int N = A.m_rows;

	std::vector<double> x_exact(N, 1.0);
	std::vector<double> rhs(N); 
	spmv(1.0, A, x_exact, 0.0, rhs);
	std::vector<double> x(N, 0.0);

	GMRES<double> solver(N);
	dummy_precond<double> P(A);
	auto [iters, error] = solver.solve(A, P, rhs, x);

	std::cout << "Iterations: " << iters << ", Relative Error: " << error << "..." << std::flush;
	assertEquals(x_exact, x, "GMRES solution incorrect!");
	std::cout << "OK" << std::endl;
}

// Right preconditioning monitors the true residual b - A x.
void test_gmres_amg_precond_1() {
	std::cout << "GMRES with amg right preconditioner..." << std::flush;
	csr::CSRMatrix<double> A = create_laplacian(5);
	const int N = A.m_rows;

	std::vector<double> x_exact(N, 1.0);
	std::vector<double> rhs(N); 
	spmv(1.0, A, x_exact, 0.0, rhs);
	std::vector<double> x(N, 0.0);

	GMRES<double>::params prm;
	prm.pside = precondSide::right;
	GMRES<double> solver(N, prm);
	amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
	amg_prm.coarse_enough = 2;
	amg_prm.coarsening.eps_strong = 0.0f;
	amg_prm.npre = 1;
	amg_prm.npost = 1;
	amg<double, ruge_stuben, damped_jacobi> P(A, amg_prm);
	auto [iters, error] = solver.solve(A, P, rhs, x);

	std::cout << "Iterations: " << iters << ", Relative Error: " << error << "..." << std::flush;
	assertEquals(x_exact, x, "GMRES solution incorrect!");
	std::cout << "OK" << std::endl;
}

// Left preconditioning monitors the preconditioned residual.
void test_gmres_amg_precond_2() {
	std::cout << "GMRES with amg left preconditioner..." << std::flush;
	csr::CSRMatrix<double> A = create_laplacian(5);
	const int N = A.m_rows;

	std::vector<double> x_exact(N, 1.0);
	std::vector<double> rhs(N); 
	spmv(1.0, A, x_exact, 0.0, rhs);
	std::vector<double> x(N, 0.0);

	GMRES<double>::params prm;
	prm.pside = precondSide::left;
	GMRES<double> solver(N, prm);
	amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
	amg_prm.coarse_enough = 2;
	amg_prm.coarsening.eps_strong = 0.0f;
	amg_prm.npre = 1;
	amg_prm.npost = 1;
	amg<double, ruge_stuben, damped_jacobi> P(A, amg_prm);
	auto [iters, error] = solver.solve(A, P, rhs, x);

	std::cout << "Iterations: " << iters << ", Relative Error: " << error << "..." << std::flush;
	assertEquals(x_exact, x, "GMRES solution incorrect!");
	std::cout << "OK" << std::endl;
}
// Force several restart cycles and verify the known solution.
void test_gmres_restart() {
	std::cout << "GMRES restarts and still converges..." << std::flush;

	csr::CSRMatrix<double> A = create_laplacian(12);
	const int N = A.m_rows;
	std::vector<double> x_exact(N, 1.0);
	std::vector<double> rhs(N);
	spmv(1.0, A, x_exact, 0.0, rhs);
	dummy_precond<double> P(A);

	for (precondSide side : {precondSide::right, precondSide::left}) {
		GMRES<double>::params prm;
		prm.pside = side;
		prm.M = 5;
		prm.maxIter = 3000;
		prm.tol = 1e-10;

		std::vector<double> x(N, 0.0);
		GMRES<double> solver(N, prm);
		auto [iters, error] = solver.solve(A, P, rhs, x);

		assertTrueMsg(iters > prm.M, "GMRES(5) never restarted - the test proves nothing!");
		assertTrueMsg(iters < prm.maxIter, "Restarted GMRES hit maxIter instead of converging!");
		assertEquals(x_exact, x, "Restarted GMRES did not reach the right solution!");
	}
	std::cout << "OK" << std::endl;
}

// Identity preconditioner scaled by a constant.
template <typename T> struct scaled_identity_precond {
	T c;
	void apply(const std::vector<T>& r, std::vector<T>& x) const {
		x.resize(r.size());
		for (size_t i = 0; i < r.size(); ++i)
			x[i] = c * r[i];
	}
};

// Scaling a left preconditioner by a constant must not change convergence or iteration
// counts. Test all overloads with c = 2^-20, 1, and 2^20 and check the true residual.
void test_gmres_left_tolerance_scale() {
	std::cout << "GMRES left preconditioning tolerance is scale-invariant..." << std::flush;
	csr::CSRMatrix<double> A = create_laplacian(5);
	const int N = A.m_rows;

	std::vector<double> x_exact(N, 1.0);
	std::vector<double> rhs(N);
	spmv(1.0, A, x_exact, 0.0, rhs);

	const auto true_res = [&](const std::vector<double>& x) {
		std::vector<double> r(N);
		residual(rhs, A, x, r);
		return norm(r) / norm(rhs);
	};

	for (int overload = 0; overload < 3; ++overload) {
		std::vector<size_t> iters;
		for (double c : {std::ldexp(1.0, -20), 1.0, std::ldexp(1.0, 20)}) {
			GMRES<double>::params prm;
			prm.pside = precondSide::left;
			prm.maxIter = 200;
			GMRES<double> solver(N, prm);
			scaled_identity_precond<double> P{c};
			std::vector<double> x(N, 0.0);
			auto [it, err] = overload == 0 ? solver.solve(A, P, rhs, x)
											: overload == 1 ? solver.solve(A, P, A, rhs, x)
																			: solver.solveRightMap(A, P, A, rhs, x);
			assertTrueMsg(true_res(x) < 1e-7, "Left-preconditioned GMRES stopped short of tol!");
			assertTrueMsg(it < prm.maxIter, "Left-preconditioned GMRES ran to maxIter!");
			iters.push_back(it);
		}
		assertTrueMsg(iters[0] == iters[1] && iters[1] == iters[2], "Stopping point depends on the scale of the preconditioner!");
	}
	std::cout << "OK" << std::endl;
}

// Identity preconditioner that rejects non-finite inputs.
template <typename T> struct finite_checking_precond {
	void apply(const std::vector<T>& r, std::vector<T>& x) const {
		for (T v : r)
			assertTrueMsg(std::isfinite(v), "Preconditioner handed a non-finite vector!");
		x = r;
	}
};

// A diagonal system with b = e_0 reaches the exact solution after one happy-breakdown
// step. Verify all overloads and preconditioning sides remain finite.
void test_gmres_happy_breakdown() {
	std::cout << "GMRES happy breakdown..." << std::flush;
	const int N = 5;
	std::vector<std::vector<double>> dense(N, std::vector<double>(N, 0.0)), eye = dense;
	for (int i = 0; i < N; ++i) { dense[i][i] = 2.0; eye[i][i] = 1.0; }
	const csr::CSRMatrix<double> A(dense), I(eye);

	std::vector<double> b(N, 0.0);
	b[0] = 1.0;
	std::vector<double> x_exact(N, 0.0);
	x_exact[0] = 0.5;

	for (precondSide side : {precondSide::right, precondSide::left}) {
		for (int overload = 0; overload < 3; ++overload) {
			GMRES<double>::params prm;
			prm.pside = side;
			GMRES<double> solver(N, prm);
			finite_checking_precond<double> P;
			std::vector<double> x(N, 0.0);
			auto [it, err] = overload == 0 ? solver.solve(A, P, b, x)
										 : overload == 1 ? solver.solve(A, P, I, b, x)
																		 : solver.solveRightMap(A, P, I, b, x);
			assertEquals<size_t>(1, it, "Breakdown should end the solve after one step!");
			assertTrueMsg(std::isfinite(err), "Non-finite residual after breakdown!");
			assertEquals(x_exact, x, "Wrong solution after breakdown!");
		}
	}
	std::cout << "OK" << std::endl;
}