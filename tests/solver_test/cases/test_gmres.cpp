#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/amg.hpp"
#include "helpers.hpp"
#include "testlib.hpp"
#include "dummy_precond.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"

void test_gmres_no_precond() {
    std::cout << "GMRES without preconditioner..." << std::flush;
    CSRMatrix<double> A = create_laplacian(5);
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

void test_gmres_amg_precond_1() {
    std::cout << "GMRES with amg right preconditioner..." << std::flush;
    CSRMatrix<double> A = create_laplacian(5);
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

void test_gmres_amg_precond_2() {
    std::cout << "GMRES with amg left preconditioner..." << std::flush;
    CSRMatrix<double> A = create_laplacian(5);
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