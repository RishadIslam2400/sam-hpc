#include "sam_hpc/precond/amg.hpp"
#include "helpers.hpp"
#include "testlib.hpp"

// standard v cycle - weighted jacobi smoothing with ruge-stuben coarsening
void test_standard_v_cycle_1() {
    std::cout << "v-cycle amg, weighted jacobi relaxation, ruge-stuben coarsening..." << std::flush;
    CSRMatrix<double> A = create_laplacian(5);
    const int N = A.m_rows;

    std::vector<double> x_exact(N, 1.0);
    std::vector<double> rhs(N); 
    spmv(1.0, A, x_exact, 0.0, rhs);
    std::vector<double> x(N, 0.0);

    amg<double, ruge_stuben, damped_jacobi>::params prm;
    prm.coarse_enough = 2;
    prm.coarsening.eps_strong = 0.0f;
    prm.npre = 2;
    prm.npost = 2;
    prm.ncycle = 1;

    amg<double, ruge_stuben, damped_jacobi> solver(A, prm);
    for (int i = 0; i < 10; ++i) {
        solver.cycle(rhs, x);
    }

    assertEquals(x_exact, x, "Solution not correct!");

    std::cout << "OK" << std::endl;
}

void test_standard_v_cycle_2() {
    std::cout << "v-cycle amg, gauss seidel relaxation, smoothed aggregation coarsening..." << std::flush;
    CSRMatrix<double> A = create_laplacian(5);
    const int N = A.m_rows;

    std::vector<double> x_exact(N, 1.0);
    std::vector<double> rhs(N); 
    spmv(1.0, A, x_exact, 0.0, rhs);
    std::vector<double> x(N, 0.0);

    amg<double, smoothed_aggregation, gauss_seidel>::params prm;
    prm.coarse_enough = 2;
    prm.coarsening.aggr.eps_strong = 0.0f;
    prm.npre = 2;
    prm.npost = 2;
    prm.ncycle = 1;

    amg<double, smoothed_aggregation, gauss_seidel> solver(A, prm);
    for (int i = 0; i < 10; ++i) {
        solver.cycle(rhs, x);
    }

    assertEquals(x_exact, x, "Solution not correct!");

    std::cout << "OK" << std::endl;
}