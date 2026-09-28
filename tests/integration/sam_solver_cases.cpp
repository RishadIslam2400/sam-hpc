#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "testlib.hpp"
#include "helpers.hpp"
#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/amg.hpp"
#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "sam_hpc/precond/coarsening/smoothed_aggregation.hpp"
#include "sam_hpc/precond/gauss_seidel.hpp"
#include "sam_hpc/precond/ilu0.hpp"
#include "sam_hpc/precond/iluk.hpp"
#include "sam_hpc/precond/ilutp.hpp"

void test_sam_solver_cd2d_1() {
    std::cout << "CD2D solve with GMRES (simple sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
    amg_prm.coarse_enough = 8;
    amg_prm.npre = 1;
    amg_prm.npost = 1;
    amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::right;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);

    // Set up preconditioner P1 for the target matrix
    amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_cd2d_2() {
    std::cout << "CD2D solve with GMRES (global threshold sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
    amg_prm.coarse_enough = 8;
    amg_prm.npre = 1;
    amg_prm.npost = 1;
    amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::left;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);

    // Set up preconditioner P1 for the target matrix
    amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    GlobalThresholdPattern thresh{0.001};
    SparsityPattern<double, GlobalThresholdPattern> pattern(sourceMatrix, targetMatrix, thresh);
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, GlobalThresholdPattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_cd2d_3() {
    std::cout << "CD2D solve with GMRES (column threshold sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, smoothed_aggregation, gauss_seidel>::params amg_prm;
    amg_prm.coarse_enough = 8;
    amg_prm.npre = 1;
    amg_prm.npost = 1;
    amg<double, smoothed_aggregation, gauss_seidel> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::left;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);

    // Set up preconditioner P1 for the target matrix
    amg<double, smoothed_aggregation, gauss_seidel> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    ColumnThresholdPattern thresh{0.9};
    SparsityPattern<double, ColumnThresholdPattern> pattern(sourceMatrix, targetMatrix, thresh);
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_cd2d_4() {
    std::cout << "CD2D solve with GMRES (fixed nnz sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, smoothed_aggregation, gauss_seidel>::params amg_prm;
    amg_prm.coarse_enough = 8;
    amg_prm.npre = 1;
    amg_prm.npost = 1;
    amg<double, smoothed_aggregation, gauss_seidel> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::right;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);

    // Set up preconditioner P1 for the target matrix
    amg<double, smoothed_aggregation, gauss_seidel> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    FixedNNZPattern thresh{5};
    SparsityPattern<double, FixedNNZPattern> pattern(sourceMatrix, targetMatrix, thresh);
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, FixedNNZPattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_top_opt_1() {
    std::cout << "Topology Optmization solve with GMRES and AMG (simple sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_1.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_1.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_2.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_2.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
    amg_prm.coarse_enough = 8;
    amg_prm.npre = 1;
    amg_prm.npost = 1;
    amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::left;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);
    GMRES<double> solver_init(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);
    std::vector<double> x_init(N, 0.0);

    // Solve initial matrix
    auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
    std::cout << "\n(A0) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


    // Set up preconditioner P1 for the target matrix
    amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_top_opt_2() {
    std::cout << "Topology Optmization solve with GMRES and AMG ILU(0) Smoother(simple sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_1.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_1.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_2.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_2.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, ruge_stuben, ilu0>::params amg_prm;
    amg_prm.npre = 2;
    amg_prm.npost = 2;
    amg<double, ruge_stuben, ilu0> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::left;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);
    GMRES<double> solver_init(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);
    std::vector<double> x_init(N, 0.0);

    // Solve initial matrix
    auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
    std::cout << "\n(A0) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


    // Set up preconditioner P1 for the target matrix
    amg<double, ruge_stuben, ilu0> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_top_opt_3() {
    std::cout << "Topology Optmization solve with GMRES and AMG ILU(k) Smoother(simple sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_1.txt").c_str(), targetMatrix);
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_1.txt").c_str(), target_rhs);

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_2.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_2.txt").c_str(), source_rhs);

    // Set up preconditioner P0 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    amg<double, ruge_stuben, ilutp>::params amg_prm;
    amg_prm.npre = 2;
    amg_prm.npost = 2;
    amg<double, ruge_stuben, ilutp> P0(targetMatrix, amg_prm);

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::right;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);
    GMRES<double> solver_init(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);
    std::vector<double> x_init(N, 0.0);

    // Solve initial matrix
    auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
    std::cout << "\n(A0) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


    // Set up preconditioner P1 for the target matrix
    amg<double, ruge_stuben, ilutp> P1(sourceMatrix, amg_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}

void test_sam_solver_weather_1() {
    std::cout << "NWP solve with GMRES and AMG (simple sparsity pattern)..." << std::flush;

    CSRMatrix<double> targetMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "weather_matrices_18x27/matrix_40.txt").c_str(), targetMatrix);
    std::cout << "Target matrix rows = " << targetMatrix.m_rows << std::endl;
    std::vector<double> target_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "weather_rhs_18x27/rhs_40.txt").c_str(), target_rhs);
    std::cout << "Target rhs rows = " << target_rhs.size() << std::endl;

    CSRMatrix<double> sourceMatrix;
    read_mat((std::string(SAM_DATA_DIR) + "weather_matrices_18x27/matrix_50.txt").c_str(), sourceMatrix);
    std::vector<double> source_rhs;
    read_vec((std::string(SAM_DATA_DIR) + "weather_rhs_18x27/rhs_50.txt").c_str(), source_rhs);

    // Set up preconditioner P40 for the target matrix
    const size_t N = targetMatrix.m_rows;    
    // amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
    // amg_prm.coarse_enough = 32;
    // amg_prm.npre = 1;
    // amg_prm.npost = 1;
    // amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);
    ilutp<double>::params ilu_prm;
    ilu_prm.fill_factor = 250;
    ilu_prm.droptol = 1e-3;
    ilutp<double> P0(targetMatrix, ilu_prm);

    std::cout << "Ilutp initialized\n";

    // Solve source matrix with GMRES solver with and without map
    GMRES<double>::params prm;
    prm.pside = precondSide::left;
    prm.M = 2000;
    prm.maxIter = 2000;
    GMRES<double> solver_with_map(N, prm);
    GMRES<double> solver_without_map(N, prm);
    GMRES<double> solver_init(N, prm);

    std::vector<double> x_with_map(N, 0.0);
    std::vector<double> x_without_map(N, 0.0);
    std::vector<double> x_init(N, 0.0);

    std::cout << "Solve initial matrix\n";
    auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
    std::cout << "\n(A40) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


    // Set up preconditioner P1 for the target matrix
    // amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);
    ilutp<double> P1(sourceMatrix, ilu_prm);

    auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
    std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

    // Compute map
    SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
    pattern.computePattern();
    CSRMatrix<double> map{};
    SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

    auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
    std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

    std::cout << "OK" << std::endl;
}