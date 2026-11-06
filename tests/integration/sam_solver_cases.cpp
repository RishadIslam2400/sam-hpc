#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/amg.hpp"
#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "sam_hpc/precond/coarsening/smoothed_aggregation.hpp"
#include "sam_hpc/precond/gauss_seidel.hpp"
#include "sam_hpc/precond/ilu0.hpp"
#include "sam_hpc/precond/iluk.hpp"
#include "sam_hpc/precond/ilut.hpp"

/// Compare mapped and independently preconditioned solves for cd2d.
/// Uses Ruge-Stuben AMG with damped Jacobi and the source structure.
void test_sam_solver_cd2d_1() {
  std::cout << "CD2D solve with GMRES (simple sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
  amg_prm.coarse_enough = 8;
  amg_prm.npre = 1;
  amg_prm.npost = 1;
  amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::right;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);

  amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}

/// Use left preconditioning and a global threshold of 0.001.
void test_sam_solver_cd2d_2() {
  std::cout << "CD2D solve with GMRES (global threshold sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
  amg_prm.coarse_enough = 8;
  amg_prm.npre = 1;
  amg_prm.npost = 1;
  amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::left;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);

  amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  GlobalThresholdPattern thresh{0.001};
  SparsityPattern<double, GlobalThresholdPattern> pattern(sourceMatrix, targetMatrix, thresh);
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, GlobalThresholdPattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}

/// Use smoothed-aggregation AMG with Gauss-Seidel, left preconditioning, and a column threshold of 0.9.
void test_sam_solver_cd2d_3() {
  std::cout << "CD2D solve with GMRES (column threshold sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, smoothed_aggregation, gauss_seidel>::params amg_prm;
  amg_prm.coarse_enough = 8;
  amg_prm.npre = 1;
  amg_prm.npost = 1;
  amg<double, smoothed_aggregation, gauss_seidel> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::left;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);

  amg<double, smoothed_aggregation, gauss_seidel> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  ColumnThresholdPattern thresh{0.9};
  SparsityPattern<double, ColumnThresholdPattern> pattern(sourceMatrix, targetMatrix, thresh);
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}

/// Use right preconditioning and a fixed five entries per row.
void test_sam_solver_cd2d_4() {
  std::cout << "CD2D solve with GMRES (fixed nnz sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/target_rhs.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_TEST_DATA_DIR) + "cd2d/source_rhs.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, smoothed_aggregation, gauss_seidel>::params amg_prm;
  amg_prm.coarse_enough = 8;
  amg_prm.npre = 1;
  amg_prm.npost = 1;
  amg<double, smoothed_aggregation, gauss_seidel> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::right;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);

  amg<double, smoothed_aggregation, gauss_seidel> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  FixedNNZPattern thresh{5};
  SparsityPattern<double, FixedNNZPattern> pattern(sourceMatrix, targetMatrix, thresh);
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, FixedNNZPattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}

/// Compare mapped and independently preconditioned top_opt solves using
/// Ruge-Stuben AMG with damped Jacobi and the source structure.
void test_sam_solver_top_opt_1() {
  std::cout << "Topology Optmization solve with GMRES and AMG (simple sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_1.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_1.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_2.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_2.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, ruge_stuben, damped_jacobi>::params amg_prm;
  amg_prm.coarse_enough = 8;
  amg_prm.npre = 1;
  amg_prm.npost = 1;
  amg<double, ruge_stuben, damped_jacobi> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::left;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);
  GMRES<double> solver_init(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);
  std::vector<double> x_init(N, 0.0);

  auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
  std::cout << "\n(A0) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


  amg<double, ruge_stuben, damped_jacobi> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}

/// Use ILU(0) smoothing with two pre- and post-smoothing steps.
void test_sam_solver_top_opt_2() {
  std::cout << "Topology Optmization solve with GMRES and AMG ILU(0) Smoother(simple sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_1.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_1.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_2.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_2.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, ruge_stuben, ilu0>::params amg_prm;
  amg_prm.npre = 2;
  amg_prm.npost = 2;
  amg<double, ruge_stuben, ilu0> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::left;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);
  GMRES<double> solver_init(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);
  std::vector<double> x_init(N, 0.0);

  auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
  std::cout << "\n(A0) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


  amg<double, ruge_stuben, ilu0> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}

/// Use right preconditioning and ILUT smoothing with two pre- and post-smoothing steps.
void test_sam_solver_top_opt_3() {
  std::cout << "Topology Optmization solve with GMRES and AMG ILU(k) Smoother(simple sparsity pattern)..." << std::flush;

  csr::CSRMatrix<double> targetMatrix = read_mat<double>((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_1.txt").c_str());
  std::vector<double> target_rhs = read_vec<double>((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_1.txt").c_str());

  csr::CSRMatrix<double> sourceMatrix = read_mat<double>((std::string(SAM_DATA_DIR) + "top_opt_matrices_small_csr/matrix_2.txt").c_str());
  std::vector<double> source_rhs = read_vec<double>((std::string(SAM_DATA_DIR) + "top_opt_rhs_small/rhs_2.txt").c_str());

  const size_t N = targetMatrix.m_rows;    
  amg<double, ruge_stuben, ilut>::params amg_prm;
  amg_prm.npre = 2;
  amg_prm.npost = 2;
  amg<double, ruge_stuben, ilut> P0(targetMatrix, amg_prm);

  GMRES<double>::params prm;
  prm.pside = precondSide::right;
  GMRES<double> solver_with_map(N, prm);
  GMRES<double> solver_without_map(N, prm);
  GMRES<double> solver_init(N, prm);

  std::vector<double> x_with_map(N, 0.0);
  std::vector<double> x_without_map(N, 0.0);
  std::vector<double> x_init(N, 0.0);

  auto [init_iters, init_error] = solver_init.solve(targetMatrix, P0, target_rhs, x_init);
  std::cout << "\n(A0) Iterations: " << init_iters << ", Relative Error: " << init_error << "..." << std::flush;


  amg<double, ruge_stuben, ilut> P1(sourceMatrix, amg_prm);

  auto [without_map_iters, without_map_error] = solver_without_map.solve(sourceMatrix, P1, source_rhs, x_without_map);
  std::cout << "\n(without map) Iterations: " << without_map_iters << ", Relative Error: " << without_map_error << "..." << std::flush;

  SparsityPattern<double, SimplePattern> pattern(sourceMatrix, targetMatrix, SimplePattern());
  pattern.computePattern();
  csr::CSRMatrix<double> map{};
  SparseApproximateMap<double, SimplePattern>::computeMap(targetMatrix, sourceMatrix, pattern, map);

  auto [with_map_iters, with_map_error] = solver_with_map.solve(sourceMatrix, P0, map, source_rhs, x_with_map);
  std::cout << "\n(with map) Iterations: " << with_map_iters << ", Relative Error: " << with_map_error << "..." << std::flush;

  std::cout << "OK" << std::endl;
}