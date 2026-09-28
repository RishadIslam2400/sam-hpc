#include "sam.hpp"
#include "config.hpp"
#include "read_mat.hpp"
#include "timer.hpp"

int main(int argc, char** argv) {
  config_t cfg;
  parseargs(argc, argv, cfg);
  cfg.print();

  // Control TBB parallelism with num threads.
  num_threads = cfg.threads;
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  // Parallel benchmarking
  // Mapping the fifth matrix A5 to the initial matrix A0
  std::string filepath = cfg.matrix_dir + "matrix_1.txt";
  CSRMatrix<double> target = read_mat<double>(filepath.c_str());
  std::cout << "Target matrix : A0" << std::endl;

  // std::string source_path = "../../../top_opt_small_csr/matrix_6.txt";
  // this should be the relative path of the input file to be included in the bash script
  filepath = cfg.matrix_dir + "matrix_6.txt";
  CSRMatrix<double> source = read_mat<double>(filepath.c_str());
  std::cout << "Source matrix: A5" << std::endl;

  // Benchmarking SAM for 4 different sparsity patterns
  // This is comparing the total SAM computation
  std::cout << "Simple Sparsity Pattern (SAM computation): " << std::flush;
  Timer simple_timer{};
  simple_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Compute sparsity pattern
      SparsityPattern<double, SimplePattern> pattern(source, target, SimplePattern{});
      pattern.computePattern();

      // Compute map
      CSRMatrix<double> map;
      SparseApproximateMap<double, SimplePattern>::computeMap(target, source, pattern, map);
  }
  std::cout << (simple_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  std::cout << "Global Sparsity Pattern (SAM computation): " << std::flush;
  Timer global_timer{};
  global_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Global sparsity pattern
      GlobalThresholdPattern thresh{0.001}; // arbitrary
      SparsityPattern<double, GlobalThresholdPattern> pattern(source, target, thresh);
      pattern.computePattern();

      // Compute map
      CSRMatrix<double> map;
      SparseApproximateMap<double, GlobalThresholdPattern>::computeMap(target, source, pattern, map);
  }
  std::cout << (global_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  std::cout << "Column/Row Sparsity Pattern (SAM computation): " << std::flush;
  Timer local_timer{};
  local_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Column/Row sparsity pattern
      ColumnThresholdPattern thresh{0.8}; // arbitrary
      SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
      pattern.computePattern();

      // Compute map
      CSRMatrix<double> map;
      SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, pattern, map);
  }
  std::cout << (local_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  std::cout << "Fixed Sparsity Pattern (SAM computation): " << std::flush;
  Timer fixed_timer{};
  fixed_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Fixed NNZ sparsity pattern
      FixedNNZPattern thresh{5}; // arbitrary
      SparsityPattern<double, FixedNNZPattern> pattern(source, target, thresh);
      pattern.computePattern();

      // Compute map
      CSRMatrix<double> map;
      SparseApproximateMap<double, FixedNNZPattern>::computeMap(target, source, pattern, map);
  }
  std::cout << (fixed_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  // This is comparing the sparsity pattern computation
  std::cout << "Simple Sparsity Pattern (Sparsity pattern computation): " << std::flush;
  simple_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Compute sparsity pattern
      SparsityPattern<double, SimplePattern> pattern(source, target, SimplePattern{});
      pattern.computePattern();
  }
  std::cout << (simple_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  std::cout << "Global Sparsity Pattern (Sparsity pattern computation): " << std::flush;
  global_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Global sparsity pattern
      GlobalThresholdPattern thresh{0.001}; // arbitrary
      SparsityPattern<double, GlobalThresholdPattern> pattern(source, target, thresh);
      pattern.computePattern();
  }
  std::cout << (global_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  std::cout << "Column/Row Sparsity Pattern (Sparsity pattern computation): " << std::flush;
  local_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Column/Row sparsity pattern
      ColumnThresholdPattern thresh{0.8}; // arbitrary
      SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
      pattern.computePattern();
  }
  std::cout << (local_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  std::cout << "Fixed Sparsity Pattern (Sparsity pattern computation): " << std::flush;
  fixed_timer.start();
  for (int i = 0; i < cfg.iters; ++i) {
      // Fixed NNZ sparsity pattern
      FixedNNZPattern thresh{5}; // arbitrary
      SparsityPattern<double, FixedNNZPattern> pattern(source, target, thresh);
      pattern.computePattern();
  }
  std::cout << (fixed_timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

  // Count the number of non zero for each pattern
  std::cout << "Simple Sparsity Pattern (nnz): " << std::flush;
  SparsityPattern<double, SimplePattern> simplePattern(source, target, SimplePattern{});
  simplePattern.computePattern();
  std::cout << simplePattern.getNNZ() << std::endl;

  std::cout << "Global Sparsity Pattern (nnz): " << std::flush;
  GlobalThresholdPattern global_thresh{0.001};
  SparsityPattern<double, GlobalThresholdPattern> globalPattern(source, target, global_thresh);
  globalPattern.computePattern();
  std::cout << globalPattern.getNNZ() << std::endl;

  std::cout << "Column/Row Sparsity Pattern (nnz): " << std::flush;
  ColumnThresholdPattern col_thresh{0.8};
  SparsityPattern<double, ColumnThresholdPattern> columnPattern(source, target, col_thresh);
  columnPattern.computePattern();
  std::cout << columnPattern.getNNZ() << std::endl;

  std::cout << "Fixed Sparsity Pattern (nnz): " << std::flush;
  FixedNNZPattern lfil{5};
  SparsityPattern<double, FixedNNZPattern> fixedPattern(source, target, lfil);
  fixedPattern.computePattern();
  std::cout << fixedPattern.getNNZ() << std::endl;
}