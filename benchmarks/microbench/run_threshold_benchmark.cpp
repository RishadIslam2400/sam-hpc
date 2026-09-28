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
    // Read the target matrix
    std::string filepath = cfg.matrix_dir + "matrix_1.txt";
    CSRMatrix<double> target = read_mat<double>(filepath.c_str());
    std::cout << "Target matrix : A0" << std::endl;

    filepath = cfg.matrix_dir + "matrix_6.txt";
    CSRMatrix<double> source = read_mat<double>(filepath.c_str());
    std::cout << "Source matrix: A5" << std::endl;

    // Benchmarking SAM with Column threshold with different threhold parameters
    // This is comparing the total SAM computation
    Timer timer;
    std::cout << "Column threshold parameter - 0.8 (SAM computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.8}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();

        // Compute map
        CSRMatrix<double> map;
        SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, pattern, map);
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row threhold parameter - 0.8 (Sparsity pattern computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.8}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row Sparsity Pattern (nnz): " << std::flush;
    ColumnThresholdPattern col_thresh1{0.8};
    SparsityPattern<double, ColumnThresholdPattern> columnPattern1(source, target, col_thresh1);
    columnPattern1.computePattern();
    std::cout << columnPattern1.getNNZ() << std::endl;

    std::cout << "Column threshold parameter - 0.85 (SAM computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.85}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();

        // Compute map
        CSRMatrix<double> map;
        SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, pattern, map);
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row threhold parameter - 0.85 (Sparsity pattern computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.85}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row Sparsity Pattern (nnz): " << std::flush;
    ColumnThresholdPattern col_thresh2{0.85};
    SparsityPattern<double, ColumnThresholdPattern> columnPattern2(source, target, col_thresh2);
    columnPattern2.computePattern();
    std::cout << columnPattern2.getNNZ() << std::endl;

    std::cout << "Column threshold parameter - 0.9 (SAM computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.9}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();

        // Compute map
        CSRMatrix<double> map;
        SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, pattern, map);
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row threhold parameter - 0.9 (Sparsity pattern computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.9}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row Sparsity Pattern (nnz): " << std::flush;
    ColumnThresholdPattern col_thresh3{0.9};
    SparsityPattern<double, ColumnThresholdPattern> columnPattern3(source, target, col_thresh3);
    columnPattern3.computePattern();
    std::cout << columnPattern3.getNNZ() << std::endl;

    std::cout << "Column threshold parameter - 0.95 (SAM computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.95}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();

        // Compute map
        CSRMatrix<double> map;
        SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, pattern, map);
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row threhold parameter - 0.95 (Sparsity pattern computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.95}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row Sparsity Pattern (nnz): " << std::flush;
    ColumnThresholdPattern col_thresh4{0.95};
    SparsityPattern<double, ColumnThresholdPattern> columnPattern4(source, target, col_thresh4);
    columnPattern4.computePattern();
    std::cout << columnPattern4.getNNZ() << std::endl;

    std::cout << "Column threshold parameter - 0.99 (SAM computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.99}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();

        // Compute map
        CSRMatrix<double> map;
        SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, pattern, map);
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row threhold parameter - 0.99 (Sparsity pattern computation): " << std::flush;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) {
        // Column/Row sparsity pattern
        ColumnThresholdPattern thresh{0.99}; // arbitrary
        SparsityPattern<double, ColumnThresholdPattern> pattern(source, target, thresh);
        pattern.computePattern();
    }
    std::cout << (timer.elapsed() / cfg.iters) / 1000000.0 << " s" << std::endl;

    std::cout << "Column/Row Sparsity Pattern (nnz): " << std::flush;
    ColumnThresholdPattern col_thresh5{0.99};
    SparsityPattern<double, ColumnThresholdPattern> columnPattern5(source, target, col_thresh5);
    columnPattern5.computePattern();
    std::cout << columnPattern5.getNNZ() << std::endl;
}