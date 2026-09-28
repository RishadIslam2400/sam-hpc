#include "sam_hpc/core/CSRMatrix.hpp"
#include "testlib.hpp"
#include "sam_hpc/precond/coarsening/plain_aggregates.hpp"
#include "sam_hpc/precond/coarsening/pointwise_aggregates.hpp"
#include "sam_hpc/precond/coarsening/tentative_prolongation.hpp"
#include "sam_hpc/precond/coarsening/smoothed_aggregation.hpp"

// test case for a basic scenerio with one aggergate and removed nodes
void test_basic_aggregation() {
    std::cout << "plain aggregation with single aggregate..." << std::flush;

    // define a 6x6 matrix
    // nodes 0, 1, 2 are strongly connected
    // nodes 3, 4 are weakly connected
    // node 5 is isolaed
    //
    // A = [[10, -2,  0,  0,   0,   0],
    //      [-2, 10, -3,  0,   0,   0],
    //      [ 0, -3, 10,  0,   0,   0],
    //      [ 0,  0,  0, 10,-0.1,   0],
    //      [ 0,  0,  0,-0.1, 10,   0],
    //      [ 0,  0,  0,  0,   0,  10]]
    std::vector<size_t> row_ptr = {0, 2, 5, 7, 9, 11, 12};
    std::vector<size_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5};
    std::vector<double> values = {10, -2, -2, 10, -3, -3, 10, 10, -0.1, -0.1, 10, 10};
    CSRMatrix<double> A(6, 6, values, row_ptr, col_ind);

    plain_aggregates::params prm;
    prm.eps_strong = 0.08f;
    plain_aggregates aggs(A, prm);

    // Expected outcome:
    // - One aggregate is formed from nodes {0, 1, 2}.
    // - Nodes 3, 4, 5 are "removed" as they have no strong connections.
    std::vector<ptrdiff_t> expected_id = {0, 0, 0, -2, -2, -2};
    assertEquals<size_t>(1, aggs.count, "Aggregate count is incorrect!");
    assertEquals(expected_id, aggs.id, "Aggregate ids incorrect!");

    std::cout << "OK" << std::endl;
}

// Test case for a matrix that should result in multiple aggregates.
void test_multiple_aggregates() {
    std::cout << "plain aggregation with mutiple aggregates..." << std::flush;

    // Define a 6x6 matrix with three distinct, strongly connected pairs
    std::vector<size_t> ptr = {0, 2, 4, 6, 8, 10, 12};
    std::vector<size_t> col = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
    std::vector<float> val = {10.f, -2.f, -2.f, 10.f, 10.f, -3.f, -3.f, 10.f, 10.f, -4.f, -4.f, 10.f};
    CSRMatrix<float> A(6, 6, val, ptr, col);
    plain_aggregates::params prm;
    prm.eps_strong = 0.08f;

    // Perform aggregation
    plain_aggregates aggs(A, prm);

    // Expected outcome:
    // - Three aggregates are formed: {0, 1}, {2, 3}, and {4, 5}.
    std::vector<ptrdiff_t> expected_id = {0, 0, 1, 1, 2, 2};
    assertEquals<size_t>(3, aggs.count, "Aggregate count is incorrect!");
    assertEquals(expected_id, aggs.id, "Aggregate ids are incorrect!");

    std::cout << "OK" << std::endl;
}

void test_remove_small_aggregates() {
    std::cout << "Removing small aggregates..." << std::flush;
    // Define a 16x16 matrix with 5 distinct groups.
    // Agg 0: {0, 1}          (size 2)
    // Agg 1: {2, 3, 4}       (size 3)
    // Agg 2: {5, 6, 7}       (size 3)
    // Agg 3: {8, 9, 10, 11}  (size 4)
    // Agg 4: {12, 13, 14, 15}(size 4)
    size_t rows = 16, cols = 16;
    size_t nnz = 2*2 + 3*3 + 3*3 + 4*4 + 4*4; // 4 + 9 + 9 + 16 + 16 = 54
    std::vector<size_t> ptr = {0, 2, 4, 7, 10, 13, 16, 19, 22, 26, 30, 34, 38, 42, 46, 50, 54};
    std::vector<size_t> col = {
        0, 1, 0, 1,                                     // Group 0 (size 2)
        2, 3, 4, 2, 3, 4, 2, 3, 4,                      // Group 1 (size 3)
        5, 6, 7, 5, 6, 7, 5, 6, 7,                      // Group 2 (size 3)
        8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, // Group 3 (size 4)
        12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15 // Group 4 (size 4)
    };
    std::vector<float> val;
    val.reserve(nnz);
    for(size_t i = 0; i < nnz; ++i) val.push_back(-2.f); // Off-diagonals
    for(int p : ptr) { // Diagonals
        if (p < nnz) val[p] = 10.f;
    }

    CSRMatrix<float> A(rows, cols, nnz, val, ptr, col);
    pointwise_aggregates::params prm;
    prm.eps_strong = 0.01f; // Ensure all connections are strong

    // Set the minimum aggregate size to 2.
    // This should remove the aggregate of size 1 formed by node 4.
    unsigned min_aggregate_size = 3;

    // Perform aggregation and filtering
    pointwise_aggregates aggs(A, prm, min_aggregate_size);

    // Expected outcome: 4 aggregates remain.
    std::vector<ptrdiff_t> expected_id = {-2, -2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    assertEquals<size_t>(4, aggs.count, "Aggregate count is incorrect!");
    assertEquals(expected_id, aggs.id, "Aggregate ids are incorrect!");
    std::cout << "OK" << std::endl;
}

// Test case for piecewise-constant prolongation (no nullspace provided).
void test_piecewise_constant_prolongation() {
    std::cout << "tentative prolongation without nullspace..." << std::flush;

    size_t n = 5;                                   // 5 fine grid nodes
    size_t naggr = 2;                               // 2 aggregates
    std::vector<ptrdiff_t> aggr = {0, 1, 0, -2, 1}; // Node mapping to aggregates
    nullspace_params nullspace;                     // No nullspace info
    int block_size = 1;

    std::shared_ptr<CSRMatrix<double>> P = tentative_prolongation<double>(n, naggr, aggr, nullspace, block_size);

    std::vector<double> val = {1, 1, 1, 1};
    std::vector<size_t> ptr = {0, 1, 2, 3, 3, 4};
    std::vector<size_t> col = {0, 1, 0, 1};
    CSRMatrix<double> exp_P(n, naggr, 4, val, ptr, col);
    assertEquals<CSRMatrix<double>>(exp_P, *P, "Pronlongation operator incorrect!");

    std::cout << "OK" << std::endl;
}

// Test case for nullspace-based prolongation.
void test_nullspace_prolongation() {
    std::cout << "tentative prolongation with nullspace..." << std::flush;

    size_t n = 4;
    size_t naggr = 2;
    std::vector<ptrdiff_t> aggr = {0, 0, 1, 1};
    int block_size = 1;

    nullspace_params nullspace;
    nullspace.cols = 2;
    // B is a 4x2 matrix (row major)
    nullspace.B = {
        1.0, 0.0, // node 0
        0.0, 1.0, // node 1
        1.0, 0.0, // node 2
        0.0, 1.0  // node 3
    };

    std::shared_ptr<CSRMatrix<double>> P = tentative_prolongation<double>(n, naggr, aggr, nullspace, block_size);

    std::vector<double> val = {1, 0, 0, 1, 1, 0, 0, 1};
    std::vector<size_t> ptr = {0, 2, 4, 6, 8};
    std::vector<size_t> col = {0, 1, 0, 1, 2, 3, 2, 3};
    CSRMatrix<double> exp_P(n, naggr * nullspace.cols, n * nullspace.cols, val, ptr, col);
    assertEquals<CSRMatrix<double>>(exp_P, *P, "Pronlongation operator incorrect!");

    std::cout << "OK" << std::endl;
}

void test_smoothed_transfer_operator() {
    std::cout << "transfer operator..." << std::flush;

    // 1. Setup the problem
    // A = [[10, -1, -4],
    //      [-1, 10, -1],
    //      [-4, -1, 10]]
    size_t n = 3;
    CSRMatrix<double> A;
    A.m_rows = n;
    A.m_cols = n;
    A.m_nnz = 9;
    A.m_row_pointers = {0, 3, 6, 9};
    A.m_col_indices  = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    A.m_vals         = {10, -1, -4, -1, 10, -1, -4, -1, 10};

    // 2. Setup parameter
    smoothed_aggregation<double>::params prm;
    // Set eps_strong so that after halving it becomes 0.1.
    // eps_strong^2 = 0.01.
    // A(0,2)^2 / (A(0,0)A(2,2)) = 16/100 = 0.16 > 0.01 -> strong
    // Other off-diagonals are weak.
    prm.aggr.eps_strong = 0.2f;
    prm.relax = 1.0f;

    // 3. Create the smoothed_aggregation object and call the method
    smoothed_aggregation<double> sa(prm);
    auto [P, R] = sa.transfer_operators(A);

    // Based on manual calculation:
    // - Aggregation should be {0, -2, 0} (nodes 0 and 2 in agg 0).
    // - P_tent should be [1, 0, 1]^T.
    // - Smoothed P should be a 3x1 matrix with values [0.629, 0, 0.629].
    std::vector<size_t> row_pointers = {0, 1, 1, 2};
    std::vector<size_t> col_indices = {0, 0};
    std::vector<double> vals = {0.62962962962962965, 0.62962962962962965};
    CSRMatrix<double> expectedP(3, 1, 2, vals, row_pointers, col_indices);

    assertEquals<CSRMatrix<double>>(expectedP, *P, "Prolongation operator incorrect!");

    std::cout << "OK" << std::endl;
}