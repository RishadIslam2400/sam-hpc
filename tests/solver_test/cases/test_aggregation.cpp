#include "sam_hpc/core/CSRMatrix.hpp"
#include "testlib.hpp"
#include "sam_hpc/precond/coarsening/plain_aggregates.hpp"
#include "sam_hpc/precond/coarsening/pointwise_aggregates.hpp"
#include "sam_hpc/precond/coarsening/tentative_prolongation.hpp"
#include "sam_hpc/precond/coarsening/smoothed_aggregation.hpp"

// Verify one aggregate and removal of weakly connected nodes.
void test_basic_aggregation() {
	std::cout << "plain aggregation with single aggregate..." << std::flush;

	// Nodes 0-2 are strongly connected; nodes 3-5 have no strong connections.
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
	csr::CSRMatrix<double> A(6, 6, values, row_ptr, col_ind);

	// With eps_strong^2 = 0.0064, links (0,1) and (1,2) are strong while (3,4) is weak.
	plain_aggregates::params prm;
	prm.eps_strong = 0.08f;
	plain_aggregates aggs(A, prm);

	// Expected: aggregate {0, 1, 2}; nodes 3-5 are removed.
	std::vector<ptrdiff_t> expected_id = {0, 0, 0, -2, -2, -2};
	assertEquals<size_t>(1, aggs.count, "Aggregate count is incorrect!");
	assertEquals(expected_id, aggs.id, "Aggregate ids incorrect!");

	std::cout << "OK" << std::endl;
}

// Verify that disconnected strong pairs form separate aggregates.
void test_multiple_aggregates() {
	std::cout << "plain aggregation with mutiple aggregates..." << std::flush;

	// Three strongly connected pairs have no links between them.
	std::vector<size_t> ptr = {0, 2, 4, 6, 8, 10, 12};
	std::vector<size_t> col = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
	std::vector<float> val = {10.f, -2.f, -2.f, 10.f, 10.f, -3.f, -3.f, 10.f, 10.f, -4.f, -4.f, 10.f};
	csr::CSRMatrix<float> A(6, 6, val, ptr, col);
	plain_aggregates::params prm;
	prm.eps_strong = 0.08f;

	// Each pair coupling is strong at eps_strong = 0.08, and no pairs are connected.
	plain_aggregates aggs(A, prm);

	// Expected: aggregates {0, 1}, {2, 3}, and {4, 5}.
	std::vector<ptrdiff_t> expected_id = {0, 0, 1, 1, 2, 2};
	assertEquals<size_t>(3, aggs.count, "Aggregate count is incorrect!");
	assertEquals(expected_id, aggs.id, "Aggregate ids are incorrect!");

	std::cout << "OK" << std::endl;
}

void test_remove_small_aggregates() {
	std::cout << "Removing small aggregates..." << std::flush;
	// Five disconnected groups with sizes 2, 3, 3, 4, and 4.
	size_t rows = 16, cols = 16;
	size_t nnz = 2*2 + 3*3 + 3*3 + 4*4 + 4*4;
	std::vector<size_t> ptr = {0, 2, 4, 7, 10, 13, 16, 19, 22, 26, 30, 34, 38, 42, 46, 50, 54};
	std::vector<size_t> col = {
		0, 1,         0, 1,
		2, 3, 4, 2, 3, 4, 2, 3, 4,
		5, 6, 7, 5, 6, 7, 5, 6, 7,
		8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
		12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	};
	std::vector<float> val;
	val.reserve(nnz);
	for(size_t i = 0; i < nnz; ++i)
		val.push_back(-2.f);
	for(int p : ptr) {
		if (p < nnz)
			val[p] = 10.f;
	}

	csr::CSRMatrix<float> A(rows, cols, nnz, val, ptr, col);
	pointwise_aggregates::params prm;
	prm.eps_strong = 0.01f;

	// Remove the only aggregate smaller than three and renumber the remaining aggregates.
	unsigned min_aggregate_size = 3;

	pointwise_aggregates aggs(A, prm, min_aggregate_size);

	// Expected: aggregate {0, 1} is removed and four aggregates remain.
	std::vector<ptrdiff_t> expected_id = {-2, -2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
	assertEquals<size_t>(4, aggs.count, "Aggregate count is incorrect!");
	assertEquals(expected_id, aggs.id, "Aggregate ids are incorrect!");
	std::cout << "OK" << std::endl;
}

// Without a nullspace, tentative prolongation is the aggregate indicator.
void test_piecewise_constant_prolongation() {
	std::cout << "tentative prolongation without nullspace..." << std::flush;

	size_t n = 5;
	size_t naggr = 2;
	std::vector<ptrdiff_t> aggr = {0, 1, 0, -2, 1};
	nullspace_params nullspace;
	int block_size = 1;

	std::shared_ptr<csr::CSRMatrix<double>> P = tentative_prolongation<double>(n, naggr, aggr, nullspace, block_size);

	std::vector<double> val = {1, 1, 1, 1};
	std::vector<size_t> ptr = {0, 1, 2, 3, 3, 4};
	std::vector<size_t> col = {0, 1, 0, 1};
	csr::CSRMatrix<double> exp_P(n, naggr, 4, val, ptr, col);
	assertEquals<csr::CSRMatrix<double>>(exp_P, *P, "Pronlongation operator incorrect!");

	std::cout << "OK" << std::endl;
}

// Verify prolongation from a two-column nullspace basis on each aggregate.
void test_nullspace_prolongation() {
	std::cout << "tentative prolongation with nullspace..." << std::flush;

	size_t n = 4;
	size_t naggr = 2;
	std::vector<ptrdiff_t> aggr = {0, 0, 1, 1};
	int block_size = 1;

	nullspace_params nullspace;
	nullspace.cols = 2;
	// B is a 4x2 row-major matrix.
	nullspace.B = {
		1.0, 0.0, // node 0
		0.0, 1.0, // node 1
		1.0, 0.0, // node 2
		0.0, 1.0  // node 3
	};

	std::shared_ptr<csr::CSRMatrix<double>> P = tentative_prolongation<double>(n, naggr, aggr, nullspace, block_size);

	std::vector<double> val = {1, 0, 0, 1, 1, 0, 0, 1};
	std::vector<size_t> ptr = {0, 2, 4, 6, 8};
	std::vector<size_t> col = {0, 1, 0, 1, 2, 3, 2, 3};
	csr::CSRMatrix<double> exp_P(n, naggr * nullspace.cols, n * nullspace.cols, val, ptr, col);
	assertEquals<csr::CSRMatrix<double>>(exp_P, *P, "Pronlongation operator incorrect!");

	std::cout << "OK" << std::endl;
}

void test_smoothed_transfer_operator() {
	std::cout << "transfer operator..." << std::flush;

	// Set up the test matrix.
	// A = [[10, -1, -4],
	//      [-1, 10, -1],
	//      [-4, -1, 10]]
	size_t n = 3;
	csr::CSRMatrix<double> A;
	A.m_rows = n;
	A.m_cols = n;
	A.m_nnz = 9;
	A.m_row_pointers = {0, 3, 6, 9};
	A.m_col_indices  = {0, 1, 2, 0, 1, 2, 0, 1, 2};
	A.m_vals         = {10, -1, -4, -1, 10, -1, -4, -1, 10};

	// Configure aggregation and relaxation.
	smoothed_aggregation<double>::params prm;
	// transfer_operators halves eps_strong for the next level, so the threshold is 0.04.
	// The (0,2) link is strong; the adjacent links are weak.
	prm.aggr.eps_strong = 0.2f;
	prm.relax = 1.0f;

	// Build the transfer operators.
	smoothed_aggregation<double> sa(prm);
	auto [P, R] = sa.transfer_operators(A);

	// The expected aggregate is {0, -2, 0}; smoothing gives 17/27 in rows 0 and 2.
	std::vector<size_t> row_pointers = {0, 1, 1, 2};
	std::vector<size_t> col_indices = {0, 0};
	std::vector<double> vals = {0.62962962962962965, 0.62962962962962965};
	csr::CSRMatrix<double> expectedP(3, 1, 2, vals, row_pointers, col_indices);

	assertEquals<csr::CSRMatrix<double>>(expectedP, *P, "Prolongation operator incorrect!");

	std::cout << "OK" << std::endl;
}