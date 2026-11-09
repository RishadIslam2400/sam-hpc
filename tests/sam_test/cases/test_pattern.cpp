#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <iostream>

/// SimplePattern preserves the source matrix structure.
void testSimpleSparsityPattern() {
	std::cout << "Simple sparsity pattern..." << std::flush;

	/*
		Input matrix:
		[ 1  0 4 5 ]
		[ 2 -1 0 0 ]
		[ 0  0 3 2 ]
		[ 7  0 0 6 ]

		Expected CSR storage:
		rows:    [ 0, 3, 5, 7, 9 ]
		columns: [ 0, 2, 3, 0, 1, 2, 3, 0, 3 ]
		values:  [ 1, 4, 5, 2, -1, 3, 2, 7, 6 ]
	*/

	std::vector<size_t> rowPointers1 = {0, 3, 5, 7, 9};
	std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3, 0, 3};
	std::vector<int> vals1 = {1, 4, 5, 2, -1, 3, 2, 7, 6};
	csr::CSRMatrix<int> m1(4, 4, vals1, rowPointers1, colIndices1);

	SparsityPattern<int, SimplePattern> simplePattern(m1, m1, SimplePattern{});
	simplePattern.computePattern();
	const csr::CSRMatrix<int>* recievedPattern = simplePattern.getPattern();

	std::vector<int> patternValuesCorrect(9, 1);

	assertEquals(recievedPattern->m_row_pointers, rowPointers1, "Incorrect internal row pointers");
	assertEquals(recievedPattern->m_col_indices, colIndices1, "Incorrect internal column indices");
	assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");

	std::cout << "OK" << std::endl;
}

/// A low threshold retains all entries; diagonal completion and S^2 determine the result.
void testGlobalSparsityPattern1() {
	std::cout << "Global sparsity pattern 1..." << std::flush;

	/*
		Input matrix:
		[ 1  0 4 5 ]
		[ 2 -1 0 0 ]
		[ 0  0 3 2 ]
		[ 7  0 2 0 ]

		Expected CSR storage:
		rows:    [ 0, 3, 5, 7, 9 ]
		columns: [ 0, 2, 3, 0, 1, 2, 3, 0, 2 ]
		values:  [ 1, 4, 5, 2, -1, 3, 2, 7, 2 ]
	*/

	std::vector<size_t> rowPointers1 = {0, 3, 5, 7, 9};
	std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3, 0, 2};
	std::vector<double> vals1 = {1, 4, 5, 2, -1, 3, 2, 7, 2};
	csr::CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

	SparsityPattern<double, GlobalThresholdPattern> globalPattern(m1, m1, GlobalThresholdPattern{0.001});
	globalPattern.computePattern();
	const csr::CSRMatrix<int> *recievedPattern = globalPattern.getPattern();

	std::vector<int> patternValuesCorrect(13, 1);
	std::vector<size_t> patternRowPointersCorrect = {0, 3, 7, 10, 13};
	std::vector<size_t> patternColIndicesCorrect = {0, 2, 3, 0, 1, 2, 3, 0, 2, 3, 0, 2, 3};

	assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
	assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
	assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values"); 

	std::cout << "OK" << std::endl;
}

/// With threshold 4, only the dominant entries survive scaling; S^2 leaves the
/// resulting pattern unchanged.
void testGlobalSparsityPattern2() {
	std::cout << "Global sparsity pattern 2..." << std::flush;

	/*
		Input matrix:
		[ 1  0 4 5 ]
		[ 2 -1 0 0 ]
		[ 0  0 3 2 ]
		[ 7  0 2 0 ]

		Expected CSR storage:
		rows:    [ 0, 3, 5, 7, 9 ]
		columns: [ 0, 2, 3, 0, 1, 2, 3, 0, 2 ]
		values:  [ 1, 4, 5, 2, -1, 3, 2, 7, 2 ]
	*/

	std::vector<size_t> rowPointers1 = {0, 3, 5, 7, 9};
	std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3, 0, 2};
	std::vector<double> vals1 = {1, 4, 5, 2, -1, 3, 2, 7, 2};
	csr::CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

	SparsityPattern<double, GlobalThresholdPattern> globalPattern(m1, m1, GlobalThresholdPattern{4.00});
	globalPattern.computePattern();
	const csr::CSRMatrix<int> *recievedPattern = globalPattern.getPattern();

	std::vector<int> patternValuesCorrect = {1, 1, 1, 1, 1, 1};
	std::vector<size_t> patternRowPointersCorrect = {0, 2, 3, 4, 6};
	std::vector<size_t> patternColIndicesCorrect = {0, 3, 1, 2, 0, 3};

	assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
	assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
	assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");

	std::cout << "OK" << std::endl;
}

/// A column threshold of 0.5 keeps entries above half the scaled row maximum.
void testColumnSparsityPattern() {
	std::cout << "Column sparsity pattern..." << std::flush;

	/*
		Input matrix:
		[ 1  0 4 5 ]
		[ 2 -1 0 8 ]
		[ 0  2 3 2 ]
		[ 7  0 0 0 ]

		Expected CSR storage:
		rows:    [ 0, 3, 6, 9, 10 ]
		columns: [ 0, 2, 3, 0, 1, 3, 1, 2, 3, 0 ]
		values:  [ 1, 4, 5, 2, -1, 8, 2, 3, 2, 7]
	*/

	std::vector<size_t> rowPointers1 = {0, 3, 6, 9, 10};
	std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 3, 1, 2, 3, 0};
	std::vector<double> vals1 = {1, 4, 5, 2, -1, 8, 2, 3, 2, 7};
	csr::CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

	SparsityPattern<double, ColumnThresholdPattern> columnPattern(m1, m1, ColumnThresholdPattern{0.5});
	columnPattern.computePattern();
	const csr::CSRMatrix<int> *recievedPattern = columnPattern.getPattern();

	// Expected pattern after symmetric diagonal scaling and S^2 extension.
	std::vector<int> patternValuesCorrect(11, 1);
	std::vector<size_t> patternRowPointersCorrect = {0, 2, 5, 9, 11};
	std::vector<size_t> patternColIndicesCorrect = {0, 3, 0, 1, 3, 0, 1, 2, 3, 0, 3};

	assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
	assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
	assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");
	std::cout << "OK" << std::endl;
}

/// The column threshold is relative to the largest OFF-diagonal entry of the row, not the
/// row maximum. On a scaled SPD row the diagonal is 1 and dominates every off-diagonal, so
/// a row-maximum rule would reduce to the global threshold 1 - tau and keep nothing here.
void testColumnThresholdIgnoresDiagonal() {
	std::cout << "Column threshold relative to max off-diagonal..." << std::flush;

	/*
		Symmetric, unit diagonal (scaling leaves it unchanged):
		[ 1    0.4  0.25 0.05 ]
		[ 0.4  1    0    0    ]
		[ 0.25 0    1    0    ]
		[ 0.05 0    0    1    ]
		tau = 0.5: row 0 keeps |a| > 0.5 * 0.4 = 0.2 -> columns 1, 2 (not 3).
		Rows 1-3 have one off-diagonal each, which is their own maximum, so it is kept.
	*/
	std::vector<size_t> rp = {0, 4, 6, 8, 10};
	std::vector<size_t> ci = {0, 1, 2, 3, 0, 1, 0, 2, 0, 3};
	std::vector<double> v = {1, 0.4, 0.25, 0.05, 0.4, 1, 0.25, 1, 0.05, 1};
	csr::CSRMatrix<double> m(4, 4, v, rp, ci);

	SparsityPattern<double, ColumnThresholdPattern> column(m, m, ColumnThresholdPattern{0.5}, 1);
	column.computePattern();
	const csr::CSRMatrix<int> *p = column.getPattern();
	assertEquals(p->m_row_pointers, std::vector<size_t>{0, 3, 5, 7, 9}, "Incorrect column-threshold row pointers");
	assertEquals(p->m_col_indices, std::vector<size_t>{0, 1, 2, 0, 1, 0, 2, 0, 3}, "Incorrect column-threshold columns");

	// The global threshold 1 - tau = 0.5 keeps only the diagonal: the two rules now differ.
	SparsityPattern<double, GlobalThresholdPattern> global(m, m, GlobalThresholdPattern{0.5}, 1);
	global.computePattern();
	assertEquals(global.getPattern()->m_row_pointers, std::vector<size_t>{0, 1, 2, 3, 4}, "Global threshold should keep the diagonal only");
	std::cout << "OK" << std::endl;
}

/// FixedNNZPattern keeps the two largest scaled entries per row before S^2 extension.
void testFixedNNZSparsityPattern() {
	std::cout << "Fixed NNZ sparsity pattern..." << std::flush;

	/*
		Input matrix:
		[ 1  0 4 5 ]
		[ 2 -1 0 8 ]
		[ 0  2 3 4 ]
		[ 7  0 0 0 ]

		Expected CSR storage:
		rows:    [ 0, 3, 6, 9, 10 ]
		columns: [ 0, 2, 3, 0, 1, 3, 1, 2, 3, 0 ]
		values:  [ 1, 4, 5, 2, -1, 8, 2, 3, 4, 7]
	*/

	std::vector<size_t> rowPointers1 = {0, 3, 6, 9, 10};
	std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 3, 1, 2, 3, 0};
	std::vector<double> vals1 = {1, 4, 5, 2, -1, 8, 2, 3, 4, 7};
	csr::CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

	SparsityPattern<double, FixedNNZPattern> fixedNNZPattern(m1, m1, FixedNNZPattern{2});
	fixedNNZPattern.computePattern();
	const csr::CSRMatrix<int> *recievedPattern = fixedNNZPattern.getPattern();

	// Expected pattern after symmetric diagonal scaling and S^2 extension.
	std::vector<int> patternValuesCorrect(10, 1);
	std::vector<size_t> patternRowPointersCorrect = {0, 3, 6, 8, 10};
	std::vector<size_t> patternColIndicesCorrect = {0, 1, 3, 0, 2, 3, 0, 3, 2, 3};

	assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
	assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
	assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");
	std::cout << "OK" << std::endl;
}