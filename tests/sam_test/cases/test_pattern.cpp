#include "CSRMatrix.hpp"
#include "sparsityPattern.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <iostream>

void testSimpleSparsityPattern()
{
    std::cout << "Simple sparsity pattern..." << std::flush;

    /*
        "Standard" matrix
        [ 1  0 4 5 ]
        [ 2 -1 0 0 ]
        [ 0  0 3 2 ]

        should be stored as
        rows:    [ 0, 3, 5, 7 ]
        columns: [ 0, 2, 3, 0, 1, 2, 3 ]
        values:  [ 1, 4, 5, 2, -1, 3, 2 ]
    */

    // Generate the matrix
    std::vector<size_t> rowPointers1 = {0, 3, 5, 7};
    std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3};
    std::vector<int> vals1 = {1, 4, 5, 2, -1, 3, 2};
    CSRMatrix<int> m1(3, 4, vals1, rowPointers1, colIndices1);

    // Generate the sparsity pattern
    SparsityPattern<int, SimplePattern> simplePattern(m1, m1, SimplePattern{});
    simplePattern.computePattern();
    const CSRMatrix<int>* recievedPattern = simplePattern.getPattern();

    std::vector<int> patternValuesCorrect(7, 1);

    assertEquals(recievedPattern->m_row_pointers, rowPointers1, "Incorrect internal row pointers");
    assertEquals(recievedPattern->m_col_indices, colIndices1, "Incorrect internal column indices");
    assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");

    std::cout << "OK" << std::endl;
}

void testGlobalSparsityPattern1()
{
    std::cout << "Global sparsity pattern 1..." << std::flush;

    /*
        "Standard" matrix
        [ 1  0 4 5 ]
        [ 2 -1 0 0 ]
        [ 0  0 3 2 ]
        [ 7  0 2 0 ]

        should be stored as
        rows:    [ 0, 3, 5, 7, 9 ]
        columns: [ 0, 2, 3, 0, 1, 2, 3, 0, 2 ]
        values:  [ 1, 4, 5, 2, -1, 3, 2, 7, 2 ]
    */

    // Generate the matrix
    std::vector<size_t> rowPointers1 = {0, 3, 5, 7, 9};
    std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3, 0, 2};
    std::vector<double> vals1 = {1, 4, 5, 2, -1, 3, 2, 7, 2};
    CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

    // Generate the sparsity pattern
    SparsityPattern<double, GlobalThresholdPattern> globalPattern(m1, m1, GlobalThresholdPattern{0.001});
    globalPattern.computePattern();
    const CSRMatrix<int> *recievedPattern = globalPattern.getPattern();

    std::vector<int> patternValuesCorrect(13, 1);
    std::vector<size_t> patternRowPointersCorrect = {0, 3, 7, 10, 13};
    std::vector<size_t> patternColIndicesCorrect = {0, 2, 3, 0, 1, 2, 3, 0, 2, 3, 0, 2, 3};

    assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
    assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
    assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values"); 

    std::cout << "OK" << std::endl;
}

void testGlobalSparsityPattern2()
{
    std::cout << "Global sparsity pattern 2..." << std::flush;

    /*
        "Standard" matrix
        [ 1  0 4 5 ]
        [ 2 -1 0 0 ]
        [ 0  0 3 2 ]
        [ 7  0 2 0 ]

        should be stored as
        rows:    [ 0, 3, 5, 7, 9 ]
        columns: [ 0, 2, 3, 0, 1, 2, 3, 0, 2 ]
        values:  [ 1, 4, 5, 2, -1, 3, 2, 7, 2 ]
    */

    // Generate the matrix
    std::vector<size_t> rowPointers1 = {0, 3, 5, 7, 9};
    std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3, 0, 2};
    std::vector<double> vals1 = {1, 4, 5, 2, -1, 3, 2, 7, 2};
    CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

    // Generate the sparsity pattern
    SparsityPattern<double, GlobalThresholdPattern> globalPattern(m1, m1, GlobalThresholdPattern{4.00});
    globalPattern.computePattern();
    const CSRMatrix<int> *recievedPattern = globalPattern.getPattern();

    std::vector<int> patternValuesCorrect = {1, 1, 1, 1, 1, 1};
    std::vector<size_t> patternRowPointersCorrect = {0, 2, 3, 4, 6};
    std::vector<size_t> patternColIndicesCorrect = {0, 3, 1, 2, 0, 3};

    assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
    assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
    assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");

    std::cout << "OK" << std::endl;
}

void testColumnSparsityPattern()
{
    std::cout << "Column sparsity pattern..." << std::flush;

    /*
        "Standard" matrix
        [ 1  0 4 5 ]
        [ 2 -1 0 8 ]
        [ 0  2 3 2 ]
        [ 7  0 0 0 ]

        should be stored as
        rows:    [ 0, 3, 6, 9, 10 ]
        columns: [ 0, 2, 3, 0, 1, 3, 1, 2, 3, 0 ]
        values:  [ 1, 4, 5, 2, -1, 8, 2, 3, 2, 7]
    */

    // Generate the matrix
    std::vector<size_t> rowPointers1 = {0, 3, 6, 9, 10};
    std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 3, 1, 2, 3, 0};
    std::vector<double> vals1 = {1, 4, 5, 2, -1, 8, 2, 3, 2, 7};
    CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

    // Generate the sparsity pattern
    SparsityPattern<double, ColumnThresholdPattern> columnPattern(m1, m1, ColumnThresholdPattern{0.5});
    columnPattern.computePattern();
    const CSRMatrix<int> *recievedPattern = columnPattern.getPattern();

    std::vector<int> patternValuesCorrect(14, 1);
    std::vector<size_t> patternRowPointersCorrect = {0, 4, 7, 11, 14};
    std::vector<size_t> patternColIndicesCorrect = {0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3, 0, 2, 3};

    assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
    assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
    assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");
    std::cout << "OK" << std::endl;
}

void testFixedNNZSparsityPattern()
{
    std::cout << "Fixed NNZ sparsity pattern..." << std::flush;

     /*
        "Standard" matrix
        [ 1  0 4 5 ]
        [ 2 -1 0 8 ]
        [ 0  2 3 4 ]
        [ 7  0 0 0 ]

        should be stored as
        rows:    [ 0, 3, 6, 9, 10 ]
        columns: [ 0, 2, 3, 0, 1, 3, 1, 2, 3, 0 ]
        values:  [ 1, 4, 5, 2, -1, 8, 2, 3, 2, 7]
    */

    // Generate the matrix
    std::vector<size_t> rowPointers1 = {0, 3, 6, 9, 10};
    std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 3, 1, 2, 3, 0};
    std::vector<double> vals1 = {1, 4, 5, 2, -1, 8, 2, 3, 4, 7};
    CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

    // Generate the sparsity pattern
    SparsityPattern<double, FixedNNZPattern> fixedNNZPattern(m1, m1, FixedNNZPattern{2});
    fixedNNZPattern.computePattern();
    const CSRMatrix<int> *recievedPattern = fixedNNZPattern.getPattern();

    std::vector<int> patternValuesCorrect(11, 1);
    std::vector<size_t> patternRowPointersCorrect = {0, 3, 6, 9, 11};
    std::vector<size_t> patternColIndicesCorrect = {0, 2, 3, 0, 2, 3, 0, 2, 3, 2, 3};

    assertEquals(recievedPattern->m_row_pointers, patternRowPointersCorrect, "Incorrect internal row pointers");
    assertEquals(recievedPattern->m_col_indices, patternColIndicesCorrect, "Incorrect internal column indices");
    assertEquals(recievedPattern->m_vals, patternValuesCorrect, "Incorrect internal values");
    std::cout << "OK" << std::endl;
}