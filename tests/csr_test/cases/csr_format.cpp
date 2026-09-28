#include "CSRMatrix.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

void testInternalStorage() {
    std::cout << "internal storage..." << std::flush;

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

    std::vector<size_t> rowPointers1 = {0, 3, 5, 7};
    std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3};
    std::vector<int> vals1 = {1, 4, 5, 2, -1, 3, 2};
    CSRMatrix<int> m1(3, 4, vals1, rowPointers1, colIndices1);

    assertEquals(rowPointers1, m1.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m1.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals1, m1.m_vals, "Incorrect internal values");

    /*
        Matrix with empty row
        [ 10 0 0 2 ]
        [  0 0 0 0 ]
        [  3 1 0 4 ]

        should be stored as
        rows:    [ 0, 2, 2, 5 ]
        columns: [ 0, 3, 0, 1, 3 ]
        values:  [ 10, 2, 3, 1, 4 ]
    */

    std::vector<size_t> rowPointers2 = {0, 2, 2, 5};
    std::vector<size_t> colIndices2 = {0, 3, 0, 1, 3};
    std::vector<int> vals2 = {10, 2, 3, 1, 4};
    CSRMatrix<int> m2(3, 4, vals2, rowPointers2, colIndices2);
    
    assertEquals(rowPointers2, m2.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices2, m2.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals2, m2.m_vals, "Incorrect internal values");

    /*
        Previous matrix after adding non-zero element to empty row

        should be stored as
        rows:    [ 0, 2, 3, 6 ]
        columns: [ 0, 3, 1, 0, 1, 3 ]
        values:  [ 10, 2, 5, 3, 1, 4 ]
     */

    CSRMatrix<int> m3 = m2;
    m3.set(5, 1, 1);

    std::vector<size_t> rowPointers3 = {0, 2, 3, 6};
    assertEquals(rowPointers3, m3.m_row_pointers, "Incorrect internal row pointers");

    std::vector<size_t> colIndices3 = {0, 3, 1, 0, 1, 3};
    assertEquals(colIndices3, m3.m_col_indices, "Incorrect internal column indices");

    std::vector<int> vals3 = {10, 2, 5, 3, 1, 4};
    assertEquals(vals3, m3.m_vals, "Incorrect internal values");

    /*
        Previous matrix with removed the only non-zero element on 2nd row (should be equal to 2nd matrix)

        should be stored as
        rows:    [ 0, 2, 2, 5 ]
        columns: [ 0, 3, 0, 1, 3 ]
        values:  [ 10, 2, 3, 1, 4 ]
     */

    CSRMatrix<int> m4 = m3;
    m4.set(0, 1, 1);

    assertEquals(rowPointers2, m4.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices2, m4.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals2, m4.m_vals, "Incorrect internal values");


    // Construct from dense 2D matrix
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

    std::vector<std::vector<int>> denseMatrix = {
        {1, 0, 4, 5},
        {2, -1, 0, 0},
        {0, 0, 3, 2}
    };

    // Construct a matrix using dense matrix
    CSRMatrix<int> m5(denseMatrix);
    assertEquals(rowPointers1, m5.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m5.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals1, m5.m_vals, "Incorrect internal values");

    // Testing row iterator
    auto row0 = m5.rowBegin(0);
    assertEquals<size_t>(0, row0.col(), "Incorrect row iterator.");
    assertEquals<int>(1, row0.value(), "Incorrect row iterator.");
    ++row0;
    assertEquals<size_t>(2, row0.col(), "Incorrect row iterator.");
    assertEquals<int>(4, row0.value(), "Incorrect row iterator.");
    ++row0;
    assertEquals<size_t>(3, row0.col(), "Incorrect row iterator.");
    assertEquals<int>(5, row0.value(), "Incorrect row iterator.");

    auto row1 = m5.rowBegin(1);
    assertEquals<size_t>(0, row1.col(), "Incorrect row iterator.");
    assertEquals<int>(2, row1.value(), "Incorrect row iterator.");
    ++row1;
    assertEquals<size_t>(1, row1.col(), "Incorrect row iterator.");
    assertEquals<int>(-1, row1.value(), "Incorrect row iterator.");

    // Construct a matrix using only vectors
    CSRMatrix<int> m6(vals1, rowPointers1, colIndices1);
    assertEquals(rowPointers1, m6.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m6.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals1, m6.m_vals, "Incorrect internal values");

    // Construct a matrix using non zero count
    CSRMatrix<int> m7(3, 4, 7, vals1, rowPointers1, colIndices1);
    assertEquals(rowPointers1, m7.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m7.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals1, m7.m_vals, "Incorrect internal values");

    // Construct a matrix using array pointers
    CSRMatrix<int> m8(3, 4, vals1.data(), rowPointers1.data(), colIndices1.data());
    assertEquals(rowPointers1, m8.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m8.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals1, m8.m_vals, "Incorrect internal values");

    // Constrcut a matrix using array pointers and non zero count
    CSRMatrix<int> m9(3, 4, 7, vals1.data(), rowPointers1.data(), colIndices1.data());
    assertEquals(rowPointers1, m9.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m9.m_col_indices, "Incorrect internal column indices");
    assertEquals(vals1, m9.m_vals, "Incorrect internal values");

    // Testing copy and move constructors
    CSRMatrix<int> testCSR(denseMatrix);

    // Using copy constructor
    CSRMatrix<int> m10(testCSR);
    assertEquals(rowPointers1, m10.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m10.m_col_indices, "Incorrect internal column indices.");
    assertEquals(vals1, m10.m_vals, "Incorrect internal values.");

    // Using copy assignment
    CSRMatrix<int> m11 = m10;
    assertEquals(rowPointers1, m11.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m11.m_col_indices, "Incorrect internal column indices.");
    assertEquals(vals1, m11.m_vals, "Incorrect internal values.");

    // Using move constructor
    CSRMatrix<int> m12(std::move(m11));
    assertEquals(rowPointers1, m12.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m12.m_col_indices, "Incorrect internal column indices.");
    assertEquals(vals1, m12.m_vals, "Incorrect internal values.");
    assertEquals(std::vector<size_t>(1, 0), m11.m_row_pointers, "Moved m_row_pointers is not empty.");
    assertEquals(std::vector<size_t>(), m11.m_col_indices, "Moved m_col_indices is not empty.");
    assertEquals(std::vector<int>(), m11.m_vals, "Moved m_vals is not empty.");

    // Using move assignment
    CSRMatrix<int> m13 = std::move(m10);
    assertEquals(rowPointers1, m13.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m13.m_col_indices, "Incorrect internal column indices.");
    assertEquals(vals1, m13.m_vals, "Incorrect internal values.");
    assertEquals(std::vector<size_t>(1, 0), m10.m_row_pointers, "Moved m_row_pointers is not empty.");
    assertEquals(std::vector<size_t>(), m10.m_col_indices, "Moved m_col_indices is not empty.");
    assertEquals(std::vector<int>(), m10.m_vals, "Moved m_vals is not empty.");

    // Construct by moving vectors
    std::vector<size_t> move_row_ptr = rowPointers1;
    std::vector<size_t> move_col_ind = colIndices1;
    std::vector<int> move_vals = vals1;
    CSRMatrix<int> m14(3, 4, std::move(move_vals), std::move(move_row_ptr), std::move(move_col_ind));
    assertEquals(rowPointers1, m14.m_row_pointers, "Incorrect internal row pointers");
    assertEquals(colIndices1, m14.m_col_indices, "Incorrect internal column indices.");
    assertEquals(vals1, m14.m_vals, "Incorrect internal values.");
    assertEquals(std::vector<size_t>(), move_row_ptr, "Moved m_row_pointers is not empty.");
    assertEquals(std::vector<size_t>(), move_col_ind, "Moved m_col_indices is not empty.");
    assertEquals(std::vector<int>(), move_vals, "Moved m_vals is not empty.");

    std::cout << " OK" << std::endl;
}