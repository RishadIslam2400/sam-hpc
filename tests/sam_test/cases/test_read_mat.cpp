#include "read_mat.hpp"
#include "helpers.hpp"
#include "testlib.hpp"

void testReadMat1() {
    std::cout << "Matrix reader 1..." << std::flush;
    std::string filename = "/home/rishad/SAM-HPC/tests/sam_test/testMatrix.txt";
    CSRMatrix<double> testMatrix;

    // Read the matrix from the file
    if (!read_mat(filename.c_str(), testMatrix)) {
        throw FailureException("Failed to read matrix from file");        
    }

    const size_t expectedRows = 5;
    const size_t expectedCols = 5;
    const size_t expectedNNZ = 9;
    const std::vector<size_t> expectedRowPointers = {0, 2, 4, 6, 8, 9};
    const std::vector<size_t> expectedColIndices = {0, 2, 1, 3, 0, 4, 1, 4, 3};
    const std::vector<double> expectedValues = {5.0, 8.0, 3.0, 6.0, 9.0, 7.0, 4.0, 2.0, 1.0};

    assertEquals<size_t>(testMatrix.m_rows, expectedRows, "Row number mismatch");
    assertEquals<size_t>(testMatrix.m_cols, expectedCols, "Column number mismatch");
    assertEquals<size_t>(testMatrix.m_nnz, expectedNNZ, "NNZ mismatch");
    assertEquals(testMatrix.m_row_pointers, expectedRowPointers, "Row pointers mismatch");
    assertEquals(testMatrix.m_col_indices, expectedColIndices, "Column indices mismatch");
    assertEquals(testMatrix.m_vals, expectedValues, "Values mismatch");
    std::cout << "OK" << std::endl;
}

void testReadMat2() {
    std::cout << "Matrix reader 2..." << std::flush;
    std::string filename = "/home/rishad/SAM-HPC/tests/sam_test/testMatrix.txt";
    CSRMatrix<double> testMatrix = read_mat<double>(filename.c_str());
    if (testMatrix.isEmpty()) {
        throw FailureException("Failed to read matrix from file");
    }

    const size_t expectedRows = 5;
    const size_t expectedCols = 5;
    const size_t expectedNNZ = 9;
    const std::vector<size_t> expectedRowPointers = {0, 2, 4, 6, 8, 9};
    const std::vector<size_t> expectedColIndices = {0, 2, 1, 3, 0, 4, 1, 4, 3};
    const std::vector<double> expectedValues = {5.0, 8.0, 3.0, 6.0, 9.0, 7.0, 4.0, 2.0, 1.0};

    assertEquals<size_t>(testMatrix.m_rows, expectedRows, "Row number mismatch");
    assertEquals<size_t>(testMatrix.m_cols, expectedCols, "Column number mismatch");
    assertEquals<size_t>(testMatrix.m_nnz, expectedNNZ, "NNZ mismatch");
    assertEquals(testMatrix.m_row_pointers, expectedRowPointers, "Row pointers mismatch");
    assertEquals(testMatrix.m_col_indices, expectedColIndices, "Column indices mismatch");
    assertEquals(testMatrix.m_vals, expectedValues, "Values mismatch");
    std::cout << "OK" << std::endl;
}