#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

/// Compare 5000 random sparse matrix-vector products with a dense reference.
void testVectorMultiplication() {
  for (int N = 0; N < 5e3; ++N) {
    std::cout << "\rvector multiplication...#" << N + 1 << std::flush;

    size_t rows = rand() % 16 + 1;
    size_t cols = rand() % 16 + 1;

    std::vector<int> vec = generateRandomVector<int>(cols);
    std::vector<std::vector<int>> classicMatrix = generateRandomMatrix<int>(rows, cols);
    std::vector<int> result(rows);
    csr::CSRMatrix<int> sparseMatrix(classicMatrix);

    std::vector<int> manualResult = multiplyMatrixByVector(classicMatrix, vec);

    spmv(1, sparseMatrix, vec, 0, result);

    assertEquals(manualResult, result, "Incorrect matrix-vector multiplication result");
  }

  std::cout << " OK" << std::endl;
}

/// Compare 5000 random sparse matrix products with a dense reference using
/// both SpGEMM accumulators.
void testMatrixMultiplication() {
  for (int N = 0; N < 5e3; ++N) {
    std::cout << "\rmatrices multiplication... #" << N + 1 << std::flush;
    
    size_t rowsA = rand() % 16 + 1;
    size_t colsArowsB = rand() % 16 + 1;
    size_t colsB = rand() % 16 + 1;

    std::vector<std::vector<int>> classicMatrixA = generateRandomMatrix<int>(rowsA, colsArowsB);
    csr::CSRMatrix<int> sparseMatrixA(classicMatrixA);

    std::vector<std::vector<int>> classicMatrixB = generateRandomMatrix<int>(colsArowsB, colsB);
    csr::CSRMatrix<int> sparseMatrixB(classicMatrixB);

    std::vector<std::vector<int>> manualResult = multiplyMatrices<int>(classicMatrixA, classicMatrixB);

    const spgemm_accumulator saved = spgemm_acc;
    spgemm_acc = spgemm_accumulator::dense;
    const auto dense_sorted = matrix_product(sparseMatrixA, sparseMatrixB, true);
    const auto dense_unsorted = matrix_product(sparseMatrixA, sparseMatrixB, false);
    spgemm_acc = spgemm_accumulator::hash;
    const auto hash_sorted = matrix_product(sparseMatrixA, sparseMatrixB, true);
    const auto hash_unsorted = matrix_product(sparseMatrixA, sparseMatrixB, false);
    spgemm_acc = saved;

    assertEquals<std::vector<std::vector<int>>, csr::CSRMatrix<int>>(
        manualResult, dense_sorted, "Incorrect matrix-matrix multiplication result (dense)");
    assertEquals<std::vector<std::vector<int>>, csr::CSRMatrix<int>>(
        manualResult, hash_sorted, "Incorrect matrix-matrix multiplication result (hash)");
    assertTrueMsg(dense_unsorted == hash_unsorted,
                  "Dense and hash accumulators disagree on the unsorted product");
  }

  std::cout << " OK" << std::endl;
}