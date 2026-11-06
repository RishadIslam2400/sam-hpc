#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

/// Tests 5000 random matrix pairs with dimensions from 1 to 16.
void testAddition() {
  for (int N = 0; N < 5e3; ++N) {
    std::cout << "\rmatrices addition...#" << N + 1 << std::flush;

    size_t rows = rand() % 16 + 1;
    size_t cols = rand() % 16 + 1;

    std::vector<std::vector<int>> classicMatrixA = generateRandomMatrix<int>(rows, cols);

    csr::CSRMatrix<int> sparseMatrixA(classicMatrixA);

    std::vector<std::vector<int>> classicMatrixB = generateRandomMatrix<int>(rows, cols);
    csr::CSRMatrix<int> sparseMatrixB(classicMatrixB);

    std::vector<std::vector<int>> manualResult = addMatrices<int>(classicMatrixA, classicMatrixB);

    assertEquals<std::vector<std::vector<int>>, csr::CSRMatrix<int>>(
      manualResult,
      matrix_sum(1, sparseMatrixA, 1, sparseMatrixB),
      "incorrect matrix summation"
    );
  }

  std::cout << " OK" << std::endl;
}