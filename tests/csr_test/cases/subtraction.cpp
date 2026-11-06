#include "sam_hpc/core/CSRMatrix.hpp"
#include "testlib.hpp"
#include "helpers.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

/// Compare 5000 random same-shaped sparse matrix pairs with a dense reference.
void testSubtraction() {
  for (int N = 0; N < 5e3; ++N) {
    std::cout << "\rmatrices subtraction...#" << N + 1 << std::flush;

    size_t rows = rand() % 16 + 1;
    size_t cols = rand() % 16 + 1;

    std::vector<std::vector<int>> classicMatrixA = generateRandomMatrix<int>(rows, cols);
    csr::CSRMatrix<int> sparseMatrixA(classicMatrixA);

    std::vector<std::vector<int>> classicMatrixB = generateRandomMatrix<int>(rows, cols);
    csr::CSRMatrix<int> sparseMatrixB(classicMatrixB);

    std::vector<std::vector<int>> manualResult = subtractMatrices<int>(classicMatrixA, classicMatrixB);

    assertEquals<std::vector<std::vector<int>>, csr::CSRMatrix<int>>(
      manualResult, matrix_subtraction(1, sparseMatrixA, 1, sparseMatrixB),
      "incorrect matrices subtraction"
    );
  }

  std::cout << " OK" << std::endl;
}