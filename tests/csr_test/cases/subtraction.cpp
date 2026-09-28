#include "sam_hpc/core/CSRMatrix.hpp"
#include "testlib.hpp"
#include "helpers.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

void testSubtraction() {
    for (int N = 0; N < 5e3; ++N) {
        std::cout << "\rmatrices subtraction...#" << N + 1 << std::flush;

        // generate random matrices
        size_t rows = rand() % 16 + 1;
        size_t cols = rand() % 16 + 1;

        std::vector<std::vector<int>> classicMatrixA = generateRandomMatrix<int>(rows, cols);
        CSRMatrix<int> sparseMatrixA(classicMatrixA);

        std::vector<std::vector<int>> classicMatrixB = generateRandomMatrix<int>(rows, cols);
        CSRMatrix<int> sparseMatrixB(classicMatrixB);

        // calculate results manually
        std::vector<std::vector<int>> manualResult = subtractMatrices<int>(classicMatrixA, classicMatrixB);

        // method
        assertEquals<std::vector<std::vector<int>>, CSRMatrix<int>>(
            manualResult,
            *matrix_subtraction(1, sparseMatrixA, 1, sparseMatrixB, true),
            "incorrect matrices subtraction");
    }

    std::cout << " OK" << std::endl;
}