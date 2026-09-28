#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

void testVectorMultiplication() {
    for (int N = 0; N < 5e3; ++N) {
        std::cout << "\rvector multiplication...#" << N + 1 << std::flush;

        // generate random matrices
        size_t rows = rand() % 16 + 1;
        size_t cols = rand() % 16 + 1;

        std::vector<int> vec = generateRandomVector<int>(cols);
        std::vector<std::vector<int>> classicMatrix = generateRandomMatrix<int>(rows, cols);
        std::vector<int> result(rows);
        CSRMatrix<int> sparseMatrix(classicMatrix);

        // calculate result manually
        std::vector<int> manualResult = multiplyMatrixByVector(classicMatrix, vec);

        // Perform spmv
        spmv(1, sparseMatrix, vec, 0, result);

        // method
        assertEquals(manualResult, result, "Incorrect matrix-vector multiplication result");
    }

    std::cout << " OK" << std::endl;
}

void testMatrixMultiplication() {
    for (int N = 0; N < 5e3; ++N) {
        std::cout << "\rmatrices multiplication... #" << N + 1 << std::flush;
        
        // generate random matrices
        size_t rowsA = rand() % 16 + 1;
        size_t colsArowsB = rand() % 16 + 1;
        size_t colsB = rand() % 16 + 1;

        std::vector<std::vector<int>> classicMatrixA = generateRandomMatrix<int>(rowsA, colsArowsB);
        CSRMatrix<int> sparseMatrixA(classicMatrixA);

        std::vector<std::vector<int>> classicMatrixB = generateRandomMatrix<int>(colsArowsB, colsB);
        CSRMatrix<int> sparseMatrixB(classicMatrixB);

        // calculate result manually
        std::vector<std::vector<int>> manualResult = multiplyMatrices<int>(classicMatrixA, classicMatrixB);

        // method
        assertEquals<std::vector<std::vector<int>>, CSRMatrix<int>>(
            manualResult,
            *matrix_product(sparseMatrixA, sparseMatrixB, true),
            "Incorrect matrix-matrix multiplication result");
    }

    std::cout << " OK" << std::endl;
}