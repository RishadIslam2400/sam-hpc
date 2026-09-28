#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"


template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<T> generateRandomVector(size_t size) {
    std::vector<T> vector(size, T());

    for (size_t i = 0; i < size; ++i) {
        vector[i] = rand() % 101;
    }

    return vector;
}

template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<std::vector<T>> generateRandomMatrix(size_t rows, size_t cols) {
    std::vector<std::vector<T>> matrix(rows, std::vector<T>(cols, T()));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 101;
        }
    }

    return matrix;
}

template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<std::vector<T>> addMatrices(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b) {
    size_t rows = a.size();
    size_t cols = a.front().size();
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols, T()));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    return result;
}

template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<std::vector<T>> subtractMatrices(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b) {
    size_t rows = a.size();
    size_t cols = a.front().size();
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols, T()));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }

    return result;
}

template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<T> multiplyMatrixByVector(const std::vector<std::vector<T>> &matrix, const std::vector<T> &vec) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<T> result(rows, T());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<std::vector<T>> multiplyMatrices(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b) {
    size_t rowsA = a.size();
    size_t colsA = a.front().size();
    size_t colsB = b.front().size();

    std::vector<std::vector<T>> result(rowsA, std::vector<T>(colsB, T()));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            result[i][j] = T();
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

template <typename T>
    requires std::is_arithmetic_v<T>
std::vector<T> multiplyMatrices(int rA, int cA, const std::vector<T>& A, int rB, int cB, const std::vector<T>& B) {
    assert(cA == rB);
    std::vector<T> C(rA * cB, 0.0);
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            for (int k = 0; k < cA; ++k) {
                C[i * cB + j] += A[i * cA + k] * B[k * cB + j];
            }
        }
    }
    return C;
}

// Helper to create a 2D Laplacian matrix for testing
CSRMatrix<double> create_laplacian(int n_rows) {
    const int N = n_rows * n_rows;
    std::vector<size_t> ptr;
    std::vector<size_t> col;
    std::vector<double> val;
    ptr.push_back(0);
    for (int i = 0; i < N; ++i) {
        val.push_back(4.0);
        col.push_back(i);
        if (i % n_rows > 0) { val.push_back(-1.0); col.push_back(i - 1); }
        if (i % n_rows < n_rows - 1) { val.push_back(-1.0); col.push_back(i + 1); }
        if (i / n_rows > 0) { val.push_back(-1.0); col.push_back(i - n_rows); }
        if (i / n_rows < n_rows - 1) { val.push_back(-1.0); col.push_back(i + n_rows); }
        ptr.push_back(val.size());
    }
    return CSRMatrix<double>(N, N, val, ptr, col);
}