#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "sam_hpc/core/CSRMatrix.hpp"

/**
 * @brief: Read a sparse matrix stored in CSR format in a text file. The first line in the
 * text file contains the number of rows, columns, and non-zero elements. The following lines contain
 * the row pointers, column indices, and values of the non-zero elements. The function populates the
 * the provided CSRMatrix object with the data read from the file.
 *
 * @param filename: A C-style string representing the path to the input text file containing the matrix.
 * @param matrix: A pointer to an uninitialized or empty CSRMatrix<double> object that will be filled
 *                with the matrix data read from the file.
 * @return true If the matrix was read successfully, false otherwise.
 */
template <typename T>
bool read_mat(const char *filename, CSRMatrix<T> &matrix) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    size_t rows, cols, nnz;
    std::string line;

    // First line: rows, cols, nnz
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading matrix dimensions from file: " << filename << std::endl;
        return false;
    }

    std::istringstream header(line);
    if (!(header >> rows >> cols >> nnz)) {
        std::cerr << "Error parsing matrix dimensions from file: " << filename << std::endl;
        return false;
    }

    matrix.m_rows = rows;
    matrix.m_cols = cols;
    matrix.m_nnz = nnz;

    // Second line: row pointers
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading row pointers from file: " << filename << std::endl;
        return false;
    }
    std::istringstream rowPtrStream(line);
    matrix.m_row_pointers.resize(rows + 1, 0);
    for (size_t i = 0; i <= rows; ++i) {
        if (!(rowPtrStream >> matrix.m_row_pointers[i])) {
            std::cerr << "Error parsing row pointers from file: " << filename << std::endl;
            return false;
        }
    }

    // Third line: column indices
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading column indices from file: " << filename << std::endl;
        return false;
    }
    std::istringstream colIdxStream(line);
    matrix.m_col_indices.resize(nnz, 0);
    for (size_t i = 0; i < nnz; ++i) {
        if (!(colIdxStream >> matrix.m_col_indices[i])) {
            std::cerr << "Error parsing column indices from file: " << filename << std::endl;
            return false;
        }
    }

    // Fourth line: values
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading values from file: " << filename << std::endl;
        return false;
    }
    std::istringstream valStream(line);
    matrix.m_vals.resize(nnz, T(0));
    for (size_t i = 0; i < nnz; ++i) {
        if (!(valStream >> matrix.m_vals[i])) {
            std::cerr << "Error parsing values from file: " << filename << std::endl;
            return false;
        }
    }

    infile.close();
    return true;
}

/**
 * @brief: Read a sparse matrix stored in CSR format in a text file. The first line in the
 * text file contains the number of rows, columns, and non-zero elements. The following lines contain
 * the row pointers, column indices, and values of the non-zero elements. The function creates a new CSRMatrix object 
 * with the data read from the file.
 *
 * @param filename: A C-style string representing the path to the input text file containing the matrix.
 * @param matrix: A pointer to an uninitialized or empty CSRMatrix<double> object that will be filled
 *                with the matrix data read from the file.
 * @return A CSRMatrix object containing the matrix data read from the file.
 */
template <typename T>
CSRMatrix<T> read_mat(const char *filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return CSRMatrix<T>();
    }

    size_t rows, cols, nnz;
    std::string line;

    // First line: rows, cols, nnz
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading matrix dimensions from file: " << filename << std::endl;
        return CSRMatrix<T>();
    }

    std::istringstream header(line);
    if (!(header >> rows >> cols >> nnz)) {
        std::cerr << "Error parsing matrix dimensions from file: " << filename << std::endl;
        return CSRMatrix<T>();
    }

    // Second line: row pointers
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading row pointers from file: " << filename << std::endl;
        return CSRMatrix<T>();
    }
    std::istringstream rowPtrStream(line);
    std::vector<size_t> row_pointers(rows + 1);
    for (size_t i = 0; i <= rows; ++i) {
        if (!(rowPtrStream >> row_pointers[i])) {
            std::cerr << "Error parsing row pointers from file: " << filename << std::endl;
            return CSRMatrix<T>();
        }
    }

    // Third line: column indices
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading column indices from file: " << filename << std::endl;
        return CSRMatrix<T>();
    }
    std::istringstream colIdxStream(line);
    std::vector<size_t> col_indices(nnz);
    for (size_t i = 0; i < nnz; ++i) {
        if (!(colIdxStream >> col_indices[i])) {
            std::cerr << "Error parsing column indices from file: " << filename << std::endl;
            return CSRMatrix<T>();
        }
    }

    // Fourth line: values
    if (!std::getline(infile, line)) {
        std::cerr << "Error reading values from file: " << filename << std::endl;
        return CSRMatrix<T>();
    }
    std::istringstream valStream(line);
    std::vector<T> vals(nnz);
    for (size_t i = 0; i < nnz; ++i) {
        if (!(valStream >> vals[i])) {
            std::cerr << "Error parsing values from file: " << filename << std::endl;
            return CSRMatrix<T>();
        }
    }

    infile.close();

    return CSRMatrix<T>(rows, cols, nnz, vals, row_pointers, col_indices);
}

/**
 * @brief Reads a vector from a text file where each line contains one element.
 *
 * @tparam T The data type of the vector elements (e.g., double, float, int).
 * @param filename A C-style string representing the path to the input text file.
 * @param vec A pointer to a std::vector<T> object that will be filled
 * with the data read from the file.
 * @return true if the file was read successfully, false if the file could not be opened.
 */
template <typename T>
bool read_vec(const char *filename, std::vector<T> &vec) {
    // Open the input file stream.
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    // Clear the vector to ensure it's empty before filling.
    vec.clear();

    std::string line;
    T value;

    // Read the file line by line.
    while (std::getline(infile, line)) {
        // Use a string stream to parse the value from the current line.
        std::istringstream valueStream(line);

        if (valueStream >> value) {
            vec.push_back(value);
        } else {
            if (!line.empty()) {
                 std::cerr << "Warning: Could not parse value from line: \"" << line << "\" in file: " << filename << std::endl;
            }
        }
    }

    // Close the file stream.
    infile.close();
    return true;
}

template <typename T>
std::vector<T> read_vec(const char *filename) {
    // Open the input file stream.
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> vec;
    std::string line;
    T value;

    // Read the file line by line.
    while (std::getline(infile, line)) {
        // Use a string stream to parse the value from the current line.
        std::istringstream valueStream(line);

        if (valueStream >> value) {
            vec.push_back(value);
        } else {
            if (!line.empty()) {
                 std::cerr << "Warning: Could not parse value from line: \"" << line << "\" in file: " << filename << std::endl;
            }
        }
    }

    // Close the file stream.
    infile.close();
    return vec;
}