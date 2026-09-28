#pragma once

#include <eigen3/Eigen/Dense>
#include <span>

inline void eigenQRSolve(std::vector<std::vector<double>>& A,
                         std::vector<double>& rhs, std::span<double> x,
                         const size_t cols, const size_t rows)
{
  if (rows == 0 || cols == 0) {
    return;
  }

  assert((A.size() == cols && A[0].size() == rows) && "Matrix A must be of size rows x cols.");
  assert(x.size() == cols && "Vector X must be equal to the number of columns.");
  assert(rhs.size() == rows && "RHS vector must be equal to the number of rows.");

  // Convert A and rhs to Eigen matrices insert into eigen matrix in row-major order
  Eigen::MatrixXd eigenA(rows, cols);
  Eigen::VectorXd eigenRhs(rows);
  for (size_t i = 0; i < cols; ++i) {
    for (size_t j = 0; j < rows; ++j) {
      eigenA(j, i) = A[i][j];
    }
  }
  for (size_t i = 0; i < rows; ++i) {
    eigenRhs(i) = rhs[i];
  }

  // Compute the QR decomposition of A
  Eigen::VectorXd eigenSol = eigenA.colPivHouseholderQr().solve(eigenRhs);

  // Convert the solution back to std::vector
  for (size_t i = 0; i < cols; ++i) {
    x[i] = eigenSol(i);
  }
}