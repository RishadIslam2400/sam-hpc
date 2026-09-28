#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

template <typename T>
std::shared_ptr<CSRMatrix<T>> galerkin(const CSRMatrix<T>& A,
                                       const CSRMatrix<T>& P,
                                       const CSRMatrix<T>& R)
{
  return matrix_product(R, *matrix_product(A, P));
}

template <typename T>
std::shared_ptr<CSRMatrix<T>> scaled_galerkin(const CSRMatrix<T>& A,
                                              const CSRMatrix<T>& P,
                                              const CSRMatrix<T>& R, float s)
{
  auto a = galerkin(A, P, R);
  scale(*a, s);
  return a;
}