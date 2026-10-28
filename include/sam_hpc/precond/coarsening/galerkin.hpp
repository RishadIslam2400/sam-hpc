#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include <memory>

/// Forms the coarse Galerkin operator `A_c = R A P`. `A` is the fine-level operator,
/// `P` prolongs coarse vectors to the fine level, and `R` restricts fine vectors to the
/// coarse level. The intermediate product is temporary; the returned matrix is owned by
/// the caller. The inputs must have compatible dimensions:
/// `A` is `n_f x n_f`, `P` is `n_f x n_c`, and `R` is `n_c x n_f`.
/// The output is an `n_c x n_c` CSR matrix whose rows are not guaranteed to be
/// sorted.
template <typename T>
std::shared_ptr<csr::CSRMatrix<T>> galerkin(const csr::CSRMatrix<T>& A,
                                            const csr::CSRMatrix<T>& P,
                                            const csr::CSRMatrix<T>& R)
{
  return std::make_shared<csr::CSRMatrix<T>>(matrix_product(R, matrix_product(A, P)));
}