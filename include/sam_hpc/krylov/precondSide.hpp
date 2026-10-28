#pragma once

#include <iostream>
#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

/// Selects whether the preconditioner is applied before or after the matrix.
enum precondSide {
  left,
  right
};

/// Writes the selected preconditioning side.
inline std::ostream& operator<<(std::ostream& os, precondSide& p) {
  switch (p) {
  case left:
    return os << "left";
  case right:
    return os << "right";
  default:
    return os << "????";
  }
}

/// Reads `left` or `right`; invalid values throw `std::invalid_argument`.
inline std::istream& operator>>(std::istream& in, precondSide& p) {
  std::string val;
  in >> val;

  if (val == "left") {
    p = left;
  } else if (val == "right") {
    p = right;
  } else {
    throw std::invalid_argument("Invalid preconditioning side. Valid choices are left and right");
  }

  return in;
}

/// Applies the selected preconditioned operator to a Krylov vector.
/// Left preconditioning computes `P(A F)`; right preconditioning computes `A(P F)`.
/// `T` is temporary storage and must be distinct from `X`.
template <typename U, class Precond>
void precond_spmv(precondSide &pside, const Precond &P, const csr::CSRMatrix<U> &A,
                  const std::vector<U> &F, std::vector<U> &X, std::vector<U> &T)
{
  if (pside == left) {
    spmv(U(1), A, F, U(0), T);
    P.apply(T, X);
  } else {
    P.apply(F, T);
    spmv(U(1), A, T, U(0), X);
  }
}

/// Applies a left-map SAM operator using `N A ≈ A_0`.
/// Left preconditioning computes `P(N(A F))`; right preconditioning
/// computes `N(A(P F))`. `T` and `temp` are overwritten temporary buffers and
/// must be distinct from each other and from `X`.
template <typename U, class Precond>
void precond_spmv_sam(precondSide &pside, const Precond &P, const csr::CSRMatrix<U> &A,
                      const csr::CSRMatrix<U>& N, const std::vector<U> &F, std::vector<U> &X,
                      std::vector<U> &T, std::vector<U>& temp)
{
  if (pside == left) {
    spmv(U(1), A, F, U(0), temp);
    spmv(U(1), N, temp, U(0), T);
    P.apply(T, X);
  } else {
    P.apply(F, T);
    spmv(U(1), A, T, U(0), temp);
    spmv(U(1), N, temp, U(0), X);
  }
}

/// Applies a right-map SAM operator using `A N ≈ A_0`.
/// Left preconditioning computes `P(A(N F))`; right preconditioning
/// computes `A(N(P F))`. `T` and `temp` are overwritten temporary buffers and
/// must be distinct from each other and from `X`.
template <typename U, class Precond>
void precond_spmv_sam_right(precondSide &pside, const Precond &P, const csr::CSRMatrix<U> &A,
                            const csr::CSRMatrix<U>& N, const std::vector<U> &F, std::vector<U> &X,
                            std::vector<U> &T, std::vector<U>& temp)
{
  if (pside == left) {
    spmv(U(1), N, F, U(0), temp);
    spmv(U(1), A, temp, U(0), T);
    P.apply(T, X);
  } else {
    P.apply(F, T);
    spmv(U(1), N, T, U(0), temp);
    spmv(U(1), A, temp, U(0), X);
  }
}