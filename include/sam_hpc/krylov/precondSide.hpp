#pragma once

#include <iostream>
#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

enum precondSide {
    left,
    right
};

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

/// @brief: Computes the preconditioned sparse matrix vector product based on the precond side.
template <typename U, class Precond>
void precond_spmv(precondSide &pside, const Precond &P, const CSRMatrix<U> &A,
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

/// @brief: Computes the preconditioned sparse matrix vector product based on the precond side.
template <typename U, class Precond>
void precond_spmv_sam(precondSide &pside, const Precond &P, const CSRMatrix<U> &A, const CSRMatrix<U>& N,
                  const std::vector<U> &F, std::vector<U> &X, std::vector<U> &T, std::vector<U>& temp)
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