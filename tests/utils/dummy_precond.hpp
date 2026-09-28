#pragma once 

#include "sam_hpc/core/CSRMatrix.hpp"

template <typename T> struct dummy_precond {
    dummy_precond(const CSRMatrix<T>&) {}
    void apply(const std::vector<T>& r, std::vector<T>& x) const { x = r; }
};