#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

// Verify construction and inverse-diagonal calculation. The dia member stores D^-1.
void test_jacobi_constructor() {
  std::cout << "constuctor... " << std::flush;
  std::vector<size_t> row_ptr = {0, 3, 6, 9};
  std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
  csr::CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);

  typename damped_jacobi<double>::params prm;
  damped_jacobi<double> smoother(A, prm);

  // The inverse of the diagonal [4, 4, 4] is [0.25, 0.25, 0.25].
  std::vector<double> inverseDiag = {0.25, 0.25, 0.25};
  assertEquals(inverseDiag, smoother.dia, "Diagonal inverse not correct!");

  std::cout<< "OK" << std::endl;
}

// Verify one pre-smoothing update.
void test_jacobi_presmoothing() {
  std::cout << "presmoothing... " << std::flush;

  std::vector<size_t> row_ptr = {0, 3, 6, 9};
  std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
  csr::CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
  std::vector<double> rhs = {2.0, 2.0, 2.0};
  std::vector<double> x = {0.0, 0.0, 0.0};
  std::vector<double> tmp(3);
  typename damped_jacobi<double>::params prm(0.5);
  damped_jacobi<double> smoother(A, prm);

  // Starting from zero, the damped update gives [0.25, 0.25, 0.25].
  smoother.apply_pre(A, rhs, x, tmp);

  std::vector<double> x_new = {0.25, 0.25, 0.25};
  assertEquals(x_new, x, "Solution vector is incorrect!");

  std::cout << "OK" << std::endl;
}

// Verify that post-smoothing applies the same update as pre-smoothing.
void test_jacobi_postsmoothing() {
  std::cout << "postsmoothing... " << std::flush;

  std::vector<size_t> row_ptr = {0, 3, 6, 9};
  std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
  csr::CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
  std::vector<double> rhs = {2.0, 2.0, 2.0};
  std::vector<double> x = {0.0, 0.0, 0.0};
  std::vector<double> tmp(3);
  typename damped_jacobi<double>::params prm(0.5);
  damped_jacobi<double> smoother(A, prm);

  // Starting from zero, the damped update gives [0.25, 0.25, 0.25].
  smoother.apply_post(A, rhs, x, tmp);

  std::vector<double> x_new = {0.25, 0.25, 0.25};
  assertEquals(x_new, x, "Solution vector is incorrect!");

  std::cout << "OK" << std::endl;
}

// Verify the direct Jacobi application without damping.
void test_direct_jacobi() {
  std::cout << "direct solver... " << std::flush;

  std::vector<size_t> row_ptr = {0, 3, 6, 9};
  std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
  csr::CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
  std::vector<double> rhs = {2.0, 2.0, 2.0};
  std::vector<double> x = {0.0, 0.0, 0.0};
  std::vector<double> tmp(3);
  typename damped_jacobi<double>::params prm(0.5);
  damped_jacobi<double> smoother(A, prm);

  // apply computes x = D^-1 * rhs, independent of the damping factor and input x.
  smoother.apply(A, rhs, x);

  std::vector<double> x_new = {0.5, 0.5, 0.5};
  assertEquals(x_new, x, "Solution vector is incorrect!");

  std::cout << "OK" << std::endl;
}