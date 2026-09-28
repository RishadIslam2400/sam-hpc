#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/gauss_seidel.hpp"
#include "testlib.hpp"

void test_gauss_siedel_presmoothing_serial() {
    std::cout << "presmoothing (serial sweep)... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp; // unused for gauss seidel

    // Force serial implementation
    typename gauss_seidel<double>::params prm(true);
    gauss_seidel<double> smoother(A, prm);

    // Manually calculated result for one forward sweep:
    // x_0 = (1/4) * (2 - (-1*0) - (-1*0)) = 0.5
    // x_1 = (1/4) * (2 - (-1*0.5) - (-1*0)) = 0.625
    // x_2 = (1/4) * (2 - (-1*0.5) - (-1*0.625)) = 0.78125
    smoother.apply_pre(A, rhs, x, tmp);

    std::vector<double> x_new = {0.5, 0.625, 0.78125};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}

void test_gauss_siedel_postsmoothing_serial() {
    std::cout << "postsmoothing (serial sweep)... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp; // unused for gauss seidel

    // Force serial implementation
    typename gauss_seidel<double>::params prm(true);
    gauss_seidel<double> smoother(A, prm);

    // Manually calculated result for one forward sweep:
    // x_0 = (1/4) * (2 - (-1*0) - (-1*0)) = 0.5
    // x_1 = (1/4) * (2 - (-1*0.5) - (-1*0)) = 0.625
    // x_2 = (1/4) * (2 - (-1*0.5) - (-1*0.625)) = 0.78125
    smoother.apply_post(A, rhs, x, tmp);

    std::vector<double> x_new = {0.78125, 0.625, 0.5};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;    
}

void test_gauss_siedel_direct_sovler_serial() {
    std::cout << "symmetric solve (serial sweep)... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp; // unused for gauss seidel

    // Force serial implementation
    typename gauss_seidel<double>::params prm(true);
    gauss_seidel<double> smoother(A, prm);

    // First, a forward sweep is applied, resulting in:
    // x = [0.5, 0.625, 0.78125]
    // Then, a backward sweep is applied to this result:
    // x_2 = (1/4) * (2 - (-1*0.5) - (-1*0.625)) = 0.78125 (no change)
    // x_1 = (1/4) * (2 - (-1*0.5) - (-1*0.78125)) = 0.8203125
    // x_0 = (1/4) * (2 - (-1*0.8203125) - (-1*0.78125)) = 0.900390625
    smoother.apply(A, rhs, x, tmp);

    std::vector<double> x_new = {0.900390625, 0.8203125, 0.78125};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}

void test_gauss_siedel_presmoothing_parallel() {
    std::cout << "presmoothing (parallel sweep)... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp; // unused for gauss seidel

    // Force parallel implementation
    typename gauss_seidel<double>::params prm(false);
    gauss_seidel<double> smoother(A, prm);

    // Manually calculated result for one forward sweep:
    // x_0 = (1/4) * (2 - (-1*0) - (-1*0)) = 0.5
    // x_1 = (1/4) * (2 - (-1*0.5) - (-1*0)) = 0.625
    // x_2 = (1/4) * (2 - (-1*0.5) - (-1*0.625)) = 0.78125
    smoother.apply_pre(A, rhs, x, tmp);

    std::vector<double> x_new = {0.5, 0.625, 0.78125};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}

void test_gauss_siedel_postsmoothing_parallel() {
    std::cout << "postsmoothing (parallel sweep)... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp; // unused for gauss seidel

    // Force parallel implementation
    typename gauss_seidel<double>::params prm(false);
    gauss_seidel<double> smoother(A, prm);

    // Manually calculated result for one forward sweep:
    // x_0 = (1/4) * (2 - (-1*0) - (-1*0)) = 0.5
    // x_1 = (1/4) * (2 - (-1*0.5) - (-1*0)) = 0.625
    // x_2 = (1/4) * (2 - (-1*0.5) - (-1*0.625)) = 0.78125
    smoother.apply_pre(A, rhs, x, tmp);

    std::vector<double> x_new = {0.5, 0.625, 0.78125};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
}

void test_gauss_siedel_direct_sovler_parallel() {
    std::cout << "symmetric solve (parallel sweep)... " << std::flush;

    std::vector<size_t> row_ptr = {0, 3, 6, 9};
    std::vector<size_t> col_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> values = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
    CSRMatrix<double> A(3, 3, values, row_ptr, col_ind);
    std::vector<double> rhs = {2.0, 2.0, 2.0};
    std::vector<double> x = {0.0, 0.0, 0.0};
    std::vector<double> tmp; // unused for gauss seidel

    // Force serial implementation
    typename gauss_seidel<double>::params prm(false);
    gauss_seidel<double> smoother(A, prm);

    // First, a forward sweep is applied, resulting in:
    // x = [0.5, 0.625, 0.78125]
    // Then, a backward sweep is applied to this result:
    // x_2 = (1/4) * (2 - (-1*0.5) - (-1*0.625)) = 0.78125 (no change)
    // x_1 = (1/4) * (2 - (-1*0.5) - (-1*0.78125)) = 0.8203125
    // x_0 = (1/4) * (2 - (-1*0.8203125) - (-1*0.78125)) = 0.900390625
    smoother.apply(A, rhs, x, tmp);

    std::vector<double> x_new = {0.900390625, 0.8203125, 0.78125};
    assertEquals(x_new, x, "Solution vector is incorrect!");

    std::cout << "OK" << std::endl;
    
}