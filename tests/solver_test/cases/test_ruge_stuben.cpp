#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"
#include "testlib.hpp"

void test_1d_laplacian_splitting_and_interpolation() {
    std::cout << "1d laplacian matrix..." << std::flush;
    // A 5-point 1D Laplacian matrix with a [2, -1] stencil
    CSRMatrix<float> A_1D_laplacian = CSRMatrix<float>(5, 5,
        {2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2},
        {0, 2, 5, 8, 11, 13},
        {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4}
    );

    ruge_stuben<float> rs;
    auto [P, R] = rs.transfer_operators(A_1D_laplacian);

    // For a 1D Laplacian, we expect a F-C-F-C-F splitting.
    // C-points: 1, 3. F-points: 0, 2, 4. Coarse size nc = 2.
    CSRMatrix<float> expectedP(5, 2, {0.5, 1, 0.5, 0.5, 1, 0.5}, {0, 1, 2, 4, 5, 6}, {0, 0, 0, 1, 1, 1});
    assertEquals<CSRMatrix<float>>(expectedP, *P, "C/F splitting interpolation not correct!");

    auto tranposeP = transpose(*P);
    assertEquals<CSRMatrix<float>>(*tranposeP, *R, "C/F splitting restriction not correct!");

    std::cout << "OK" << std::endl;
}