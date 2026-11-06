#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "testlib.hpp"

/// Verify that sortRows orders columns and preserves their associated values.
void testSortRows() {
  std::cout << "sort rows..." << std::flush;
  /*
    Matrix
    [ 1  0 4 5 ]
    [ 2 -1 0 0 ]
    [ 0  0 3 2 ]

    should be stored as
    rows:    [ 0, 3, 5, 7 ]
    columns: [ 0, 2, 3, 0, 1, 2, 3 ]
    values:  [ 1, 4, 5, 2, -1, 3, 2 ]
  */

  std::vector<size_t> rowPointers1 = {0, 3, 5, 7};
  std::vector<size_t> colIndices1 = {2, 3, 0, 1, 0, 3, 2};
  std::vector<int> vals1 = {4, 5, 1, -1, 2, 2 , 3};
  csr::CSRMatrix<int> m1(3, 4, vals1, rowPointers1, colIndices1);

  std::vector<size_t> rowPointersCorrect = {0, 3, 5, 7};
  std::vector<size_t> colIndicesCorrect = {0, 2, 3, 0, 1, 2, 3};
  std::vector<int> valsCorrect = {1, 4, 5, 2, -1, 3, 2};

  m1.sortRows();
  assertEquals(rowPointersCorrect, m1.m_row_pointers, "Incorrect internal row pointers");
  assertEquals(colIndicesCorrect, m1.m_col_indices, "Incorrect internal column indices");
  assertEquals(valsCorrect, m1.m_vals, "Incorrect internal values");

  std::cout << "OK" << std::endl;
}

/// Verify the transpose of a rectangular matrix, including sorted output rows.
void testTranspose() {
  std::cout << "transpose..." << std::flush;
  /*
    Matrix
    [ 1  0 4 5 ]
    [ 2 -1 0 0 ]
    [ 0  0 3 2 ]

    should be stored as
    rows:    [ 0, 3, 5, 7 ]
    columns: [ 0, 2, 3, 0, 1, 2, 3 ]
    values:  [ 1, 4, 5, 2, -1, 3, 2 ]
  */

  std::vector<size_t> rowPointers1 = {0, 3, 5, 7};
  std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3};
  std::vector<int> vals1 = {1, 4, 5, 2, -1, 3, 2};
  csr::CSRMatrix<int> m1(3, 4, vals1, rowPointers1, colIndices1);

  csr::CSRMatrix<int> m1_transpose = transpose(m1);

  /*
    Transposed matrix
    [ 1  2 0 ]
    [ 0 -1 0 ]
    [ 4  0 3 ]
    [ 5  0 2 ]

    should be stored as
    rows:    [ 0, 2, 3, 5, 7 ]
    columns: [ 0, 1, 1, 0, 2, 0, 2 ]
    values:  [ 1, 2, -1, 4, 3, 5, 2 ]
  */

  std::vector<size_t> rowPointers2 = {0, 2, 3, 5, 7};
  std::vector<size_t> colIndices2 = {0, 1, 1, 0, 2, 0, 2};
  std::vector<int> vals2 = {1, 2, -1, 4, 3, 5, 2};
  csr::CSRMatrix<int> m2(4, 3, vals2, rowPointers2, colIndices2);

  assertEquals<csr::CSRMatrix<int>>(m2, m1_transpose, "Incorrect transpose");

  std::cout << "OK" << std::endl;
}

/// Verify raw diagonal extraction and scaling values for an empty diagonal entry.
void testDiagonal() {
  std::cout << "diagonal..." << std::flush;

  /*
    Matrix
    [ 1  0 4 5 ]
    [ 2 -1 0 0 ]
    [ 0  0 4 2 ]
    [ 0  0 0 0 ]

    should be stored as
    rows:    [ 0, 3, 5, 7, 7 ]
    columns: [ 0, 2, 3, 0, 1, 2, 3 ]
    values:  [ 1, 4, 5, 2, -1, 4, 2 ]
  */

  std::vector<size_t> rowPointers1 = {0, 3, 5, 7, 7};
  std::vector<size_t> colIndices1 = {0, 2, 3, 0, 1, 2, 3};
  std::vector<double> vals1 = {1, 4, 5, 2, -1, 4, 2};
  csr::CSRMatrix<double> m1(4, 4, vals1, rowPointers1, colIndices1);

  std::vector<double> m1_diag = csr::diagonal<csr::diagonalType::simple>(m1);
  std::vector<double> m1_diag_for_scaling = csr::diagonal<csr::diagonalType::forScaling>(m1);

  std::vector<double> originalDiag = {1, -1, 4, 0};
  std::vector<double> forScalingDiag = {1, 1, 0.5, 1};

  assertEquals(m1_diag, originalDiag, "Incorrect diagonal");
  assertEquals(m1_diag_for_scaling, forScalingDiag, "Incorrect diagonal for scaling");

  std::cout << "OK" << std::endl;
}

/// Verify the inner product against an independently evaluated reference sum.
void testInnerProduct() {
  std::cout << "inner product..." << std::flush;
  
  std::vector<double> a{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8,
                        9.9, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7};
  std::vector<double> b{0.5, 1.5, -1.0, 2.0, -0.5, 3.0, -2.0, 1.0,
                        0.0, -1.0, 0.25, 0.75, -0.25, -0.5, 1.5, 2.5};

  double expected =  1.1 * 0.5 + 2.2 * 1.5 + 3.3 * -1.0 + 4.4 * 2.0 + 5.5 * -0.5 + 
                     6.6 * 3.0 + 7.7 * -2.0 + 8.8 * 1.0 + 9.9 * 0.0 + 10.1 * -1.0 +
                     11.2 * 0.25 + 12.3 * 0.75 + 13.4 * -0.25 + 14.5 * -0.5 + 
                     15.6 * 1.5 + 16.7 * 2.5;

  assertEquals<double>(expected, innerProduct(a, b), "Incorrect inner product.");
  std::cout << "OK" << std::endl;

}

/// Verify deterministic inner products across repeated runs and thread counts.
/// Exact comparison detects changes in the reduction order.
void testInnerProductDeterministic() {
  std::cout << "inner product determinism..." << std::flush;

  const size_t n = 1000003;
  std::vector<double> a(n), b(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = std::sin(0.37 * static_cast<double>(i)) * std::pow(10.0, static_cast<double>(i % 7) - 3.0);
    b[i] = std::cos(0.11 * static_cast<double>(i)) - 0.5;
  }

  double reference;
  {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);
    reference = innerProduct(a, b);
  }
  for (size_t threads : {2, 4, 8, 16}) {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
    for (int rep = 0; rep < 3; ++rep) {
      const double computed = innerProduct(a, b);
      assertTrueMsg(computed == reference, "innerProduct differs from the 1-thread result at " +
                    std::to_string(threads) + " threads");
    }
  }
  std::cout << "OK" << std::endl;
}

/// Verify y <- a*x + b*y, including the zero-coefficient case.
void testAXPBY() {
  std::cout << "axpby..." << std::flush;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> y = {5.0, 6.0, 7.0, 8.0};
  double a = 2.0;
  double b = 3.0;
  std::vector<double> result = {17.0, 22.0, 27.0, 32.0};
  axpby(a, x, b, y);
  assertEquals(result, y, "Incorrect axpby function.");

  y = {5.0, 6.0, 7.0, 8.0};
  result = {2.0, 4.0, 6.0, 8.0};
  axpby(a, x, 0.0, y);
  assertEquals(result, y, "Incorrect axpby function.");
  std::cout << "OK" << std::endl;
}

/// Verify z <- a*x + b*y + c*z, including the zero-coefficient case.
void testAXPBYPCZ() {
  std::cout << "axpbypcz..." << std::flush;
  std::vector<int> x = {1, 2, 3, 4};
  std::vector<int> y = {5, 6, 7, 8};
  std::vector<int> z = {9, 10, 11, 12};
  int a = 1, b = 2, c = 3;
  std::vector<int> result = {38, 44, 50, 56};
  axpbypcz(a, x, b, y, c, z);
  assertEquals(result, z, "Incorrect axpbypcz function.");

  z = {9, 10, 11, 12};
  result = {11, 14, 17, 20};
  c = 0;
  axpbypcz(a, x, b, y, c, z);
  assertEquals(result, z, "Incorrect axpbypcz function.");
  std::cout << "OK" << std::endl;
}

/// Verify the linear combination of multiple vectors with an initial y term.
void testLinComb() {
  std::cout << "linear combination..." << std::flush;
  std::vector<std::vector<double>> v = {
    {1.0, 2.0, 3.0, 4.0},
    {0.5, 1.5, 2.5, 3.5},
    {2.0, 1.0, 0.0, -1.0}
  };
  std::vector<double> c = {2.0, 1.0, -1.0};
  std::vector<double> y = {10.0, 20.0, 30.0, 40.0};
  double alpha = 0.5;
  std::vector<double> result = {5.5, 14.5, 23.5, 32.5};
  linComb(c.size(), c, v, alpha, y);
  assertEquals(result, y, "Incorrect lincomb function");
  std::cout << "OK" << std::endl;
}

/// Verify the residual rhs - A*x for a rectangular matrix.
void testResidual() {
  std::cout << "residual..." << std::flush;
  // A =
  // [1 0 2]
  // [0 3 0]
  // [4 0 5]
  // [0 6 0]
  std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<size_t> col_idx = {0, 2, 1, 0, 2, 1};
  std::vector<size_t> row_ptr = {0, 2, 3, 5, 6};
  csr::CSRMatrix<double> A(4, 3, values, row_ptr, col_idx);
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> rhs = {10.0, 5.0, 25.0, 14.0};
  std::vector<double> res(4);
  std::vector<double> result = {3.0, -1.0, 6.0, 2.0};

  residual(rhs, A, x, res);
  assertEquals(result, res, "Incorrect residual calculation");
  std::cout << "OK" << std::endl;
}