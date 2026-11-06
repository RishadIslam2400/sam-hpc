#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <cmath>
#include <vector>

namespace {

using dense = std::vector<std::vector<double>>;

dense to_dense(const csr::CSRMatrix<double>& A) {
  dense D(A.m_rows, std::vector<double>(A.m_cols, 0.0));
  for (size_t i = 0; i < A.m_rows; ++i)
    for (size_t p = A.m_row_pointers[i]; p < A.m_row_pointers[i + 1]; ++p)
      D[i][A.m_col_indices[p]] += A.m_vals[p];
  return D;
}

dense mul(const dense& X, const dense& Y) {
  const size_t m = X.size(), k = Y.size(), n = Y[0].size();
  dense Z(m, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < m; ++i)
    for (size_t p = 0; p < k; ++p)
      for (size_t j = 0; j < n; ++j)
        Z[i][j] += X[i][p] * Y[p][j];
  return Z;
}

dense sub(const dense& X, const dense& Y) {
  dense Z = X;
  for (size_t i = 0; i < X.size(); ++i)
    for (size_t j = 0; j < X[0].size(); ++j)
      Z[i][j] -= Y[i][j];
  return Z;
}

dense tr(const dense& X) {
  dense Z(X[0].size(), std::vector<double>(X.size(), 0.0));
  for (size_t i = 0; i < X.size(); ++i)
    for (size_t j = 0; j < X[0].size(); ++j)
      Z[j][i] = X[i][j];
  return Z;
}

double fro(const dense& X) {
  double s = 0.0;
  for (const auto& row : X)
    for (double v : row)  
      s += v * v;
  return std::sqrt(s);
}

/// Return the largest gradient magnitude on entries allowed by the pattern.
double max_gradient_on_pattern(const csr::CSRMatrix<double>& map, const dense& G) {
  double worst = 0.0;
  for (size_t i = 0; i < map.m_rows; ++i)
      for (size_t p = map.m_row_pointers[i]; p < map.m_row_pointers[i + 1]; ++p)
          worst = std::max(worst, std::abs(G[i][map.m_col_indices[p]]));
  return worst;
}

/// Build a nonsymmetric Laplacian by changing the superdiagonal weights.
csr::CSRMatrix<double> nonsymmetric_source(int n_rows) {
  csr::CSRMatrix<double> A = create_laplacian(n_rows);
  A.sortRows();
  for (size_t i = 0; i < A.m_rows; ++i)
    for (size_t p = A.m_row_pointers[i]; p < A.m_row_pointers[i + 1]; ++p)
      if (A.m_col_indices[p] == i + 1)
        A.m_vals[p] = -0.25;
  return A;
}

csr::CSRMatrix<double> shifted(const csr::CSRMatrix<double>& A, double d) {
  csr::CSRMatrix<double> B = A;
  for (size_t i = 0; i < B.m_rows; ++i)
    for (size_t p = B.m_row_pointers[i]; p < B.m_row_pointers[i + 1]; ++p)
      if (B.m_col_indices[p] == i)
        B.m_vals[p] += d;
  return B;
}

void build_maps(const csr::CSRMatrix<double>& source, const csr::CSRMatrix<double>& target,
                csr::CSRMatrix<double>& left, csr::CSRMatrix<double>& right,
                sam_transpose_times* tt = nullptr)
{
  SparsityPattern<double, SimplePattern> pattern(source, target, SimplePattern());
  pattern.computePattern();
  using Map = SparseApproximateMap<double, SimplePattern>;
  Map::computeMap(target, source, *pattern.getPattern(), left);
  Map::computeMapRight(target, source, *pattern.getPattern(), right, tt);
}

}

/// Each map satisfies the normal equations for its own objective.
void testMapSidesAreMinimizers() {
  std::cout << "Left and right maps each minimize their own objective..." << std::flush;

  csr::CSRMatrix<double> source = nonsymmetric_source(5);
  csr::CSRMatrix<double> target = shifted(source, 2.0);

  csr::CSRMatrix<double> left, right;
  build_maps(source, target, left, right);

  const dense Ak = to_dense(source), A0 = to_dense(target);
  const dense AkT = tr(Ak);
  const dense NL = to_dense(left), NR = to_dense(right);

  // Left objective: ||N A_k - A_0||_F.
  const dense RL = sub(mul(NL, Ak), A0);
  assertEquals(0.0, max_gradient_on_pattern(left, mul(RL, AkT)),
                "Left map does not satisfy its normal equations on the pattern!", 1e-8);

  // Right objective: ||A_k N - A_0||_F.
  const dense RR = sub(mul(Ak, NR), A0);
  assertEquals(0.0, max_gradient_on_pattern(right, mul(AkT, RR)),
                "Right map does not satisfy its normal equations on the pattern!", 1e-8);

  std::cout << "OK" << std::endl;
}

/// On a nonsymmetric problem, each map must differ and minimize its own objective.
void testMapSidesDifferWhenNonsymmetric() {
  std::cout << "Left and right maps differ on a nonsymmetric problem..." << std::flush;

  csr::CSRMatrix<double> source = nonsymmetric_source(5);
  csr::CSRMatrix<double> target = shifted(source, 2.0);

  csr::CSRMatrix<double> left, right;
  build_maps(source, target, left, right);

  const dense Ak = to_dense(source), A0 = to_dense(target);
  const dense NL = to_dense(left), NR = to_dense(right);

  assertTrueMsg(fro(sub(NL, NR)) > 1e-6, "Left and right maps are identical!");

  // Each map must outperform the other for its own objective.
  const double left_at_left   = fro(sub(mul(NL, Ak), A0));
  const double right_at_left  = fro(sub(mul(NR, Ak), A0));
  const double right_at_right = fro(sub(mul(Ak, NR), A0));
  const double left_at_right  = fro(sub(mul(Ak, NL), A0));

  assertTrueMsg(left_at_left < right_at_left,
                "Left map is not the better minimizer of ||N A_k - A_0||!");
  assertTrueMsg(right_at_right < left_at_right,
                "Right map is not the better minimizer of ||A_k N - A_0||!");

  std::cout << "OK" << std::endl;
}

/// Symmetry makes the two formulations transposes of one another.
void testMapSidesTransposeWhenSymmetric() {
  std::cout << "Right map is the transpose of the left map when symmetric..." << std::flush;

  csr::CSRMatrix<double> source = create_laplacian(5);
  source.sortRows();
  csr::CSRMatrix<double> target = shifted(source, 2.0);

  csr::CSRMatrix<double> left, right;
  build_maps(source, target, left, right);

  assertEquals(0.0, fro(sub(to_dense(right), tr(to_dense(left)))),
               "Right map is not the transpose of the left map on a symmetric problem!", 1e-8);

  std::cout << "OK" << std::endl;
}

/// Verify sam_residual_exact_right against a dense ||A_k N - A_0||_F calculation.
void testRightResidualMatchesDense() {
  std::cout << "sam_residual_exact_right matches a dense ||A_k N - A_0||_F..." << std::flush;

  csr::CSRMatrix<double> source = nonsymmetric_source(5);
  csr::CSRMatrix<double> target = shifted(source, 2.0);

  csr::CSRMatrix<double> left, right;
  sam_transpose_times tt;
  build_maps(source, target, left, right, &tt);

  const dense Ak = to_dense(source), A0 = to_dense(target);
  const double expected = fro(sub(mul(Ak, to_dense(right)), A0));
  const double computed = sam_residual_exact_right(right, source, target).residual_frobenius;

  assertEquals(expected, computed,
               "sam_residual_exact_right disagrees with the dense product!",
               1e-8 * std::max(1.0, expected));

  // Ensure transpose timing is recorded for the CSR implementation.
  assertTrueMsg(tt.total_ns() > 0, "computeMapRight reported no transpose time!");

  std::cout << "OK" << std::endl;
}