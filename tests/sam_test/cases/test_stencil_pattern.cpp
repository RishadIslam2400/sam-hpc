#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <iostream>
#include <stdexcept>

namespace {

// A 6 x 6 matrix whose values and structure the stencil must ignore.
csr::CSRMatrix<double> stencil_host() {
  return csr::CSRMatrix<double>(6, 6, 6, {1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5});
}

csr::CSRMatrix<int> pattern_of(const csr::CSRMatrix<double>& host, const StencilPattern& s, int level = 2) {
  SparsityPattern<double, StencilPattern> p(host, host, s, level);
  p.computePattern();
  return *p.getPattern();
}

}

/// Row s keeps {s + d} inside [0, n); unsorted and duplicate offsets are normalized.
void testStencilPatternOffsets() {
  std::cout << "Stencil pattern offsets..." << std::flush;

  // Offsets {-2, 0, 1}, given unsorted and with a duplicate.
  const csr::CSRMatrix<int> S = pattern_of(stencil_host(), StencilPattern{{1, 0, -2, 1}});

  const std::vector<size_t> rp = {0, 2, 4, 7, 10, 13, 15};
  const std::vector<size_t> ci = {0, 1,  1, 2,  0, 2, 3,  1, 3, 4,  2, 4, 5,  3, 5};
  assertEquals(S.m_row_pointers, rp, "Incorrect stencil row pointers");
  assertEquals(S.m_col_indices, ci, "Incorrect stencil column indices");
  assertEquals(S.m_vals, std::vector<int>(ci.size(), 1), "Incorrect stencil values");

  // Offsets that never land in range give empty rows, not out-of-range columns.
  const csr::CSRMatrix<int> far = pattern_of(stencil_host(), StencilPattern{{-100, 100}});
  assertEquals<size_t>(0, far.m_nnz, "Out-of-range offsets should contribute nothing");

  std::cout << "OK" << std::endl;
}

/// restrict_to_structure keeps only offsets the source couples: a wide stencil over a
/// tridiagonal matrix is the tridiagonal pattern, and the unrestricted one is not.
void testStencilPatternRestricted() {
  std::cout << "Stencil pattern restricted to structure..." << std::flush;

  // 6 x 6 tridiagonal, plus (0, 5) and (5, 0) so the restriction is not just a band.
  const csr::CSRMatrix<double> tri(6, 6, 18, std::vector<double>(18, 1.0),
                                   {0, 3, 6, 9, 12, 15, 18},
                                   {0, 1, 5,  0, 1, 2,  1, 2, 3,  2, 3, 4,  3, 4, 5,  0, 4, 5});
  const csr::CSRMatrix<int> R = pattern_of(tri, StencilPattern{{-2, -1, 0, 1, 2, 5, -5}, true});
  assertEquals(R.m_row_pointers, tri.m_row_pointers, "Restricted stencil row pointers");
  assertEquals(R.m_col_indices, tri.m_col_indices, "Restricted stencil column indices");

  const csr::CSRMatrix<int> U = pattern_of(tri, StencilPattern{{-2, -1, 0, 1, 2, 5, -5}});
  assertTrueMsg(U.m_nnz > R.m_nnz, "Unrestricted stencil should keep uncoupled offsets");

  // The flag survives transposition.
  assertTrueMsg(transposed(StencilPattern{{1}, true}).restrict_to_structure,
                "transposed() dropped restrict_to_structure");

  std::cout << "OK" << std::endl;
}

/// Stencils are used as given: the level argument does not extend them.
void testStencilPatternIgnoresLevel() {
  std::cout << "Stencil pattern ignores level..." << std::flush;

  const StencilPattern s{{-1, 0, 1}};
  const csr::CSRMatrix<int> l1 = pattern_of(stencil_host(), s, 1);
  const csr::CSRMatrix<int> l3 = pattern_of(stencil_host(), s, 3);
  assertTrueMsg(l1 == l3, "Stencil pattern changed with the extension level");
  assertEquals<size_t>(3 * 6 - 2, l1.m_nnz, "Tridiagonal stencil has the wrong nnz");

  static_assert(!pattern_is_extended_v<StencilPattern>);
  static_assert(!pattern_is_extended_v<ExplicitPattern>);
  static_assert(!pattern_is_extended_v<SimplePattern>);
  static_assert(pattern_is_extended_v<ColumnThresholdPattern>);

  std::cout << "OK" << std::endl;
}

/// transposed() negates every offset, so a symmetric stencil is its own transpose.
void testStencilTransposed() {
  std::cout << "Stencil transpose..." << std::flush;

  const StencilPattern s{{0, 1, -3}};
  const StencilPattern t = transposed(s);
  assertEquals(std::vector<long>{0, -1, 3}, t.offsets, "transposed() should negate offsets");

  // As patterns, the stencil of -D is the transpose of the stencil of D.
  const csr::CSRMatrix<int> S = pattern_of(stencil_host(), s);
  const csr::CSRMatrix<int> St = pattern_of(stencil_host(), t);
  assertTrueMsg(transpose(S) == St, "Stencil of -D is not the transpose of the stencil of D");

  std::cout << "OK" << std::endl;
}

/// An explicit pattern is copied verbatim (values reset to 1) and validated.
void testExplicitPattern() {
  std::cout << "Explicit pattern..." << std::flush;

  const csr::CSRMatrix<double> host = stencil_host();
  csr::CSRMatrix<int> given = pattern_of(host, StencilPattern{{-2, 0, 1}});
  given.m_vals.assign(given.m_nnz, 7);

  SparsityPattern<double, ExplicitPattern> p(host, host, ExplicitPattern{&given}, 3);
  p.computePattern();
  const csr::CSRMatrix<int>& S = *p.getPattern();
  assertEquals(given.m_row_pointers, S.m_row_pointers, "Explicit pattern row pointers changed");
  assertEquals(given.m_col_indices, S.m_col_indices, "Explicit pattern columns changed");
  assertEquals(S.m_vals, std::vector<int>(S.m_nnz, 1), "Explicit pattern values should be 1");

  const auto rejects = [&](const csr::CSRMatrix<int>& bad, const std::string& what) {
    SparsityPattern<double, ExplicitPattern> q(host, host, ExplicitPattern{&bad});
    bool threw = false;
    try {
      q.computePattern();
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    assertTrueMsg(threw, "Explicit pattern accepted " + what);
  };

  csr::CSRMatrix<int> unsorted = given;
  std::swap(unsorted.m_col_indices[0], unsorted.m_col_indices[1]);
  rejects(unsorted, "an unsorted row");

  csr::CSRMatrix<int> out_of_range = given;
  out_of_range.m_col_indices[given.m_nnz - 1] = 6;
  rejects(out_of_range, "an out-of-range column");

  const csr::CSRMatrix<int> wrong_size = pattern_of(
      csr::CSRMatrix<double>(3, 3, 3, {1, 1, 1}, {0, 1, 2, 3}, {0, 1, 2}), StencilPattern{{0}});
  rejects(wrong_size, "a pattern of the wrong size");

  std::cout << "OK" << std::endl;
}

/// Maps over a stencil pass the same two-way check as every other pattern.
void testStencilMapCD2D() {
  // The fixture is a 10 x 10 grid: {0, +-1, +-10} is its five-point stencil.
  check_map("CD2D stencil 5-point", read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"),
            StencilPattern{{0, 1, -1, 10, -10}});
  check_map("CD2D stencil diagonal", read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"),
            StencilPattern{{0}});
}

/// A map over an explicit copy of a pattern equals the map over the pattern itself.
void testExplicitMapMatchesDirect() {
  std::cout << "Explicit pattern map matches direct map..." << std::flush;

  const csr::CSRMatrix<double> target = read_fixture("cd2d/target.txt");
  const csr::CSRMatrix<double> source = read_fixture("cd2d/source.txt");
  SparsityPattern<double, ColumnThresholdPattern> col(source, target, ColumnThresholdPattern{0.8});
  col.computePattern();

  SparsityPattern<double, ExplicitPattern> ex(source, target, ExplicitPattern{col.getPattern()});
  ex.computePattern();

  csr::CSRMatrix<double> direct, via_explicit;
  SparseApproximateMap<double, ColumnThresholdPattern>::computeMap(target, source, *col.getPattern(), direct);
  SparseApproximateMap<double, ExplicitPattern>::computeMap(target, source, ex, via_explicit);
  assertTrueMsg(direct == via_explicit, "Map over an explicit pattern differs from the direct map");

  std::cout << "OK" << std::endl;
}
