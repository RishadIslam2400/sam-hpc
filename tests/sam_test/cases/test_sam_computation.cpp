#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <string>

namespace {

// Small nonsymmetric target used by the sanity checks.
csr::CSRMatrix<double> small_target() {
  return csr::CSRMatrix<double>(5, 5, 7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                {0, 2, 3, 5, 6, 7}, {0, 2, 1, 0, 4, 3, 2});
}

// The source is nonsymmetric and omits the diagonal in rows 0, 1, and 4.
csr::CSRMatrix<double> small_source() {
  return csr::CSRMatrix<double>(5, 5, 8, {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0},
                                {0, 1, 3, 5, 7, 8}, {1, 0, 4, 1, 2, 0, 3, 2});
}

// Compute the map with TBB limited to the requested worker count.
csr::CSRMatrix<double> map_at(int threads, const csr::CSRMatrix<double>& target,
                              const csr::CSRMatrix<double>& source,
                              const csr::CSRMatrix<int>& pattern, sam_residual& residual)
{
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
  csr::CSRMatrix<double> map;
  SparseApproximateMap<double, SimplePattern>::computeMapResidual(target, source, pattern, map, residual);
  return map;
}

/// Verify the exact residual and thread-independent map construction.
template <typename PatternType>
void check_map(const std::string& label, const csr::CSRMatrix<double>& target,
               const csr::CSRMatrix<double>& source, const PatternType& params)
{
  std::cout << label << "..." << std::flush;

  SparsityPattern<double, PatternType> pattern(source, target, params);
  pattern.computePattern();
  const csr::CSRMatrix<int>& S = *pattern.getPattern();

  sam_residual qr_residual_1, qr_residual_8;
  const csr::CSRMatrix<double> map_1 = map_at(1, target, source, S, qr_residual_1);
  const csr::CSRMatrix<double> map_8 = map_at(8, target, source, S, qr_residual_8);

  assertEquals(S.m_nnz, map_1.m_nnz, label + ": map does not have the pattern's nonzeros");

  const sam_accuracy exact = sam_residual_exact(map_1, source, target);
  const double scale = std::max(1.0, exact.target_frobenius);
  assertEquals(exact.residual_frobenius / scale, qr_residual_1.frobenius() / scale,
               label + ": QR residual disagrees with exact ||N A - A_0||_F", 1e-10);

  // Compare dimensions, row boundaries, and entries to detect parallel write errors.
  assertEquals<size_t>(map_1.m_rows, map_8.m_rows, label + ": row count differs at 8 threads");
  assertEquals<size_t>(map_1.m_cols, map_8.m_cols, label + ": column count differs at 8 threads");
  assertEquals<size_t>(map_1.m_nnz, map_8.m_nnz, label + ": nnz differs at 8 threads");

  for (size_t i = 0; i <= map_1.m_rows; ++i) {
    if (map_1.m_row_pointers[i] != map_8.m_row_pointers[i]) {
      throw FailureException(label + ": row pointer " + std::to_string(i) +
                             " differs between 1 and 8 threads (" +
                             std::to_string(map_1.m_row_pointers[i]) + " vs " +
                             std::to_string(map_8.m_row_pointers[i]) + ")");
    }
  }
  for (size_t k = 0; k < map_1.m_nnz; ++k) {
    if (map_1.m_col_indices[k] != map_8.m_col_indices[k] || map_1.m_vals[k] != map_8.m_vals[k]) {
      throw FailureException(label + ": map differs between 1 and 8 threads at entry " +
                             std::to_string(k));
    }
  }

  std::cout << "OK (relative residual " << exact.relative() << ")" << std::endl;
}

// read_mat reports missing or malformed fixtures instead of returning an empty matrix.
csr::CSRMatrix<double> read_fixture(const std::string& name) {
  return read_mat<double>((std::string(SAM_TEST_DATA_DIR) + name).c_str());
}

}

void testSAMSanityCheck1() { check_map("SAM sanity, simple",        small_target(), small_source(), SimplePattern{}); }
void testSAMSanityCheck2() { check_map("SAM sanity, global 0.001",  small_target(), small_source(), GlobalThresholdPattern{0.001}); }
void testSAMSanityCheck3() { check_map("SAM sanity, column 0.9",    small_target(), small_source(), ColumnThresholdPattern{0.9}); }
void testSAMSanityCheck4() { check_map("SAM sanity, fixed nnz 2",   small_target(), small_source(), FixedNNZPattern{2}); }

void testCD2D1() { check_map("CD2D simple",       read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), SimplePattern{}); }
void testCD2D2() { check_map("CD2D global 0.001", read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), GlobalThresholdPattern{0.001}); }
void testCD2D3() { check_map("CD2D column 0.9",   read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), ColumnThresholdPattern{0.9}); }
void testCD2D4() { check_map("CD2D fixed nnz 3",  read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), FixedNNZPattern{3}); }
