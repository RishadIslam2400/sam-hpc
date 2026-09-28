#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <string>

namespace {

// The 5x5 pair every sanity check maps between.
CSRMatrix<double> small_target() {
    return CSRMatrix<double>(5, 5, 7, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                             {0, 2, 3, 5, 6, 7}, {0, 2, 1, 0, 4, 3, 2});
}

CSRMatrix<double> small_source() {
    return CSRMatrix<double>(5, 5, 8, {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0},
                             {0, 1, 3, 5, 7, 8}, {1, 0, 4, 1, 2, 0, 3, 2});
}

CSRMatrix<double> map_at(int threads, const CSRMatrix<double>& target, const CSRMatrix<double>& source,
                         const CSRMatrix<int>& pattern, sam_residual& residual) {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
    CSRMatrix<double> map;
    // The pattern type only selects how the pattern was built; the map itself takes the
    // finished pattern matrix, so any instantiation computes the same thing.
    SparseApproximateMap<double, SimplePattern>::computeMapResidual(target, source, pattern, map, residual);
    return map;
}

/// Builds the pattern and the map, then checks the two properties every map must have:
///
///   - the QR's own least squares residual equals the independently computed
///     ||N A_k - A_0||_F. J covers each row's whole residual support, so they agree to
///     roundoff; if they ever disagree one of the two implementations is wrong.
///   - the map does not depend on the thread count. Rows are independent solves, so the
///     values must be bitwise identical; a difference is a race.
template <typename PatternType>
void check_map(const std::string& label, const CSRMatrix<double>& target,
               const CSRMatrix<double>& source, const PatternType& params) {
    std::cout << label << "..." << std::flush;

    SparsityPattern<double, PatternType> pattern(source, target, params);
    pattern.computePattern();
    const CSRMatrix<int>& S = *pattern.getPattern();

    sam_residual qr_residual_1, qr_residual_8;
    const CSRMatrix<double> map_1 = map_at(1, target, source, S, qr_residual_1);
    const CSRMatrix<double> map_8 = map_at(8, target, source, S, qr_residual_8);

    assertEquals(S.m_nnz, map_1.m_nnz, label + ": map does not have the pattern's nonzeros");

    const sam_accuracy exact = sam_residual_exact(map_1, source, target);
    const double scale = std::max(1.0, exact.target_frobenius);
    assertEquals(exact.residual_frobenius / scale, qr_residual_1.frobenius() / scale,
                 label + ": QR residual disagrees with exact ||N A - A_0||_F", 1e-10);

    for (size_t k = 0; k < map_1.m_nnz; ++k) {
        if (map_1.m_col_indices[k] != map_8.m_col_indices[k] || map_1.m_vals[k] != map_8.m_vals[k]) {
            throw FailureException(label + ": map differs between 1 and 8 threads at entry " +
                                   std::to_string(k));
        }
    }

    std::cout << "OK (relative residual " << exact.relative() << ")" << std::endl;
}

CSRMatrix<double> read_fixture(const std::string& name) {
    CSRMatrix<double> A;
    read_mat((std::string(SAM_TEST_DATA_DIR) + name).c_str(), A);
    return A;
}

} // namespace

void testSAMSanityCheck1() { check_map("SAM sanity, simple",        small_target(), small_source(), SimplePattern{}); }
void testSAMSanityCheck2() { check_map("SAM sanity, global 0.001",  small_target(), small_source(), GlobalThresholdPattern{0.001}); }
void testSAMSanityCheck3() { check_map("SAM sanity, column 0.9",    small_target(), small_source(), ColumnThresholdPattern{0.9}); }
void testSAMSanityCheck4() { check_map("SAM sanity, fixed nnz 2",   small_target(), small_source(), FixedNNZPattern{2}); }

void testCD2D1() { check_map("CD2D simple",       read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), SimplePattern{}); }
void testCD2D2() { check_map("CD2D global 0.001", read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), GlobalThresholdPattern{0.001}); }
void testCD2D3() { check_map("CD2D column 0.9",   read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), ColumnThresholdPattern{0.9}); }
void testCD2D4() { check_map("CD2D fixed nnz 3",  read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"), FixedNNZPattern{3}); }
