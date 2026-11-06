#include "sam_hpc/sam/sam.hpp"
#include "testlib.hpp"

namespace {

template <typename PatternType>
void check_fused(const std::string& label, const csr::CSRMatrix<double>& target,
                 const csr::CSRMatrix<double>& source, const PatternType& params)
{
	std::cout << label << "..." << std::flush;
	using Map = SparseApproximateMap<double, PatternType>;

	// Build the level-1 pattern; fused mapping performs higher-level extension internally.
	SparsityPattern<double, PatternType> sp(source, target, params, 1, patternMatrix::union_max);
	sp.computePattern();
	const csr::CSRMatrix<int> base = *sp.getPattern();

	const int saved_threads = num_threads;
	const spgemm_accumulator saved_acc = spgemm_acc;
	for (int level = 1; level <= 3; ++level) {
		csr::CSRMatrix<int> extended = base;
		sam::extend_pattern(extended, level);

		for (mapSide side : {mapSide::left, mapSide::right}) {
			csr::CSRMatrix<double> reference;
			Map::computeMapSided(side, target, source, extended, reference);

			for (spgemm_accumulator acc : {spgemm_accumulator::dense, spgemm_accumulator::hash}) {
				for (int threads : {1, 8}) {
					tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
					num_threads = threads;
					spgemm_acc = acc;
					csr::CSRMatrix<double> fused;
					Map::computeMapSidedFused(side, target, source, base, level, fused);
					assertTrueMsg(fused == reference,
												label + ": fused " + to_string(side) + " map differs at level " +
												std::to_string(level) + ", " + to_string(acc) + " accumulator, " +
												std::to_string(threads) + " threads");
				}
			}
		}

		sam_residual fused_res, reference_res;
		csr::CSRMatrix<double> a, b;
		Map::computeMapFusedResidual(target, source, base, level, a, fused_res);
		Map::computeMapResidual(target, source, extended, b, reference_res);
		const double rel = std::abs(fused_res.sum_sq - reference_res.sum_sq) /
											 std::max(1e-300, reference_res.sum_sq);
		assertTrueMsg(rel < 1e-12, label + ": fused residual differs at level " +
									std::to_string(level) + " (relative " + std::to_string(rel) + ")");
	}
	num_threads = saved_threads;
	spgemm_acc = saved_acc;
	std::cout << "OK" << std::endl;
}

}

void testFusedMapSmall() {
  check_fused("fused map, small pair, column 0.9", small_target(), small_source(), ColumnThresholdPattern{0.9});
}

void testFusedMapCD2D() {
	check_fused("fused map, cd2d, column 0.9", read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"),
							ColumnThresholdPattern{0.9});
	check_fused("fused map, cd2d, fixed nnz 3", read_fixture("cd2d/target.txt"), read_fixture("cd2d/source.txt"),
							FixedNNZPattern{3});
}
