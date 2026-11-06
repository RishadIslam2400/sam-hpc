#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/util/config.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/util/measure.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace {

using bench::measure;
using bench::sample;

// Print one benchmark row for a specific phase.
void row(const std::string& phase, const sample& s, double total_ms) {
	std::cout << "  " << std::left << std::setw(18) << phase << std::right
						<< std::setw(11) << std::fixed << std::setprecision(2) << s.min_ms << " ms"
						<< std::setw(11) << s.median_ms << " ms"
						<< std::setw(9) << std::setprecision(1) << (total_ms > 0 ? 100.0 * s.min_ms / total_ms : 0.0) << " %"
						<< std::setw(8) << s.reps << " reps"
						<< std::setw(9) << std::setprecision(1) << s.spread_pct() << " % spr"
						<< "\n";
}

template <typename PatternType>
void run_pattern(const std::string& label, const config_t& cfg, int threads,
                 const csr::CSRMatrix<double>& target, const csr::CSRMatrix<double>& source,
                 const PatternType& params)
{
	using Map = SparseApproximateMap<double, PatternType>;
	const pattern_variant pv = parse_pattern_variant(cfg.variant);
	const patternMatrix pm = pv.union_candidate ? patternMatrix::union_max : patternMatrix::source;

	pattern_phase_times pattern_split{};
	const sample pattern_total = measure(cfg.iters, [&] {
		SparsityPattern<double, PatternType> sp(source, target, params, 2, pm);
		sp.computePattern();
		pattern_split = sp.getPhaseTimes();
	});

	SparsityPattern<double, PatternType> sp(source, target, params, 2, pm);
	sp.computePattern();
	const size_t pattern_nnz = sp.getNNZ();

	const csr::CSRMatrix<int>* merged = sp.getPattern();

	csr::CSRMatrix<double> map;
	Map::computeMap(target, source, *merged, map);
	const sample map_s = measure(cfg.iters, [&] {
		csr::CSRMatrix<double> m;
		Map::computeMap(target, source, *merged, m);
	});

	// Drops entries with |n_ij| < 0.01.
	csr::CSRMatrix<double> filtered;
	Map::post_filtration(map, filtered, 0.01);
	const sample filt_s = measure(cfg.iters, [&] {
		csr::CSRMatrix<double> f;
		Map::post_filtration(map, f, 0.01);
	});

	const sample total_s = measure(cfg.iters, [&] {
		SparsityPattern<double, PatternType> p(source, target, params, 2, pm);
		p.computePattern();
		csr::CSRMatrix<double> m;
		Map::computeMap(target, source, *p.getPattern(), m);
		csr::CSRMatrix<double> f;
		Map::post_filtration(m, f, 0.01);
	});

	// Measure accuracy
	sam_residual cheap;
	{
		csr::CSRMatrix<double> m;
		Map::computeMapResidual(target, source, *merged, m, cheap);
	}
	const sam_accuracy exact = sam_residual_exact(map, source, target);

	sam_phase_times prof;
	{
		csr::CSRMatrix<double> m;
		Map::computeMapProfiled(target, source, *merged, m, prof);
	}

	const double split_total = static_cast<double>(pattern_split.total_ns());
	const auto share = [&](uint64_t part) {
		return split_total > 0 ? pattern_total.min_ms * part / split_total : 0.0;
	};
	const double merge_ms = share(pattern_split.merge_ns);
	const double filter_ms = share(pattern_split.filter_ns);
	const double extend_ms = pattern_total.min_ms - merge_ms - filter_ms;

	const double T = total_s.min_ms;
	std::cout << "\n[" << label << " | " << pv.name << "]  threads " << threads
						<< "   pattern nnz " << pattern_nnz
						<< " (" << std::fixed << std::setprecision(1) << (double)pattern_nnz / target.m_rows << "/row)"
						<< "   final nnz " << merged->m_nnz
						<< " (" << (double)merged->m_nnz / target.m_rows << "/row)"
						<< "   map nnz after filtration " << filtered.m_nnz << "\n"
						<< "  " << std::left << std::setw(18) << "phase" << std::right
						<< std::setw(14) << "min" << std::setw(14) << "median" << std::setw(11) << "of total"
						<< std::setw(13) << "reps" << "\n";
	
	row("pattern merge", {merge_ms, merge_ms, merge_ms, pattern_total.reps}, T);
	row("pattern filter", {filter_ms, filter_ms, filter_ms, pattern_total.reps}, T);
	row("pattern extend", {extend_ms, extend_ms, extend_ms, pattern_total.reps}, T);
	row("map", map_s, T);
	row("post filtration", filt_s, T);
	row("TOTAL (measured)", total_s, T);

	const double phase_sum = merge_ms + filter_ms + extend_ms + map_s.min_ms + filt_s.min_ms;
	std::cout << "  " << std::left << std::setw(18) << "(sum of phases)" << std::right
						<< std::setw(11) << std::fixed << std::setprecision(2) << phase_sum << " ms"
						<< "   delta vs measured total: " << std::setprecision(1)
						<< (T > 0 ? 100.0 * (phase_sum - T) / T : 0.0) << " %\n";

	const double tot = static_cast<double>(prof.total_ns());
	if (tot > 0) {
			const double gflops = prof.lsq_flops / (map_s.min_ms * 1e-3) / 1e9;
			std::cout << "  map internals (share of map thread-time):"
								<< "  symbolic " << std::setprecision(1) << 100.0 * prof.symbolic_ns / tot << "%"
								<< " | rhs " << 100.0 * prof.rhs_ns / tot << "%"
								<< " | submatrix " << 100.0 * prof.submatrix_ns / tot << "%"
								<< " | solve " << 100.0 * prof.solve_ns / tot << "%\n"
								<< "  local problems: rows " << prof.rows
								<< ", mean |I| " << std::setprecision(1) << (double)prof.sum_I / prof.rows
								<< " (max " << prof.max_I << ")"
								<< ", mean |J| " << (double)prof.sum_J / prof.rows
								<< " (max " << prof.max_J << ")"
								<< ", re-solved with pivoting " << prof.pivoted_rows
								<< "\n  least squares work: " << std::setprecision(1) << prof.lsq_flops / 1e9
								<< " GFLOP over " << map_s.min_ms << " ms = "
								<< std::setprecision(2) << gflops << " GFLOP/s\n";
	}

	std::cout << "  accuracy: ||N A_k - A_0||_F / ||A_0||_F  exact " << std::scientific << std::setprecision(4)
						<< exact.relative() << "   from QR residual " << (cheap.frobenius() / exact.target_frobenius);
	if (exact.relative() > 0) {
		std::cout << "   (QR value understates by " << std::fixed << std::setprecision(1)
							<< 100.0 * (exact.relative() - cheap.frobenius() / exact.target_frobenius) / exact.relative() << "%)";
	}
	std::cout << std::fixed << "\n";

	if (parse_fused(cfg.extension) && !std::is_same_v<PatternType, SimplePattern>) {
		const sample materialized_s = measure(cfg.iters, [&] {
			SparsityPattern<double, PatternType> p(source, target, params, 2, pm);
			p.computePattern();
			csr::CSRMatrix<double> m;
			Map::computeMap(target, source, *p.getPattern(), m);
		});
		const sample fused_s = measure(cfg.iters, [&] {
			SparsityPattern<double, PatternType> p(source, target, params, 1, pm);
			p.computePattern();
			csr::CSRMatrix<double> m;
			Map::computeMapFused(target, source, *p.getPattern(), 2, m);
		});
		std::cout << "  fused extension: pattern + map " << std::setprecision(2)
							<< materialized_s.min_ms << " ms materialized, " << fused_s.min_ms
							<< " ms fused (" << std::setprecision(3)
							<< (fused_s.min_ms > 0 ? materialized_s.min_ms / fused_s.min_ms : 0.0) << "x)\n";
		std::cout << "CSVFUSED," << label << "," << pv.name << "," << threads << ","
							<< to_string(spgemm_acc) << "," << std::setprecision(4)
							<< materialized_s.min_ms << "," << fused_s.min_ms << "\n";
	}

	const char* names[] = {"pattern_merge", "pattern_filter", "pattern_extend", "map", "post_filtration", "total"};
	const double vals[] = {merge_ms, filter_ms, extend_ms, map_s.min_ms, filt_s.min_ms, T};
	for (int i = 0; i < 6; ++i) {
		std::cout << "CSV," << label << "," << pv.name << "," << threads << "," << names[i] << ","
							<< std::setprecision(4) << vals[i] << "\n";
	}
	std::cout << "CSVACC," << label << "," << pv.name << "," << threads << ","
						<< merged->m_nnz << "," << std::scientific << std::setprecision(6)
						<< exact.relative() << "," << (cheap.frobenius() / exact.target_frobenius)
						<< std::fixed << "\n";
}

/// Whether -p selects `key`. A substring test on the raw list, so separators are not checked.
bool enabled(const config_t& cfg, const std::string& key) {
  return cfg.patterns == "all" || cfg.patterns.find(key) != std::string::npos;
}

void run_all(const config_t& cfg, int threads, const csr::CSRMatrix<double>& target,
			       const csr::CSRMatrix<double>& source)
{
	tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
	num_threads = threads;

	if (enabled(cfg, "simple"))   run_pattern("simple", cfg, threads, target, source, SimplePattern{});
	if (enabled(cfg, "global"))   run_pattern("global 0.001", cfg, threads, target, source, GlobalThresholdPattern{0.001});
	if (enabled(cfg, "column"))   run_pattern("column 0.8", cfg, threads, target, source, ColumnThresholdPattern{0.8});
	if (enabled(cfg, "fixed"))    run_pattern("fixed nnz 5", cfg, threads, target, source, FixedNNZPattern{5});
	if (enabled(cfg, "combined")) run_pattern("combined", cfg, threads, target, source, combinedThresholdPattern{0.001, 0.8});
}

} // namespace

int main(int argc, char** argv) {
	config_t cfg;
	parseargs(argc, argv, cfg);
	cfg.print();

	// Before any parallel work, so every thread TBB ever brings in gets its own core.
	ThreadPinner pinner;
	spgemm_acc = parse_accumulator(cfg.accumulator);

	// config_t leaves the indices unset so each benchmark picks its own default.
	const int target_idx = cfg.target_idx > 0 ? cfg.target_idx : 1;
	const int source_idx = cfg.source_idx > 0 ? cfg.source_idx : 6;
	const std::string target_path = cfg.matrix_dir + "matrix_" + std::to_string(target_idx) + ".txt";
	const std::string source_path = cfg.matrix_dir + "matrix_" + std::to_string(source_idx) + ".txt";
	csr::CSRMatrix<double> target = read_mat<double>(target_path.c_str());
	csr::CSRMatrix<double> source = read_mat<double>(source_path.c_str());
	std::cout << "target " << target_path << "  " << target.m_rows << " rows, " << target.m_nnz << " nnz\n"
						<< "source " << source_path << "  " << source.m_rows << " rows, " << source.m_nnz << " nnz\n";

	std::vector<int> sweep;
	if (cfg.threads > 0) {
		sweep.push_back(cfg.threads);
	} else {
		// -t 0 (or negative). The pinner finds no cores off Linux; fall back to logical CPUs.
		sweep = bench::thread_sweep(pinner.cores() > 0 ? pinner.cores()
																: static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
	}

	for (int t : sweep) {
		std::cout << "\n================ threads: " << t << " ================\n";
		bench::report_pinning(t);
		pinner.report(t);
		run_all(cfg, t, target, source);
		// Cumulative: one pinner serves the whole sweep, and a thread keeps its first core.
		std::cout << "# threads pinned so far: " << pinner.pinned() << "\n";
	}
	return 0;
}
