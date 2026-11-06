#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/ilut.hpp"
#include "sam_hpc/util/config.hpp"
#include "sam_hpc/util/measure.hpp"
#include "sam_hpc/util/timer.hpp"

#include <iomanip>
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace {
constexpr int kMinReps = 3;
constexpr double kBudgetMs = 60000.0;
constexpr int kDefaultTarget = 40;
constexpr int kDefaultSource = 50;

struct BenchmarkResults {
	size_t map_nnz = 0;
	bench::sample map;
	bench::sample solver;
	double relative_error = 0.0;
	size_t iterations = 0;
	double residual = 0.0;
	sam_transpose_times transposes;
	bool solved = false;
	bool converged = false;
};

/// Benchmark settings
template <typename Precond>
struct problem {
	const csr::CSRMatrix<double>& target;
	const csr::CSRMatrix<double>& source;
	const Precond& P_0;
	const GMRES<double>::params& prm;
	const std::vector<double>& rhs;
	patternMatrix pm;
	mapSide side;
	int max_reps;
	bool fused;
};

template <typename PatternType, typename Precond, typename... PatternArgs>
void run_benchmark(const problem<Precond>& p, std::map<std::string, BenchmarkResults>& results,
                   const std::string& pattern_name, PatternArgs&&... args)
{
	std::cout << "Running benchmark: [" << pattern_name << "]..." << std::flush;

	using Map = SparseApproximateMap<double, PatternType>;
	BenchmarkResults res;

	// Benchmark sparsity pattern construction and SAM computation
	constexpr int level = std::is_same_v<PatternType, SimplePattern> ? 1 : 2;
	csr::CSRMatrix<double> map;
	res.map = bench::measure(p.max_reps, [&] {
		PatternType params(args...);
		SparsityPattern<double, PatternType> pattern(p.source, p.target, params, p.fused ? 1 : level, p.pm);
		pattern.computePattern();
		csr::CSRMatrix<double> m;
		sam_transpose_times tt;
		if (p.fused) {
			Map::computeMapSidedFused(p.side, p.target, p.source, *pattern.getPattern(), level, m, &tt);
		} else {
			Map::computeMapSided(p.side, p.target, p.source, *pattern.getPattern(), m, &tt);
		}
		map = std::move(m);
		res.transposes = tt;
	}, kBudgetMs, kMinReps);
	res.map_nnz = map.m_nnz;

	// Benchmark the solver
	GMRES<double> solver(p.source.m_cols, p.prm);
	res.solver = bench::measure(p.max_reps, [&] {
		std::vector<double> x(p.source.m_cols, 0.0);
		auto [iters, error] = p.side == mapSide::left
																		? solver.solve(p.source, p.P_0, map, p.rhs, x)
																		: solver.solveRightMap(p.source, p.P_0, map, p.rhs, x);
		res.iterations = iters;
		res.relative_error = error;
	}, kBudgetMs, kMinReps);
	res.solved = true;

	res.converged = res.iterations < (size_t)p.prm.maxIter && res.relative_error <= 1e-6;
	res.residual = p.side == mapSide::left
												? sam_residual_exact(map, p.source, p.target).relative()
												: sam_residual_exact_right(map, p.source, p.target).relative();

	results[pattern_name] = res;
	std::cout << " map " << std::fixed << std::setprecision(1) << res.map.min_ms << " ms";
	if (res.transposes.total_ns() > 0)
		std::cout << " (transposes " << res.transposes.total_ms() << " ms)";
	if (res.solved) {
		std::cout << ", " << res.iterations << " iters";
		if (!res.converged)
			std::cout << " [DID NOT CONVERGE]";
	}
	std::cout << ", residual " << std::scientific << std::setprecision(3) << res.residual
						<< std::fixed << "\n";
}

/// Prints the summary table and the CSV rows.
void report(const std::map<std::string, BenchmarkResults>& results, const config_t& cfg,
            size_t n, const char* variant, mapSide side)
{
	const bool right = side == mapSide::right;
	std::cout << "\n\nBenchmark Summary   (min over repetitions; times in ms)\n"
						<< "# n " << n << ", threads " << cfg.threads
						<< ", variant " << variant << ", map " << to_string(side) << "\n"
						<< "# residual is " << (right ? "||A_k N - A_0||_F" : "||N A_k - A_0||_F")
						<< " / ||A_0||_F\n";
	if (right)
		std::cout << "# 'trans' is the serial transpose time inside 'map': the price of "
								 "holding A_k in CSR\n"
							<< "# solve uses operator P_0 A_k N with x = N u (GMRES::solveRightMap), "
								 "not the left map's P_0 N A_k\n";
	std::cout << "\n"
						<< std::left << std::setw(52) << "pattern" << std::right
						<< std::setw(11) << "nnz/row" << std::setw(11) << "map"
						<< std::setw(8) << "spr" << std::setw(10) << (right ? "trans" : "")
						<< std::setw(9) << (right ? "trans%" : "")
						<< std::setw(12) << "solve"
						<< std::setw(8) << "spr" << std::setw(11) << "total"
						<< std::setw(8) << "iters" << std::setw(13) << "residual" << "\n";

	for (const auto& [name, r] : results) {
		const double tms = r.transposes.total_ms();
		std::cout << std::left << std::setw(52) << name << std::right << std::fixed
							<< std::setprecision(2) << std::setw(11) << (double)r.map_nnz / n
							<< std::setprecision(1)
							<< std::setw(11) << r.map.min_ms
							<< std::setw(7) << r.map.spread_pct() << "%";
		if (right) {
			std::cout << std::setw(10) << tms << std::setw(8)
								<< (r.map.min_ms > 0 ? 100.0 * tms / r.map.min_ms : 0.0) << "%";
		} else {
			std::cout << std::setw(10) << "" << std::setw(9) << "";
		}
		if (r.solved) {
			const std::string it_s = std::to_string(r.iterations) + (r.converged ? "" : "!");
			std::cout << std::setw(12) << r.solver.min_ms
								<< std::setw(7) << r.solver.spread_pct() << "%"
								<< std::setw(11) << (r.map.min_ms + r.solver.min_ms)
								<< std::setw(8) << it_s;
		} else {
			std::cout << std::setw(12) << "-" << std::setw(8) << "-"
								<< std::setw(11) << "-" << std::setw(8) << "-";
		}
		std::cout << std::setw(13) << std::scientific << std::setprecision(3) << r.residual
							<< std::fixed << "\n";
	}

	std::cout << "\n# CSV,pattern,variant,map_side,transpose_ms,threads,map_nnz,map_min_ms,map_med_ms,"
							 "solve_min_ms,solve_med_ms,total_min_ms,iters,converged,residual,map_reps,solve_reps\n";
	for (const auto& [name, r] : results) {
		std::cout << "CSV," << name << "," << variant
							<< "," << to_string(side)
							<< "," << std::fixed << std::setprecision(4) << r.transposes.total_ms()
							<< "," << cfg.threads
							<< "," << r.map_nnz
							<< "," << std::fixed << std::setprecision(4) << r.map.min_ms
							<< "," << r.map.median_ms
							<< "," << r.solver.min_ms << "," << r.solver.median_ms
							<< "," << (r.map.min_ms + r.solver.min_ms)
							<< "," << r.iterations
							<< "," << (r.converged ? 1 : 0)
							<< "," << std::scientific << std::setprecision(6) << r.residual
							<< "," << r.map.reps << "," << r.solver.reps << "\n";
	}
}

/// The matrices and right hand sides of one source/target pair, loaded once.
struct pair_data {
	csr::CSRMatrix<double> target;
	csr::CSRMatrix<double> source;
	std::vector<double> r_target;
	std::vector<double> r_source;
};

pair_data load(const config_t& cfg) {
	const int target_idx = cfg.target_idx > 0 ? cfg.target_idx : kDefaultTarget;
	const int source_idx = cfg.source_idx > 0 ? cfg.source_idx : kDefaultSource;

	pair_data d;
	d.target = read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(target_idx) + ".txt").c_str());
	d.source = read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(source_idx) + ".txt").c_str());
	d.r_target = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(target_idx) + ".txt").c_str());
	d.r_source = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(source_idx) + ".txt").c_str());

	const auto require = [](const std::vector<double>& v, const csr::CSRMatrix<double>& A, const char* what) {
		if (v.size() != A.m_rows) {
			std::cerr << "error: " << what << " has " << v.size() << " entries but the matrix has "
								<< A.m_rows << " rows\n";
			std::exit(1);
		}
	};
	require(d.r_target, d.target, "target rhs");
	require(d.r_source, d.source, "source rhs");

	std::cout << "target matrix_" << target_idx << ", source matrix_" << source_idx
						<< ", n = " << d.target.m_rows << ", nnz " << d.target.m_nnz << "\n";
	return d;
}

template <typename Precond>
std::unique_ptr<Precond> build_preconditioner(const pair_data& d,
                                              const typename Precond::params& ilu_prm) {
	Timer setup;
	setup.start();
	auto P = std::make_unique<Precond>(d.target, ilu_prm);
	std::cout << "Preconditioner built in " << std::fixed << std::setprecision(1)
						<< setup.elapsed() / 1000.0 << " ms\n";
	return P;
}

/// Baseline solve of the target matrix
template <typename Precond>
void reference_solve(const pair_data& d, const Precond& P_target, const GMRES<double>::params& prm) {
	std::vector<double> x(d.target.m_cols, 0.0);
	GMRES<double> solver(d.target.m_cols, prm);
	Timer t;
	t.start();
	auto [iters, error] = solver.solve(d.target, P_target, d.r_target, x);
	std::cout << "Target system with its own preconditioner: " << iters << " iters, "
						<< std::scientific << std::setprecision(3) << error << " rel error, "
						<< std::fixed << std::setprecision(1) << t.elapsed() / 1000.0 << " ms\n\n";
}

void run_top_opt(const config_t& cfg, const pair_data& d, patternMatrix pm, mapSide side,
                 std::map<std::string, BenchmarkResults>& results)
{
	// TODO: Fix all the parameter for benchmark setting
	ilut<double>::params ilu_prm;
	ilu_prm.fill_factor = 250;
	ilu_prm.droptol = 1e-3;

	// TODO: I think the restart will work now after the big fixes.
	GMRES<double>::params prm;
	prm.pside = precondSide::right;
	prm.M = 50;
	prm.maxIter = 2000;

	const auto P_target = build_preconditioner<ilut<double>>(d, ilu_prm);
	reference_solve(d, *P_target, prm);
	const problem<ilut<double>> p{d.target, d.source, *P_target, prm, d.r_source, pm, side,
																cfg.iters, parse_fused(cfg.extension)};


	run_benchmark<SimplePattern>(p, results, "S(A_k) unsparsified");
	run_benchmark<ColumnThresholdPattern>(p, results, "Column threshold 0.7", 0.7);
	run_benchmark<ColumnThresholdPattern>(p, results, "Column threshold 0.8", 0.8);
	run_benchmark<ColumnThresholdPattern>(p, results, "Column threshold 0.9", 0.9);
	run_benchmark<FixedNNZPattern>(p, results, "Fixed nnz 3", 3);
	run_benchmark<FixedNNZPattern>(p, results, "Fixed nnz 5", 5);
	run_benchmark<FixedNNZPattern>(p, results, "Fixed nnz 7", 7);
	run_benchmark<combinedThresholdPattern>(p, results, "Combined 0.1 / 0.8", 0.1, 0.8);
	run_benchmark<combinedThresholdPattern>(p, results, "Combined 0.1 / 0.9", 0.1, 0.9);
}

} // namespace

int main(int argc, char** argv) {
	config_t cfg;
	parseargs(argc, argv, cfg);
	cfg.print();

	// Before any parallel work, so every thread TBB ever brings in gets its own core.
	ThreadPinner pinner;

	num_threads = cfg.threads;
	spgemm_acc = parse_accumulator(cfg.accumulator);
	tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
	bench::report_pinning(cfg.threads);
	pinner.report(cfg.threads);

	mapSide side = mapSide::left;
	if (cfg.map_side == "right") {
		side = mapSide::right;
	} else if (cfg.map_side != "left") {
		std::cerr << "Unknown map side '" << cfg.map_side << "', using left. Valid: left | right\n";
	}

	const pattern_variant pv = parse_pattern_variant(cfg.variant);
	const patternMatrix pm = pv.union_candidate ? patternMatrix::union_max : patternMatrix::source;

	const pair_data d = load(cfg);

	std::map<std::string, BenchmarkResults> results;
	run_top_opt(cfg, d, pm, side, results);

	report(results, cfg, d.target.m_rows, pv.name, side);
	std::cout << "# threads pinned: " << pinner.pinned() << " of " << cfg.threads << "\n";
	return 0;
}
