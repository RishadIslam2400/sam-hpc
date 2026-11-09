#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/amg.hpp"
#include "sam_hpc/precond/coarsening/ruge_stuben.hpp"
#include "sam_hpc/precond/damped_jacobi.hpp"
#include "sam_hpc/precond/gauss_seidel.hpp"
#include "sam_hpc/precond/ilu0.hpp"
#include "sam_hpc/precond/ilut.hpp"
#include "sam_hpc/util/config.hpp"
#include "sam_hpc/util/measure.hpp"
#include "sam_hpc/util/timer.hpp"

#include "bench_patterns.hpp"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
constexpr int kMinReps = 3;
constexpr double kBudgetMs = 60000.0;
constexpr int kDefaultTarget = 40;
constexpr int kDefaultSource = 50;
constexpr double kConvergedRelres = 1e-6;

using Matrix = csr::CSRMatrix<double>;

/// One (step, mode, pattern) measurement.
struct step_record {
	int step = 0;               // Source index k.
	int target = 0;             // Index of the matrix whose preconditioner was used.
	std::string mode;           // target | sam | reuse | recompute | periodic.
	std::string pattern = "-";  // Pattern label, "-" when no map is involved.
	size_t map_nnz = 0;
	double precond_ms = 0;      // Preconditioner built at this step; 0 when P is reused.
	bench::sample map;          // Pattern + map, min over repetitions.
	double transpose_ms = 0;    // Right map only: serial transposes inside `map`.
	bench::sample solver;
	size_t iterations = 0;
	double relres = 0;          // True ||b - A x|| / ||b||.
	std::optional<double> map_residual;  // ||N A_k - A_0||_F / ||A_0||_F (or the right form).
	bool converged = false;

	double total_ms() const { return precond_ms + map.min_ms + solver.min_ms; }
};

/// Settings shared by every record of a run.
struct run_settings {
	const config_t& cfg;
	const char* variant;
	patternMatrix pm;
	mapSide side;
	bool fused;
	GMRES<double>::params prm;
	bench::pattern_set patterns;
	std::string precond_label;
};

/// One loaded system.
struct system_data {
	Matrix A;
	std::vector<double> b;
};

system_data load_system(const config_t& cfg, int idx) {
	system_data s;
	s.A = read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(idx) + ".txt").c_str());
	s.b = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(idx) + ".txt").c_str());
	if (s.b.size() != s.A.m_rows) {
		std::cerr << "error: rhs_" << idx << " has " << s.b.size() << " entries but matrix_" << idx
							<< " has " << s.A.m_rows << " rows\n";
		std::exit(1);
	}
	return s;
}

/// The n x n identity.
Matrix identity(size_t n) {
	Matrix I;
	I.m_rows = I.m_cols = I.m_nnz = n;
	I.m_row_pointers.resize(n + 1);
	I.m_col_indices.resize(n);
	I.m_vals.resize(n);
	for (size_t i = 0; i < n; ++i) {
		I.m_row_pointers[i] = i;
		I.m_col_indices[i] = static_cast<csr::index_t>(i);
		I.m_vals[i] = 1.0;
	}
	I.m_row_pointers[n] = n;
	return I;
}

// Preconditioner construction
using Ilut = ilut<double>;
template <template <class> class Relax>
using AmgRS = amg<double, ruge_stuben, Relax>;

std::shared_ptr<const Ilut> make_precond(const Matrix& A, const config_t& cfg, const Ilut*) {
	Ilut::params p;
	p.fill_factor = cfg.ilut_fill;
	p.droptol = cfg.ilut_droptol;
	return std::make_shared<const Ilut>(A, p);
}

template <template <class> class Relax>
typename AmgRS<Relax>::params amg_params(const config_t& cfg) {
	typename AmgRS<Relax>::params p;
	p.coarsening.eps_strong = static_cast<float>(cfg.amg_strong);
	p.npre = cfg.amg_npre;
	p.npost = cfg.amg_npost;
	if (cfg.amg_coarse > 0)
		p.coarse_enough = cfg.amg_coarse;
	p.allow_rebuild = false;
	if constexpr (std::is_same_v<Relax<double>, Ilut>) {
		p.relax.fill_factor = cfg.amg_ilut_fill;
		p.relax.droptol = cfg.amg_ilut_droptol;
	}
	return p;
}

template <template <class> class Relax>
std::shared_ptr<const AmgRS<Relax>> make_precond(const Matrix& A, const config_t& cfg,
                                                 const AmgRS<Relax>*)
{
	return std::make_shared<const AmgRS<Relax>>(A, amg_params<Relax>(cfg));
}

/// Builds a preconditioner once and returns it with its setup time.
template <class Precond>
std::pair<std::shared_ptr<const Precond>, double> build_preconditioner(const Matrix& A,
                                                                       const config_t& cfg)
{
	Timer t;
	t.start();
	auto P = make_precond(A, cfg, static_cast<const Precond*>(nullptr));
	return {std::move(P), t.elapsed() / 1000.0};
}

// Records
const char* kCsvHeader =
		"# CSV,step,target,mode,pattern,variant,map_side,threads,level,precond,map_nnz,nnz_per_row,"
		"precond_ms,map_min_ms,map_med_ms,transpose_ms,solve_min_ms,solve_med_ms,total_min_ms,iters,"
		"converged,solve_relres,map_residual,map_reps,solve_reps\n";

void emit(const run_settings& s, const step_record& r, size_t n) {
	std::cout << "CSV," << r.step << "," << r.target << "," << r.mode << "," << r.pattern << ","
						<< s.variant << "," << to_string(s.side) << "," << s.cfg.threads << ","
						<< s.patterns.level << "," << s.precond_label << "," << r.map_nnz
						<< std::fixed << std::setprecision(4) << "," << (double)r.map_nnz / n
						<< "," << r.precond_ms << "," << r.map.min_ms << "," << r.map.median_ms
						<< "," << r.transpose_ms << "," << r.solver.min_ms << "," << r.solver.median_ms
						<< "," << r.total_ms() << "," << r.iterations << "," << (r.converged ? 1 : 0)
						<< std::scientific << std::setprecision(6) << "," << r.relres << ",";
	if (r.map_residual)
		std::cout << *r.map_residual;
	std::cout << std::fixed << "," << r.map.reps << "," << r.solver.reps << "\n";
}

/// Prints the one-line progress summary of a record, then its CSV row.
void finish(const run_settings& s, step_record& r, std::vector<step_record>& all, size_t n) {
	r.converged = r.iterations < s.prm.maxIter && r.relres <= kConvergedRelres;
	std::cout << "  [" << r.mode << " | " << r.pattern << "] " << std::fixed << std::setprecision(1);
	if (r.precond_ms > 0)
		std::cout << "precond " << r.precond_ms << " ms, ";
	if (r.map.reps > 0) {
		std::cout << "map " << r.map.min_ms << " ms";
		if (r.transpose_ms > 0)
			std::cout << " (transposes " << r.transpose_ms << " ms)";
		std::cout << ", ";
	}
	std::cout << "solve " << r.solver.min_ms << " ms, " << r.iterations << " iters";
	if (!r.converged)
		std::cout << " [DID NOT CONVERGE]";
	std::cout << std::scientific << std::setprecision(3) << ", relres " << r.relres;
	if (r.map_residual)
		std::cout << ", map residual " << *r.map_residual;
	std::cout << std::fixed << "\n";
	emit(s, r, n);
	std::cout << std::flush;
	all.push_back(r);
}

/// Times GMRES on A x = b with preconditioner P and, if given, the map N.
template <class Precond>
void solve_into(const run_settings& s, step_record& r, const Matrix& A, const Precond& P,
                const Matrix* N, const std::vector<double>& b)
{
	GMRES<double> solver(A.m_cols, s.prm);
	std::vector<double> t;
	for (int rep = 0; rep < std::max(1, s.cfg.solve_reps); ++rep) {
		std::vector<double> x(A.m_cols, 0.0);
		const auto begin = bench::clock_type::now();
		auto [iters, error] = !N ? solver.solve(A, P, b, x) : s.side == mapSide::left 
																												? solver.solve(A, P, *N, b, x)
		                                              			: solver.solveRightMap(A, P, *N, b, x);
		t.push_back(bench::ms_since(begin));
		r.iterations = iters;
		r.relres = error;
	}
	r.solver = bench::summarize(std::move(t));
}

/// Pattern + map from A_k onto A_ref, then the solve with P_ref N.
template <class Precond, typename PatternType>
void map_and_solve(const run_settings& s, step_record& r, const Matrix& A_ref, const Precond& P_ref,
                   const system_data& src, const PatternType& params)
{
	using Map = SparseApproximateMap<double, PatternType>;
	const int level = pattern_is_extended_v<PatternType> ? s.patterns.level : 1;
	const bool fused = s.fused && pattern_is_extended_v<PatternType> && level > 1;

	Matrix map;
	r.map = bench::measure(s.cfg.iters, [&] {
		SparsityPattern<double, PatternType> pattern(src.A, A_ref, params, fused ? 1 : level, s.pm);
		pattern.computePattern();
		Matrix m;
		sam_transpose_times tt;
		if (fused)
			Map::computeMapSidedFused(s.side, A_ref, src.A, *pattern.getPattern(), level, m, &tt);
		else
			Map::computeMapSided(s.side, A_ref, src.A, *pattern.getPattern(), m, &tt);
		map = std::move(m);
		r.transpose_ms = tt.total_ms();
	}, kBudgetMs, kMinReps);
	r.map_nnz = map.m_nnz;
	r.map_residual = s.side == mapSide::left ? sam_residual_exact(map, src.A, A_ref).relative()
	                                         : sam_residual_exact_right(map, src.A, A_ref).relative();
	solve_into(s, r, src.A, P_ref, &map, src.b);
}

/// SAM records for one step, every selected pattern, mapped onto (A_ref, P_ref).
template <class Precond>
void sam_records(const run_settings& s, const char* mode, int step, int ref_idx, const Matrix& A_ref,
                 const Precond& P_ref, const system_data& src, std::vector<step_record>& all)
{
	// S(A_ref) for the `target` family: the matrix being mapped onto, which under periodic
	// recomputation is not the original target.
	const csr::CSRMatrix<int> ref_structure = bench::structure_of(A_ref);
	bench::for_each_pattern(s.patterns, &ref_structure, [&](const std::string& name, const auto& params) {
		step_record r;
		r.step = step;
		r.target = ref_idx;
		r.mode = mode;
		r.pattern = name;
		map_and_solve(s, r, A_ref, P_ref, src, params);
		finish(s, r, all, A_ref.m_rows);
	});
}

/// Prints every record once more, as one table.
void report(const run_settings& s, const std::vector<step_record>& all, size_t n) {
	const bool right = s.side == mapSide::right;
	std::cout << "\n\nBenchmark Summary   (map: min over repetitions; solve: min of "
						<< s.cfg.solve_reps << "; precond: one build; ms)\n"
						<< "# n " << n << ", threads " << s.cfg.threads << ", variant " << s.variant
						<< ", map " << to_string(s.side) << ", precond " << s.precond_label << "\n"
						<< "# map residual is " << (right ? "||A_k N - A_0||_F" : "||N A_k - A_0||_F")
						<< " / ||A_0||_F, against the matrix in column 'tgt'\n"
						<< "# relres is the true ||b - A_k x|| / ||b||; '!' marks a run that did not converge\n\n"
						<< std::right << std::setw(5) << "step" << std::setw(5) << "tgt" << "  " << std::left
						<< std::setw(10) << "mode" << std::setw(24) << "pattern" << std::right
						<< std::setw(9) << "nnz/row" << std::setw(11) << "precond" << std::setw(10) << "map"
						<< std::setw(11) << "solve" << std::setw(11) << "total" << std::setw(7) << "iters"
						<< std::setw(12) << "relres" << std::setw(12) << "map res" << "\n";
	for (const step_record& r : all) {
		const std::string it_s = std::to_string(r.iterations) + (r.converged ? "" : "!");
		std::cout << std::right << std::setw(5) << r.step << std::setw(5) << r.target << "  " << std::left
							<< std::setw(10) << r.mode << std::setw(24) << r.pattern.substr(0, 23) << std::right
							<< std::fixed << std::setprecision(2) << std::setw(9) << (double)r.map_nnz / n
							<< std::setprecision(1) << std::setw(11) << r.precond_ms << std::setw(10) << r.map.min_ms
							<< std::setw(11) << r.solver.min_ms << std::setw(11) << r.total_ms() << std::setw(7) << it_s
							<< std::scientific << std::setprecision(3) << std::setw(12) << r.relres;
		if (r.map_residual)
			std::cout << std::setw(12) << *r.map_residual;
		else
			std::cout << std::setw(12) << "-";
		std::cout << std::fixed << "\n";
	}
}

template <class Precond>
void run(const run_settings& s) {
	const config_t& cfg = s.cfg;
	const int target_idx = cfg.target_idx > 0 ? cfg.target_idx : kDefaultTarget;
	const int first = cfg.source_idx > 0 ? cfg.source_idx : kDefaultSource;
	const int last = cfg.last_idx > 0 ? cfg.last_idx : first;
	if (last < first) {
		std::cerr << "error: last source " << last << " is before the first " << first << "\n";
		std::exit(1);
	}

	const bool do_sam = list_selects(cfg.modes, "sam");
	const bool do_reuse = list_selects(cfg.modes, "reuse");
	const bool do_recompute = list_selects(cfg.modes, "recompute");
	const bool do_periodic = list_selects(cfg.modes, "periodic");
	if (!do_sam && !do_reuse && !do_recompute && !do_periodic) {
		std::cerr << "error: -r '" << cfg.modes << "' selects no mode. Valid: sam, reuse, recompute, "
																							 "periodic, all\n";
		std::exit(1);
	}
	if (do_periodic && cfg.period <= 0) {
		std::cerr << "error: the periodic mode needs a recompute period, -R <steps> > 0\n";
		std::exit(1);
	}

	const system_data target = load_system(cfg, target_idx);
	const size_t n = target.A.m_rows;
	std::cout << "target matrix_" << target_idx << ", sources matrix_" << first << " .. matrix_" << last
						<< ", n = " << n << ", nnz " << target.A.m_nnz << "\n";
	for (const auto& [name, p] : s.patterns.explicit_patterns) {
		if (p.m_rows != n) {
			std::cerr << "error: " << name << " has " << p.m_rows << " rows, the matrices " << n << "\n";
			std::exit(1);
		}
	}

	std::cout << kCsvHeader;
	std::vector<step_record> all;

	// The target with its own preconditioner.
	auto [P_target, P_target_ms] = build_preconditioner<Precond>(target.A, cfg);
	{
		std::cout << "\n=== target " << target_idx << " ===\n";
		step_record r;
		r.step = target_idx;
		r.target = target_idx;
		r.mode = "target";
		r.precond_ms = P_target_ms;
		solve_into(s, r, target.A, *P_target, nullptr, target.b);
		finish(s, r, all, n);
	}

	// Periodic state: the matrix the map currently targets, and its preconditioner.
	int ref_idx = target_idx;
	std::shared_ptr<const Matrix> A_ref(&target.A, [](const Matrix*) {});
	std::shared_ptr<const Precond> P_ref = P_target;

	for (int k = first; k <= last; ++k) {
		std::cout << "\n=== step " << k << " ===\n";
		const auto src = std::make_shared<const system_data>(load_system(cfg, k));
		const bool periodic_recompute = do_periodic && k - ref_idx >= cfg.period;

		// One fresh preconditioner serves both recompute and a periodic recompute step.
		std::shared_ptr<const Precond> P_own;
		double P_own_ms = 0;
		std::optional<step_record> own;
		if (do_recompute || periodic_recompute) {
			std::tie(P_own, P_own_ms) = build_preconditioner<Precond>(src->A, cfg);
			step_record r;
			r.step = k;
			r.target = k;
			r.mode = "recompute";
			r.precond_ms = P_own_ms;
			solve_into(s, r, src->A, *P_own, nullptr, src->b);
			own = r;
			if (do_recompute)
				finish(s, r, all, n);
		}

		if (do_reuse) {
			step_record r;
			r.step = k;
			r.target = target_idx;
			r.mode = "reuse";
			// No map is N = I, so its "map residual" is the drift ||A_k - A_0||_F / ||A_0||_F.
			r.map_residual = sam_residual_exact(identity(src->A.m_rows), src->A, target.A).relative();
			solve_into(s, r, src->A, *P_target, nullptr, src->b);
			finish(s, r, all, n);
		}

		const size_t sam_begin = all.size();
		if (do_sam)
			sam_records(s, "sam", k, target_idx, target.A, *P_target, *src, all);
		const size_t sam_end = all.size();

		if (do_periodic) {
			if (periodic_recompute) {
				step_record r = *own;
				r.mode = "periodic";
				finish(s, r, all, n);
				ref_idx = k;
				A_ref = std::shared_ptr<const Matrix>(src, &src->A);
				P_ref = P_own;
			} else if (ref_idx == target_idx && do_sam) {
				// Same map onto the same target as the sam rows just measured: relabel them.
				for (size_t i = sam_begin; i < sam_end; ++i) {
					step_record r = all[i];
					r.mode = "periodic";
					finish(s, r, all, n);
				}
			} else {
				sam_records(s, "periodic", k, ref_idx, *A_ref, *P_ref, *src, all);
			}
		}
	}

	report(s, all, n);
}

/// Dispatches on --precond / --amg-relax to the matching instantiation of run().
void dispatch(run_settings& s) {
	const config_t& cfg = s.cfg;
	if (cfg.precond == "ilut") {
		s.precond_label = "ILUT(" + bench::fmt(cfg.ilut_fill) + ":" + bench::fmt(cfg.ilut_droptol) + ")";
		return run<Ilut>(s);
	}
	if (cfg.precond == "amg") {
		s.precond_label = "AMG-RS(" + cfg.amg_relax + ")";
		if (cfg.amg_relax == "jacobi")
			return run<AmgRS<damped_jacobi>>(s);
		if (cfg.amg_relax == "gauss_seidel")
			return run<AmgRS<gauss_seidel>>(s);
		if (cfg.amg_relax == "ilu0")
			return run<AmgRS<ilu0>>(s);
		if (cfg.amg_relax == "ilut")
			return run<AmgRS<ilut>>(s);
		std::cerr << "error: unknown --amg-relax '" << cfg.amg_relax
							<< "'. Valid: jacobi | gauss_seidel | ilu0 | ilut\n";
		std::exit(2);
	}
	std::cerr << "error: unknown --precond '" << cfg.precond << "'. Valid: ilut | amg\n";
	std::exit(2);
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
		std::cerr << "error: unknown map side '" << cfg.map_side << "'. Valid: left | right\n";
		return 2;
	}

	GMRES<double>::params prm;
	if (cfg.precond_side == "right") {
		prm.pside = precondSide::right;
	} else if (cfg.precond_side == "left") {
		prm.pside = precondSide::left;
	} else {
		std::cerr << "error: unknown --precond-side '" << cfg.precond_side << "'. Valid: left | right\n";
		return 2;
	}
	prm.M = static_cast<unsigned>(cfg.gmres_restart);
	prm.maxIter = static_cast<unsigned>(cfg.gmres_maxit);
	prm.tol = cfg.gmres_tol;

	const pattern_variant pv = parse_pattern_variant(cfg.variant);
	try {
		run_settings s{cfg, pv.name, pv.union_candidate ? patternMatrix::union_max : patternMatrix::source,
		               side, parse_fused(cfg.extension), prm, bench::pattern_set(cfg), {}};
		dispatch(s);
	} catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
		return 1;
	}

	std::cout << "# threads pinned: " << pinner.pinned() << " of " << cfg.threads << "\n";
	return 0;
}
