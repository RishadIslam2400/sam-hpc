// End-to-end SAM benchmark: build a SAM from A_source to A_target, then solve
// A_source x = b_source with GMRES preconditioned by P_target * N.
//
// Every configuration is repeated under a time budget and reported as
// min / median with the observed spread, never as elapsed()/iters. See measure.hpp for why.
// Matrix loading and preconditioner setup happen once, outside every timed region.
//
// Usage: pipeline_bench -d <top_opt|weather> -x <matrix dir> -y <rhs dir> [-k max reps]
//                       [-t threads] [-a target idx] [-b source idx] [-u variant]
//
// The dataset preset fixes the preconditioner, the GMRES parameters and the pattern list;
// see run_top_opt() and run_weather().

#include "sam_hpc/sam/sam.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/krylov/gmres.hpp"
#include "sam_hpc/precond/iluk.hpp"
#include "sam_hpc/precond/ilutp.hpp"
#include "sam_hpc/util/config.hpp"
#include "sam_hpc/util/measure.hpp"
#include "sam_hpc/util/timer.hpp"

#include <iomanip>
#include <map>
#include <memory>
#include <vector>

namespace {

// A single GMRES solve here runs for tens of seconds, so measure.hpp's default 4 s budget
// would give exactly one repetition and therefore no spread at all. Force at least three.
constexpr int kMinReps = 3;
constexpr double kBudgetMs = 60000.0;

// Both sequences are mapped from A_50 back onto A_40 unless -a / -b say otherwise.
constexpr int kDefaultTarget = 40;
constexpr int kDefaultSource = 50;

struct BenchmarkResults {
    size_t map_nnz = 0;
    bench::sample map;      // ms
    bench::sample solver;   // ms
    double relative_error = 0.0;
    size_t iterations = 0;
    double residual = 0.0;  // ||N A_k - A_0||_F / ||A_0||_F, untimed diagnostic
};

template <typename Precond>
struct problem {
    const CSRMatrix<double>& target;
    const CSRMatrix<double>& source;
    const Precond& P_0;
    const GMRES<double>::params& prm;
    const std::vector<double>& rhs;
    patternMatrix pm;
    int max_reps;
};

/// Times the map and the solve separately, each with its own repetitions, then records the
/// accuracy of the map from an untimed run.
template <typename PatternType, typename Precond, typename... PatternArgs>
void run_benchmark(const problem<Precond>& p, std::map<std::string, BenchmarkResults>& results,
                   const std::string& pattern_name, PatternArgs&&... args)
{
    std::cout << "Running benchmark: [" << pattern_name << "]..." << std::flush;

    using Map = SparseApproximateMap<double, PatternType>;
    BenchmarkResults res;

    // Each repetition builds the pattern and the map from scratch, exactly as a real
    // sequence step would - reusing a warm output buffer would hide the allocation cost.
    CSRMatrix<double> map;
    res.map = bench::measure(p.max_reps, [&] {
        PatternType params(args...);
        SparsityPattern<double, PatternType> pattern(p.source, p.target, params, 2, p.pm);
        pattern.computePattern();
        CSRMatrix<double> m;
        Map::computeMap(p.target, p.source, *pattern.getPattern(), m);
        map = std::move(m);
    }, kBudgetMs, kMinReps);
    res.map_nnz = map.m_nnz;

    res.solver = bench::measure(p.max_reps, [&] {
        std::vector<double> x(p.source.m_cols, 0.0);
        GMRES<double> solver(p.source.m_cols, p.prm);
        auto [iters, error] = solver.solve(p.source, p.P_0, map, p.rhs, x);
        res.iterations = iters;
        res.relative_error = error;
    }, kBudgetMs, kMinReps);

    res.residual = sam_residual_exact(map, p.source, p.target).relative();

    results[pattern_name] = res;
    std::cout << " map " << std::fixed << std::setprecision(1) << res.map.min_ms << " ms, "
              << res.iterations << " iters\n";
}

void report(const std::map<std::string, BenchmarkResults>& results, const config_t& cfg,
            size_t n, const char* variant)
{
    std::cout << "\n\nBenchmark Summary   (min over repetitions; times in ms)\n"
              << "# dataset " << cfg.dataset << ", n " << n << ", threads " << cfg.threads
              << ", variant " << variant << "\n\n"
              << std::left << std::setw(52) << "pattern" << std::right
              << std::setw(11) << "nnz/row" << std::setw(11) << "map"
              << std::setw(8) << "spr" << std::setw(12) << "solve"
              << std::setw(8) << "spr" << std::setw(11) << "total"
              << std::setw(8) << "iters" << std::setw(13) << "residual" << "\n";

    for (const auto& [name, r] : results) {
        std::cout << std::left << std::setw(52) << name << std::right << std::fixed
                  << std::setprecision(2) << std::setw(11) << (double)r.map_nnz / n
                  << std::setprecision(1)
                  << std::setw(11) << r.map.min_ms
                  << std::setw(7) << r.map.spread_pct() << "%"
                  << std::setw(12) << r.solver.min_ms
                  << std::setw(7) << r.solver.spread_pct() << "%"
                  << std::setw(11) << (r.map.min_ms + r.solver.min_ms)
                  << std::setw(8) << r.iterations
                  << std::setw(13) << std::scientific << std::setprecision(3) << r.residual
                  << std::fixed << "\n";
    }

    // machine readable
    std::cout << "\n# CSV,pattern,variant,threads,map_nnz,map_min_ms,map_med_ms,"
                 "solve_min_ms,solve_med_ms,total_min_ms,iters,residual,map_reps,solve_reps\n";
    for (const auto& [name, r] : results) {
        std::cout << "CSV," << name << "," << variant << "," << cfg.threads
                  << "," << r.map_nnz
                  << "," << std::fixed << std::setprecision(4) << r.map.min_ms
                  << "," << r.map.median_ms
                  << "," << r.solver.min_ms << "," << r.solver.median_ms
                  << "," << (r.map.min_ms + r.solver.min_ms)
                  << "," << r.iterations
                  << "," << std::scientific << std::setprecision(6) << r.residual
                  << "," << r.map.reps << "," << r.solver.reps << "\n";
    }
}

/// The matrices and right hand sides of one source/target pair, loaded once.
struct pair_data {
    CSRMatrix<double> target;
    CSRMatrix<double> source;
    std::vector<double> r_target;
    std::vector<double> r_source;
};

pair_data load(const config_t& cfg) {
    const int target_idx = cfg.target_idx > 0 ? cfg.target_idx : kDefaultTarget;
    const int source_idx = cfg.source_idx > 0 ? cfg.source_idx : kDefaultSource;

    pair_data d;
    read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(target_idx) + ".txt").c_str(), d.target);
    read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(source_idx) + ".txt").c_str(), d.source);

    // matrix_i pairs with rhs_i. This used to read rhs_<source+1> for the source system
    // while reading rhs_<target> for the target - an off-by-one that, on a 50 matrix
    // dataset with source 50, asked for a file that does not exist. read_vec returns an
    // EMPTY vector in that case rather than failing, so GMRES was handed a zero length
    // right hand side and every solver number for the SAM cases was meaningless.
    d.r_target = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(target_idx) + ".txt").c_str());
    d.r_source = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(source_idx) + ".txt").c_str());

    const auto require = [](const std::vector<double>& v, const CSRMatrix<double>& A, const char* what) {
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

/// Builds P_target, timing the setup, so the map can reuse it.
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

/// Reference point: solve the target system with its own preconditioner, no map involved.
template <typename Precond>
void reference_solve(const pair_data& d, const Precond& P_target, const GMRES<double>::params& prm) {
    std::vector<double> x(d.target.m_cols, 0.0);
    GMRES<double> solver(d.target.m_cols, prm);
    Timer t; t.start();
    auto [iters, error] = solver.solve(d.target, P_target, d.r_target, x);
    std::cout << "Target system with its own preconditioner: " << iters << " iters, "
              << std::scientific << std::setprecision(3) << error << " rel error, "
              << std::fixed << std::setprecision(1) << t.elapsed() / 1000.0 << " ms\n\n";
}

void run_top_opt(const config_t& cfg, const pair_data& d, patternMatrix pm,
                 std::map<std::string, BenchmarkResults>& results)
{
    ilutp<double>::params ilu_prm;
    ilu_prm.fill_factor = 250;
    ilu_prm.droptol = 1e-3;

    GMRES<double>::params prm;
    prm.pside = precondSide::right;
    prm.M = 2000;
    prm.maxIter = 2000;

    const auto P_target = build_preconditioner<ilutp<double>>(d, ilu_prm);
    reference_solve(d, *P_target, prm);
    const problem<ilutp<double>> p{d.target, d.source, *P_target, prm, d.r_source, pm, cfg.iters};

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

void run_weather(const config_t& cfg, const pair_data& d, patternMatrix pm,
                 std::map<std::string, BenchmarkResults>& results)
{
    iluk<double>::params ilu_prm;
    ilu_prm.k = 5;

    GMRES<double>::params prm;
    prm.pside = precondSide::right;
    prm.M = 50;
    prm.maxIter = 10000;

    const auto P_target = build_preconditioner<iluk<double>>(d, ilu_prm);
    reference_solve(d, *P_target, prm);
    const problem<iluk<double>> p{d.target, d.source, *P_target, prm, d.r_source, pm, cfg.iters};

    run_benchmark<SimplePattern>(p, results, "S(A_k) unsparsified");
    run_benchmark<GlobalThresholdPattern>(p, results, "Global threshold 0.01", 0.01);
    run_benchmark<GlobalThresholdPattern>(p, results, "Global threshold 0.001", 0.001);
    run_benchmark<GlobalThresholdPattern>(p, results, "Global threshold 0.0001", 0.0001);
    run_benchmark<ColumnThresholdPattern>(p, results, "Column threshold 0.7", 0.7);
    run_benchmark<ColumnThresholdPattern>(p, results, "Column threshold 0.8", 0.8);
    run_benchmark<ColumnThresholdPattern>(p, results, "Column threshold 0.9", 0.9);
    run_benchmark<FixedNNZPattern>(p, results, "Fixed nnz 3", 3);
    run_benchmark<FixedNNZPattern>(p, results, "Fixed nnz 5", 5);
    run_benchmark<FixedNNZPattern>(p, results, "Fixed nnz 7", 7);
}

} // namespace

int main(int argc, char** argv) {
    config_t cfg;
    parseargs(argc, argv, cfg);
    cfg.print();

    if (cfg.dataset != "top_opt" && cfg.dataset != "weather") {
        std::cerr << "error: unknown dataset '" << cfg.dataset << "'. Valid: top_opt | weather\n";
        return 1;
    }

    num_threads = cfg.threads;
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
    bench::report_pinning(cfg.threads);

    const pattern_variant pv = parse_pattern_variant(cfg.variant);
    const patternMatrix pm = pv.union_candidate ? patternMatrix::union_max : patternMatrix::source;

    const pair_data d = load(cfg);

    std::map<std::string, BenchmarkResults> results;
    if (cfg.dataset == "top_opt") run_top_opt(cfg, d, pm, results);
    else                          run_weather(cfg, d, pm, results);

    report(results, cfg, d.target.m_rows, pv.name);
    return 0;
}
