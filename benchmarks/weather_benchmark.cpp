// End-to-end weather benchmark: build a SAM from A_source to A_target, then solve
// A_source x = b_source with GMRES preconditioned by P_target * N.
//
// Same methodology as topology_benchmark (C8): budgeted repetitions reported as min /
// median with the observed spread, never elapsed()/iters. See measure.hpp.
//
// Usage: weather_bench -x <matrix dir> -y <rhs dir> [-k max reps] [-t threads]
//                      [-a target idx] [-b source idx] [-u variant]

#include "config.hpp"
#include "sam.hpp"
#include "read_mat.hpp"
#include "measure.hpp"
#include "amg.hpp"
#include "ruge_stuben.hpp"
#include "damped_jacobi.hpp"
#include "smoothed_aggregation.hpp"
#include "gauss_seidel.hpp"
#include "gmres.hpp"
#include "timer.hpp"
#include "ilu0.hpp"
#include "iluk.hpp"
#include "ilutp.hpp"

#include <iomanip>
#include <map>
#include <vector>

namespace {

// A single GMRES solve here runs for tens of seconds, so measure.hpp's default 4 s budget
// would give exactly one repetition and therefore no spread at all. Force at least three.
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
};

struct problem {
    const CSRMatrix<double>& target;
    const CSRMatrix<double>& source;
    const iluk<double>& P_0;
    const GMRES<double>::params& prm;
    const std::vector<double>& rhs;
    patternMatrix pm;
    int max_reps;
};

template <typename PatternType, typename... PatternArgs>
void run_benchmark(const problem& p, std::map<std::string, BenchmarkResults>& results,
                   const std::string& pattern_name, PatternArgs&&... args)
{
    std::cout << "Running benchmark: [" << pattern_name << "]..." << std::flush;

    using Map = SparseApproximateMap<double, PatternType>;
    BenchmarkResults res;

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
              << "# n " << n << ", threads " << cfg.threads
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

} // namespace

int main(int argc, char** argv) {
    config_t cfg;
    parseargs(argc, argv, cfg);
    cfg.print();

    num_threads = cfg.threads;
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
    bench::report_pinning(cfg.threads);

    const int target_idx = cfg.target_idx > 0 ? cfg.target_idx : kDefaultTarget;
    const int source_idx = cfg.source_idx > 0 ? cfg.source_idx : kDefaultSource;

    const pattern_variant pv = parse_pattern_variant(cfg.variant);
    const patternMatrix pm = pv.union_candidate ? patternMatrix::union_max : patternMatrix::source;

    CSRMatrix<double> targetMatrix;
    read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(target_idx) + ".txt").c_str(), targetMatrix);
    CSRMatrix<double> sourceMatrix;
    read_mat<double>((cfg.matrix_dir + "matrix_" + std::to_string(source_idx) + ".txt").c_str(), sourceMatrix);

    // matrix_i pairs with rhs_i. This used to read rhs_<source+1> for the source system
    // while reading rhs_<target> for the target, so the SAM cases were solving A_50 x = b_51.
    std::vector<double> r_target = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(target_idx) + ".txt").c_str());
    std::vector<double> r_source = read_vec<double>((cfg.rhs_dir + "rhs_" + std::to_string(source_idx) + ".txt").c_str());

    const auto require = [](const std::vector<double>& v, const CSRMatrix<double>& A, const char* what) {
        if (v.size() != A.m_rows) {
            std::cerr << "error: " << what << " has " << v.size() << " entries but the matrix has "
                      << A.m_rows << " rows\n";
            std::exit(1);
        }
    };
    require(r_target, targetMatrix, "target rhs");
    require(r_source, sourceMatrix, "source rhs");

    std::cout << "target matrix_" << target_idx << ", source matrix_" << source_idx
              << ", n = " << targetMatrix.m_rows << ", nnz " << targetMatrix.m_nnz << "\n";

    iluk<double>::params ilu_prm;
    ilu_prm.k = 5;
    Timer setup;
    setup.start();
    iluk<double> P_target(targetMatrix, ilu_prm);
    std::cout << "Preconditioner built in " << std::fixed << std::setprecision(1)
              << setup.elapsed() / 1000.0 << " ms\n";

    GMRES<double>::params prm;
    prm.pside = precondSide::right;
    prm.M = 50;
    prm.maxIter = 10000;

    {
        std::vector<double> x(targetMatrix.m_cols, 0.0);
        GMRES<double> solver(targetMatrix.m_cols, prm);
        Timer t; t.start();
        auto [iters, error] = solver.solve(targetMatrix, P_target, r_target, x);
        std::cout << "Target system with its own preconditioner: " << iters << " iters, "
                  << std::scientific << std::setprecision(3) << error << " rel error, "
                  << std::fixed << std::setprecision(1) << t.elapsed() / 1000.0 << " ms\n\n";
    }

    const problem p{targetMatrix, sourceMatrix, P_target, prm, r_source, pm, cfg.iters};

    std::map<std::string, BenchmarkResults> results;
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

    report(results, cfg, targetMatrix.m_rows, pv.name);
    return 0;
}
