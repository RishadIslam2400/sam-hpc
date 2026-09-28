#pragma once

// Repetition discipline shared by every benchmark in this repository.
//
// Two rules, both learned the hard way on this code base:
//
//   1. Report the MINIMUM, with the median beside it. A single run's elapsed time is the
//      true cost plus non-negative noise, so the minimum is the least contaminated
//      estimate; the median tells you whether that minimum was a fluke. Never report
//      elapsed()/iters - that is a mean, and a mean is dragged around by exactly the
//      outliers you want to exclude.
//
//   2. Budget the repetitions rather than fixing them. Phases in this pipeline differ in
//      cost by three orders of magnitude; a fixed count either runs for hours on the
//      expensive ones or measures pure noise on the cheap ones.
//
// A warning the spread field exists for: minimum-over-repetitions protects against noise
// WITHIN a process, not against drift BETWEEN processes. Two runs of the same binary
// minutes apart have been seen to differ by 10% on this machine. Anything below that is
// not a result - compare variants inside one process, or report the spread and say so.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace bench {

using clock_type = std::chrono::steady_clock;

inline double ms_since(clock_type::time_point t) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

struct sample {
    double min_ms = 0;
    double median_ms = 0;
    double max_ms = 0;
    int reps = 0;

    /// (max - min) / min, in percent. Above ~10% treat differences of that size as noise.
    double spread_pct() const { return min_ms > 0 ? 100.0 * (max_ms - min_ms) / min_ms : 0.0; }
};

inline sample summarize(std::vector<double> t) {
    if (t.empty()) return {};
    std::sort(t.begin(), t.end());
    return {t.front(), t[t.size() / 2], t.back(), static_cast<int>(t.size())};
}

/// One untimed warm-up, then repetitions inside `budget_ms`, capped at `max_reps`
/// and never fewer than `min_reps`.
inline sample measure(int max_reps, const std::function<void()>& f,
                      double budget_ms = 4000.0, int min_reps = 1) {
    const auto warm_begin = clock_type::now();
    f();
    const double warm_ms = ms_since(warm_begin);

    int reps = max_reps;
    if (warm_ms > 0) {
        reps = std::clamp(static_cast<int>(budget_ms / warm_ms), min_reps, std::max(min_reps, max_reps));
    }

    std::vector<double> t;
    t.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        const auto begin = clock_type::now();
        f();
        t.push_back(ms_since(begin));
    }
    return summarize(std::move(t));
}

/// The thread counts a sweep should visit, capped so it never silently crosses a socket.
///
/// The target cluster is 4x Xeon Platinum 8160: 24 cores per socket, 96 total, four NUMA
/// nodes. Every large array here is created with resize(n, 0) and therefore first-touched
/// by the master thread, which places the whole working set on one node - so the moment a
/// sweep passes the socket boundary the numbers stop being a scaling curve and start being
/// a remote-memory measurement. Cap at one socket, or drop the single-node claim.
inline std::vector<int> thread_sweep(int hardware, int cap = 24) {
    const int top = std::min(hardware, cap);
    std::vector<int> s;
    for (int t = 1; t < top; t *= 2) s.push_back(t);
    s.push_back(top);
    return s;
}

/// Warn once if the process is free to migrate across NUMA nodes.
inline void report_pinning(int threads) {
    std::printf("# threads %d", threads);
#ifdef __linux__
    if (const char* pin = std::getenv("SAM_PINNED"); pin && *pin) {
        std::printf(", pinned: %s", pin);
    } else {
        std::printf(", NOT pinned - run under `numactl --cpunodebind=0 --membind=0` "
                    "for a scaling curve that means anything");
    }
#endif
    std::printf("\n");
}

} // namespace bench
