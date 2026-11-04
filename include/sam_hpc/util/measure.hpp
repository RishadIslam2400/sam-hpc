#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace bench {

/// Monotonic clock used for benchmark samples.
using clock_type = std::chrono::steady_clock;

/// Milliseconds (fractional) from t to now.
inline double ms_since(clock_type::time_point t) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

/// Summary of one measurement.
struct sample {
  double min_ms = 0;
  double median_ms = 0;
  double max_ms = 0;
  int reps = 0;

  /// Relative min-to-max spread in percent.
  double spread_pct() const { return min_ms > 0 ? 100.0 * (max_ms - min_ms) / min_ms : 0.0; }
};

/// Summarizes timings into a sample.
inline sample summarize(std::vector<double> t) {
  if (t.empty())
    return {};
  std::sort(t.begin(), t.end());
  return {t.front(), t[t.size() / 2], t.back(), static_cast<int>(t.size())};
}

/// Runs one untimed warm-up followed by timed repetitions.
inline sample measure(int max_reps, const std::function<void()>& f,
                      double budget_ms = 5000.0, int min_reps = 1)
{
  // Perform the warm up run
  const auto warm_begin = clock_type::now();
  f();
  const double warm_ms = ms_since(warm_begin);

  int reps = max_reps;
  if (warm_ms > 0) {
    // Find the ideal number of reps based on warm up time
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

/// Returns the physical-core thread sweep based on availability.
inline std::vector<int> thread_sweep(int cores) {
  std::vector<int> s{1};
  for (int t : {12, 24, 48, 72, 96}) {
    if (t < cores)
      s.push_back(t);
  }
  if (cores > 1)
    s.push_back(cores);
  return s;
}

/// Reports the thread count and the configured NUMA memory policy.
///
/// `SAM_PINNED` is set by the benchmark script when memory interleaving is enabled.
inline void report_pinning(int threads) {
  std::printf("# threads %d", threads);
#ifdef __linux__
  if (const char* pin = std::getenv("SAM_PINNED"); pin && *pin) {
    std::printf(", memory policy: %s", pin);
  } else {
    std::printf(", NO memory policy - run under `numactl --interleave=all` "
                "(scripts/run_benchmark.sh) for a scaling curve that means anything");
  }
#endif
  std::printf("\n");
}

} // namespace bench
