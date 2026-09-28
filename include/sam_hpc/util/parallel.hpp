#pragma once

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>

#include <thread>
#include <vector>

// Thread count the algorithms are tuned for. It picks the SpGEMM variant in extend_pattern,
// sizes the level-scheduled tasks in ilu_solve and gauss_seidel, and gates the parallel
// paths in linearAlgebra. It does NOT limit TBB: each benchmark's main must also construct
// a tbb::global_control with the same value.
inline int num_threads = 16;

/// Runs f(thread_id) on num_threads dedicated std::threads and joins them.
///
/// Used by gauss_seidel, whose parallel sweep synchronizes on a std::barrier sized to
/// num_threads. That needs every participant running at once, which a TBB parallel_for
/// does not guarantee (it may run the tasks on fewer workers and deadlock the barrier).
template <typename Func>
void launchThreadsWithID(Func&& f) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(f, t);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}
