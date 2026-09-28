#pragma once

#include <thread>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>

// GLobal thread variable for parallelization


/* #ifdef SEQUENTIAL_ENABLED
constexpr bool SEQUENTIAL = true;
int num_threads = 1;
#else
constexpr bool SEQUENTIAL = false;
int num_threads = 16;
#endif */
// constexpr bool SEQUENTIAL = false;
inline int num_threads = 16;

template <typename Func>
void launchThreads(size_t row_num, Func&& f) {
    std::vector<std::thread> threads;
    size_t rows_per_thread = (row_num + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * rows_per_thread;
        size_t end = std::min(row_num, start + rows_per_thread);
        threads.emplace_back(std::forward<Func>(f), start, end);
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

template <typename Func>
void launchThreadsWithID(size_t row_num, Func&& f) {
    std::vector<std::thread> threads;
    size_t rows_per_thread = (row_num + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * rows_per_thread;
        size_t end = std::min(row_num, start + rows_per_thread);
        threads.emplace_back(std::forward<Func>(f), start, end, t);
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

template <typename Func>
void launchThreadsWithID(Func&& f) {
    std::vector<std::thread> threads;
    

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(std::forward<Func>(f), t);
    }

    for (auto &thread : threads) {
        thread.join();
    }
}