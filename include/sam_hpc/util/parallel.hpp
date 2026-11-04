#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

/// Thread count used by algorithm-specific scheduling (default 16).
inline int num_threads = 16;

/// Accumulator selected by sparse matrix products and pattern extension.
enum class spgemm_accumulator { dense, hash };
inline spgemm_accumulator spgemm_acc = spgemm_accumulator::dense;

inline const char* to_string(spgemm_accumulator a) {
  return a == spgemm_accumulator::dense ? "dense" : "hash";
}

/// Minimum TBB chunk sizes for vector loops, row-wise matrix loops, and reductions.
inline size_t vector_grain = 4096;
inline size_t row_grain = 256;
inline size_t reduce_grain = 1024;

/// One logical CPU per physical core in the calling thread's affinity mask, ordered socket
/// by socket and then by core.
inline std::vector<int> physical_cores() {
  std::vector<int> cores;
#ifdef __linux__
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return cores;

  struct cpu { int package, core, id; };
  std::vector<cpu> cpus;
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (!CPU_ISSET(c, &mask))
      continue;
    const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
    int package = -1, core = -1;
    std::ifstream(dir + "physical_package_id") >> package;
    std::ifstream(dir + "core_id") >> core;
    if (package < 0 || core < 0) { 
      package = 0; core = c; 
    }
    cpus.push_back({package, core, c});
  }
  std::sort(cpus.begin(), cpus.end(), [](const cpu& a, const cpu& b) {
    if (a.package != b.package) 
      return a.package < b.package;
    if (a.core != b.core) 
      return a.core < b.core;
    return a.id < b.id;
  });
  for (size_t i = 0; i < cpus.size(); ++i) {
    if (i == 0 || cpus[i].package != cpus[i - 1].package || cpus[i].core != cpus[i - 1].core) {
      cores.push_back(cpus[i].id);
    }
  }
#endif
  return cores;
}

/// Binds TBB workers and dedicated worker threads to distinct physical cores.
/// Construct one before parallel work and keep it alive for the duration of the run.
/// Cores are assigned in the order returned by physical_cores().
class ThreadPinner : public tbb::task_scheduler_observer {
public:
  ThreadPinner() : m_cores(physical_cores()), m_generation(++s_generations) {
    // Initialize the arena before enabling the observer
    (void)tbb::this_task_arena::max_concurrency();
    s_active = this;
    observe(true);
    claim();
  }
  ~ThreadPinner() override {
    observe(false);
    s_active = nullptr;
  }
  
  // Copying is disabled
  ThreadPinner(const ThreadPinner&) = delete;
  ThreadPinner& operator=(const ThreadPinner&) = delete;

  void on_scheduler_entry(bool) override { claim(); }

  int cores() const { return static_cast<int>(m_cores.size()); }

  /// Returns the number of threads assigned a physical core so far.
  int pinned() const { return std::min(m_next.load(), cores()); }

  void report(int threads) const {
    std::printf("# thread pinning: one thread per physical core, %d cores available", cores());
    if (threads > cores()) {
      std::printf(" - WARNING: %d threads requested, %d will share cores or float",
                  threads, threads - cores());
    }
    std::printf("\n");
  }

  /// Binds a thread with a fixed id to core `slot` (launchThreadsWithID).
  static void bind_slot(int slot) {
    if (s_active && slot < s_active->cores()) {
      bind(s_active->m_cores[slot]);
      owner() = s_active->m_generation;
    }
  }

private:
  /// Claim a core for a thread
  void claim() {
    // Claim only one slot for the current thread pinner on each re-entry
    if (owner() == m_generation)
      return;
    
    // Mark as claimed and take the next available slot
    owner() = m_generation;
    const int slot = m_next.fetch_add(1);
    
    // Bind the thread to the core if available
    if (slot < cores())
      bind(m_cores[slot]);
  }

  // Bind a thread to a physical core
  static void bind(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
  }

  // Per thread state
  static unsigned& owner() {
    thread_local unsigned generation = 0;
    return generation;
  }

  static inline ThreadPinner* s_active = nullptr;
  static inline std::atomic<unsigned> s_generations{0};

  std::vector<int> m_cores;
  std::atomic<int> m_next{0};
  unsigned m_generation;
};

/// Runs `f(thread_id)` on num_threads dedicated threads and joins them.
///
/// This is used when the caller requires all participants to be active simultaneously,
/// which TBB parallel_for does not guarantee.
///
/// Each thread binds to the corresponding physical core when a ThreadPinner is active.
template <typename Func>
void launchThreadsWithID(Func&& f) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([f, t]() mutable {
            ThreadPinner::bind_slot(t);
            f(t);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}
