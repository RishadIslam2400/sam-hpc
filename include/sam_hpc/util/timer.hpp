#pragma once

#include <chrono>
#include <cstdint>

/// Wall-clock timer that reports durations in microseconds.
///
/// Call start() before elapsed() or split(). The timer is not thread-safe, so each
/// thread should use its own instance. Because high_resolution_clock is not guaranteed
/// to be steady, system clock adjustments can affect a measurement.
class Timer {
private:
  using clock = std::chrono::high_resolution_clock;
  clock::time_point _start{};

public:
  Timer() = default;

  void start() {
    _start = clock::now();
  }

  /// Returns the time since start() or the previous split(), then restarts the timer.
  uint64_t split() {
    auto now = clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
    _start = now;
    return diff.count();
  }

  /// Returns the time since start() or the last split() without resetting the timer.
  uint64_t elapsed() {
    auto now = clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
    return diff.count();
  }
};