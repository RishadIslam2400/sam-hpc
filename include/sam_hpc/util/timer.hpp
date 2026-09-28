#pragma once

#include <chrono>
#include <cstdint>

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point _start{};

public:
    Timer() = default;

    void start() {
        _start = clock::now();
    }

    // return the split time in microseconds
    uint64_t split() {
        auto now = clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
        _start = now;
        return diff.count();
    }

    // return the elapsed time in microseconds
    uint64_t elapsed() {
        auto now = clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
        return diff.count();
    }
};