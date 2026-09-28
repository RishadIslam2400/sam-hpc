#pragma once

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// N-dimensional dense matrix
template <class T, int N>
class multiArray {
  static_assert(N > 0, "Wrong number of dimensions");

 public:
  template <class... I>
  multiArray(I... n) {
    static_assert(sizeof...(I) == N, "Wrong number of dimensions.");
    buf.resize(init(n...));
  }

  size_t size() const { return buf.size(); }
  int stride(int i) const { return strides[i]; }

  template <class... I>
  T operator()(I... i) const {
    static_assert(sizeof...(I) == N, "Wrong number of indices.");
    return buf[index(i...)];
  }

  template <class... I>
  T& operator()(I... i) {
    static_assert(sizeof...(I) == N, "Wrong number of indices.");
    return buf[index(i...)];
  }

  const T* data() const { return buf.data(); }

 private:
  std::array<int, N> strides;
  std::vector<T> buf;

  // recursive index function
  template <class... I>
  int index(int i, I... tail) const {
    return strides[N - sizeof...(I) - 1] * i + index(tail...);
  }

  // base case with one parameter
  int index(int i) const { return strides[N - 1] * i; }

  // recursive init function
  template <class... I>
  int init(int i, I... tail) {
    int size = init(tail...);
    strides[N - sizeof...(I) - 1] = size;
    return i * size;
  }

  // base case
  int init(int i) {
    strides[N - 1] = 1;
    return i;
  }
};

template <class T>
class circularBuffer {
 public:
  circularBuffer(size_t n) : start(0) { buf.reserve(n); }

  size_t size() const { return buf.size(); }

  void push_back(const T& v) {
    if (buf.size() < buf.capacity()) {
      buf.push_back(v);
    } else {
      buf[start] = v;
      start = (start + 1) % buf.capacity();
    }
  }

  const T& operator[](size_t i) const {
    return buf[(start + i) % buf.capacity()];
  }

  T& operator[](size_t i) { return buf[(start + i) % buf.capacity()]; }

  void clear() {
    buf.clear();
    start = 0;
  }

 private:
  size_t start;
  std::vector<T> buf;
};

template <class T>
T eps(size_t n) {
  return 2 * std::numeric_limits<T>::epsilon() * n;
}

inline std::string human_readable_memory(size_t bytes) {
  static const char* suffix[] = {"B", "KB", "MB", "GB", "TB"};

  int i = 0;
  double m = static_cast<double>(bytes);
  for (; i < 4 && m >= 1024.0; ++i, m /= 1024.0);

  std::ostringstream s;
  s << std::fixed << std::setprecision(2) << m << " " << suffix[i];
  return s.str();
}