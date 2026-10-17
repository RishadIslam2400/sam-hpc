#pragma once

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

/// Fixed-size dense N-dimensional array stored in contiguous row-major order.
///
/// The constructor accepts one extent per dimension. The number of supplied
/// indices is checked at compile time, and elements are value-initialized.
template <class T, int N>
class multiArray {
  static_assert(N > 0, "Wrong number of dimensions");

 public:
  template <class... I>
  multiArray(I... n) {
    static_assert(sizeof...(I) == N, "Wrong number of dimensions.");
    buf.resize(init(n...));
  }

  /// Returns the total number of elements.
  size_t size() const { return buf.size(); }

  /// Returns the row-major stride for dimension `i`.
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

  /// Returns a pointer to the underlying contiguous storage.
  const T* data() const { return buf.data(); }

 private:
  std::array<int, N> strides;
  std::vector<T> buf;

  // Converts a multidimensional index to a row-major offset.
  template <class... I>
  int index(int i, I... tail) const {
    return strides[N - sizeof...(I) - 1] * i + index(tail...);
  }

  // Final index contribution.
  int index(int i) const { return strides[N - 1] * i; }

  // Initializes strides from the last dimension and returns the total size.
  template <class... I>
  int init(int i, I... tail) {
    int size = init(tail...);
    strides[N - sizeof...(I) - 1] = size;
    return i * size;
  }

  // The last dimension has unit stride.
  int init(int i) {
    strides[N - 1] = 1;
    return i;
  }
};

/// Fixed-capacity FIFO buffer that overwrites the oldest value when full.
///
/// Elements are indexed from oldest to newest. The capacity must be nonzero,
/// and indexing requires a valid position among the stored elements.
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

/// Returns a roundoff tolerance scaled by `n` machine epsilons for type `T`.
template <class T>
T eps(size_t n) {
  return 2 * std::numeric_limits<T>::epsilon() * n;
}

/// Formats a byte count using binary units and two decimal places.
///
/// Unit suffixes range from bytes through terabytes.
inline std::string human_readable_memory(size_t bytes) {
  static const char* suffix[] = {"B", "KB", "MB", "GB", "TB"};

  int i = 0;
  double m = static_cast<double>(bytes);
  for (; i < 4 && m >= 1024.0; ++i, m /= 1024.0);

  std::ostringstream s;
  s << std::fixed << std::setprecision(2) << m << " " << suffix[i];
  return s.str();
}