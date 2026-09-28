#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <ostream>
#include <vector>

/// Index types and the allocator every large CSR array uses.
namespace csr {

/// Column indices are 32-bit
using index_t = std::uint32_t;

/// Offsets into the value/index arrays (row pointers, nnz) are 64 bit.
using offset_t = std::size_t;
inline constexpr offset_t max_index = std::numeric_limits<index_t>::max();

/// Allocator that leaves trivially-constructible elements uninitialized on resize(n).
template <typename T>
struct default_init_allocator {
  using value_type = T;

  default_init_allocator() noexcept = default;

  template <typename U>
  constexpr default_init_allocator(const default_init_allocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(p); }

  /// A value-initialising construct() becomes a no-op for trivial types and no arguments.
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible_v<U>) {
      (void)p;
    } else {
      ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
  }

  template <typename U>
  bool operator==(const default_init_allocator<U>&) const noexcept { return true; }
  template <typename U>
  bool operator!=(const default_init_allocator<U>&) const noexcept { return false; }
};

/// The vector type used for every large CSR array.
template <typename T>
using vec = std::vector<T, default_init_allocator<T>>;

/// Copy any contiguous integer range into a csr::vec.
template <typename Dst, typename Src>
inline void assign_converted(Dst& dst, const Src& src) {
  dst.resize(src.size());
  std::copy(src.begin(), src.end(), dst.begin());
}

template <typename Dst, typename It>
inline void assign_converted(Dst& dst, It first, It last) {
  dst.resize(static_cast<std::size_t>(std::distance(first, last)));
  std::copy(first, last, dst.begin());
}

/// Comparison between two vectors, where at least one is csr::vec.
template <typename V>
inline constexpr bool is_csr_vec = false;
template <typename T>
inline constexpr bool is_csr_vec<std::vector<T, default_init_allocator<T>>> = true;

// Disables the overload when both vectors have same element type or same allocators.
template <typename T, typename A1, typename U, typename A2>
  requires (is_csr_vec<std::vector<T, A1>> || is_csr_vec<std::vector<U, A2>>)
    && (!std::is_same_v<A1, A2> || !std::is_same_v<T, U>)
bool operator==(const std::vector<T, A1>& a, const std::vector<U, A2>& b) {
  if (a.size() != b.size())
    return false;
  // Compare every elemnt in terms of common type
  for (std::size_t i = 0; i < a.size(); ++i) {
    using wide = std::common_type_t<T, U>;
    if (static_cast<wide>(a[i]) != static_cast<wide>(b[i]))
      return false;
  }
  return true;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T, default_init_allocator<T>>& v) {
  os << "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i)
      os << ", ";
    os << v[i];
  }
  return os << "]";
}

} // namespace csr
