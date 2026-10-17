#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <ostream>
#include <vector>

/// Index types and allocator for CSR matrix arrays.
namespace csr {

/// Column indices are 32-bit.
using index_t = std::uint32_t;

/// Offsets into the value/index arrays are 64 bit.
using offset_t = std::size_t;

/// Largest column index representable in index_t.
inline constexpr offset_t max_index = std::numeric_limits<index_t>::max();

/// Custom allocator for CSR matrix vectors that are uninitialied on resize(n).
template <typename T>
struct default_init_allocator {
  using value_type = T;

  default_init_allocator() noexcept = default;

  template <typename U>
  constexpr default_init_allocator(const default_init_allocator<U>&) noexcept {}

  // Memory allocated with new and deallocated with delete
  T* allocate(std::size_t n) {
    // Return bad_alloc if the memory overflows the limit.
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(p); }

  /// If there are no arguments and U is trivially default-constructible, 
  /// then allocation is skipped as no-op.
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

/// The vector type used for CSR matrix arrays with the custom allocator
template <typename T>
using vec = std::vector<T, default_init_allocator<T>>;

/// Copy any source with a size() and range (begin(),end()) into dst
template <typename Dst, typename Src>
inline void assign_converted(Dst& dst, const Src& src) {
  // Resize the dest vector to the source vector
  dst.resize(src.size());
  std::copy(src.begin(), src.end(), dst.begin());
}

/// Copy any source with a size() and iterator range (begin(),end()) into dst
template <typename Dst, typename It>
inline void assign_converted(Dst& dst, It first, It last) {
  dst.resize(static_cast<std::size_t>(std::distance(first, last)));
  std::copy(first, last, dst.begin());
}

/// True for csr::vec<T>, false for every other type.
template <typename V>
inline constexpr bool is_csr_vec = false;
template <typename T>
inline constexpr bool is_csr_vec<std::vector<T, default_init_allocator<T>>> = true;

/// Element-wise equality between two vectors of which at least one is a csr::vec.
/// Both are same types then std::vector operator== overload is called instead.
template <typename T, typename A1, typename U, typename A2>
  requires (is_csr_vec<std::vector<T, A1>> || is_csr_vec<std::vector<U, A2>>)
    && (!std::is_same_v<A1, A2> || !std::is_same_v<T, U>)
bool operator==(const std::vector<T, A1>& a, const std::vector<U, A2>& b) {
  if (a.size() != b.size())
    return false;
  // Compare every elemnt in terms of common type
  using wide = std::common_type_t<T, U>;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (static_cast<wide>(a[i]) != static_cast<wide>(b[i]))
      return false;
  }
  return true;
}

/// Prints a csr::vec as std::vector
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
