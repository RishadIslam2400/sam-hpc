#pragma once

#include <cmath>

/// Computes a Givens rotation that eliminates the second component.
/// The coefficients are normalized and computed using the larger-magnitude
/// input as the divisor to improve numerical stability.
template <typename T>
inline void generatePlaneRotation(T dx, T dy, T& cs, T& sn) {
  if (dy == T(0)) {
    cs = 1;
    sn = 0;
  } else if (std::abs(dy) > std::abs(dx)) {
    T tmp = dx / dy;
    sn = 1 / sqrt(T(1) + tmp * tmp);
    cs = tmp * sn;
  } else {
    T tmp = dy / dx;
    cs = 1 / sqrt(T(1) + tmp * tmp);
    sn = tmp * cs;
  }
}

/// Applies the Givens rotation in place:
/// `(dx, dy) <- (cs * dx + sn * dy, -sn * dx + cs * dy)`.
template <typename T>
void applyPlaneRotation(T& dx, T& dy, T cs, T sn) {
  T tmp = cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = tmp;
}