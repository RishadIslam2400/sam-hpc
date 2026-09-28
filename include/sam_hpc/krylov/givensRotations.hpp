#pragma once

#include <cmath>

/**
 * @brief Computes the components of a Givens plane rotation.
 *
 * This function calculates the cosine (cs) and sine (sn) parameters for a
 * Givens rotation matrix. The goal is to create a rotation that, when applied
 * to the vector [dx, dy], will zero out the second component (dy).
 *
 * The implementation includes checks to maintain numerical stability,
 * especially when one of the input values is much larger than the other, to
 * avoid overflow or division by a very small number.
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @param dx The first component of the vector to be rotated.
 * @param dy The second component of the vector to be rotated.
 * @param cs [out] A reference to store the computed cosine component.
 * @param sn [out] A reference to store the computed sine component.
 */
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

template <typename T>
void applyPlaneRotation(T& dx, T& dy, T cs, T sn) {
  T tmp = cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = tmp;
}