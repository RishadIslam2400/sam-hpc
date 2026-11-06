#pragma once

#include <stdexcept>
#include <string>
#include <functional>
#include <sstream>
#include <vector>
#include "sam_hpc/core/CSRMatrix.hpp"

// Format a vector for assertion failure messages.
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
	os << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		if (i > 0) {
			os << ", ";
		}
		os << vec[i];
	}
	os << "]";
	return os;
}

// Exception type used for failed assertions.
class FailureException : public std::runtime_error {
public:
	FailureException(const std::string &message) : std::runtime_error(message) {}
};

// Assert a condition with a descriptive failure message.
inline void assertTrueMsg(bool condition, const std::string &message) {
  if (!condition)
		throw FailureException(message);
}

// Compare non-floating-point values using their operator==.
template <typename T>
  requires(!std::floating_point<T>)
void assertEquals(const T &expected, const T &computed, const std::string &message = "") {
	if (!(expected == computed)) {
		std::ostringstream oss;
		if (message.empty()) {
			oss << "Objects not equal when they should be\n";
		} else {
			oss << message << "\n";
		}
		oss << expected << "\nexpected, but\n" << computed << " given";
		throw FailureException(oss.str());
	}
}


// Compare floating-point values using absolute tolerance.
template <typename T>
  requires std::floating_point<T>
void assertEquals(const T &expected, const T &computed, const std::string &message = "", T epsilon = T(1e-6)) {
	if (std::abs(expected - computed) > epsilon) {
		std::ostringstream oss;
		if (message.empty()) {
			oss << "Floating-point values differ more than epsilon\n";
		} else {
			oss << message << "\n";
		}
		oss << expected << "\nexpected, but\n" << computed << " given";
		throw FailureException(oss.str());
	}
}

// Compare floating-point vectors entry by entry using absolute tolerance.
template <typename T>
  requires std::floating_point<T>
void assertEquals(const std::vector<T> &expected, const std::vector<T> &computed,
                  const std::string &message = "", T epsilon = T(1e-6))
{
	bool equal = expected.size() == computed.size();
	for (size_t i = 0; equal && i < expected.size(); ++i) {
		equal = std::abs(expected[i] - computed[i]) <= epsilon;
	}

	if (!equal) {
		std::ostringstream oss;
		if (message.empty()) {
			oss << "Floating-point vectors differ more than epsilon\n";
		} else {
			oss << message << "\n";
		}
		oss << expected << "\nexpected, but\n" << computed << " given";
		throw FailureException(oss.str());
	}
}

// Compare a CSR matrix with a dense reference entry by entry.
template <typename T>
bool operator==(const csr::CSRMatrix<T> &sparse, const std::vector<std::vector<T>> &classical) {
	for (size_t i = 0; i < classical.size(); ++i) {
		for (size_t j = 0; j < classical[i].size(); ++j) {
			if (sparse.get(i, j) != classical[i][j])
				return false;
		}
	}

	return true;
}

// Compare compatible types through their operator== overload.
template <typename X, typename Y>
void assertEquals(const X &expected, const Y &computed, const std::string &message = "") {
	if (!(expected == computed)) {
		std::ostringstream oss;
		if (message.empty()) {
			oss << "Objects not equal when they should be" << std::endl;
		} else {
			oss << message << std::endl;
		}

		oss << expected << std::endl << "expected, but" << std::endl << computed << " given";

		throw FailureException(oss.str());
	}
}