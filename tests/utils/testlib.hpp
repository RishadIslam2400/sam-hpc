#pragma once

#include <stdexcept>
#include <string>
#include <functional>
#include <sstream>
#include "sam_hpc/core/CSRMatrix.hpp"

class FailureException : public std::runtime_error {
public:
    FailureException(const std::string &message) : std::runtime_error(message) {}
};

/* void assertException(const std::string &exceptionClass, const std::function<void()> &callback) {
    try {
        callback();
    } catch (const std::exception &e) {
        std::string actualClass(typeid(e).name());

        if (actualClass.find(exceptionClass) == std::string::npos) {
            throw FailureException("Exception class " + std::string(exceptionClass) + " expected, but " + std::string(actualClass) + " thrown");
        }

        return;
    }

    throw FailureException("Exception expected, but none thrown");
} */

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


// For floating-point types (uses epsilon-based comparison)
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

template <typename T>
bool operator==(const CSRMatrix<T> &sparse, const std::vector<std::vector<T>> &classical) {
    for (size_t i = 0; i < classical.size(); ++i) {
        for (size_t j = 0; j < classical[i].size(); ++j) {
            if (sparse.get(i, j) != classical[i][j])
                return false;
        }
    }

    return true;
}

template <typename X, typename Y>
void assertEquals(const X &expected, const Y &computed, const std::string &message = "") {
    if (!(expected == computed)) {
        std::ostringstream oss;
        if (message.empty()) {
            oss << "Objects not equal when they should be" << std::endl;
        } else {
            oss << message << std::endl;
        }

        oss << expected << std::endl
            << "expected, but" << std::endl
            << computed << " given";

        throw FailureException(oss.str());
    }
}