#include <cmath>
#include <vector>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/precond/ilut.hpp"
#include "testlib.hpp"

namespace {

// Build an upwinded convection-diffusion matrix on a k x k grid.
csr::CSRMatrix<double> convection_diffusion(size_t k, double c) {
	std::vector<size_t> ptr{0}, col;
	std::vector<double> val;
	for (size_t i = 0; i < k; ++i) {
		for (size_t j = 0; j < k; ++j) {
			const size_t r = i * k + j;
			if (i > 0) {
				col.push_back(r - k);
				val.push_back(-1.0);
			}
			if (j > 0) {
				col.push_back(r - 1);
				val.push_back(-1.0 - c);
			}
			col.push_back(r);
			val.push_back(4.0 + c);
			if (j + 1 < k) { 
				col.push_back(r + 1);
				val.push_back(-1.0);
			}
			if (i + 1 < k) { 
				col.push_back(r + k); 
				val.push_back(-1.0); 
			}
			ptr.push_back(col.size());
		}
	}
	const size_t n = k * k;
	return csr::CSRMatrix<double>(n, n, std::move(val), std::move(ptr), std::move(col));
}

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
	double num = 0, den = 0;
	for (size_t i = 0; i < a.size(); ++i) {
		num = std::max(num, std::abs(a[i] - b[i]));
		den = std::max(den, std::abs(b[i]));
	}
	return num / den;
}

// Compare serial or level-scheduled solves and verify aliased application.
void check_ilut_against_eigen(int threads) {
	const int saved = num_threads;
	num_threads = threads;
	tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);

	const auto A = convection_diffusion(40, 2.0);
	const size_t n = A.m_rows;

	ilut<double>::params prm;
	prm.droptol = 1e-3;
	prm.fill_factor = 5;
	ilut<double> ours(A, prm);
	prm.eigen_solve = true;
	ilut<double> eigen(A, prm);

	std::vector<double> b(n);
	for (size_t i = 0; i < n; ++i)
		b[i] = std::sin(0.7 * static_cast<double>(i)) + 0.1;

	std::vector<double> x_ours(n), x_eigen(n);
	ours.apply(b, x_ours);
	eigen.apply(b, x_eigen);
	assertTrueMsg(max_rel_diff(x_ours, x_eigen) < 1e-12,
								"ILUT apply differs from Eigen's solve at " + std::to_string(threads) +
								" threads: " + std::to_string(max_rel_diff(x_ours, x_eigen)));

	// apply_pre and apply_post use this in-place form.
	std::vector<double> y = b;
	ours.apply(y, y);
	assertTrueMsg(max_rel_diff(y, x_eigen) < 1e-12,
								"Aliased ILUT apply differs from Eigen's solve at " + std::to_string(threads) + " threads");

	num_threads = saved;
}

}

void test_ilut_matches_eigen_serial() {
	std::cout << "ILUT apply vs Eigen, serial triangular solves..." << std::flush;
	check_ilut_against_eigen(1);
	std::cout << "OK" << std::endl;
}

void test_ilut_matches_eigen_parallel() {
	std::cout << "ILUT apply vs Eigen, level-scheduled triangular solves..." << std::flush;
	check_ilut_against_eigen(8);
	std::cout << "OK" << std::endl;
}