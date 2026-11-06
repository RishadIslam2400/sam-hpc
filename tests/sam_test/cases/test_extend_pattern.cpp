#include <random>
#include <vector>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/sam/extendPattern.hpp"
#include "testlib.hpp"

namespace {

using bool_matrix = std::vector<std::vector<char>>;

/// Convert a dense boolean pattern to a CSR matrix with unit nonzero values.
csr::CSRMatrix<int> to_pattern(const bool_matrix& D) {
	const size_t n = D.size();
	std::vector<size_t> ptr{0}, col;
	std::vector<int> val;
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			if (D[i][j])
				{ col.push_back(j); val.push_back(1); }
		}
		ptr.push_back(col.size());
	}
	return csr::CSRMatrix<int>(n, n, std::move(val), std::move(ptr), std::move(col));
}

/// Compute a boolean matrix power using the original pattern at every step.
bool_matrix dense_power(const bool_matrix& D, int level) {
	const size_t n = D.size();
	bool_matrix P = D;
	for (int l = 2; l <= level; ++l) {
		bool_matrix Q(n, std::vector<char>(n, 0));
		for (size_t i = 0; i < n; ++i)
			for (size_t k = 0; k < n; ++k)
				if (P[i][k])
					for (size_t j = 0; j < n; ++j)
							if (D[k][j])
								Q[i][j] = 1;
		P = Q;
	}
	return P;
}

/// Generate an n x n random pattern with an optional full diagonal.
bool_matrix random_pattern(size_t n, double density, bool diagonal, unsigned seed) {
	std::mt19937 g(seed);
	std::bernoulli_distribution b(density);
	bool_matrix D(n, std::vector<char>(n, 0));
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			D[i][j] = b(g);
	for (size_t i = 0; i < n; ++i)
		if (diagonal)
			D[i][i] = 1;
	if (!diagonal)
		D[0][0] = 0;
	return D;
}

/// Compare sparse extension with the dense reference across levels, accumulators, and thread counts.
void check_power(const bool_matrix& D, const std::string& label) {
	const csr::CSRMatrix<int> S = to_pattern(D);
	const int saved_threads = num_threads;
	const spgemm_accumulator saved_acc = spgemm_acc;

	for (int level = 1; level <= 4; ++level) {
		const csr::CSRMatrix<int> expected = to_pattern(dense_power(D, level));
		for (spgemm_accumulator acc : {spgemm_accumulator::dense, spgemm_accumulator::hash}) {
			for (int threads : {1, 8}) {
				tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
				num_threads = threads;
				spgemm_acc = acc;
				csr::CSRMatrix<int> P = S;
				sam::extend_pattern(P, level);
				assertTrueMsg(P == expected, label + ": S^" + std::to_string(level) + " wrong with " +
											to_string(acc) + " accumulator at " + std::to_string(threads) + " threads");
			}
		}
	}
	num_threads = saved_threads;
	spgemm_acc = saved_acc;
}

} // namespace

/// Verify frontier expansion when every row contains a diagonal entry.
void testExtendPatternFrontier() {
	std::cout << "pattern extension, full diagonal (frontier path)..." << std::flush;
	for (unsigned seed = 1; seed <= 20; ++seed) {
		const bool_matrix D = random_pattern(40, 0.05, true, seed);
		assertTrueMsg(sam::has_full_diagonal(to_pattern(D)), "fixture should have a full diagonal");
		check_power(D, "frontier seed " + std::to_string(seed));
	}
	std::cout << "OK" << std::endl;
}

/// Verify exact-length walks when the pattern is missing a diagonal entry.
void testExtendPatternWalks() {
	std::cout << "pattern extension, missing diagonal (exact-length walks)..." << std::flush;
	for (unsigned seed = 1; seed <= 20; ++seed) {
		const bool_matrix D = random_pattern(40, 0.05, false, seed);
		assertTrueMsg(!sam::has_full_diagonal(to_pattern(D)), "fixture should lack a diagonal entry");
		check_power(D, "walks seed " + std::to_string(seed));
	}
	std::cout << "OK" << std::endl;
}

/// Verify the nonzero count and row boundaries for powers of a large tridiagonal pattern.
void testExtendPatternTridiagonal() {
	std::cout << "pattern extension, tridiagonal closed form..." << std::flush;
	const size_t n = 20000;
	std::vector<size_t> ptr{0}, col;
	std::vector<int> val;
	for (size_t i = 0; i < n; ++i) {
		if (i > 0)
			col.push_back(i - 1);
		col.push_back(i);
		if (i + 1 < n)
			col.push_back(i + 1);
		val.resize(col.size(), 1);
		ptr.push_back(col.size());
	}
	const csr::CSRMatrix<int> S(n, n, std::move(val), std::move(ptr), std::move(col));
	const spgemm_accumulator saved_acc = spgemm_acc;
	for (spgemm_accumulator acc : {spgemm_accumulator::dense, spgemm_accumulator::hash}) {
		spgemm_acc = acc;
		for (size_t l = 2; l <= 5; ++l) {
			csr::CSRMatrix<int> P = S;
			sam::extend_pattern(P, static_cast<int>(l));
			assertEquals<size_t>((2 * l + 1) * n - l * (l + 1), P.m_nnz,
														"tridiagonal nnz(S^" + std::to_string(l) + ") with " + to_string(acc));
			for (size_t i = 0; i < n; ++i) {
				const size_t lo = i >= l ? i - l : 0;
				assertEquals<size_t>(lo, P.m_col_indices[P.m_row_pointers[i]], "tridiagonal row start");
			}
		}
	}
	spgemm_acc = saved_acc;
	std::cout << "OK" << std::endl;
}
