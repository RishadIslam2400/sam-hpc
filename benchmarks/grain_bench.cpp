#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <thread>
#include <cmath>
#include <algorithm>

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/util/config.hpp"
#include "sam_hpc/util/measure.hpp"
#include "sam_hpc/util/parallel.hpp"

namespace {

/// 5-point Laplacian on a k x k grid, rows sorted.
csr::CSRMatrix<double> laplacian_2d(size_t k) {
	const size_t n = k * k;
	std::vector<size_t> ptr{0};
	std::vector<size_t> col;
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
				val.push_back(-1.0); 
			}
			col.push_back(r); 
			val.push_back(4.0);
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
	return csr::CSRMatrix<double>(n, n, std::move(val), std::move(ptr), std::move(col));
}

template <typename Set, typename F>
void time_grains(const std::string& kernel, size_t n, int threads,
                 const std::vector<size_t>& grains, Set&& set_grain, F&& call)
{
	constexpr int rounds = 5;
	const size_t batch = std::max<size_t>(1, 1000000 / n);
	const double per_call_us = 1000.0 / static_cast<double>(batch);

	std::vector<std::vector<double>> mins(grains.size());
	for (int r = 0; r < rounds; ++r) {
		for (size_t g = 0; g < grains.size(); ++g) {
			set_grain(grains[g]);
			const bench::sample s = bench::measure(100, [&] {
				for (size_t b = 0; b < batch; ++b)
					call();
			}, 80.0, 3);
			mins[g].push_back(s.min_ms * per_call_us);
		}
	}
	for (size_t g = 0; g < grains.size(); ++g) {
		std::sort(mins[g].begin(), mins[g].end());
		std::cout << "CSV," << kernel << "," << n << "," << threads << "," << grains[g] << ","
							<< mins[g].front() << "," << mins[g][mins[g].size() / 2] << "\n";
	}
}

} // namespace

int main(int argc, char** argv) {
	config_t cfg;
	parseargs(argc, argv, cfg);

	// Before any parallel work, so every thread TBB ever brings in gets its own core.
	ThreadPinner pinner;

	const int idx = cfg.target_idx > 0 ? cfg.target_idx : 40;
	const std::string path = cfg.matrix_dir + "matrix_" + std::to_string(idx) + ".txt";

	std::vector<csr::CSRMatrix<double>> matrices;
	matrices.push_back(laplacian_2d(32));    // n = 1,024
	matrices.push_back(laplacian_2d(100));   // n = 10,000
	matrices.push_back(read_mat<double>(path.c_str()));
	std::cout << "# loaded " << path << ": " << matrices.back().m_rows << " rows, "
						<< matrices.back().m_nnz << " nnz\n";

	// Powers of two up to the core count, plus the cluster sweep points below it.
	const int cores = pinner.cores() > 0 ? pinner.cores()
										: static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
	std::set<int> sweep_set;
	for (int t = 1; t <= cores; t *= 2)
		sweep_set.insert(t);
	for (int t : bench::thread_sweep(cores))
		sweep_set.insert(t);
	if (cfg.threads > 0)
		sweep_set = {cfg.threads};

	const std::vector<size_t> element_grains{1, 256, 1024, 4096, 16384};
	const std::vector<size_t> row_grains{1, 16, 64, 256, 1024};

	std::cout << "CSV,kernel,n,threads,grain,min_us,median_us\n";
	for (int t : sweep_set) {
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, t);
		num_threads = t;
		bench::report_pinning(t);
		pinner.report(t);

		for (const auto& A : matrices) {
			const size_t n = A.m_rows;
			std::vector<double> x(n), y(n), z(n);
			for (size_t i = 0; i < n; ++i) {
				x[i] = std::sin(0.1 * static_cast<double>(i));
				y[i] = std::cos(0.1 * static_cast<double>(i));
				z[i] = 1.0;
			}

			const auto set_vector = [](size_t g) { vector_grain = g; };
			const auto set_reduce = [](size_t g) { reduce_grain = g; };
			const auto set_row = [](size_t g) { row_grain = g; };
			volatile double sink = 0;

			time_grains("axpby", n, t, element_grains, set_vector, [&] { axpby(0.5, x, 0.5, y); });
			time_grains("axpbypcz", n, t, element_grains, set_vector,
									[&] { axpbypcz(0.3, x, 0.3, y, 0.4, z); });
			time_grains("innerProduct", n, t, element_grains, set_reduce,
									[&] { sink = sink + innerProduct(x, y); });
			time_grains("spmv", n, t, row_grains, set_row, [&] { spmv(1.0, A, x, 0.0, y); });
		}
	}
	return 0;
}
