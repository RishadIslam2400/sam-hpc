#include <iostream>

#include "cases/test_qr_opt.cpp"

int main()
{
	std::cout << "Running QR tests..." << std::endl;
	test_factorization();
	test_solve_overdetermined();
	test_solve_underdetermined();
	test_solve_rank_deficient();
	test_solve_nearly_rank_deficient();
	test_solve_nearly_rank_deficient_wide();
	test_solve_pivot_order();
	test_try_solve_unpivoted();
}