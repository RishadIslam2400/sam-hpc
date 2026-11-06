#include "sam_hpc/precond/cuthill_mckee.hpp"
#include "testlib.hpp"

// Cuthill-McKee uses matrix structure as a graph and ignores values. Traversal starts
// at node 0, orders each level by degree, and restarts between disconnected components.
// The reverse flag changes the level ordering but does not reverse the final permutation.
// Test a simple path graph: 0 -- 1 -- 2 -- 3.
void test_simple_path_graph_cm() {
	std::cout << "simple cuthill mckee..." << std::flush;
	csr::CSRMatrix<double> A(4, 4,
		{1, 1, 1, 1, 1, 1},
		{0, 1, 3, 5, 6},            
		{1, 0, 2, 1, 3, 2}          
	);

	std::vector<ptrdiff_t> perm(4);
	cuthill_mckee<double, false>::get(A, perm);
	std::vector<ptrdiff_t> expected_perm = {0, 1, 2, 3};
	
	assertEquals(expected_perm, perm, "Permution incorrect!");
	std::cout << "OK" << std::endl;
}

// Test the decreasing-degree variant on an asymmetric tree. Node 3 is reached only
// after traversal restarts because no row from the initial component lists it.
void test_reverse_ordering_rcm() {
	std::cout << "reverse cuthill mckee..." << std::flush;
	csr::CSRMatrix<double> A(6, 6,
		std::vector<double>(9, 1.0),
		{0, 1, 4, 6, 7, 8, 9},                 
		{1, 0, 2, 4, 1, 5, 0, 1, 2}
	);

	std::vector<ptrdiff_t> perm(6);
	cuthill_mckee<double, true>::get(A, perm);

	std::vector<ptrdiff_t> expected_perm = {0, 1, 2, 4, 5, 3};  
	assertEquals(expected_perm, perm, "Permution incorrect!");

	std::cout << "OK" << std::endl;
}

// Test traversal across two disconnected components: {0, 1} and {2, 3, 4}.
void test_disconnected_graph() {
	std::cout << "disconnected graph..." << std::flush;
	csr::CSRMatrix<double> A(5, 5,
		std::vector<double>(6, 1.0),
		{0, 1, 2, 3, 5, 6},    
		{1, 0, 3, 2, 4, 3}    
	);

	std::vector<ptrdiff_t> perm(5);
	cuthill_mckee<double, false>::get(A, perm);

	std::vector<ptrdiff_t> expected_perm = {0, 1, 2, 3, 4};
	assertEquals(expected_perm, perm, "Permution incorrect!");

	std::cout << "OK" << std::endl;
}