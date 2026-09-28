#include "sam_hpc/precond/cuthill_mckee.hpp"
#include "testlib.hpp"

/**
 * @brief Tests the standard Cuthill-McKee algorithm on a simple line graph.
 *
 * Graph: 0 -- 1 -- 2 -- 3
 * The algorithm should produce a straightforward sequential permutation.
 */
void test_simple_path_graph_cm() {
    std::cout << "simple cuthill mckee..." << std::flush;
    // Adjacency matrix for the graph
    CSRMatrix<double> A(4, 4,
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

/**
 * @brief Tests the reverse Cuthill-McKee by checking the intra-level set ordering.
 *
 * Graph:
 * 3   4
 * | /
 * 0 -- 1 -- 2 -- 5
 *
 * When exploring from node 1, its neighbors are {0, 2, 4}.
 * Degrees: deg(0)=1, deg(2)=2, deg(4)=1.
 * CM (reverse=false) should order them by increasing degree: {0, 4, 2} or {4, 0, 2}.
 * RCM (reverse=true) should order them by decreasing degree: {2, 0, 4} or {2, 4, 0}.
 * This test verifies the RCM case.
 */
void test_reverse_ordering_rcm() {
    std::cout << "reverse cuthill mckee..." << std::flush;
    // Adjacency matrix for the graph
    CSRMatrix<double> A(6, 6,
        std::vector<double>(9, 1.0),
        {0, 1, 4, 6, 7, 8, 9},                 
        {1, 0, 2, 4, 1, 5, 0, 1, 2}
    );

    // A more precise adjacency list for clarity:
    // 0: [1]        (deg 1)
    // 1: [0, 2, 4]  (deg 3)
    // 2: [1, 5]      (deg 2)
    // 3: []           (deg 0, we start here)
    // 4: [1]        (deg 1)
    // 5: [2]        (deg 1)
    
    // We will manually set initialNode to 0 to make the path predictable.
    // The code as written starts at node 0 by default.
    // Level 0: {0} (deg 1) -> perm: [0]
    // Level 1: {1} (deg 3) -> perm: [0, 1]
    // Level 2: {2, 4} (deg(2)=2, deg(4)=1)
    // RCM (reverse=true) sorts by decreasing degree, so 2 comes before 4.
    // perm: [0, 1, 2, 4]
    // Level 3: {5} (from 2) -> perm: [0, 1, 2, 4, 5]
    // The node 3 is disconnected and will be picked up last.
    std::vector<ptrdiff_t> perm(6);
    cuthill_mckee<double, true>::get(A, perm);

    // The disconnected node 3 will be appended at the end.
    std::vector<ptrdiff_t> expected_perm = {0, 1, 2, 4, 5, 3};  
    assertEquals(expected_perm, perm, "Permution incorrect!");

    std::cout << "OK" << std::endl;
}

/**
 * @brief Tests the algorithm's ability to handle graphs with multiple connected components.
 *
 * Graph: 0 -- 1   and   2 -- 3 -- 4
 * The algorithm should process the first component, then find an unvisited node
 * in the second component and process it.
 */
void test_disconnected_graph() {
    std::cout << "disconnected graph..." << std::flush;
    CSRMatrix<double> A(5, 5,
        std::vector<double>(6, 1.0),
        {0, 1, 2, 3, 5, 6},    
        {1, 0, 3, 2, 4, 3}    
    );

    std::vector<ptrdiff_t> perm(5);
    cuthill_mckee<double, false>::get(A, perm);

    // Expected: processes {0, 1}, then finds {2} and processes {2, 3, 4}
    std::vector<ptrdiff_t> expected_perm = {0, 1, 2, 3, 4};
    assertEquals(expected_perm, perm, "Permution incorrect!");

    std::cout << "OK" << std::endl;
}