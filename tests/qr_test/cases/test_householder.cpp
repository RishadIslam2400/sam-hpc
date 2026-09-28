#include "householderQR.hpp"
#include "testlib.hpp"
#include "helpers.hpp"

#include <iostream>

void testHouseholderQRSolve1()
{
    std::cout << "Sanity Check..." << std::flush;
    std::vector<std::vector<double>> A = {
        {2, 1},
        {1, 3}};
    std::vector<double> rhs = {5, 10};
    std::vector<double> x(2);
    std::vector<double> expectedSolution = {1, 3};
    householderQRSolve(A, rhs, x, 2, 2);

    assertEquals(expectedSolution, x, "Incorrect solution");
    std::cout << "OK" << std::endl;
}

void testHouseholderQRSolve2()
{
    std::cout << "Orthogoanl Matrix..." << std::flush;
    std::vector<std::vector<double>> A = {
        {1, 0},
        {0, -1}};
    std::vector<double> rhs = {1, -2};
    std::vector<double> x(2);
    std::vector<double> expectedSolution = {1.0, 2.0};
    householderQRSolve(A, rhs, x, 2, 2);

    assertEquals(expectedSolution, x, "Incorrect solution");
    std::cout << "OK" << std::endl;
}

void testHouseholderQRSolve3()
{
    std::cout << "Square Matrix Full Rank..." << std::flush;
    std::vector<std::vector<double>> A = {
        {5, 4, 4},
        {-2, -3, -6},
        {2, 4, 7}};
    std::vector<double> rhs = {1, 2, 3};
    std::vector<double> x(3);
    std::vector<double> expectedSolution = {0.06666667, 0.4, 0.73333333};
    householderQRSolve(A, rhs, x, 3, 3);

    assertEquals(expectedSolution, x, "Incorrect solution");
    std::cout << "OK" << std::endl;
}

void testHouseholderQRSolve4()
{
    std::cout << "Diaognal Dominant Matrix..." << std::flush;
    std::vector<std::vector<double>> A = {
        {10, 2, 2},
        {1, 10, 2},
        {1, 1, 10}};
    std::vector<double> rhs = {12, 13, 14};
    std::vector<double> x(3);
    std::vector<double> expectedSolution = {1, 1, 1};
    householderQRSolve(A, rhs, x, 3, 3);

    assertEquals(expectedSolution, x, "Incorrect solution");
    std::cout << "OK" << std::endl;
}

void testHouseholderQRSolve5()
{
    std::cout << "Overdetermined Linear System, Full Rank..." << std::flush;
    std::vector<std::vector<double>> A = {
        {1, 1, 1, 1},
        {1, 2, 3, 4},
    };
    std::vector<double> rhs = {6, 5, 7, 10};
    std::vector<double> x(2);
    std::vector<double> expectedSolution = {3.5, 1.4};
    householderQRSolve(A, rhs, x, 2, 4);

    assertEquals(expectedSolution, x, "Incorrect solution");
    std::cout << "OK" << std::endl;
}

void testHouseholderQRSolve6()
{
    std::cout << "Nearly Singular Matrix..." << std::flush;
    std::vector<std::vector<double>> A = {
        {1, 1},
        {1, 1.00001}};
    std::vector<double> rhs = {2, 2.00001};
    std::vector<double> x(2);
    std::vector<double> expectedSolution = {1.0, 1.0};
    householderQRSolve(A, rhs, x, 2, 2);

    assertEquals(expectedSolution, x, "Incorrect solution");
    std::cout << "OK" << std::endl;
}