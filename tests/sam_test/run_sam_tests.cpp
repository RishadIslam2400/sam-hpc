#include "cases/test_pattern.cpp"
#include "cases/test_read_mat.cpp"
#include "cases/test_sam_computation.cpp"

int main()
{
    std::cout << "Running sparsity pattern tests..." << std::endl;
    testSimpleSparsityPattern();
    testGlobalSparsityPattern1();
    testGlobalSparsityPattern2();
    testColumnSparsityPattern();
    testFixedNNZSparsityPattern();

    std::cout << "\nRunning matrix reader tests..." << std::endl;
    testReadMat1();
    testReadMat2();

    std::cout << "\nRunning SAM computation tests..." << std::endl;
    testSAMSanityCheck1();
    testSAMSanityCheck2();
    testSAMSanityCheck3();
    testSAMSanityCheck4();
    testCD2D1();
    testCD2D2();
    testCD2D3();
    testCD2D4();
    return 0;
}
