#include "cases/test_pattern.cpp"
#include "cases/test_extend_pattern.cpp"
#include "cases/test_read_mat.cpp"
#include "cases/test_sam_computation.cpp"
#include "cases/test_sam_map_sides.cpp"
#include "cases/test_fused_map.cpp"
#include "cases/test_stencil_pattern.cpp"

int main()
{
	std::cout << "Running sparsity pattern tests..." << std::endl;
	testSimpleSparsityPattern();
	testGlobalSparsityPattern1();
	testGlobalSparsityPattern2();
	testColumnSparsityPattern();
	testFixedNNZSparsityPattern();

	std::cout << "\nRunning pattern extension tests..." << std::endl;
	testExtendPatternFrontier();
	testExtendPatternWalks();
	testExtendPatternTridiagonal();

	std::cout << "\nRunning matrix reader tests..." << std::endl;
	testReadMat1();
	testReadMat2();
	testReadMatErrors();
	testReadVec();

	std::cout << "\nRunning SAM computation tests..." << std::endl;
	testSAMSanityCheck1();
	testSAMSanityCheck2();
	testSAMSanityCheck3();
	testSAMSanityCheck4();
	testCD2D1();
	testCD2D2();
	testCD2D3();
	testCD2D4();

	std::cout << "\nRunning left vs right map tests..." << std::endl;
	testMapSidesAreMinimizers();
	testMapSidesDifferWhenNonsymmetric();
	testMapSidesTransposeWhenSymmetric();
	testRightResidualMatchesDense();

	std::cout << "\nRunning fused pattern extension tests..." << std::endl;
	testFusedMapSmall();
	testFusedMapCD2D();

	std::cout << "\nRunning stencil and explicit pattern tests..." << std::endl;
	testStencilPatternOffsets();
	testStencilPatternRestricted();
	testStencilPatternIgnoresLevel();
	testStencilTransposed();
	testExplicitPattern();
	testStencilMapCD2D();
	testExplicitMapMatchesDirect();
	return 0;
}
