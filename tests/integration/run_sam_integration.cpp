// SAM + GMRES on real matrix sets, printing iteration counts with and without the map.
// These report rather than assert; the cd2d group uses the fixtures in tests/data, the
// others need the full sets in data/.
//
// Usage: sam_integration [cd2d|top_opt|weather|all]      (default cd2d)
//        The weather case does not converge with the current GMRES + AMG configuration.

#include "sam_solver_cases.cpp"

#include <string>

int main(int argc, char** argv)
{
    const std::string group = argc > 1 ? argv[1] : "cd2d";
    const bool all = group == "all";

    if (all || group == "cd2d") {
        test_sam_solver_cd2d_1();
        test_sam_solver_cd2d_2();
        test_sam_solver_cd2d_3();
        test_sam_solver_cd2d_4();
    }
    if (all || group == "top_opt") {
        test_sam_solver_top_opt_1();
        test_sam_solver_top_opt_2();
        test_sam_solver_top_opt_3();
    }
    if (all || group == "weather") {
        test_sam_solver_weather_1();
    }
    return 0;
}
