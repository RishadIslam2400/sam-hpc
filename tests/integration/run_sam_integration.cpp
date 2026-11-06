#include "sam_solver_cases.cpp"

#include <string>

int main(int argc, char** argv)
{
  // Default to the cd2d cases; pass "top_opt" or "all" to select other groups.
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
  return 0;
}