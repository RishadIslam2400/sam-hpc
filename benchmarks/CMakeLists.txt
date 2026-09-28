# End to end: pattern + map + GMRES solve, top_opt or weather preset (-d).
sam_add_executable(pipeline_bench pipeline_bench.cpp)

# Phase-resolved timing of the SAM pipeline across a thread sweep.
sam_add_executable(phase_bench phase_bench.cpp)
