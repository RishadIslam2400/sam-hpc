
# End to end: pattern + map + GMRES solve on the top_opt sequence.
sam_add_executable(pipeline_bench pipeline_bench.cpp)

# Phase-resolved timing of the SAM pipeline across a thread sweep.
sam_add_executable(phase_bench phase_bench.cpp)

# Grain-size (parallelism cutoff) sweep for the vector and SpMV kernels in linearAlgebra.
sam_add_executable(grain_bench grain_bench.cpp)
