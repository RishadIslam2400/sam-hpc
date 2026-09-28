find_package(TBB REQUIRED)

add_executable(microbench run_microbench.cpp)
target_link_libraries(microbench PUBLIC sam common TBB::tbb)
target_compile_options(microbench
    PRIVATE

    # Debug options
    $<$<CONFIG:Debug>:-Wall>
    $<$<CONFIG:Debug>:-Werror>
    $<$<CONFIG:Debug>:-fsanitize=address>

    # Release options
    $<$<CONFIG:Release>:${SAM_RELEASE_OPTS}>
)
target_link_options(microbench
    PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)

add_executable(threshold_benchmark run_threshold_benchmark.cpp)
target_link_libraries(threshold_benchmark PUBLIC sam common TBB::tbb)
target_compile_options(threshold_benchmark
    PRIVATE

    # Debug options
    $<$<CONFIG:Debug>:-Wall>
    $<$<CONFIG:Debug>:-Werror>
    $<$<CONFIG:Debug>:-fsanitize=address>

    # Release options
    $<$<CONFIG:Release>:${SAM_RELEASE_OPTS}>
)
target_link_options(threshold_benchmark
    PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)

# Phase-resolved benchmark
add_executable(phase_bench run_phase_bench.cpp)
target_link_libraries(phase_bench PUBLIC sam common TBB::tbb)
target_compile_options(phase_bench
    PRIVATE

    # Debug options
    $<$<CONFIG:Debug>:-Wall>
    $<$<CONFIG:Debug>:-fsanitize=address>

    # Release options
    $<$<CONFIG:Release>:${SAM_RELEASE_OPTS}>
)
target_link_options(phase_bench
    PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)
