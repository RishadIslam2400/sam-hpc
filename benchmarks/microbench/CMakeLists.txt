find_package(TBB REQUIRED)

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
