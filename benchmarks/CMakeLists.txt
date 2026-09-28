# Find the TBB library. This is crucial for parallel execution.
# This command defines the TBB::tbb target.
find_package(TBB REQUIRED)

add_executable(top_opt_bench topology_benchmark.cpp)
target_link_libraries(top_opt_bench 
    PUBLIC 
    sam 
    common 
    solver
    TBB::tbb
)
target_compile_options(top_opt_bench
    PRIVATE
    # Debug options
    $<$<CONFIG:Debug>:-Wall;-Werror;-fsanitize=address>

    # Release options
    $<$<CONFIG:Release>:${SAM_RELEASE_OPTS}>
)
target_link_options(top_opt_bench
    PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)

add_subdirectory(microbench)

add_executable(weather_bench weather_benchmark.cpp)
target_link_libraries(weather_bench PUBLIC sam common solver TBB::tbb)
target_compile_options(weather_bench
    PRIVATE

    # Debug options
    $<$<CONFIG:Debug>:-Wall>
    $<$<CONFIG:Debug>:-Werror>
    $<$<CONFIG:Debug>:-fsanitize=address>

    # Release options
    $<$<CONFIG:Release>:${SAM_RELEASE_OPTS}>
)
target_link_options(weather_bench
    PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)