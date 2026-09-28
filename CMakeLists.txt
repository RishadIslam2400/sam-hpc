cmake_minimum_required(VERSION 3.14)
project(SAM_HPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Without an explicit build type, CMake passes no optimization flags at all AND every
# $<CONFIG:Release> option below is dropped, so an unconfigured build silently produces
# unoptimized binaries. Default to Release; configure with -DCMAKE_BUILD_TYPE=Debug to override.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified, defaulting to Release")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_TESTS "Enable tests" ON)

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)   # ilutp wraps Eigen::IncompleteLUT

# The whole library is header-only: one target, included as "sam_hpc/<module>/<header>.hpp".
add_library(sam_hpc INTERFACE)
target_include_directories(sam_hpc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sam_hpc INTERFACE TBB::tbb Threads::Threads Eigen3::Eigen)

# Large matrix sets live in data/ (git-ignored). Tests and benchmarks that need them get the
# path at compile time instead of hardcoding a home directory.
set(SAM_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Directory holding the matrix sets")

# Every executable gets the same flags, so measurements are comparable across binaries.
# Debug links AddressSanitizer: never take timings from it (see CLAUDE.md).
function(sam_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE sam_hpc)
    target_compile_options(${name} PRIVATE
        # vptr is off: TBB 2020 placement-news its task objects, which UBSan misreports.
        $<$<CONFIG:Debug>:-Wall -g -fsanitize=address,undefined -fno-sanitize=vptr>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    )
    target_link_options(${name} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endfunction()

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
