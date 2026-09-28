cmake_minimum_required(VERSION 3.10)
project(SAM_HPC LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified, defaulting to Release")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
set(SAM_RELEASE_OPTS -O3 -DNDEBUG -march=native)

# Output settings
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Options
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_TESTS "Enable tests" ON)

# Subdirectories
set(SUBDIRS
    sam_hpc
    csr_matrix
    common
    tools
    solver
    preconditioner
    relaxation
    coarsening
)

foreach(subdir ${SUBDIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt")
        add_subdirectory(${subdir})
    else()
        message(WARNING "Directory '${subdir}' not found, skipping.")
    endif()
endforeach()

# Conditionally include tests
if(ENABLE_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        message(STATUS "Building tests")
        add_subdirectory(tests)
    else()
        message(WARNING "Tests directory not found.")
    endif()
else()
    message(STATUS "Skipping tests")
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
        message(STATUS "Building benchmarks")
        add_subdirectory(benchmarks)
    else()
        message(WARNING "Benchmarks directory not found.")
    endif()
else()
    message(STATUS "Skipping benchmarks")
endif()