cmake_minimum_required(VERSION 3.14)
project(SAM_HPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "SAM-HPC is built with GCC; got ${CMAKE_CXX_COMPILER_ID} "
                      "(${CMAKE_CXX_COMPILER}). Use 'cmake --preset release' or pass "
                      "-DCMAKE_CXX_COMPILER=g++ to a fresh build directory.")
endif()
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR "GCC ${CMAKE_CXX_COMPILER_VERSION} is too old; SAM-HPC needs GCC 11 or newer.")
endif()
message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default release build if build type is not set and a single config generator
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "No build type specified, defaulting to Release")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_TESTS "Enable tests" ON)

# Required packages
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# The whole library is header-only
add_library(sam_hpc INTERFACE)
target_include_directories(sam_hpc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sam_hpc INTERFACE TBB::tbb Threads::Threads Eigen3::Eigen)

# Dataset directory
set(SAM_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Directory holding the matrix sets")

# Generate uniform executable for all experiments
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
