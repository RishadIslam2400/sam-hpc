# ilutp wraps Eigen's IncompleteLUT.
find_package(Eigen3 REQUIRED)

add_library(relaxation INTERFACE)
target_include_directories(relaxation
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../relaxation/include>
)
target_link_libraries(relaxation INTERFACE csr_matrix Eigen3::Eigen)