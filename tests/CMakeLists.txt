# Unit tests: small, checked-in fixtures under tests/data, run by `ctest`.
function(sam_add_test name source)
    sam_add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
    target_compile_definitions(${name} PRIVATE
        SAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
        SAM_DATA_DIR="${SAM_DATA_DIR}/")
    add_test(NAME ${name} COMMAND ${name})
endfunction()

sam_add_test(csr_test    csr_test/run_csr_tests.cpp)
sam_add_test(qr_test     qr_test/run_qr_tests.cpp)
sam_add_test(sam_test    sam_test/run_sam_tests.cpp)
sam_add_test(solver_test solver_test/run_solver_tests.cpp)

# Integration checks against the full matrix sets in data/. Minutes per case and not
# assertions, so they are not registered with ctest; run them by hand.
sam_add_executable(sam_integration integration/run_sam_integration.cpp)
target_include_directories(sam_integration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_compile_definitions(sam_integration PRIVATE
    SAM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
    SAM_DATA_DIR="${SAM_DATA_DIR}/")
