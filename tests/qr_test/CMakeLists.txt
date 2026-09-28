add_executable(qr_test run_qr_tests.cpp)
target_link_libraries(qr_test 
    PUBLIC
    test_utils
    solver
)
target_compile_options(qr_test
    PRIVATE

    # Debug options
    $<$<CONFIG:Debug>:-Wall>
    $<$<CONFIG:Debug>:-Werror>
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
    $<$<CONFIG:Debug>:-fsanitize=leak>

    # $<$<CONFIG:Debug>:-fsanitize=thread>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-DDEBUG>

    # Release options
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
)

target_link_options(qr_test
    PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
    $<$<CONFIG:Debug>:-fsanitize=leak>
    # $<$<CONFIG:Debug>:-fsanitize=thread>
)