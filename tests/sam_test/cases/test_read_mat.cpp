#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include "sam_hpc/core/read_mat.hpp"
#include "helpers.hpp"
#include "testlib.hpp"

namespace {

/// Write content to a temporary file and return its path.
std::string write_temp(const std::string& name, const std::string& content) {
  const auto path = std::filesystem::temp_directory_path() / ("sam_hpc_read_test_" + name + ".txt");
  std::ofstream(path) << content;
  return path.string();
}

/// Verify that loading fails with an error containing the expected text.
void expect_rejected(const std::string& label, const std::function<void()>& load, const std::string& expected) {
  try {
    load();
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    assertTrueMsg(what.find(expected) != std::string::npos,
                  label + ": wrong error, expected '" + expected + "' in: " + what);
    return;
  }
  throw FailureException(label + ": loaded without an error");
}

void expect_matrix_rejected(const std::string& name, const std::string& content, const std::string& expected) {
  const std::string path = write_temp(name, content);
  expect_rejected("read_mat " + name, [&] { read_mat<double>(path.c_str()); }, expected);
  std::filesystem::remove(path);
}

}

void testReadMat1() {
  std::cout << "Matrix reader..." << std::flush;
  std::string filename = std::string(SAM_TEST_DATA_DIR) + "testMatrix.txt";
  csr::CSRMatrix<double> testMatrix = read_mat<double>(filename.c_str());

  const size_t expectedRows = 5;
  const size_t expectedCols = 5;
  const size_t expectedNNZ = 9;
  const std::vector<size_t> expectedRowPointers = {0, 2, 4, 6, 8, 9};
  const std::vector<size_t> expectedColIndices = {0, 2, 1, 3, 0, 4, 1, 4, 3};
  const std::vector<double> expectedValues = {5.0, 8.0, 3.0, 6.0, 9.0, 7.0, 4.0, 2.0, 1.0};

  assertEquals<size_t>(testMatrix.m_rows, expectedRows, "Row number mismatch");
  assertEquals<size_t>(testMatrix.m_cols, expectedCols, "Column number mismatch");
  assertEquals<size_t>(testMatrix.m_nnz, expectedNNZ, "NNZ mismatch");
  assertEquals(testMatrix.m_row_pointers, expectedRowPointers, "Row pointers mismatch");
  assertEquals(testMatrix.m_col_indices, expectedColIndices, "Column indices mismatch");
  assertEquals(testMatrix.m_vals, expectedValues, "Values mismatch");
  std::cout << "OK" << std::endl;
}

/// Loading sorts row entries with their values and accepts CRLF and blank trailing lines.
void testReadMat2() {
  std::cout << "Matrix reader, unsorted row and CRLF..." << std::flush;
  const std::string path = write_temp("unsorted", "2 3 4\r\n0 3 4\r\n2 0 1 1\r\n30 10 20 5\r\n\r\n");
  const csr::CSRMatrix<double> A = read_mat<double>(path.c_str());
  std::filesystem::remove(path);

  assertEquals(A.m_col_indices, std::vector<size_t>{0, 1, 2, 1}, "unsorted row not sorted");
  assertEquals(A.m_vals, std::vector<double>{10.0, 20.0, 30.0, 5.0}, "values did not move with their columns");
  assertTrueMsg(A.hasSortedRows(), "loaded matrix breaks the sorted-rows invariant");
  std::cout << "OK" << std::endl;
}

void testReadMatErrors() {
  std::cout << "Matrix reader, malformed files..." << std::flush;
  expect_rejected("missing file", [] { read_mat<double>("/nonexistent/sam_hpc/matrix.txt"); }, "cannot open file");

  // Validate header, line-count, and row-pointer errors.
  expect_matrix_rejected("missing_line",    "2 2 2\n0 1 2\n0 1\n",           "missing line: values");
  expect_matrix_rejected("bad_header",      "2 two 2\n0 1 2\n0 1\n1 1\n",   "cannot parse header");
  expect_matrix_rejected("few_indices",     "2 2 2\n0 1 2\n0\n1 1\n",       "expected 2 column indices, found 1");
  expect_matrix_rejected("many_values",     "2 2 2\n0 1 2\n0 1\n1 1 1\n",   "more than 2 values");
  expect_matrix_rejected("first_ptr",       "2 2 2\n1 1 2\n0 1\n1 1\n",     "first row pointer is 1");
  expect_matrix_rejected("last_ptr",        "2 2 2\n0 1 1\n0 1\n1 1\n",     "last row pointer is 1");
  expect_matrix_rejected("decreasing_ptr",  "3 3 2\n0 2 1 2\n0 1\n1 1\n",   "row pointers non-monotonic at row 1");

  // Validate index, value, duplicate-entry, trailing-content, and size checks.
  expect_matrix_rejected("col_range",       "2 2 2\n0 1 2\n0 2\n1 1\n",     "outside [0, 2)");
  expect_matrix_rejected("negative_col",    "2 2 2\n0 1 2\n-1 1\n1 1\n",    "cannot parse column indices #0");
  expect_matrix_rejected("nan",             "2 2 2\n0 1 2\n0 1\n1 nan\n",   "value #1 is not finite");
  expect_matrix_rejected("duplicate",       "1 3 2\n0 2\n1 1\n1 1\n",        "column 1 appears twice in row 0");
  expect_matrix_rejected("trailing",        "2 2 2\n0 1 2\n0 1\n1 1\nextra\n", "unexpected content");
  expect_matrix_rejected("too_dense",       "1 2 5\n0 5\n0 1 0 1 0\n1 1 1 1 1\n", "exceeds rows * cols");
  std::cout << "OK" << std::endl;
}

void testReadVec() {
  std::cout << "Vector reader..." << std::flush;

  // Blank lines and surrounding whitespace are ignored.
  const std::string good = write_temp("vec_good", "1.5\n\n-2e-3\r\n  7  \n");
  assertEquals(read_vec<double>(good.c_str()), std::vector<double>{1.5, -2e-3, 7.0}, "read_vec values");
  std::filesystem::remove(good);

  const std::string two = write_temp("vec_two", "1.5\n2 3\n");
  expect_rejected("read_vec two values on a line", [&] { read_vec<double>(two.c_str()); }, ":2: more than 1 value");
  std::filesystem::remove(two);

  const std::string bad = write_temp("vec_bad", "1.5\nabc\n3\n");
  expect_rejected("read_vec bad line", [&] { read_vec<double>(bad.c_str()); }, ":2: cannot parse value");
  std::filesystem::remove(bad);

  const std::string empty = write_temp("vec_empty", "\n\n");
  expect_rejected("read_vec empty", [&] { read_vec<double>(empty.c_str()); }, "no values");
  std::filesystem::remove(empty);

  expect_rejected("read_vec missing", [] { read_vec<double>("/nonexistent/sam_hpc/rhs.txt"); }, "cannot open file");
  std::cout << "OK" << std::endl;
}