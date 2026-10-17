#pragma once

/// Loaders for the matrices and right-hand sides of a sequence, as stored under data/ and
/// tests/data/. The benchmarks and the tests load all their input through this file.
///
/// Matrix files are text CSR in exactly four lines, tokens separated by whitespace:
///
///   line 1   rows cols nnz
///   line 2   rows + 1 row pointers: 0-based offsets, first 0, last nnz
///   line 3   nnz column indices, 0-based
///   line 4   nnz values
///
/// Right-hand-side files hold one value per line; blank lines are skipped.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "sam_hpc/core/CSRMatrix.hpp"

namespace read_detail {

/// Helper to print failure message with file name and line number
[[noreturn]] inline void fail(const char* filename, size_t line, const std::string& what) {
  throw std::runtime_error(std::string(filename) + (line ? ":" + std::to_string(line) : "") + ": " + what);
}

// Helper to read a file and handle file errors
inline std::string read_file(const char* filename) {
  // Open the file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    fail(filename, 0, "cannot open file");
  
  // Seek to the end of the file to find file size
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0)
    fail(filename, 0, "cannot determine file size");
  
  // Copy the file contents into a string
  std::string text(static_cast<size_t>(size), '\0');
  in.seekg(0, std::ios::beg);
  if (!in.read(text.data(), size))
    fail(filename, 0, "read failed");
  
  return text;
}

/// Helper to check the character is a whitespace character
inline bool is_space(char c) { 
  return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f'; 
}

/// Helper struct to store the text string line by line.
struct lines {
  const char* p;
  const char* end;
  size_t number = 0;

  explicit lines(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

  /// Returns pointers to the next line after stripping '\n'
  bool next(const char*& b, const char*& e) {
    // Return false if the last line is reached
    if (p == end)
      return false;
    
    b = p;
    e = std::find(p, end, '\n');
    
    // Advance the internal read pointer
    p = (e == end) ? end : e + 1;
    ++number;
    return true;
  }

  /// Returns true if every character in the string is a whitespace
  static bool blank(const char* b, const char* e) { 
    return std::all_of(b, e, is_space); 
  }
};

/// Parse a single line for exactly `count` whitespace-separated tokens of type V from
/// [b, e). V is the value type and Store is the storage type.
template <typename V, typename Store>
void parse_line(const char* filename, size_t line, const char* b, const char* e,
                size_t count, const char* what, Store&& store)
{
  for (size_t i = 0; i < count; ++i) {
    // Skip the leading whitespace
    while (b != e && is_space(*b))
      ++b;

    // Return failure if the string ended before count tokens are reached
    if (b == e)
      fail(filename, line, "expected " + std::to_string(count) + " " + what + ", found " + std::to_string(i));

    // Parse the string and store the value in v without allocating any local memory.
    // std::from_chars returns the error code and pointer to the first character not 
    // matching the pattern or `last`. The numerical value is written to v if it exists.
    V v{};  
    const auto [ptr, ec] = std::from_chars(b, e, v);
    
    // Parsing fails of the conversion failed or the parsed value is immediately followed
    // by characters without any whitespace.
    if (ec != std::errc() || (ptr != e && !is_space(*ptr))) {
      // Find the the end of the current token that failed
      const char* tok_end = std::find_if(b, e, is_space);
      fail(filename, line, std::string("cannot parse ") + what + " #" + std::to_string(i) 
           + ": '" + std::string(b, std::min<size_t>(tok_end - b, 40)) + "'");
    }

    // Store the value after a successful parse and advance the pointer to the next token.
    store(i, v);
    b = ptr;
  }

  // Fail if there are more tokens in the line
  if (!lines::blank(b, e))
    fail(filename, line, "more than " + std::to_string(count) + " " + what);
}

// Helper to validate a floating point value and reject NaN or infinity values.
template <typename T>
void check_finite(const char* filename, size_t line, T v, size_t i) {
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(v))
      fail(filename, line, "value #" + std::to_string(i) + " is not finite");
  }
}

} // namespace read_detail

/// Reads a sparse matrix stored as text CSR.
template <typename T>
csr::CSRMatrix<T> read_mat(const char* filename) {
  using namespace read_detail;

  // Read the entire file in the text string
  const std::string text = read_file(filename);
  lines in(text);
  const char *b, *e;

  // Request the pointers to the next line
  const auto next_line = [&](const char* what) {
    if (!in.next(b, e))
      fail(filename, in.number + 1, std::string("missing line: ") + what);
  };

  // Line 1: rows cols nnz.
  size_t dims[3];
  next_line("rows cols nnz");
  parse_line<size_t>(filename, in.number, b, e, 3, "header values (rows cols nnz)",
                     [&](size_t i, size_t v) { dims[i] = v; });
  const size_t rows = dims[0], cols = dims[1], nnz = dims[2];
  if (cols > static_cast<size_t>(csr::max_index) + 1)
    fail(filename, in.number, std::to_string(cols) + " columns do not fit csr::index_t");
  if (rows > 0 && cols > 0 && nnz / rows > cols)
    fail(filename, in.number, "nnz " + std::to_string(nnz) + " exceeds rows * cols");
  if ((rows == 0 || cols == 0) && nnz != 0)
    fail(filename, in.number, "nnz is nonzero for an empty matrix");

  csr::CSRMatrix<T> A;
  A.m_rows = rows;
  A.m_cols = cols;
  A.m_nnz = nnz;

  // Line 2: row pointers, 0 first, nnz last.
  next_line("row pointers");
  A.m_row_pointers.resize(rows + 1);
  parse_line<size_t>(filename, in.number, b, e, rows + 1, "row pointers",
                     [&](size_t i, size_t v) { A.m_row_pointers[i] = v; });
  if (A.m_row_pointers[0] != 0)
    fail(filename, in.number, "first row pointer is " + std::to_string(A.m_row_pointers[0]) + ", not 0");
  if (A.m_row_pointers[rows] != nnz)
    fail(filename, in.number, "last row pointer is "
      + std::to_string(A.m_row_pointers[rows]) + ", not nnz = " + std::to_string(nnz));
  for (size_t i = 0; i < rows; ++i) {
    if (A.m_row_pointers[i + 1] < A.m_row_pointers[i])
      fail(filename, in.number, "row pointers non-monotonic at row " + std::to_string(i));
  }

  // Line 3: column indices, each in [0, cols).
  next_line("column indices");
  A.m_col_indices.resize(nnz);
  parse_line<size_t>(filename, in.number, b, e, nnz, "column indices",
    [&](size_t i, size_t c) {
      if (c >= cols)
        fail(filename, in.number, "column index #" + std::to_string(i) + " is " 
          + std::to_string(c) + ", outside [0, " + std::to_string(cols) + ")");
      A.m_col_indices[i] = static_cast<csr::index_t>(c);
    }
  );

  // Line 4: values, finite.
  next_line("values");
  A.m_vals.resize(nnz);
  parse_line<T>(filename, in.number, b, e, nnz, "values", [&](size_t i, T v) {
    check_finite(filename, in.number, v, i);
    A.m_vals[i] = v;
  });

  while (in.next(b, e)) {
    if (!lines::blank(b, e))
      fail(filename, in.number, "unexpected content after the values line");
  }

  // Rows: sort any that are out of order, then reject repeated columns.
  for (size_t i = 0; i < rows; ++i) {
    const size_t rb = A.m_row_pointers[i], re = A.m_row_pointers[i + 1];
    csr::index_t* cols_i = A.m_col_indices.data() + rb;
    if (!std::is_sorted(cols_i, cols_i + (re - rb)))
      csr::detail::sortRow(cols_i, A.m_vals.data() + rb, static_cast<int>(re - rb));
    for (size_t j = rb + 1; j < re; ++j) {
      if (A.m_col_indices[j] == A.m_col_indices[j - 1])
        fail(filename, 0, "column " + std::to_string(A.m_col_indices[j]) 
             + " appears twice in row " + std::to_string(i));
    }
  }

  return A;
}

/// Reads a vector stored one value per line. 
template <typename T>
std::vector<T> read_vec(const char* filename) {
  using namespace read_detail;
  const std::string text = read_file(filename);
  lines in(text);
  const char *b, *e;

  std::vector<T> vec;
  vec.reserve(static_cast<size_t>(std::count(text.begin(), text.end(), '\n')) + 1);
  while (in.next(b, e)) {
    if (lines::blank(b, e))
      continue;
    parse_line<T>(filename, in.number, b, e, 1, "value", [&](size_t, T v) {
      check_finite(filename, in.number, v, vec.size());
      vec.push_back(v);
    });
  }
  if (vec.empty())
    fail(filename, 0, "no values");
  return vec;
}
