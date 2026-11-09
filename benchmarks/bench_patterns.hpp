#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/read_mat.hpp"
#include "sam_hpc/sam/sparsityPattern.hpp"
#include "sam_hpc/util/config.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bench {

/// Splits a comma-separated list, skipping empty items.
inline std::vector<std::string> split_list(const std::string &s, char sep = ',') {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);)
    if (!item.empty())
      out.push_back(item);
  return out;
}

/// Parses a number or exits.
inline double parse_number(const std::string &s, const char *what) {
  char *end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || *end != '\0') {
    std::cerr << "error: " << what << " value '" << s << "' is not a number" << std::endl;
    std::exit(2);
  }
  return v;
}

/// Shortest decimal form of a parameter.
inline std::string fmt(double v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

/// The structure of a matrix as a pattern.
template <typename T>
csr::CSRMatrix<int> structure_of(const csr::CSRMatrix<T> &m) {
  csr::CSRMatrix<int> p;
  p.m_rows = m.m_rows;
  p.m_cols = m.m_cols;
  p.m_nnz = m.m_nnz;
  p.m_row_pointers = m.m_row_pointers;
  p.m_col_indices = m.m_col_indices;
  p.m_vals.assign(m.m_nnz, 1);
  return p;
}

/// Everything the pattern families need, parsed once from the configuration.
struct pattern_set {
  std::string families;
  std::vector<double> global, column;
  std::vector<size_t> fixed;
  std::vector<std::pair<double, double>> combined;
  std::vector<std::pair<std::string, std::vector<long>>> stencils;
  std::vector<std::pair<std::string, csr::CSRMatrix<int>>> explicit_patterns;
  int level = 2;

  explicit pattern_set(const config_t &cfg) : families(cfg.patterns), level(cfg.level) {
    for (const auto &v : split_list(cfg.global_list))
      global.push_back(parse_number(v, "--global"));
    for (const auto &v : split_list(cfg.column_list))
      column.push_back(parse_number(v, "--column"));
    for (const auto &v : split_list(cfg.fixed_list))
      fixed.push_back(static_cast<size_t>(parse_number(v, "--fixed")));
    for (const auto &v : split_list(cfg.combined_list)) {
      const auto gt = split_list(v, ':');
      if (gt.size() != 2) {
        std::cerr << "error: --combined item '" << v << "' is not global:tau" << std::endl;
        std::exit(2);
      }
      combined.emplace_back(parse_number(gt[0], "--combined"), parse_number(gt[1], "--combined"));
    }
    for (const auto &item : split_list(cfg.offsets, ';')) {
      const auto eq = item.find('=');
      const std::vector<long> d = parse_offsets(eq == std::string::npos ? item : item.substr(eq + 1));
      std::string name = eq == std::string::npos ? "stencil[" + std::to_string(d.size()) + "]"
                                                 : item.substr(0, eq);
      if (name.empty() || d.empty()) {
        std::cerr << "error: -o item '" << item << "' is not [name=]offset,offset,..." << std::endl;
        std::exit(2);
      }
      stencils.emplace_back(std::move(name), d);
    }
    for (const auto &file : split_list(cfg.pattern_file)) {
      const auto m = read_mat<double>(file.c_str());
      std::string base = file.substr(file.find_last_of('/') + 1);
      explicit_patterns.emplace_back("explicit " + base, structure_of(m));
    }
    if (level < 1) {
      std::cerr << "error: --level must be >= 1" << std::endl;
      std::exit(2);
    }
  }

  bool selects(const char *family) const { return list_selects(families, family); }
};

/// Calls f(label, params) for every selected pattern.
template <typename F>
void for_each_pattern(const pattern_set &ps, const csr::CSRMatrix<int> *target_structure, F &&f) {
  if (ps.selects("target") && target_structure)
    f(std::string("S(A_0)"), ExplicitPattern{target_structure});
  if (ps.selects("simple"))
    f(std::string("S(A_k)"), SimplePattern{});
  if (ps.selects("global"))
    for (double g : ps.global)
      f("global " + fmt(g), GlobalThresholdPattern{g});
  if (ps.selects("column"))
    for (double t : ps.column)
      f("column " + fmt(t), ColumnThresholdPattern{t});
  if (ps.selects("fixed"))
    for (size_t l : ps.fixed)
      f("fixed " + std::to_string(l), FixedNNZPattern{l});
  if (ps.selects("combined"))
    for (const auto &[g, t] : ps.combined)
      f("combined " + fmt(g) + ":" + fmt(t), combinedThresholdPattern{g, t});
  // Mesh stencils are restricted to the source's structure, so offsets that wrap past a mesh
  // boundary onto a non-neighbour are dropped (Carr et al.'s boundary adjustment).
  if (ps.selects("stencil"))
    for (const auto &[name, d] : ps.stencils)
      f(name, StencilPattern{d, true});
  if (ps.selects("explicit"))
    for (const auto &[name, p] : ps.explicit_patterns)
      f(name, ExplicitPattern{&p});
}

} // namespace bench
