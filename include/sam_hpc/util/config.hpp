#pragma once

#include <iostream>
#include <string>
#include <unistd.h>

#include "sam_hpc/util/parallel.hpp"

/// Command-line configuration shared by the benchmarks.
struct config_t {
  std::string name;        // Experiment label (-n).
  std::string matrix_dir;  // Matrix directory (-x).
  std::string rhs_dir;     // Right-hand-side directory (-y).
  int iters;               // Maximum repetitions per measurement (-k).
  int threads;             // Thread count (-t); zero requests a supported sweep.
  int target_idx;          // Target matrix index (-a); nonpositive uses the benchmark default.
  int source_idx;          // Source matrix index (-b); nonpositive uses the benchmark default.
  std::string patterns;    // Comma-separated pattern names, or "all" (-p).
  std::string variant;     // Pattern variant: source or union (-u).
  std::string map_side;    // Map formulation: left or right (-m).
  std::string accumulator; // Sparse-product accumulator: dense or hash (-s).
  std::string extension;   // Pattern extension: materialized or fused (-e).

  config_t () {
    name = "default name";
    matrix_dir = "default path";
    rhs_dir = "default path";
    iters = 5;
    threads = 1;
    target_idx = -1;
    source_idx = -1;
    patterns = "all";
    variant = "source";
    map_side = "left";
    accumulator = "dense";
    extension = "materialized";
  }

  /// Prints the active configuration as comment-prefixed benchmark metadata.
  void print() {
    std::cout << "# name, matrix_dir, rhs_dir, iters, threads:\n";
    std::cout << name << ", " << matrix_dir << ", " << rhs_dir << ", " << iters << ", " << threads << std::endl;
    std::cout << "# accumulator: " << accumulator << ", extension: " << extension << "\n";
    std::cout << "# pattern variant: " << variant << ", map side: " << map_side
              << std::endl;
  }
};

/// Prints the supported command-line options.
inline void usage() {
  std::cout << "Command line options:\n"
            << "-n <string> : name of the experiment\n"
            << "-x <string> : path to the matrix directory\n"
            << "-y <string> : path to the rhs directory\n" 
            << "-k <int> : number of test iterations\n"
            << "-t : number of threads for the execution (0 = sweep, where supported)\n"
            << "-a <int> : index of the target matrix file (benchmark specific default)\n"
            << "-b <int> : index of the source matrix file (benchmark specific default)\n"
            << "-p <list> : comma separated patterns to run, or all (default all)\n"
            << "-u <name> : matrix to sparsify - source | union (default source)\n"
            << "-m <name> : map side - left (||N A_k - A_0||) | right (||A_k N - A_0||),\n"
            << "-s <name> : sparse product accumulator - dense | hash\n"
            << "-e <name> : pattern extension - materialized | fused (computed inside the map)\n"
            << "            default left; right needs transposes, see sam.hpp mapSide\n"
            << "-h : display help message"
            << std::endl;
}

/// Parses command-line options with getopt().
///
/// Unspecified fields retain config_t's defaults. `-h` prints help and parsing continues.
inline void parseargs(int argc, char** argv, config_t& cfg) {
  int opt;
  while ((opt = getopt(argc, argv, "n:x:y:k:t:a:b:p:u:m:s:e:h")) != -1) {
    switch (opt) {
    case 'n':
      cfg.name = std::string(optarg);
      break;
    case 'x':
      cfg.matrix_dir = std::string(optarg);
      break;
    case 'y':
      cfg.rhs_dir = std::string(optarg);
      break;
    case 'k':
      cfg.iters = atoi(optarg);
      break;
    case 't':
      cfg.threads = atoi(optarg);
      break;
    case 'a':
      cfg.target_idx = atoi(optarg);
      break;
    case 'b':
      cfg.source_idx = atoi(optarg);
      break;
    case 'p':
      cfg.patterns = std::string(optarg);
      break;
    case 'u':
      cfg.variant = std::string(optarg);
      break;
    case 'm':
      cfg.map_side = std::string(optarg);
      break;
    case 's':
      cfg.accumulator = std::string(optarg);
      break;
    case 'e':
      cfg.extension = std::string(optarg);
      break;
    case 'h':
      usage();
      break;
    }
  }
}

/// Selected sparsification variant.
/// `source` sparsifies A_k. `union` merges A_k and A_0 by magnitude before sparsification.
struct pattern_variant {
  bool union_candidate = false;
  const char* name = "source";
};

/// Parses -u, accepting `source`, `union`.
/// Invalid values produce a warning and select `source`.
inline pattern_variant parse_pattern_variant(const std::string &s) {
  if (s == "source")
    return {false, "source"};
  if (s == "union")
    return {true,  "union"};
  std::cerr << "Unknown pattern variant '" << s << "', using source. Valid: source | union"
            << std::endl;
  return {};
}


/// Parses -s. Invalid values produce a warning and select the dense accumulator.
inline spgemm_accumulator parse_accumulator(const std::string &s) {
  if (s == "dense")
    return spgemm_accumulator::dense;
  if (s == "hash") 
    return spgemm_accumulator::hash;
  std::cerr << "Unknown accumulator '" << s << "', using dense. Valid: dense | hash" << std::endl;
  return spgemm_accumulator::dense;
}

/// Parses -e. Invalid values produce a warning and select materialized extension.
inline bool parse_fused(const std::string &s) {
  if (s == "fused")
    return true;
  if (s == "materialized")
    return false;
  std::cerr << "Unknown extension '" << s << "', using materialized. Valid: materialized | fused"
            << std::endl;
  return false;
}
