#pragma once

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "sam_hpc/util/parallel.hpp"

/// Command-line configuration shared by the benchmarks.
struct config_t {
  std::string name;        // Experiment label (-n).
  std::string matrix_dir;  // Matrix directory (-x).
  std::string rhs_dir;     // Right-hand-side directory (-y).
  int iters;               // Maximum repetitions per measurement (-k).
  int threads;             // Thread count (-t).
  int target_idx;          // Target matrix index (-a).
  int source_idx;          // Source matrix index (-b).
  std::string patterns;    // Comma-separated pattern names, or "all" (-p).
  std::string variant;     // Pattern variant: source or union (-u).
  std::string map_side;    // Map formulation: left or right (-m).
  std::string accumulator; // Sparse-product accumulator: dense or hash (-s).
  std::string extension;   // Pattern extension: materialized or fused (-e).
  int last_idx;            // Last source index of a sequence (-c).
  std::string modes;       // Comma-separated update modes, or "all" (-r).
  int period;              // Recompute period of the periodic mode, in steps (-R).
  std::string offsets;     // Stencils (-o).
  std::string pattern_file;// Comma-separated CSR files whose structure is an explicit pattern (-f).

  // Pattern parameters.
  std::string global_list;
  std::string column_list;
  std::string fixed_list;    
  std::string combined_list;
  int level;

  // Solver and preconditioner.
  std::string precond;       // ilut | amg
  int ilut_fill;             // fill factor of the ILUT preconditioner
  double ilut_droptol;       // drop tolerance of the ILUT preconditioner
  std::string amg_relax;     // jacobi | gauss_seidel | ilu0 | ilut
  double amg_strong;         // Ruge-Stuben strength threshold
  int amg_npre;              // pre-smoothing sweeps
  int amg_npost;             // post-smoothing sweeps
  int amg_coarse;            // direct solve below this many rows
  int amg_ilut_fill;         // fill factor of the ILUT smoother
  double amg_ilut_droptol;   // drop tolerance of the ILUT smoother
  int gmres_restart;         // Krylov dimension M before a restart
  int gmres_maxit;           // total iteration cap
  double gmres_tol;          // relative tolerance
  std::string precond_side;  // left | right
  int solve_reps;

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
    last_idx = -1;
    modes = "sam";
    period = 0;
    global_list = "0.3";
    column_list = "0.7,0.8,0.9";
    fixed_list = "3,5,7";
    combined_list = "0.1:0.8";
    level = 2;
    precond = "ilut";
    ilut_fill = 250;
    ilut_droptol = 1e-3;
    amg_relax = "gauss_seidel";
    amg_strong = 0.25;
    amg_npre = 1;
    amg_npost = 1;
    amg_coarse = 0;
    amg_ilut_fill = 3;
    amg_ilut_droptol = 1e-3;
    gmres_restart = 2000;
    gmres_maxit = 2000;
    gmres_tol = 1e-8;
    precond_side = "right";
    solve_reps = 3;
  }

  /// Prints the active configuration as comment-prefixed benchmark metadata.
  void print() {
    std::cout << "# name, matrix_dir, rhs_dir, iters, threads:\n";
    std::cout << name << ", " << matrix_dir << ", " << rhs_dir << ", " << iters << ", " << threads << std::endl;
    std::cout << "# accumulator: " << accumulator << ", extension: " << extension << "\n";
    std::cout << "# pattern variant: " << variant << ", map side: " << map_side
              << std::endl;
    std::cout << "# patterns: " << patterns << ", modes: " << modes << ", period: " << period
              << ", last source: " << last_idx << "\n";
    if (!offsets.empty())
      std::cout << "# stencil offsets: " << offsets << "\n";
    if (!pattern_file.empty())
      std::cout << "# explicit pattern file: " << pattern_file << "\n";
    std::cout << "# pattern values: global " << global_list << " | column " << column_list
              << " | fixed " << fixed_list << " | combined " << combined_list << " | level "
              << level << "\n";
    std::cout << "# precond: " << precond;
    if (precond == "amg")
      std::cout << " (ruge_stuben eps_strong " << amg_strong << ", relax " << amg_relax << ", npre "
                << amg_npre << ", npost " << amg_npost << ", coarse " << amg_coarse
                << (amg_relax == "ilut" ? ", smoother ILUT(" + std::to_string(amg_ilut_fill) + ", " +
                                              std::to_string(amg_ilut_droptol) + ")"
                                        : std::string())
                << ")";
    else
      std::cout << " ILUT(fill " << ilut_fill << ", droptol " << ilut_droptol << ")";
    std::cout << "\n# gmres: M " << gmres_restart << ", maxit " << gmres_maxit << ", tol "
              << gmres_tol << ", precond side " << precond_side << ", solve reps " << solve_reps
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
            << "            default left; right needs transposes, see sam.hpp mapSide\n"
            << "-e <name> : pattern extension - materialized | fused (computed inside the map)\n"
            << "-c <int> : last source index; sources -b..-c form the sequence (default: -b only)\n"
            << "-r <list> : update modes - sam, reuse, recompute, periodic, or all (default sam)\n"
            << "-R <int> : periodic mode recomputes the preconditioner every R steps\n"
            << "-o <list> : stencils, [name=]d,d,... separated by ';' - one StencilPattern run each\n"
            << "-f <files> : comma separated CSR matrices, each structure run as an ExplicitPattern\n"
            << "--global <list>     global thresholds (default 0.3)\n"
            << "--column <list>     row-relative thresholds tau (default 0.7,0.8,0.9)\n"
            << "--fixed <list>      fixed nonzeros per row lfil (default 3,5,7)\n"
            << "--combined <list>   global:tau pairs (default 0.1:0.8)\n"
            << "--level <int>       extension power S^level (default 2)\n"
            << "--precond <name>    ilut | amg (default ilut)\n"
            << "--ilut-fill <int>, --ilut-droptol <real>   ILUT preconditioner (250, 1e-3)\n"
            << "--amg-relax <name>  jacobi | gauss_seidel | ilu0 | ilut (default gauss_seidel)\n"
            << "--amg-strong <real>, --amg-npre <int>, --amg-npost <int>, --amg-coarse <int>\n"
            << "--amg-ilut-fill <int>, --amg-ilut-droptol <real>   ILUT smoother (3, 1e-3)\n"
            << "--gmres-restart <int>, --gmres-maxit <int>, --gmres-tol <real> (2000, 2000, 1e-8)\n"
            << "--precond-side <name>   left | right (default right)\n"
            << "--solve-reps <int>  timed GMRES solves per record (default 3)\n"
            << "-h : display help message"
            << std::endl;
}

/// Parses command-line options with getopt_long().
inline void parseargs(int argc, char** argv, config_t& cfg) {
  enum : int {
    o_global = 1000, o_column, o_fixed, o_combined, o_level, o_precond, o_ilut_fill,
    o_ilut_droptol, o_amg_relax, o_amg_strong, o_amg_npre, o_amg_npost, o_amg_coarse,
    o_amg_ilut_fill, o_amg_ilut_droptol, o_gmres_restart, o_gmres_maxit, o_gmres_tol,
    o_precond_side, o_solve_reps
  };
  static const option long_opts[] = {
    {"global", required_argument, nullptr, o_global},
    {"column", required_argument, nullptr, o_column},
    {"fixed", required_argument, nullptr, o_fixed},
    {"combined", required_argument, nullptr, o_combined},
    {"level", required_argument, nullptr, o_level},
    {"precond", required_argument, nullptr, o_precond},
    {"ilut-fill", required_argument, nullptr, o_ilut_fill},
    {"ilut-droptol", required_argument, nullptr, o_ilut_droptol},
    {"amg-relax", required_argument, nullptr, o_amg_relax},
    {"amg-strong", required_argument, nullptr, o_amg_strong},
    {"amg-npre", required_argument, nullptr, o_amg_npre},
    {"amg-npost", required_argument, nullptr, o_amg_npost},
    {"amg-coarse", required_argument, nullptr, o_amg_coarse},
    {"amg-ilut-fill", required_argument, nullptr, o_amg_ilut_fill},
    {"amg-ilut-droptol", required_argument, nullptr, o_amg_ilut_droptol},
    {"gmres-restart", required_argument, nullptr, o_gmres_restart},
    {"gmres-maxit", required_argument, nullptr, o_gmres_maxit},
    {"gmres-tol", required_argument, nullptr, o_gmres_tol},
    {"precond-side", required_argument, nullptr, o_precond_side},
    {"solve-reps", required_argument, nullptr, o_solve_reps},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "n:x:y:k:t:a:b:p:u:m:s:e:c:r:R:o:f:h", long_opts, 
          nullptr)) != -1)
  {
    switch (opt) {
    case o_global: cfg.global_list = optarg; break;
    case o_column: cfg.column_list = optarg; break;
    case o_fixed: cfg.fixed_list = optarg; break;
    case o_combined: cfg.combined_list = optarg; break;
    case o_level: cfg.level = atoi(optarg); break;
    case o_precond: cfg.precond = optarg; break;
    case o_ilut_fill: cfg.ilut_fill = atoi(optarg); break;
    case o_ilut_droptol: cfg.ilut_droptol = atof(optarg); break;
    case o_amg_relax: cfg.amg_relax = optarg; break;
    case o_amg_strong: cfg.amg_strong = atof(optarg); break;
    case o_amg_npre: cfg.amg_npre = atoi(optarg); break;
    case o_amg_npost: cfg.amg_npost = atoi(optarg); break;
    case o_amg_coarse: cfg.amg_coarse = atoi(optarg); break;
    case o_amg_ilut_fill: cfg.amg_ilut_fill = atoi(optarg); break;
    case o_amg_ilut_droptol: cfg.amg_ilut_droptol = atof(optarg); break;
    case o_gmres_restart: cfg.gmres_restart = atoi(optarg); break;
    case o_gmres_maxit: cfg.gmres_maxit = atoi(optarg); break;
    case o_gmres_tol: cfg.gmres_tol = atof(optarg); break;
    case o_precond_side: cfg.precond_side = optarg; break;
    case o_solve_reps: cfg.solve_reps = atoi(optarg); break;
    case '?':
      std::cerr << "error: unknown or incomplete option; see -h" << std::endl;
      std::exit(2);
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
    case 'c':
      cfg.last_idx = atoi(optarg);
      break;
    case 'r':
      cfg.modes = std::string(optarg);
      break;
    case 'R':
      cfg.period = atoi(optarg);
      break;
    case 'o':
      cfg.offsets = std::string(optarg);
      break;
    case 'f':
      cfg.pattern_file = std::string(optarg);
      break;
    case 'h':
      usage();
      break;
    }
  }
}

/// Selected sparsification variant.
struct pattern_variant {
  bool union_candidate = false;
  const char* name = "source";
};

/// Parses -u, accepting `source`, `union`. Default is `source`.
inline pattern_variant parse_pattern_variant(const std::string &s) {
  if (s == "source")
    return {false, "source"};
  if (s == "union")
    return {true,  "union"};
  std::cerr << "Unknown pattern variant '" << s << "', using source. Valid: source | union"
            << std::endl;
  return {};
}

inline spgemm_accumulator parse_accumulator(const std::string &s) {
  if (s == "dense")
    return spgemm_accumulator::dense;
  if (s == "hash") 
    return spgemm_accumulator::hash;
  std::cerr << "Unknown accumulator '" << s << "', using dense. Valid: dense | hash" << std::endl;
  return spgemm_accumulator::dense;
}

inline bool parse_fused(const std::string &s) {
  if (s == "fused")
    return true;
  if (s == "materialized")
    return false;
  std::cerr << "Unknown extension '" << s << "', using materialized. Valid: materialized | fused"
            << std::endl;
  return false;
}

/// Whether a comma-separated list selects `key`: "all", or `key` as one of its items.
inline bool list_selects(const std::string &list, const std::string &key) {
  if (list == "all")
    return true;
  std::stringstream ss(list);
  for (std::string item; std::getline(ss, item, ',');) {
    if (item == key)
      return true;
  }
  return false;
}

inline std::vector<long> parse_offsets(const std::string &s) {
  std::vector<long> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) {
    char *end = nullptr;
    const long d = std::strtol(item.c_str(), &end, 10);
    if (item.empty() || *end != '\0') {
      std::cerr << "error: stencil offset '" << item << "' is not an integer" << std::endl;
      std::exit(1);
    }
    out.push_back(d);
  }
  return out;
}
