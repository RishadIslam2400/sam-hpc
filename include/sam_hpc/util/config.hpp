#pragma once

#include <iostream>
#include <string>
#include <unistd.h>

// store all our command-line arguments
struct config_t {
    std::string name;        // name of the program
    std::string matrix_dir; // path to the source matrix
    std::string rhs_dir; // path to the taget matrix
    int iters;               // number of iteration for the tests
    int threads;             // specify number of threads
    int target_idx;          // matrix_<n>.txt used as the target (A0); <0 = benchmark default
    int source_idx;          // matrix_<n>.txt used as the source (Ai); <0 = benchmark default
    std::string patterns;    // comma separated pattern names to run, or "all"
    std::string variant;     // which matrix gets sparsified: source | union
    std::string dataset;     // pipeline_bench preset: top_opt | weather

    config_t () {
        name = "default name";
        matrix_dir = "default path";
        rhs_dir = "default path";
        iters = 10;
        threads = 1; // single thread execution by default
        target_idx = -1;   // unset; each benchmark applies its own default
        source_idx = -1;
        patterns = "all";
        variant = "source";
        dataset = "top_opt";
    }

    void print() {
        std::cout << "# name, matrix_dir, rhs_dir, iters, threads:\n";
        std::cout << name << ", " << matrix_dir << ", " << rhs_dir << ", " << iters << ", " << threads << std::endl;
        std::cout << "# pattern variant: " << variant << ", dataset: " << dataset << std::endl;
    }
};

// report on how to use the command line to configure this program
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
              << "-d <name> : pipeline_bench dataset preset - top_opt | weather (default top_opt)\n"
              << "-h : display help message"
              << std::endl;
}

// parse command line arguments using get-opt()
inline void parseargs(int argc, char** argv, config_t& cfg) {
    int opt;
    while ((opt = getopt(argc, argv, "n:x:y:k:t:a:b:p:u:d:h")) != -1) {
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
        case 'd':
            cfg.dataset = std::string(optarg);
            break;
        case 'h':
            usage();
            break;
        }
    }
}

/// Which matrix the drop rules are applied to.
///
///   source  sparsify A_k alone
///   union   merge A_k and A_0 keeping the larger-magnitude entry, then sparsify the result
struct pattern_variant {
    bool union_candidate = false;
    const char* name = "source";
};

inline pattern_variant parse_pattern_variant(const std::string &s) {
    if (s == "source")                    return {false, "source"};
    if (s == "union" || s == "union_max") return {true,  "union"};
    std::cerr << "Unknown pattern variant '" << s << "', using source. Valid: source | union"
              << std::endl;
    return {};
}

