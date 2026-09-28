#pragma once

#include "sam_hpc/core/CSRMatrix.hpp"

template <typename T, bool reverse = false>
struct cuthill_mckee {
    static void get(const CSRMatrix<T> &A, std::vector<ptrdiff_t> &perm) {
        const ptrdiff_t n = A.m_rows;

        /* The data structure used to sort and traverse the level sets:
         *
         * The current level set is currentLevelSet;
         * In this level set, there are nodes with degrees from 0 (not really
         * useful) to maxDegreeInCurrentLevelSet.
         * firstWithDegree[i] points to a node with degree i, or to -1 if it
         * does not exist. nextSameDegree[firstWithDegree[i]] points to the
         * second node with that degree, etc.
         * While the level set is being traversed, the structure for the next
         * level set is generated; nMDICLS will be the next
         * maxDegreeInCurrentLevelSet and nFirstWithDegree will be
         * firstWithDegree.
         */
        ptrdiff_t initialNode = 0; // node to start search
        ptrdiff_t maxDegree   = 0;

        std::vector<ptrdiff_t> degree(n);
        // Tracks which level set each node belongs to.
        std::vector<ptrdiff_t> levelSet(n, 0);
        // nextSameDegree[i] gives the next node in the same level set that has the same degree as node i
        std::vector<ptrdiff_t> nextSameDegree(n, -1);

        maxDegree = tbb::parallel_reduce(tbb::blocked_range<ptrdiff_t>(0, n), ptrdiff_t(0),
            [&](const tbb::blocked_range<ptrdiff_t>& r, ptrdiff_t local_max) -> ptrdiff_t {
                for (ptrdiff_t i = r.begin(); i < r.end(); ++i) {
                    ptrdiff_t row_width = 0;
                    for (auto a = A.rowBegin(i); a; ++a) {
                        row_width++;
                    }
                    degree[i] = row_width;
                    local_max = std::max(local_max, degree[i]);
                }
                return local_max;
            },
            [](ptrdiff_t a, ptrdiff_t b) -> ptrdiff_t {
                return std::max(a, b);
            }
        );

        // firstWithDegree[d] gives the first node in the current level set with degree d
        std::vector<ptrdiff_t> firstWithDegree(maxDegree + 1, -1);
        // used to build the next level set while the current one is being processed
        std::vector<ptrdiff_t> nFirstWithDegree(maxDegree + 1);

        // Initialize the first level set, made up by initialNode alone
        perm[0] = initialNode;
        ptrdiff_t currentLevelSet = 1;
        levelSet[initialNode] = currentLevelSet;
        ptrdiff_t maxDegreeInCurrentLevelSet = degree[initialNode];
        firstWithDegree[maxDegreeInCurrentLevelSet] = initialNode;

        // Main loop
        for (ptrdiff_t next = 1; next < n; ) {
            ptrdiff_t nMDICLS = 0;
            std::fill(nFirstWithDegree.begin(), nFirstWithDegree.end(), -1);
            bool empty = true; // used to detect different connected components

            ptrdiff_t firstVal  = reverse ? maxDegreeInCurrentLevelSet : 0;
            ptrdiff_t finalVal  = reverse ? -1 : maxDegreeInCurrentLevelSet + 1;
            ptrdiff_t increment = reverse ? -1 : 1;

            for(ptrdiff_t soughtDegree = firstVal; soughtDegree != finalVal; soughtDegree += increment) {
                ptrdiff_t node = firstWithDegree[soughtDegree];
                while (node > 0) {
                    // Visit neighbors
                    for(auto a = A.rowBegin(node); a; ++a) {
                        ptrdiff_t c = a.col();
                        if (levelSet[c] == 0) {
                            levelSet[c] = currentLevelSet + 1;
                            perm[next] = c;
                            ++next;
                            empty = false; // this level set is not empty
                            nextSameDegree[c] = nFirstWithDegree[degree[c]];
                            nFirstWithDegree[degree[c]] = c;
                            nMDICLS = std::max(nMDICLS, degree[c]);
                        }
                    }
                    node = nextSameDegree[node];
                }
            }

            ++currentLevelSet;
            maxDegreeInCurrentLevelSet = nMDICLS;
            for(ptrdiff_t i = 0; i <= nMDICLS; ++i)
                firstWithDegree[i] = nFirstWithDegree[i];

            if (empty) {
                // The graph contains another connected component that we
                // cannot reach.  Search for a node that has not yet been
                // included in a level set, and start exploring from it.
                bool found = false;
                for(ptrdiff_t i = 0; i < n; ++i) {
                    if (levelSet[i] == 0) {
                        perm[next] = i;
                        ++next;
                        levelSet[i] = currentLevelSet;
                        maxDegreeInCurrentLevelSet = degree[i];
                        firstWithDegree[maxDegreeInCurrentLevelSet] = i;
                        found = true;
                        break;
                    }
                }
                assert(found && "Internal consistency error at skyline_lu");
            }
        }
    }
};