#pragma once

#include <list>

#include "sam_hpc/precond/skyline_lu.hpp"
#include "sam_hpc/core/CSRMatrix.hpp"
#include "sam_hpc/core/linearAlgebra.hpp"

template <typename T, template <class> class Coarsening, template <class> class Relax>
class amg {
public:
    typedef Coarsening<T> coarsening_type;
    typedef Relax<T> relax_type;

    // Parameters of the method
    /**
     * the params struct contains paramters for each component of the method as well as
     * some universal parameters
     */
    struct params {
        typedef typename coarsening_type::params coarsening_params;
        typedef typename relax_type::params relax_params;

        coarsening_params coarsening; // Coarsening parameters
        relax_params relax;          // Relaxation parameters

        // Specifies when level is coarse enough to be solved directly
        /**
         * If number  of variables at a next level in the hierarchy becomes lower than this
         * threshold, then the hierarchy construction is stopped and the linear system is solved
         * directly at this level
         */
        unsigned coarse_enough;

        // Use direct solver at the coarse level.
        /**
         * When set the coarsest level is solved with a direct solver. Otherwise a smoother is used as a solver
         */
        bool direct_coarse;

        // Maximum number of levels
        /**
         * If this number is reached while the size of the last level is greater than coarse_enough,
         * then the corasest level will not be solved exactly, but will use a smoother
         */
        unsigned max_levels;

        // Number of pre-relxations
        unsigned npre;

        // Number of post-relaxations
        unsigned npost;

        // Number of cycles(1 for V-cycle, 2 for W-cycle, etc.).
        unsigned ncycle;

        // Number of cycles to make as part of preconditioning
        unsigned pre_cycles;

        // Keep matrices in internal format to allow for quick rebuild of the hierarchy
        bool allow_rebuild;

        params() : coarse_enough(skyline_lu<T>::coarse_enough()),
                   direct_coarse(true),
                   max_levels(std::numeric_limits<unsigned>::max()),
                   npre(1), npost(1), ncycle(1), pre_cycles(1),
                   allow_rebuild(true)
        { }
    } prm;

    // Builds the AMG hierarchy for the system matrix
    /**
     * The input matrix is copied here and is safe to delete afterwards
     * 
     * \param A The system matrix. This is a csr matrix
     * \param p AMG parameters
     */
    amg(const CSRMatrix<T>& M, const params& p = params()) : prm(p) {
        auto A = std::make_shared<CSRMatrix<T>>(M);
        sortRows(*A);
        do_init(A);
    }

    /// Builds the AMG hierarchy for the system matrix.
    /**
     * The shared pointer to the input matrix is passed here. The matrix will not be copied 
     * and should out-live the amg instance. The matrix should be in CSR format.
     *
     * \param A The system matrix.
     * \param p AMG parameters.
     */
    amg(std::shared_ptr<CSRMatrix<T>> A, const params &p = params()) : prm(p) {
        do_init(A);
    }

    /// Rebuild the hierarchy using the new system matrix.
    /**
     * This requires for prm.allow_rebuild to be set. The transfer
     * operators created during the initial setup are reused.
     */
    void rebuild(const CSRMatrix<T>& M) {
        auto A = std::make_shared<CSRMatrix<T>>(M);
        sortRows(*A);
        rebuild(A);
    }

    /// Rebuild the hierarchy using the new system matrix.
    /**
     * This requires for prm.allow_rebuild to be set. The transfer
     * operators created during the initial setup are reused.
     */
    void rebuild(std::shared_ptr<CSRMatrix<T>> A) {
        assert(prm.allow_rebuild && "allow_rebuild is not set");
        assert((A->m_rows == system_matrix().m_rows && A->m_rows == A->m_cols) && "Matrix dimensions differ from original ones!");

        coarsening_type C(prm.coarsening);
        for (auto & level : levels) {
            A = level.rebuild(A, C, prm);
        }
    }

    /// Performs single V-cycle for the given right-hand side and solution.
    /**
     * \param rhs Right-hand side vector.
     * \param x   Solution vector.
     */
    void cycle(const std::vector<T> &rhs, std::vector<T> &x) const {
        cycle(levels.begin(), rhs, x);
    }

    /// Performs single V-cycle after clearing x.
    /**
     * This is intended for use as a preconditioning procedure.
     *
     * \param rhs Right-hand side vector.
     * \param x   Solution vector.
     */
    void apply(const std::vector<T> &rhs, std::vector<T> &x) const {
        if (prm.pre_cycles) {
            std::fill(x.begin(), x.end(), T(0));
            for (unsigned i = 0; i < prm.pre_cycles; ++i)
                cycle(rhs, x);
        } else {
            x = rhs; // not using parallel copy
        }
    }

    /// return the system matrix from the finest level
    std::shared_ptr<CSRMatrix<T>> system_matrix_ptr() const {
        return levels.front().A;
    }

    const CSRMatrix<T>& system_matrix() const {
        return *system_matrix_ptr();
    }

    // TODO: implement bytes() method

private:
    struct level {
        size_t m_rows, m_nonzeros;

        mutable std::vector<T> f; // rhs vector. Holds the residual from finer levels in coarser levels
        mutable std::vector<T> u; // solution vector that is interpolated to finer levels
        mutable std::vector<T> t; // temprary storage

        std::shared_ptr<CSRMatrix<T>> A; // System matrix for the level
        std::shared_ptr<CSRMatrix<T>> P; // Prologation operator
        std::shared_ptr<CSRMatrix<T>> R; // Restriction operator
 
        std::shared_ptr<CSRMatrix<T>> bP; // used for rebuilding
        std::shared_ptr<CSRMatrix<T>> bR; // copies of P and R
 
        std::shared_ptr<skyline_lu<T>> solve; 
        std::shared_ptr<relax_type> relax;

        // todo: implement bytes() method

        level() : m_rows(0), m_nonzeros(0) {}
        level(std::shared_ptr<CSRMatrix<T>>& A, params& prm) : m_rows(A->m_rows), m_nonzeros(A->m_nnz), f(m_rows, T(0)), u(m_rows, T(0)), t(m_rows, T(0)) {
            this->A = A;
            relax = std::make_shared<relax_type>(*A, prm.relax);
        }

        // Performs one step of coarsening
        std::shared_ptr<CSRMatrix<T>> step_down(std::shared_ptr<CSRMatrix<T>> A, coarsening_type& C, bool allow_rebuild) {
            std::shared_ptr<CSRMatrix<T>> P, R;

            try {
                // Generate the transfer operators
                std::tie(P, R) = C.transfer_operators(*A);
            } catch (const std::runtime_error& e) {
                return std::shared_ptr<CSRMatrix<T>>();
            }

            sortRows(*P);
            sortRows(*R);

            if (allow_rebuild) {
                bP = P;
                bR = R;
            }

            this->P = P;
            this->R = R;

            // Compute the system matrix from the next coarser level
            A = C.coarse_operator(*A, *P, *R);
            sortRows(*A);

            return A;
        }

        // Coarsest grid
        void create_coarse(std::shared_ptr<CSRMatrix<T>> A, bool single_level) {
            m_rows = A->m_rows;
            m_nonzeros = A->m_nnz;

            u.resize(m_rows, T(0));
            f.resize(m_rows, T(0));

            // direct solver for coarsest grid
            solve = std::make_shared<skyline_lu<T>>(*A);
            if (single_level) {
                this->A = A;
            }
        }

        std::shared_ptr<CSRMatrix<T>> rebuild(std::shared_ptr<CSRMatrix<T>> A, const coarsening_type& C, const params& prm) {
            if (this->A) {
                this->A = A;
            }

            if (relax) {
                relax = std::make_shared<relax_type>(*A, prm.relax);
            }

            if (solve) {
                solve = std::make_shared<skyline_lu<T>>(*A);
            }

            if (bP && bR) {
                A = C.coarse_operator(*A, *bP, *bR);
                sortRows(*A);
            }

            return A;
        }

        size_t rows() const {
            return m_rows;
        }

        size_t nonzeros() const {
            return m_nonzeros;
        }
    };

    typedef typename std::list<level>::const_iterator level_iterator;
    std::list<level> levels;

    // Set up the multigrid grid hierarchy
    void do_init(std::shared_ptr<CSRMatrix<T>> A) {
        assert(A->m_rows == A->m_cols && "Matrix should be square!");
        bool direct_coarse_solve = true;

        coarsening_type C(prm.coarsening);

        while(A->m_rows > prm.coarse_enough) {
            levels.push_back(level(A, prm));

            if (levels.size() >= prm.max_levels)
                break;

            // calculate the transfer operators and the coarse grid system matrix for next level
            A = levels.back().step_down(A, C, prm.allow_rebuild);
            if (!A) {
                // Zero-sized coarse level. Probably the system matrix on
                // this level is diagonal, should be easily solvable with a
                // couple of smoother iterations.
                direct_coarse_solve = false;
                break;
            }
        }

        // For early exit the final coarsest grid can't be solved directly
        if (!A || A->m_rows > prm.coarse_enough) {
            // The coarse matrix is still to be solved directly
            direct_coarse_solve = false;
        }

        if (direct_coarse_solve) {
            if (prm.direct_coarse) {
                level l;
                // create the specialized coarsest level
                l.create_coarse(A, levels.empty());
                levels.push_back(l);
            } else {
                levels.push_back(level(A, prm));
            }
        }
    }

    void cycle(level_iterator lvl, const std::vector<T>& rhs, std::vector<T>& x) const {
        level_iterator nxt = lvl, end = levels.end();
        ++nxt;

        if (nxt == end) {
            if (lvl->solve) {
                (*lvl->solve)(rhs, x);

            } else {
                for(size_t i = 0; i < prm.npre;  ++i)
                    lvl->relax->apply_pre(*lvl->A, rhs, x, lvl->t);
                for(size_t i = 0; i < prm.npost; ++i)
                    lvl->relax->apply_post(*lvl->A, rhs, x, lvl->t);
            }
        } else {
            // determines V or W cycle
            for (size_t j = 0; j < prm.ncycle; ++j) {
                for(size_t i = 0; i < prm.npre; ++i)
                    lvl->relax->apply_pre(*lvl->A, rhs, x, lvl->t); // pre-smoothing

                residual(rhs, *lvl->A, x, lvl->t); // calculate residual
                spmv(T(1), *lvl->R, lvl->t, T(0), nxt->f); //  restrict the residual

                // recursive call to apply multigrid step to the next level
                std::fill(nxt->u.begin(), nxt->u.end(), T(0)); // parallel execution?
                cycle(nxt, nxt->f, nxt->u);

                // Apply prolongation operator
                spmv(T(1), *lvl->P, nxt->u, T(1), x);

                // post smoothing
                for(size_t i = 0; i < prm.npost; ++i)
                    lvl->relax->apply_post(*lvl->A, rhs, x, lvl->t);
            }
        }
    }

    template <class B, template <class> class C, template <class> class R>
    friend std::ostream& operator<<(std::ostream &os, const amg<B, C, R> &a);
};

/// Sends information about the AMG hierarchy to output stream.
template <class B, template <class> class C, template <class> class R>
std::ostream& operator<<(std::ostream &os, const amg<B, C, R> &a) {
    // Not required for now
    /**
     * typedef typename amg<B, C, R>::level level;
    ios_saver ss(os);

    size_t sum_dof = 0;
    size_t sum_nnz = 0;
    size_t sum_mem = 0;

    for(const level &lvl : a.levels) {
        sum_dof += lvl.rows();
        sum_nnz += lvl.nonzeros();
        sum_mem += lvl.bytes();
    }

    os << "Number of levels:    "   << a.levels.size()
        << "\nOperator complexity: " << std::fixed << std::setprecision(2)
        << 1.0 * sum_nnz / a.levels.front().nonzeros()
        << "\nGrid complexity:     " << std::fixed << std::setprecision(2)
        << 1.0 * sum_dof / a.levels.front().rows()
        << "\nMemory footprint:    " << human_readable_memory(sum_mem)
        << "\n\n"
           "level     unknowns       nonzeros      memory\n"
           "---------------------------------------------\n";

    size_t depth = 0;
    for(const level &lvl : a.levels) {
        os << std::setw(5)  << depth++
            << std::setw(13) << lvl.rows()
            << std::setw(15) << lvl.nonzeros()
            << std::setw(12) << human_readable_memory(lvl.bytes())
            << " (" << std::setw(5) << std::fixed << std::setprecision(2)
            << 100.0 * lvl.nonzeros() / sum_nnz
            << "%)" << std::endl;
    }
    */
    return os;
}