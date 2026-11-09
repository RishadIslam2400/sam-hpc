#!/usr/bin/env bash
# Usage:
#   scripts/run_experiments.sh [-p PROBLEMS] [-e EXPERIMENTS] [-o OUTDIR] [-n] [-s]
#
#   -p  problems, comma separated      cd2d,topopt,topopt_big,elast3d (default: all four)
#   -e  experiments, comma separated   see the list below     (default: all)
#   -o  output directory               (default: results/paper_<date>_<host>)
#   -n  dry run: print every command and the total count, run nothing
#   -s  skip the build and the unit tests (re-using an existing build-release/)
#
# Experiments:
#
#   params    Parameter study per sparsification family. One source/target pair, every
#             value in *_SWEEP, source and union variants. Gives nnz/row, map time,
#             map residual, iterations and GMRES residual per value, which is what picks
#             the operating point used everywhere else.
#   sequence  The main result. Every step of the sequence, at THREADS_MAIN, for:
#             baselines  S(A_0), S(A_k), mesh patterns Patt-0/Patt-1 (top_opt),
#												 `reuse` (P_0, no map) and `recompute` (a new preconditioner for every A_k)
#             proposed   global / column / fixed at the chosen values, source AND union variants.
#   scaling   Phase_bench on the pair at every count in THREADS, one process per count.
#             Phase split (merge / filter / extend / map), |I|, |J|, GFLOP/s, residual.
#   e2e       End-to-end time vs threads for one step.
#   level     Extension power S^l, l in LEVELS (E5), chosen column and fixed patterns.
#   side      Left (row-wise, ||N A_k - A_0||) vs right (column-wise) map.
#   amg       (cd2d only) the same pair with AMG as P_0 (sam / reuse / recompute). AMG setup is
#             about as cheap as the map, so recompute wins.
#   periodic  Recompute P every R steps on the long sequence: never recompute (sam) vs periodic for each R in PERIODS.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------------------
# Machine-level settings
# ---------------------------------------------------------------------------------------
THREADS="${THREADS:-1 12 24 48 72 96}"   # scaling and e2e sweeps
THREADS_MAIN="${THREADS_MAIN:-24}"       # params, sequence, level, periodic
THREADS_SIDE="${THREADS_SIDE:-1 24 96}"  # left vs right map
LEVELS="${LEVELS:-1 2 3}"
MAP_REPS="${MAP_REPS:-5}"                # max map repetitions (min 3, 60 s budget)
SOLVE_REPS="${SOLVE_REPS:-3}"            # timed solves per record for params/e2e/level/side
SEQ_SOLVE_REPS="${SEQ_SOLVE_REPS:-1}"    # timed solves per record for sequence/periodic
PHASE_REPS="${PHASE_REPS:-5}"            # phase_bench repetitions
PARAMS_FAMILIES="${PARAMS_FAMILIES:-global,column,fixed}"  # families swept by params
NUMA="${NUMA:-1}"                        # 0 = no numactl memory policy
BUILD="${BUILD:-$ROOT/build-release}"

# ---------------------------------------------------------------------------------------
# Problem definitions.
# ---------------------------------------------------------------------------------------
problem_cd2d() {
	MATRIX_DIR="$ROOT/data/cd2d_seq/"
	RHS_DIR="$ROOT/data/cd2d_seq_rhs/"
	TARGET=1             # A_0: the first Newton system
	PAIR=11              # representative source for params/scaling/e2e/level/side
	SEQ_FIRST=2; SEQ_LAST=21
	LONG_TARGET=1; LONG_FIRST=2; LONG_LAST=21
	PERIODS="${CD2D_PERIODS:-4 8}"

	# ILUT(50, 1e-3)
	SOLVER="${CD2D_SOLVER:---precond ilut --ilut-fill 50 --ilut-droptol 1e-3 --gmres-restart 1000 --gmres-maxit 1000 --gmres-tol 1e-8 --precond-side right}"
	AMG_SOLVERS=(
		"--precond amg --amg-relax ilu0 --amg-strong 0.25 --amg-npre 1 --amg-npost 1"
		"--precond amg --amg-relax ilut --amg-ilut-fill 3 --amg-ilut-droptol 1e-3 --amg-strong 0.25 --amg-npre 1 --amg-npost 1"
	)
	AMG_GMRES="--gmres-restart 1000 --gmres-maxit 1000 --gmres-tol 1e-8 --precond-side right"

	GLOBAL_SWEEP="${CD2D_GLOBAL_SWEEP:-0.1,0.2,0.3,0.4,0.5}"
	COLUMN_SWEEP="${CD2D_COLUMN_SWEEP:-0.05,0.1,0.2,0.3,0.5}"
	FIXED_SWEEP="${CD2D_FIXED_SWEEP:-1,2,3,4,5}"
	
	# Choose after paramter sweep
	GLOBAL="${CD2D_GLOBAL:-0.2}"
	COLUMN="${CD2D_COLUMN:-0.3}"
	FIXED="${CD2D_FIXED:-2}"
	STENCIL=""           # the 5-point stencil IS S(A_0) here
	EXPLICIT="${CD2D_EXPLICIT:-}"
}

problem_topopt() {
	MATRIX_DIR="$ROOT/data/top_opt_matrices_small_csr/"
	RHS_DIR="$ROOT/data/top_opt_rhs_small/"
	TARGET=40            
	PAIR=50
	SEQ_FIRST=41; SEQ_LAST=50
	LONG_TARGET=10; LONG_FIRST=11; LONG_LAST=50
	PERIODS="${TOPOPT_PERIODS:-5 10 20}"

	SOLVER="${TOPOPT_SOLVER:---precond ilut --ilut-fill 250 --ilut-droptol 1e-3 --gmres-restart 2000 --gmres-maxit 2000 --gmres-tol 1e-8 --precond-side right}"
	AMG_SOLVERS=()
	AMG_GMRES=""

	GLOBAL_SWEEP="${TOPOPT_GLOBAL_SWEEP:-0.01,0.05,0.1,0.2,0.3,0.5}"
	COLUMN_SWEEP="${TOPOPT_COLUMN_SWEEP:-0.05,0.1,0.2,0.3,0.4,0.5}"
	FIXED_SWEEP="${TOPOPT_FIXED_SWEEP:-2,3,4,5,6,7}"
	
	GLOBAL="${TOPOPT_GLOBAL:-0.2}"
	COLUMN="${TOPOPT_COLUMN:-0.4}"
	FIXED="${TOPOPT_FIXED:-4}"
	# Patt-0: 7-point stencil {0,+-3,+-300,+-6300}. 
	# Patt-1: custom stencil {0,+-1,+-300,+-6000}.
	STENCIL="${TOPOPT_STENCIL:-Patt-0=0,-3,3,-300,300,-6300,6300;Patt-1=0,-1,1,-300,300,-6000,6000}"
	EXPLICIT="${TOPOPT_EXPLICIT:-}"
}

problem_topopt_big() {
	MATRIX_DIR="$ROOT/data/top_opt_matrices_paper_big_csr/"
	RHS_DIR="$ROOT/data/top_opt_paper_rhs_big/"
	TARGET=20
	PAIR=31
	SEQ_FIRST=21; SEQ_LAST=31
	LONG_TARGET=20; LONG_FIRST=21; LONG_LAST=31
	PERIODS="${TOPOPT_BIG_PERIODS:-5}"
	SUPPORTED="${TOPOPT_BIG_EXPERIMENTS:-sequence,scaling,e2e}"
	SOLVER="${TOPOPT_BIG_SOLVER:---precond ilut --ilut-fill 80 --ilut-droptol 1e-3 --gmres-restart 2000 --gmres-maxit 2000 --gmres-tol 1e-8 --precond-side right}"
	AMG_SOLVERS=()
	AMG_GMRES=""

	GLOBAL_SWEEP="${TOPOPT_BIG_GLOBAL_SWEEP:-0.1,0.3,0.5}"
	COLUMN_SWEEP="${TOPOPT_BIG_COLUMN_SWEEP:-0.1,0.4,0.5}"
	FIXED_SWEEP="${TOPOPT_BIG_FIXED_SWEEP:-3,5,7}"
	
	GLOBAL="${TOPOPT_BIG_GLOBAL:-${TOPOPT_GLOBAL:-0.2}}"
	COLUMN="${TOPOPT_BIG_COLUMN:-${TOPOPT_COLUMN:-0.4}}"
	FIXED="${TOPOPT_BIG_FIXED:-${TOPOPT_FIXED:-4}}"
	STENCIL="${TOPOPT_BIG_STENCIL:-Patt-0=0,-3,3,-450,450,-13950,13950;Patt-1=0,-1,1,-450,450,-13500,13500}"
	EXPLICIT="${TOPOPT_BIG_EXPLICIT:-}"
}

problem_elast3d() {
	MATRIX_DIR="$ROOT/data/elast3d_seq/"
	RHS_DIR="$ROOT/data/elast3d_seq_rhs/"
	TARGET=1             
	PAIR=35              
	SEQ_FIRST=2; SEQ_LAST=35
	LONG_TARGET=1; LONG_FIRST=2; LONG_LAST=35
	PERIODS="${ELAST3D_PERIODS:-5 10}"

	SOLVER="${ELAST3D_SOLVER:---precond ilut --ilut-fill 80 --ilut-droptol 1e-3 --gmres-restart 2000 --gmres-maxit 2000 --gmres-tol 1e-8 --precond-side right}"
	AMG_SOLVERS=()
	AMG_GMRES=""
	SUPPORTED="${ELAST3D_EXPERIMENTS:-params,sequence,scaling,e2e,level,periodic}"

	GLOBAL_SWEEP="${ELAST3D_GLOBAL_SWEEP:-0.01,0.05,0.1,0.2,0.3,0.5}"
	COLUMN_SWEEP="${ELAST3D_COLUMN_SWEEP:-0.05,0.1,0.2,0.3,0.4,0.5}"
	FIXED_SWEEP="${ELAST3D_FIXED_SWEEP:-2,3,4,5,6,7,8,10,12}"
	
	GLOBAL="${ELAST3D_GLOBAL:-0.2}"
	COLUMN="${ELAST3D_COLUMN:-0.4}"
	FIXED="${ELAST3D_FIXED:-8}"
	STENCIL="${ELAST3D_STENCIL:-Patt-0=0,-3,3,-288,288,-7200,7200;Patt-1=0,-1,1,-288,288,-6912,6912}"
	EXPLICIT="${ELAST3D_EXPLICIT:-}"
}

# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------
PROBLEMS="cd2d,topopt,topopt_big,elast3d"
EXPERIMENTS="params,sequence,scaling,e2e,level,side,amg,periodic"
OUTDIR=""
DRY=0
SKIP_BUILD=0
while getopts "p:e:o:nsh" opt; do
	case "$opt" in
		p) PROBLEMS="$OPTARG" ;;
		e) EXPERIMENTS="$OPTARG" ;;
		o) OUTDIR="$OPTARG" ;;
		n) DRY=1 ;;
		s) SKIP_BUILD=1 ;;
		h) sed -n '2,60p' "$0"; exit 0 ;;
		*) exit 2 ;;
	esac
done
OUTDIR="${OUTDIR:-$ROOT/results/paper_$(date +%Y-%m-%d_%H-%M)_$(hostname -s)}"
PIPE="$BUILD/benchmarks/pipeline_bench"
PHASE="$BUILD/benchmarks/phase_bench"

has() { [[ ",$1," == *",$2,"* ]]; }

# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
PHYS_CORES=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
for t in $THREADS $THREADS_MAIN $THREADS_SIDE; do
	if [ "$t" -gt "$PHYS_CORES" ]; then
		echo "error: $t threads requested but this machine has $PHYS_CORES physical cores" >&2
		exit 1
	fi
done
if [ "$NUMA" != "0" ] && ! command -v numactl >/dev/null 2>&1; then
	echo "error: numactl not found; install it or set NUMA=0" >&2
	exit 1
fi

# ---------------------------------------------------------------------------------------
# Running one benchmark invocation
# ---------------------------------------------------------------------------------------
NRUNS=0
# run <logfile> <binary> <args...>: appends the command and its full output to <logfile>.
run() {
	local log="$1"; shift
	NRUNS=$((NRUNS + 1))
	if [ "$DRY" = 1 ]; then
		echo "[$NRUNS] $*"
		return
	fi
	mkdir -p "$(dirname "$log")"
	{
		echo "#### $(date '+%F %T')  $*"
	} >> "$log"
	echo "[$NRUNS $(date '+%T')] $(basename "$log" .log): ${*:2}" | cut -c1-200
	local rc=0
	if [ "$NUMA" != "0" ]; then
		SAM_PINNED="numactl --interleave=all" numactl --interleave=all "$@" >> "$log" 2>&1 || rc=$?
	else
		"$@" >> "$log" 2>&1 || rc=$?
	fi
	if [ "$rc" != 0 ]; then
		echo "    FAILED (exit $rc), see $log" | tee -a "$OUTDIR/FAILURES"
	fi
}

# Arguments every pipeline_bench call shares for the current problem.
pipe_common() {
  echo -x "$MATRIX_DIR" -y "$RHS_DIR" -k "$MAP_REPS" $SOLVER
}
# Pattern values for the chosen operating point.
chosen() {
	local a="--global $GLOBAL --column $COLUMN --fixed $FIXED"
	[ -n "$STENCIL" ] && a="$a -o $STENCIL"
	[ -n "$EXPLICIT" ] && a="$a -f $EXPLICIT"
	echo "$a"
}
# Baseline families present for this problem.
baselines() {
	local p="target,simple"
	[ -n "$STENCIL" ] && p="$p,stencil"
	[ -n "$EXPLICIT" ] && p="$p,explicit"
	echo "$p"
}

# ---------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------
exp_params() {
	local d="$OUTDIR/$1/params"
	# Baselines once, then each family's sweep per variant.
	run "$d/baselines.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SOLVE_REPS" \
		-a "$TARGET" -b "$PAIR" -r sam,reuse,recompute -u source -p "$(baselines)" $(chosen)
	for v in source union; do
		run "$d/sweep_$v.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SOLVE_REPS" \
			-a "$TARGET" -b "$PAIR" -r sam -u "$v" -p "$PARAMS_FAMILIES" \
			--global "$GLOBAL_SWEEP" --column "$COLUMN_SWEEP" --fixed "$FIXED_SWEEP"
	done
}

exp_sequence() {
	local d="$OUTDIR/$1/sequence"
	run "$d/baselines.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SEQ_SOLVE_REPS" \
		-a "$TARGET" -b "$SEQ_FIRST" -c "$SEQ_LAST" -r sam,reuse,recompute -u source \
		-p "$(baselines)" $(chosen)
	for v in source union; do
		run "$d/proposed_$v.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SEQ_SOLVE_REPS" \
				-a "$TARGET" -b "$SEQ_FIRST" -c "$SEQ_LAST" -r sam -u "$v" -p global,column,fixed $(chosen)
	done
}

exp_scaling() {
	local d="$OUTDIR/$1/scaling"
	for v in source union; do
		local fam="global,column,fixed"
		[ "$v" = source ] && fam="target,$fam"
		for t in $THREADS; do
			run "$d/phase_${v}_t$t.log" "$PHASE" -x "$MATRIX_DIR" -k "$PHASE_REPS" -t "$t" \
				-a "$TARGET" -b "$PAIR" -u "$v" -p "$fam" $(chosen)
		done
	done
}

exp_e2e() {
	local d="$OUTDIR/$1/e2e"
	for t in $THREADS; do
		run "$d/e2e_t$t.log" "$PIPE" $(pipe_common) -t "$t" --solve-reps "$SOLVE_REPS" \
			-a "$TARGET" -b "$PAIR" -r sam,reuse,recompute -u union \
			-p "$(baselines),global,column,fixed" $(chosen)
	done
}

exp_level() {
	local d="$OUTDIR/$1/level"
	for l in $LEVELS; do
		run "$d/level_$l.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SOLVE_REPS" \
				-a "$TARGET" -b "$PAIR" -r sam -u union -p global,column,fixed $(chosen) --level "$l"
	done
}

exp_side() {
	local d="$OUTDIR/$1/side"
	for t in $THREADS_SIDE; do
		for m in left right; do
			run "$d/side_${m}_t$t.log" "$PIPE" $(pipe_common) -t "$t" --solve-reps "$SOLVE_REPS" \
				-a "$TARGET" -b "$PAIR" -r sam -u union -m "$m" -p "target,column" $(chosen)
		done
	done
}

exp_amg() {
	local d="$OUTDIR/$1/amg"
	local i=0
	for solver in "${AMG_SOLVERS[@]}"; do
		i=$((i + 1))
		run "$d/amg_$i.log" "$PIPE" -x "$MATRIX_DIR" -y "$RHS_DIR" -k "$MAP_REPS" $solver $AMG_GMRES \
			-t "$THREADS_MAIN" --solve-reps "$SOLVE_REPS" -a "$TARGET" -b "$PAIR" -c "$SEQ_LAST" \
			-r sam,reuse,recompute -u union -p "target,column" $(chosen)
	done
}

exp_periodic() {
	local d="$OUTDIR/$1/periodic"
	run "$d/periodic_Rinf.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SEQ_SOLVE_REPS" \
		-a "$LONG_TARGET" -b "$LONG_FIRST" -c "$LONG_LAST" -r sam -u union -p column --column "$COLUMN"
	for R in $PERIODS; do
		run "$d/periodic_R$R.log" "$PIPE" $(pipe_common) -t "$THREADS_MAIN" --solve-reps "$SEQ_SOLVE_REPS" \
			-a "$LONG_TARGET" -b "$LONG_FIRST" -c "$LONG_LAST" -r periodic -R "$R" -u union \
			-p column --column "$COLUMN"
	done
}

# ---------------------------------------------------------------------------------------
# Collecting CSV rows: one file per (problem, experiment), with the run's file name as the
# first column so every row knows which invocation (thread count, variant, ...) it came from.
# ---------------------------------------------------------------------------------------
collect() {
	local dir="$1" exp="$2"
	[ -d "$dir" ] || return 0
	local first_log
	first_log=$(ls "$dir"/*.log 2>/dev/null | head -1) || true
	[ -n "$first_log" ] || return 0
	if [ "$exp" = scaling ]; then
		{ echo "run,pattern,variant,threads,phase,ms"
			for f in "$dir"/*.log; do grep '^CSV,' "$f" | sed "s|^CSV,|$(basename "$f" .log),|"; done
		} > "$dir/phases.csv"
		{ echo "run,pattern,variant,threads,map_nnz,residual_exact,residual_qr"
			for f in "$dir"/*.log; do grep '^CSVACC,' "$f" | sed "s|^CSVACC,|$(basename "$f" .log),|"; done
		} > "$dir/accuracy.csv"
		# The per-pattern |I|, |J|, GFLOP/s lines are not CSV; keep them greppable in one place.
		grep -h -B1 -A3 "map internals" "$dir"/*.log > "$dir/map_internals.txt" 2>/dev/null || true
	else
		{ grep -h -m1 '^# CSV,' "$first_log" | sed 's/^# CSV,/run,/'
			for f in "$dir"/*.log; do grep '^CSV,' "$f" | sed "s|^CSV,|$(basename "$f" .log),|"; done
		} > "$dir/$exp.csv"
	fi
}

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
if [ "$DRY" = 0 ]; then
	mkdir -p "$OUTDIR"
	{
		echo "SAM-HPC paper experiments"
		echo "date:        $(date)"
		echo "host:        $(hostname)"
		echo "commit:      $(git rev-parse HEAD 2>/dev/null || echo unknown)"
		echo "dirty files: $(git status --short 2>/dev/null | wc -l)"
		echo "problems:    $PROBLEMS"
		echo "experiments: $EXPERIMENTS"
		echo "THREADS='$THREADS' THREADS_MAIN=$THREADS_MAIN THREADS_SIDE='$THREADS_SIDE' LEVELS='$LEVELS'"
		echo "MAP_REPS=$MAP_REPS SOLVE_REPS=$SOLVE_REPS SEQ_SOLVE_REPS=$SEQ_SOLVE_REPS PHASE_REPS=$PHASE_REPS NUMA=$NUMA"
		echo "compiler:    $(${CXX:-g++} --version | head -1)"
		echo "physical cores: $PHYS_CORES, logical: $(nproc --all)"
		echo; lscpu
		echo; command -v numactl >/dev/null && numactl --hardware
	} > "$OUTDIR/manifest.txt"
	git diff > "$OUTDIR/uncommitted.diff" 2>/dev/null || true
	for p in ${PROBLEMS//,/ }; do
		"problem_$p"
		{
			echo "[$p] matrices $MATRIX_DIR rhs $RHS_DIR target $TARGET pair $PAIR sequence $SEQ_FIRST..$SEQ_LAST"
			echo "[$p] solver: $SOLVER"
			echo "[$p] chosen: global $GLOBAL column $COLUMN fixed $FIXED stencil '$STENCIL' explicit '$EXPLICIT'"
			echo "[$p] sweeps: global $GLOBAL_SWEEP | column $COLUMN_SWEEP | fixed $FIXED_SWEEP"
		} >> "$OUTDIR/manifest.txt"
	done

	if [ "$SKIP_BUILD" = 0 ]; then
		echo "Building (Release) and running the unit tests..."
		cmake --preset release > "$OUTDIR/build.log" 2>&1
		cmake --build --preset release -j >> "$OUTDIR/build.log" 2>&1
		ctest --preset release >> "$OUTDIR/build.log" 2>&1 || {
			echo "error: unit tests failed - see $OUTDIR/build.log; not benchmarking a broken build" >&2
			exit 1
		}
	fi
fi

for p in ${PROBLEMS//,/ }; do
	SUPPORTED="params,sequence,scaling,e2e,level,side,amg,periodic"
	"problem_$p"
	if [ ! -d "$MATRIX_DIR" ] && [ "$p" = cd2d ] && [ "$DRY" = 0 ]; then
		echo "Generating the cd2d sequence into $MATRIX_DIR ..."
		python3 "$ROOT/scripts/gen_cd2d.py" --grid 450 --out-matrix "$MATRIX_DIR" --out-rhs "$RHS_DIR" \
				> "$OUTDIR/gen_cd2d.log" 2>&1
	fi
	if [ ! -d "$MATRIX_DIR" ] && [ "$p" = elast3d ] && [ "$DRY" = 0 ]; then
		echo "Generating the elast3d sequence into $MATRIX_DIR (needs pyamg; ~1 h, 13 GB) ..."
		python3 "$ROOT/scripts/gen_elast3d.py" --out-matrix "$MATRIX_DIR" --out-rhs "$RHS_DIR" \
				> "$OUTDIR/gen_elast3d.log" 2>&1 || { echo "error: gen_elast3d.py failed, see $OUTDIR/gen_elast3d.log" >&2; exit 1; }
	fi
	for d in "$MATRIX_DIR" "$RHS_DIR"; do
		if [ "$DRY" = 0 ] && [ ! -d "$d" ]; then
			echo "error: no such directory: $d (problem $p)" >&2
			exit 1
		fi
	done
	for e in params sequence scaling e2e level side amg periodic; do
		has "$EXPERIMENTS" "$e" || continue
		has "$SUPPORTED" "$e" || { echo "[$p] skipping $e (not in this problem's experiment list)"; continue; }
		"exp_$e" "$p"
		[ "$DRY" = 0 ] && collect "$OUTDIR/$p/$e" "$e"
	done
done

if [ "$DRY" = 1 ]; then
	echo "dry run: $NRUNS benchmark invocations"
	exit 0
fi

tar -czf "$OUTDIR.tar.gz" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")"
echo "Done. $NRUNS invocations; results in $OUTDIR"
echo "Copy back with:  scp $(hostname):$OUTDIR.tar.gz ."
[ -f "$OUTDIR/FAILURES" ] && { echo "Some runs FAILED:"; cat "$OUTDIR/FAILURES"; }
exit 0
