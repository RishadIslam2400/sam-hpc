#!/usr/bin/env python3
"""Collapse the E1 CSV lines into the comparison tables the paper needs.

Reads every top_opt_*.txt / weather_*.txt in this directory, pulls the CSV rows, and emits:
  1. a per-variant pattern comparison (cost vs accuracy vs iterations)
  2. a source-vs-union delta table, which is E4
  3. a markdown block ready to paste into the manuscript

Usage: python3 results/e1/summarize.py [directory] [--legacy]

By default only current-format rows are read. --legacy also reads the pre-C3 runs, whose
J was masked to the pattern; they are labelled SUPERSEDED and must not be quoted.
"""
import csv
import glob
import os
import sys

# Current pipeline_bench CSV. J is always the full residual support.
FIELDS = ["pattern", "variant", "threads", "map_nnz", "map_min_ms", "map_med_ms",
          "solve_min_ms", "solve_med_ms", "total_min_ms", "iters", "residual",
          "map_reps", "solve_reps"]
# Pre-C3 CSV, with the removed rowSet::pattern column. Its numbers are not comparable to
# current ones (J was masked to the pattern), so they are only read with --legacy.
LEGACY_FIELDS = FIELDS[:2] + ["row_set"] + FIELDS[2:]
NUMERIC = {"threads", "map_nnz", "map_min_ms", "map_med_ms", "solve_min_ms", "solve_med_ms",
           "total_min_ms", "iters", "residual", "map_reps", "solve_reps"}


def load(directory, legacy=False):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        with open(path) as fh:
            for line in fh:
                if not line.startswith("CSV,"):
                    continue
                parts = next(csv.reader([line[4:]]))
                if len(parts) == len(FIELDS):
                    rec = dict(zip(FIELDS, parts))
                    rec["row_set"] = "support"
                elif legacy and len(parts) == len(LEGACY_FIELDS):
                    rec = dict(zip(LEGACY_FIELDS, parts))
                    rec["row_set"] += " (SUPERSEDED)"
                else:
                    continue
                for k in NUMERIC:
                    rec[k] = float(rec[k])
                rec["source_file"] = os.path.basename(path)
                rows.append(rec)
    return rows


def table(rows, n):
    hdr = f"{'pattern':<26}{'nnz/row':>10}{'map ms':>11}{'solve ms':>11}{'total ms':>11}{'iters':>8}{'residual':>13}{'reps':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["map_nnz"]):
        print(f"{r['pattern']:<26}{r['map_nnz']/n:>10.2f}{r['map_min_ms']:>11.1f}"
              f"{r['solve_min_ms']:>11.1f}{r['total_min_ms']:>11.1f}{int(r['iters']):>8}"
              f"{r['residual']:>13.3e}"
              f"{str(int(r['map_reps'])) + '/' + str(int(r['solve_reps'])):>10}")


def main():
    args = [a for a in sys.argv[1:] if a != "--legacy"]
    directory = args[0] if args else os.path.dirname(os.path.abspath(__file__))
    rows = load(directory, legacy="--legacy" in sys.argv[1:])
    if not rows:
        print(f"no CSV rows found in {directory}", file=sys.stderr)
        return 1

    n = 132300  # top_opt small
    for variant in sorted({r["variant"] for r in rows}):
        for row_set in sorted({r["row_set"] for r in rows if r["variant"] == variant}):
            sel = [r for r in rows if r["variant"] == variant and r["row_set"] == row_set]
            print(f"\n=== variant {variant} | J from {row_set} | {int(sel[0]['threads'])} threads ===")
            table(sel, n)

    # E4: does the union step earn its extra nonzeros?
    by_pattern = {}
    for r in rows:
        by_pattern.setdefault((r["pattern"], r["row_set"]), {})[r["variant"]] = r
    pairs = {k: v for k, v in by_pattern.items() if {"source", "union"} <= set(v)}
    if pairs:
        print("\n=== E4: union vs source ===")
        hdr = f"{'pattern':<26}{'nnz/row s->u':>16}{'map ms':>16}{'residual':>24}{'iters':>14}"
        print(hdr)
        print("-" * len(hdr))
        for (pattern, _), v in sorted(pairs.items()):
            s, u = v["source"], v["union"]
            print(f"{pattern:<26}{s['map_nnz']/n:>7.2f} ->{u['map_nnz']/n:>6.2f}"
                  f"{s['map_min_ms']:>8.0f} ->{u['map_min_ms']:>6.0f}"
                  f"{s['residual']:>13.3e} ->{u['residual']:>10.3e}"
                  f"{int(s['iters']):>7} ->{int(u['iters']):>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
