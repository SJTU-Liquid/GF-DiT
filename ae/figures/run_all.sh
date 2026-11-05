#!/usr/bin/env bash
# Regenerate every data figure and table of the paper from the raw logs in
# ae/figures/logs, then compare the underlying numbers with ae/figures/expected
# (the CSV/Markdown files the paper's figures and tables were made from).
# Needs Python >= 3.10 with matplotlib and numpy.  ~1 min.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export MPLBACKEND=Agg PYTHONPATH=scripts
mkdir -p figs
fail=0
run() { echo "--- $*"; "$PY" "$@" > /dev/null || { echo "FAILED: $*"; fail=1; }; }
run scripts/plot_motivation.py              # Fig. 3
run scripts/plot_main_result_combined.py    # Fig. 6
run scripts/plot_workload_traces.py         # Fig. 7
run scripts/plot_gfc_h20_bench.py           # Fig. 8
run scripts/plot_scaling_arrival_rate.py    # Fig. 9
run scripts/plot_validation_microbench.py   # Fig. 10 (and the NCCL setup plot)
run scripts/plot_legacy_vs_fcfs_sp4.py      # Sec. 6.5 runtime overhead
run scripts/summarize_group_setup.py        # Table 1 + GFC ~60 us
run scripts/summarize_breakdown.py          # Table 2 + 40% load numbers
run scripts/headline_numbers.py             # Abstract / Sec. 6.3: 6.01x, 95.2%, 89.6%
# Identical content, ignoring line endings (git may store the CSVs with LF while
# Python's csv module writes CRLF).
same() {
  "$PY" -c 'import sys; a, b = (open(f, "rb").read().replace(b"\r\n", b"\n") for f in sys.argv[1:]); sys.exit(a != b)' "$1" "$2"
}
echo
echo "Comparison with the numbers behind the paper (ae/figures/expected):"
for e in expected/*; do
  f="figs/$(basename "$e")"
  if [ ! -f "$f" ]; then echo "MISSING   $f"; fail=1
  elif same "$e" "$f"; then echo "IDENTICAL $(basename "$e")"
  else echo "DIFFERS   $(basename "$e")  (diff $e $f)"; fail=1; fi
done
echo; [ "$fail" = 0 ] && echo "ALL FIGURE/TABLE DATA MATCH THE PAPER" || echo "SOME CHECKS FAILED"
exit "$fail"
