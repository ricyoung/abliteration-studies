#!/usr/bin/env bash
# make_figures.sh -- rebuild the paper-33 figure data and every figure.
#
#   bash scripts/figures/make_figures.sh            # data + all fig*.py
#   bash scripts/figures/make_figures.sh --data     # data only
#
# 1. extract_data.py  (Heretic venv: optuna)  -> paper/figures/data/{trials,results,runs}.csv
# 2. every scripts/figures/fig*.py, in natural order (fig2 before fig10), with the figures venv
#    (matplotlib/numpy/pandas)              -> paper/figures/figN_*.pdf + preview/*.png + captions.md
#    fig10-fig14 (held-out, judged) read results/analysis/*.csv, the output of scripts/analysis/run_all.sh
#    (shared loaders in judged.py, which is not a figure script); re-run run_all.sh first after new judgements.
# A failing figure script does not stop the others; the exit status is non-zero if any step failed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DATA_PY=/data/heretic_2/heretic/.venv/bin/python
PLOT_PY=$HERE/.venv/bin/python

for py in "$DATA_PY" "$PLOT_PY"; do
  [ -x "$py" ] || { echo "missing interpreter: $py" >&2; exit 2; }
done

echo "== extract_data.py"
if ! "$DATA_PY" "$HERE/extract_data.py"; then
  echo "extract_data.py FAILED; figures not rebuilt" >&2
  exit 1
fi
[ "${1:-}" = "--data" ] && exit 0

mapfile -t scripts < <(find "$HERE" -maxdepth 1 -name 'fig*.py' -printf '%f\n' | sort -V)
if [ ${#scripts[@]} -eq 0 ]; then
  echo "no fig*.py scripts in ${HERE#$ROOT/} yet"
  exit 0
fi

failed=()
for s in "${scripts[@]}"; do
  echo "== $s"
  # run from scripts/figures so 'import style' resolves; -I is NOT used because style.py is local
  if ! (cd "$HERE" && MPLBACKEND=Agg "$PLOT_PY" "$s"); then
    failed+=("$s")
    echo "   FAILED: $s" >&2
  fi
done

echo
echo "== figures in ${ROOT}/paper/figures"
ls -1 "$ROOT"/paper/figures/fig*.pdf 2>/dev/null | sed "s|$ROOT/||"
if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED (${#failed[@]}): ${failed[*]}" >&2
  exit 1
fi
echo "all ${#scripts[@]} figure scripts OK"
