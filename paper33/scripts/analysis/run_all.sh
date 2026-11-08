#!/usr/bin/env bash
# Paper 33 -- re-run the whole held-out analysis, in order. No GPU, no network; read-only on every input,
# writes only results/analysis/. Re-run after every judge batch (judge batch 2: OLMo/Mistral, gamma-MPOA,
# community, seed-2): newly judged cells are picked up automatically, everything else is listed as pending.
#
#   scripts/analysis/run_all.sh             # full run, ~6-10 min (h1h2.py VB fits take most of it)
#   scripts/analysis/run_all.sh --no-vb     # extra arguments go to h1h2.py (here: skip the VB mixed models)
#
# Steps
#   1. assemble.py     responses + judgments -> responses_judged.csv.gz, cell_rates.csv, dir_status.csv
#   2. AdvBench map    secondary.py --build-advbench-map (needs pyarrow: the Heretic venv; skipped with a note if
#                      that Python is missing -- the map is static, the stored CSV is then used)
#   3. h1h2.py         H1/H2 (McNemar, Newcombe CIs, Holm, GEE, VB mixed model; judges wg, wg_class, kw, kw256, q3g)
#   4. h3.py           H3 (keyword vs judges, kappa, gap models)
#   5. truncation256.py  256-token judge sensitivity of the labels and of the H3 gap (trunc256_*.csv,
#                      trunc256_summary.md; paper Appendix app:trunc256; added 2026-11-06; needs steps 1 and 4)
#   6. secondary.py    over-refusal, dev vs held-out, harm/degenerate/capped, AdvBench sensitivity
#   7. final.py        gamma-MPOA control, community vs selected, seed 2 vs seed 1, dev vs held-out (all cells),
#                      search convergence, keyword-selection bias (final_*.csv, final_log.txt; added 2026-11-06)
#   8. dev_rescore_summary.py  dev-set re-score of the 46 released exports vs the selected trials
#                      (dev_rescore_*.csv, dev_rescore_summary.md; paper Appendix app:dev-rescore; added 2026-11-06;
#                      imports helpers from final.py and reads final_kl_vs_capability_cells.csv, so runs after step 7)
#   9. review_selection.py  reviewer-response analyses: KL-cap binding, pair as the unit (random-effects pooling incl.
#                      seed variance, leave-one-family-out), both scales, seed-2 batch-size confound, ARA export drift
#                      (review_sel_*.csv, review_sel_summary.md; paper Results H1/H2, Appendix app:kl-cap; added 2026-11-06;
#                      runs after final.py (step 7) and dev_rescore_summary.py (step 8): reads final_seed_cells.csv,
#                      final_pooled_robustness.csv, dev_rescore_cells.csv, and seednoise_sigma.csv, which seednoise_flags.py
#                      writes outside this script -- see the note below)
#  10. review_outcomes.py   reviewer-response analyses: item source (public sets vs code_weapon), harmful compliance as the
#                      outcome, H3 at Heretic's 100-token keyword window (review_out_*.csv, review_out_summary.md; paper
#                      Results H1/H3, Appendices app:item-source, app:harm, app:trunc256; added 2026-11-06; its tokeniser
#                      stage calls the Heretic venv (python -I) by itself when review_out_c4_kw_rows.csv is missing or stale)
#  11. freshness.py    content check that the result trees did not change during the run (exit 3 if they did)
# Console output of every step: results/analysis/run_all_log.txt
# Not called here (2026-11-06): make_hypothesis_tables.py (writes the paper's hypothesis tables) and therefore not
# seednoise_flags.py, which make_hypothesis_tables.py needs first. After this script, run by hand, in this order:
#   scripts/figures/.venv/bin/python scripts/analysis/seednoise_flags.py      (needs final_seed_cells.csv, step 7)
#   scripts/figures/.venv/bin/python scripts/analysis/make_hypothesis_tables.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/scripts/figures/.venv/bin/python"
PYARROW_PY="${PYARROW_PY:-/data/heretic_2/heretic/.venv/bin/python}"
LOG="$ROOT/results/analysis/run_all_log.txt"
mkdir -p "$ROOT/results/analysis"
: > "$LOG"

step() { echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
run() { "$@" >> "$LOG" 2>&1 || { echo "FAILED: $* (see $LOG)" | tee -a "$LOG"; exit 1; }; }

step "1/11 assemble.py"
run "$PY" scripts/analysis/assemble.py
grep -E "^## Totals|char_run recheck" "$LOG" | sed "s/^/  /" || true

step "2/11 AdvBench overlap map (all held-out items, harmful and benign)"
if [ -x "$PYARROW_PY" ] && "$PYARROW_PY" -I -c "import pyarrow, pandas" 2>/dev/null; then
    run "$PYARROW_PY" -I scripts/analysis/secondary.py --build-advbench-map
else
    echo "  no Python with pyarrow at $PYARROW_PY; using the stored results/analysis/secondary_advbench_overlap_splits.csv" \
        | tee -a "$LOG"
fi

step "3/11 h1h2.py $*"
run "$PY" scripts/analysis/h1h2.py "$@"

step "4/11 h3.py"
run "$PY" scripts/analysis/h3.py

step "5/11 truncation256.py"
run "$PY" scripts/analysis/truncation256.py

step "6/11 secondary.py"
run "$PY" scripts/analysis/secondary.py

step "7/11 final.py"
run "$PY" scripts/analysis/final.py

step "8/11 dev_rescore_summary.py"
run "$PY" scripts/analysis/dev_rescore_summary.py

step "9/11 review_selection.py"
run "$PY" scripts/analysis/review_selection.py

step "10/11 review_outcomes.py"
run "$PY" scripts/analysis/review_outcomes.py

step "11/11 freshness check"
if "$PY" scripts/analysis/freshness.py >> "$LOG" 2>&1; then
    echo "  outputs match the result trees" | tee -a "$LOG"
else
    echo "  result trees changed during the run (new responses/judgments synced); outputs are consistent with the" \
         "assembly of step 1 but already stale -- re-run" | tee -a "$LOG"
    exit 3
fi
step "done: results/analysis/{h1h2,h3,secondary,trunc256,dev_rescore,review_sel,review_out}_summary.md, final_log.txt; FINAL_NOTES.md is hand-written (update its numbers)"
