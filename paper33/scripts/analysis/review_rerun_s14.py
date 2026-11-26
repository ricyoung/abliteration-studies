#!/usr/bin/env python3
"""Reviewer item S14 (REV-22): how much could resuming a study have changed the selected trial? (added 2026-11-11)

Read-only on existing journals; writes two new CSVs, edits no existing result.

1. For the two Qwen3.8-27B studies that were interrupted and resumed (MPOA, evaluated; ARA, retained),
   find where each interruption fell, whether the selected trial was drawn before or after it, and
   whether the selected trial lies in the random start-up phase (first n_startup_trials = 30 trials,
   configs/core_*.toml), in which the TPE sampler draws independently of earlier results.
2. For the one model/method with both a resumed and a clean study (Qwen3.8-27B ARA, seed 1), compare
   the two selections at KL <= 0.1.
3. Context over all evaluated main-study runs: how often the selected trial is a start-up trial
   (column sel_is_startup_trial of results/analysis/review_sel_kl_runs.csv).
4. Replay check for the resumed MPOA study: on resumption the sampler is re-created from seed 1, so the
   start-up draws restart. Compare trial parameters with the study's own first trials and with an
   uninterrupted seed-1 MPOA study on another model (OLMo-3.1-32B, same search space), and compare the
   dev refusals of the trials whose parameters were evaluated twice.

Outputs: results/analysis/review_rerun_s14_studies.csv, results/analysis/review_rerun_s14_summary.csv,
         results/analysis/review_rerun_s14_replay.csv
Run: ../heretic/.venv/bin/python scripts/analysis/review_rerun_s14.py  (from the paper-33 root)
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import select_trial as st  # noqa: E402

from optuna.trial import TrialState  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "mc-h200-1", "runs")
OUT = os.path.join(ROOT, "results", "analysis")
N_STARTUP = 30
KL_MAX = 0.1

# (label, final study dir, [pre-resume journal dirs in order], evaluated?)
STUDIES = [
    ("qwen3.8-27b MPOA (resumed)", "qwen3.8-27b__mpoa__core-s1-resume1", ["qwen3.8-27b__mpoa__core-s1"], True),
    ("qwen3.8-27b ARA (resumed)", "qwen3.8-27b__ara__core-s1-resume2",
     ["qwen3.8-27b__ara__core-s1", "qwen3.8-27b__ara__core-s1-resume1"], False),
    ("qwen3.8-27b ARA (clean)", "qwen3.8-27b__ara__core-s1-clean", [], True),
]


def load(d):
    trials, _, _ = st.load_study(st.find_journal(os.path.join(RUNS, d)))
    return trials


def n_complete(trials):
    return sum(t.state == TrialState.COMPLETE for t in trials)


def best_in(trials, lo, hi):
    """Fewest refusals at KL <= KL_MAX among complete trials with lo <= number < hi."""
    c = [t for t in trials if t.state == TrialState.COMPLETE and lo <= t.number < hi]
    s = st.select(c, KL_MAX)
    return (st.score(s, st.REFUSAL_OBJECTIVE), st.score(s, st.KL_OBJECTIVE), s.number) if s else (None, None, None)


def main():
    rows = []
    for label, final, pre, evaluated in STUDIES:
        trials = load(final)
        complete = [t for t in trials if t.state == TrialState.COMPLETE]
        sel = st.select(complete, KL_MAX)
        # interruption points = number of trials recorded in each pre-resume journal (Optuna numbers continue)
        cuts = [len(load(p)) for p in pre]
        last_cut = max(cuts) if cuts else 0
        b_start = best_in(trials, 0, N_STARTUP)
        b_tpe = best_in(trials, N_STARTUP, 10 ** 6)
        rows.append({
            "study": label, "run_dir": final, "evaluated": evaluated,
            "trials_total": len(trials), "trials_complete": len(complete),
            "interruptions_after_trials": ";".join(map(str, cuts)) or "none",
            "sel_optuna_number": sel.number, "sel_trial_index": sel.user_attrs["index"],
            "sel_refusals_dev": round(st.score(sel, st.REFUSAL_OBJECTIVE) * 100),
            "sel_kl": round(st.score(sel, st.KL_OBJECTIVE), 4),
            "sel_after_last_interruption": sel.number >= last_cut if cuts else "n/a",
            "sel_in_startup_phase": sel.number < N_STARTUP,
            "best_startup_refusals": round(b_start[0] * 100) if b_start[0] is not None else "",
            "best_startup_kl": round(b_start[1], 4) if b_start[1] is not None else "",
            "best_tpe_refusals": round(b_tpe[0] * 100) if b_tpe[0] is not None else "",
            "best_tpe_kl": round(b_tpe[1], 4) if b_tpe[1] is not None else "",
        })
    with open(os.path.join(OUT, "review_rerun_s14_studies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # context: selected trial in the start-up phase, over evaluated main-study runs
    with open(os.path.join(OUT, "review_sel_kl_runs.csv")) as f:
        runs = [r for r in csv.DictReader(f) if r["study"] == "main"]
    n = len(runs)
    k = sum(r["sel_is_startup_trial"] == "True" for r in runs)
    ara_res = next(r for r in rows if r["study"].endswith("ARA (resumed)"))
    ara_cln = next(r for r in rows if r["study"].endswith("ARA (clean)"))
    summ = [
        {"quantity": "main-study runs (review_sel_kl_runs.csv, study == main)", "value": n},
        {"quantity": "of which selected trial is a start-up trial (number < 30)", "value": k},
        {"quantity": "ARA resumed vs clean: dev keyword refusals of selected trial",
         "value": f"{ara_res['sel_refusals_dev']} vs {ara_cln['sel_refusals_dev']}"},
        {"quantity": "ARA resumed minus clean (pp, dev)",
         "value": ara_res["sel_refusals_dev"] - ara_cln["sel_refusals_dev"]},
    ]
    with open(os.path.join(OUT, "review_rerun_s14_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value"])
        w.writeheader()
        w.writerows(summ)
    # 4. replay check (MPOA): parameters of resumed trials vs the study's first trials and vs OLMo-3.1-32B MPOA
    q = load("qwen3.8-27b__mpoa__core-s1-resume1")
    o = load("olmo-3.1-32b__mpoa__core-s1")
    cut = len(load("qwen3.8-27b__mpoa__core-s1"))
    par = lambda t: t.user_attrs.get("parameters") or t.params  # noqa: E731
    rep = []
    for i in range(len(q)):
        j = i - cut if i >= cut else i  # position in an uninterrupted seed-1 sequence
        dup = i + cut if i < cut and i + cut < len(q) else None
        rep.append({
            "optuna_number": i, "segment": "before_resume" if i < cut else "after_resume",
            "uninterrupted_position": j,
            "params_equal_olmo31_same_position": j < len(o) and par(q[i]) == par(o[j]),
            "replayed_as_trial": dup if dup is not None and par(q[i]) == par(q[dup]) else "",
            "refusals_dev": "" if st.score(q[i], st.REFUSAL_OBJECTIVE) is None else round(st.score(q[i], st.REFUSAL_OBJECTIVE) * 100),
            "refusals_dev_replay": "" if dup is None or st.score(q[dup], st.REFUSAL_OBJECTIVE) is None
            else round(st.score(q[dup], st.REFUSAL_OBJECTIVE) * 100),
            "state": str(q[i].state).split(".")[-1],
        })
    with open(os.path.join(OUT, "review_rerun_s14_replay.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0]))
        w.writeheader()
        w.writerows(rep)
    start = [r for r in rep if r["optuna_number"] < N_STARTUP]
    pairs = [r for r in rep if r["refusals_dev"] != "" and r["refusals_dev_replay"] != ""]
    summ += [
        {"quantity": "MPOA resume point (trials in the pre-resume journal)", "value": cut},
        {"quantity": "MPOA start-up trials (0-29) whose params equal OLMo-3.1 MPOA at the uninterrupted position",
         "value": f"{sum(r['params_equal_olmo31_same_position'] for r in start)}/{len(start)}"},
        {"quantity": "MPOA trials evaluated twice (same params before and after resume)", "value": len(pairs)},
        {"quantity": "max |refusal difference| between the two evaluations (dev, of 100)",
         "value": max(abs(r["refusals_dev"] - r["refusals_dev_replay"]) for r in pairs)},
    ]
    with open(os.path.join(OUT, "review_rerun_s14_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value"])
        w.writeheader()
        w.writerows(summ)

    for r in rows:
        print(r)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
