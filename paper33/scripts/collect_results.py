#!/usr/bin/env python3
"""Aggregate heretic 2.0 runs into one long-format CSV for the paper tables.

One row per (run, KL threshold), using the rule in select_trial.py (fewest refusals
among COMPLETE trials with KL <= threshold; ties -> lower KL, lower trial index).
Reads the journals directly (read-only, via a temp copy); does not need selection.json.

Usage:
    python collect_results.py RUN_DIR_OR_PARENT [...] [-o results/selection.csv]
        [--thresholds 0.05,0.1,0.2,0.5]
A directory is treated as a run if it contains checkpoints/*.jsonl; otherwise its
immediate subdirectories are scanned (so `runs/mc-h200-1/runs` works).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_trial as st  # noqa: E402

FIELDS = ["model", "model_short", "method", "tag", "seed", "kl_threshold", "status",
          "trial_index", "pareto_position", "refusals", "refusals_display", "kl",
          "baseline_refusals", "n_trials_complete", "n_trials_planned", "finished",
          "pareto_front_size", "headless_index0_trial", "headless_index0_refusals",
          "headless_index0_kl", "run", "run_dir"]


def is_run(d):
    ck = os.path.join(d, "checkpoints")
    return os.path.isdir(ck) and any(f.endswith(".jsonl") for f in os.listdir(ck))


def expand(paths):
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if is_run(p):
            out.append(p)
        elif os.path.isdir(p):
            out += [os.path.join(p, d) for d in sorted(os.listdir(p)) if is_run(os.path.join(p, d))]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--output", default="-", help="CSV path (default stdout)")
    ap.add_argument("--thresholds", default=",".join(f"{x:g}" for x in st.REPORT_THRESHOLDS))
    a = ap.parse_args(argv)
    thresholds = sorted({float(x) for x in a.thresholds.split(",") if x.strip()})

    rows, rc = [], 0
    for d in expand(a.paths):
        try:
            r = st.analyse(d, st.KL_MAX, thresholds)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {d}: {e}", file=sys.stderr); rc = 1; continue
        base = r["baselines"].get(st.REFUSAL_OBJECTIVE)
        h0 = r["headless_index0_trial"] or {}
        for thr in thresholds:
            t = r["per_threshold"][f"{thr:g}"]
            rows.append({
                "model": r["model"], "model_short": r["model_short"], "method": r["method"],
                "tag": r["tag"], "seed": r["seed"], "kl_threshold": thr,
                "status": "selected" if t else "none_qualifies",
                "trial_index": t and t["trial_index"], "pareto_position": t and t["pareto_position"],
                "refusals": t and t["refusals"], "refusals_display": t and t["refusals_display"],
                "kl": t and t["kl"], "baseline_refusals": base,
                "n_trials_complete": r["n_trials_complete"], "n_trials_planned": r["n_trials_planned"],
                "finished": r["finished"], "pareto_front_size": len(r["pareto_front"]),
                "headless_index0_trial": h0.get("trial_index"),
                "headless_index0_refusals": h0.get("refusals"), "headless_index0_kl": h0.get("kl"),
                "run": r["run"], "run_dir": r["run_dir"],
            })

    f = sys.stdout if a.output == "-" else open(a.output, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader(); w.writerows(rows)
    if f is not sys.stdout:
        f.close(); print(f"wrote {len(rows)} rows -> {a.output}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
