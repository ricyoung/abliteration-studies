#!/usr/bin/env python3
"""Paper 33 -- is the assembled held-out table still current? (content-based, not mtime-based)

    scripts/figures/.venv/bin/python scripts/analysis/freshness.py     # exit 1 and list the reasons if stale

Used by h1h2.py, h3.py and secondary.py (import) and by run_all.sh (CLI). It compares what assemble.py recorded in
results/analysis/dir_status.csv with what the result trees hold now:
  - a result dir that dir_status.csv does not list (new cell, new copy) or that has disappeared;
  - a change of state: responses.jsonl or judgments.jsonl appeared or disappeared since the assembly
    (pending_generation -> responses, pending_judging -> judged, including the jump pending_generation -> judged);
  - for judged dirs: a different number of judgment rows or a different `scored` timestamp in refusal/summary.json
    (re-judged); for pending-judging dirs: a different number of response rows (generation still running).
File modification times are not used: `rsync -a` keeps the remote mtimes, so a judgments file written before the
assembly but synced after it would look older than the table (issue found in the 2026-11-05 verification).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
EXCLUDED_ROOT_NAMES = {"ref_import"}  # same as assemble.py
HAS_JUDGMENTS = {"judged", "partially_judged", "judgments_mismatch"}


def _nlines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for line in f if line.strip())


def _scan_dirs() -> set[str]:
    roots = {p for p in ROOT.glob("runs/*/results/*") if p.is_dir()} | {p for p in ROOT.glob("results/bench*")
                                                                         if p.is_dir()}
    return {str(d.relative_to(ROOT)) for r in roots if r.name not in EXCLUDED_ROOT_NAMES
            for d in r.iterdir() if d.is_dir()}


def check() -> list[str]:
    """Reasons why results/analysis/* no longer reflects the result trees ([] = current)."""
    ds_path = OUT / "dir_status.csv"
    if not ds_path.exists():
        return ["dir_status.csv missing: run assemble.py"]
    ds = pd.read_csv(ds_path, low_memory=False)
    problems = []
    listed = set(ds["source_dir"].dropna())
    now = _scan_dirs()
    # a new dir whose model prefix is not a paper-33 model (e.g. a Qwen2.5-Coder baseline evaluated on the same box for
    # another study) can never become a cell -- assemble.py would skip it as "unknown model" -- so it does not make the
    # tables stale (added 2026-11-06: runs/mc-rtx-1/results/bench/qwen2.5-coder-32b__base appeared mid-run)
    known_models = set(ds.loc[ds["used"].astype(str) == "True", "model"].dropna())
    for s in sorted(now - listed):
        if known_models and Path(s).name.split("__")[0] not in known_models:
            continue
        problems.append(f"{s}: result dir not seen by the last assembly")
    for s in sorted(listed - now):
        problems.append(f"{s}: listed in dir_status.csv but no longer present")
    for r in ds[ds["judge_status"] != "skipped"].itertuples():
        d = ROOT / r.source_dir / "refusal"
        rf, jf = d / "responses.jsonl", d / "judgments.jsonl"
        was = r.judge_status
        if jf.exists() and was not in HAS_JUDGMENTS:
            problems.append(f"{r.source_dir}: judgments.jsonl now present (assembled as {was})")
            continue
        if not jf.exists() and was in HAS_JUDGMENTS:
            problems.append(f"{r.source_dir}: judgments.jsonl gone (assembled as {was})")
            continue
        if rf.exists() and was == "pending_generation":
            problems.append(f"{r.source_dir}: responses.jsonl now present (assembled as pending_generation)")
            continue
        if was in HAS_JUDGMENTS:
            n = _nlines(jf)
            if pd.notna(r.n_judgments) and n != int(r.n_judgments):
                problems.append(f"{r.source_dir}: judgments rows {n} != {int(r.n_judgments)} at assembly")
            sp = d / "summary.json"
            scored = json.load(open(sp)).get("scored", "") if sp.exists() else ""
            rec = "" if pd.isna(r.judge_scored) else str(r.judge_scored)
            if scored != rec:
                problems.append(f"{r.source_dir}: judge 'scored' {scored or '-'} != {rec or '-'} at assembly")
        elif was == "pending_judging" and rf.exists():
            n = _nlines(rf)
            if pd.notna(r.n_responses) and n != int(r.n_responses):
                problems.append(f"{r.source_dir}: responses rows {n} != {int(r.n_responses)} at assembly")
    return problems


def summary_line(problems: list[str]) -> str:
    if not problems:
        return ("Input check (freshness.py, content-based): the assembled tables match the result trees "
                "(no new, newly judged, re-judged or removed result dirs).")
    return (f"**WARNING (freshness.py): the assembled tables are stale -- {len(problems)} change(s) since the last "
            "assemble.py run; re-run `scripts/analysis/run_all.sh`.** " + "; ".join(problems[:8])
            + (" ..." if len(problems) > 8 else ""))


if __name__ == "__main__":
    p = check()
    print(summary_line(p))
    for x in p:
        print("  " + x)
    sys.exit(1 if p else 0)
