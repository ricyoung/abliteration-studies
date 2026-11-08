"""Aggregate bench summary.json files into one CSV with deltas vs the base model.

Usage:
  python summarize.py <dir_or_summary.json> [...] [-o capability.csv] [--labels labels.csv]

Each row: model, method, base_model, limit_mode, benchmark, metric, score, stderr, n,
          base_score, delta_vs_base, delta_pp, wall_seconds, backend, out_dir.
method / base_model come from summary.json (run.sh --method / --base-model).
labels.csv (optional; columns out_dir,method,base_model) overrides them per run.
A run counts as the base of model X when its `model` equals X's base_model and
(method == "base" or base_model is empty). Deltas are only computed between runs with the
same limit_mode and protocol_version. Standard library only.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def find(paths):
    for p in map(Path, paths):
        if p.is_file():
            yield p
        else:
            yield from sorted(p.rglob("summary.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="-")
    ap.add_argument("--labels")
    args = ap.parse_args()

    labels = {}
    if args.labels:
        with open(args.labels) as f:
            for r in csv.DictReader(f):
                labels[str(Path(r["out_dir"]).resolve())] = r

    runs = []
    for p in find(args.paths):
        s = json.loads(p.read_text())
        d = str(p.parent.resolve())
        lab = labels.get(d, {})
        method = lab.get("method") or s.get("method") or ""
        base = lab.get("base_model") or s.get("base_model") or ""
        if not method and not base:
            method = "base"
        runs.append(dict(s=s, dir=d, model=s.get("model", ""), method=method, base=base))

    # index base runs: (model, limit_mode, protocol) -> {benchmark: score}
    base_idx = {}
    for r in runs:
        if r["method"] == "base" or not r["base"]:
            key = (r["model"], r["s"].get("limit_mode"), r["s"].get("protocol_version"))
            if key in base_idx:
                print(f"warning: several base runs for {key}; using {r['dir']}", file=sys.stderr)
            base_idx[key] = {b: e.get("score") for b, e in r["s"].get("benchmarks", {}).items()}

    cols = ["model", "method", "base_model", "limit_mode", "benchmark", "metric", "score", "stderr", "n",
            "base_score", "delta_vs_base", "delta_pp", "wall_seconds", "backend", "out_dir"]
    rows = []
    for r in runs:
        s = r["s"]
        base_model = r["base"] or r["model"]
        bscores = base_idx.get((base_model, s.get("limit_mode"), s.get("protocol_version")), {})
        for b, e in s.get("benchmarks", {}).items():
            if "error" in e:
                rows.append(dict(model=r["model"], method=r["method"], base_model=base_model,
                                 limit_mode=s.get("limit_mode"), benchmark=b, metric="ERROR",
                                 backend=s.get("backend"), out_dir=r["dir"]))
                continue
            score, bs = e.get("score"), bscores.get(b)
            delta = score - bs if score is not None and bs is not None else None
            rows.append(dict(
                model=r["model"], method=r["method"], base_model=base_model,
                limit_mode=s.get("limit_mode"), benchmark=b, metric=e.get("metric"),
                score=score, stderr=e.get("stderr"), n=e.get("n"), base_score=bs,
                delta_vs_base=delta, delta_pp=round(100 * delta, 2) if delta is not None else None,
                wall_seconds=e.get("wall_seconds"), backend=s.get("backend"), out_dir=r["dir"]))
    rows.sort(key=lambda x: (x["base_model"], x["method"] != "base", x["method"], x["model"], x["benchmark"]))

    f = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    if f is not sys.stdout:
        f.close()
        print(f"wrote {len(rows)} rows from {len(runs)} runs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
