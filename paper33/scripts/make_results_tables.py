#!/usr/bin/env python3
"""make_results_tables.py -- build the paper-33 results tables (T1-T4) from the raw result files.

Stdlib only. Run with plain python3 from anywhere:
    python3 scripts/make_results_tables.py
Reads (never writes) runs/, results/, scripts/bench/refusal_sets.json; writes ONLY paper/tables/.

INPUTS
  main study   runs/mc-rtx-1/results/bench/<label>/          (preferred copy)
               runs/mc-h200-1/results/bench/<label>/         (used only if the RTX copy lacks the label)
  small pair   results/bench-4090/<label>/
  <label> = <model>__<method>, method in base/dim/mpoa/ara/som/jspace. Any further suffix after the method
  ('.u2581-bug', '.hf-fallback-aborted', '.publish-403', '-heldout', ...) marks a superseded attempt; such
  dirs are never used for numbers (reasons: SUPERSEDED below, reported in coverage.md).
  A base dir may keep its held-out generation + teacher-forced scores in refusal_run/summary.json.
  Dev-set selection: selection.json (written by scripts/select_trial.py). Preferred copy = the eval dir
  (eval_one*.sh copies it from the run it evaluated); otherwise runs/mc-h200-1/runs/<run>/,
  runs/local-4090/<run>/, runs/mc-rtx-1/runs/<run>/. All copies found are compared.

OUTPUTS (paper/tables/)
  t1_dev_refusals.csv     + t1_dev_refusals_{main,small}.tex    keyword refusals (dev set) at KL <= tau
  t2_capability.csv       + t2_capability_{main,small}.tex      capability scores (%), delta vs original
  t3_teacher_forced.csv   + t3_teacher_forced_{main,small}.tex  teacher-forced mean log-prob, shift vs original
  coverage.md             which cells exist / are pending / were excluded, every consistency check, the evaluated
                          artefacts (merged exports; byte totals incl. non-weight files; tensor headers where local),
                          and the optimisation runs' own outputs (adapter_t0/ = LoRA adapter of Pareto position 0)

EXACTNESS
  JSON is parsed with parse_float=Decimal, so CSV values are the exact decimal text of the source files;
  derived values (x100, differences) are exact Decimal arithmetic. Rounding (ROUND_HALF_UP) happens only in
  the LaTeX: 1 decimal for percentages and percentage-point deltas, 3 decimals for log-probabilities.
  Missing cells are '' in the CSVs (with a status column) and '--' in the LaTeX.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import struct
import sys
from decimal import Decimal, ROUND_HALF_UP, getcontext

getcontext().prec = 50

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "tables")

# ---------------------------------------------------------------------------------------------------------
# Study design (EXECUTION_PLAN_v2.md): family pairs old -> new; methods per study.
MODELS = [  # short label, family, generation, study, column header used in the LaTeX
    ("qwen2.5-32b", "Qwen", "old", "main", "2.5-32B"),
    ("qwen3.8-27b", "Qwen", "new", "main", "3.8-27B"),
    ("gemma-2-27b", "Gemma", "old", "main", "2-27B"),
    ("gemma-4-31b", "Gemma", "new", "main", "4-31B"),
    ("olmo-2-32b", "OLMo", "old", "main", "2-32B"),
    ("olmo-3.1-32b", "OLMo", "new", "main", "3.1-32B"),
    ("mistral-small-2409", "Mistral Small", "old", "main", "2409"),
    ("mistral-small-3.2", "Mistral Small", "new", "main", "3.2-24B"),
    ("qwen2.5-7b", "Qwen (small pair)", "old", "small", "2.5-7B"),
    ("qwen3.5-4b", "Qwen (small pair)", "new", "small", "3.5-4B"),
]
MODEL_INFO = {m[0]: {"family": m[1], "generation": m[2], "study": m[3], "col": m[4]} for m in MODELS}
METHODS = {"main": ["dim", "mpoa", "ara", "som"], "small": ["dim", "mpoa", "ara", "som", "jspace"]}
METHOD_TEX = {"base": "Original", "dim": "DiM", "mpoa": "MPOA", "ara": "ARA", "som": "SOM", "jspace": "J-space"}
BENCHES = [("mmlu_pro", "MMLU-Pro"), ("gsm8k", "GSM8K"), ("ifeval", "IFEval"), ("humaneval", "HumanEval"),
           ("mbpp", "MBPP")]
KL_ALL = ["0.05", "0.1", "0.2", "0.5"]     # thresholds stored by select_trial.py (per_threshold keys)
KL_TEX = ["0.05", "0.1", "0.2"]            # thresholds shown in the LaTeX table

BENCH_ROOTS = {  # study -> ordered list of (box, dir); earlier entries win on duplicate labels
    "main": [("mc-rtx-1", os.path.join(ROOT, "runs", "mc-rtx-1", "results", "bench")),
             ("mc-h200-1", os.path.join(ROOT, "runs", "mc-h200-1", "results", "bench"))],
    "small": [("local-4090", os.path.join(ROOT, "results", "bench-4090"))],
}
RUN_ROOTS = [  # (box, dir) holding <run>/ directories; the first two are the boxes that ran the optimisation
    ("mc-h200-1", os.path.join(ROOT, "runs", "mc-h200-1", "runs")),
    ("local-4090", os.path.join(ROOT, "runs", "local-4090")),
    ("mc-rtx-1", os.path.join(ROOT, "runs", "mc-rtx-1", "runs")),   # journal mirrors used by the RTX evals
]
QUEUES = {  # eval queues: label -> run named on the queue line
    "mc-rtx-1": os.path.join(ROOT, "runs", "mc-rtx-1", "bench_queue.txt"),
    "local-4090": os.path.join(ROOT, "runs", "local-4090", "bench_queue.txt"),
}
H200_RUN_QUEUE = os.path.join(ROOT, "runs", "mc-h200-1", "queue.txt")
LOCAL_RUN_QUEUE = os.path.join(ROOT, "scripts", "local_queue.sh")   # runner that wrote the 4090 runs' meta.txt
REFUSAL_SETS = os.path.join(ROOT, "scripts", "bench", "refusal_sets.json")
HERETIC_DEFAULTS = os.path.join(os.path.dirname(ROOT), "heretic", "config.default.toml")

# Known superseded attempt dirs (suffix -> reason). Source: notes/overnight_log.md + the dirs themselves.
SUPERSEDED = {
    ".u2581-bug": ("vLLM 0.33 detokenizer left U+2581 in Gemma-2 output (code benchmarks near 0); "
                   "re-run with the U+2581->space post-processing into the unsuffixed dir "
                   "(notes/overnight_log.md, 23:45 and 00:22 entries)"),
    ".hf-fallback-aborted": ("vLLM refused to load Qwen3.8 (max_num_seqs > Mamba cache blocks), bench fell back to "
                             "the HF backend and was aborted; no benchmark scores; re-run on vLLM with "
                             "max_num_seqs=256 into the unsuffixed dir (notes/overnight_log.md, 23:08 entry)"),
    ".publish-403": ("4090 derived-model eval attempt that failed at Hugging Face repo creation (HTTP 403) after "
                     "re-export, before any benchmark; no summary.json; re-run with --no-publish into the "
                     "unsuffixed dir (notes/overnight_log.md, 03:20 entry)"),
}

LABEL_RE = re.compile(r"^(?P<model>.+?)__(?P<method>[a-z0-9]+)(?P<suffix>[.\-].*)?$")
RUN_RE = re.compile(r"^(?P<model>.+?)__(?P<method>[a-z0-9]+)__(?P<tag>[^.]+)(?P<suffix>\..+)?$")


# ---------------------------------------------------------------------------------------------------------
def rel(p):
    return os.path.relpath(p, ROOT) if p else ""


def load_json(path):
    with open(path) as f:
        return json.load(f, parse_float=Decimal)


def dstr(x):
    """Exact text for CSV: Decimal without exponent and without trailing zeros; '' for None."""
    if x is None:
        return ""
    if isinstance(x, Decimal):
        s = format(x, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    return str(x)


def rnd(x, places):
    v = x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if v == 0:
        v = abs(v)
    return format(v, "f")


def tex_num(x, places):
    s = rnd(x, places)
    return "$-$" + s[1:] if s.startswith("-") else s


def tex_delta(x, places):
    s = rnd(x, places)
    if s.startswith("-"):
        return "$-$" + s[1:]
    return ("+" + s) if Decimal(s) != 0 else s


def tex_escape(s):
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def read_meta(path):
    meta = {}
    if os.path.isfile(path):
        for line in open(path, errors="replace"):
            if ":" in line:
                k, v = line.split(":", 1)
                meta.setdefault(k.strip(), v.strip())
    return meta


def parse_display(s):
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def label_from_path(p):
    """'/workspace/paper33/results/bench/<label>/...' or '.../results/bench-4090/<label>/...' -> <label>."""
    m = re.search(r"/(?:bench|bench-4090)/([^/]+)/", str(p))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.anomalies = []   # (category, text)
        self.excluded = []    # (path, reason)
        self.ignored_files = []
        self.checks = []      # (check, result)

    def anomaly(self, cat, text):
        self.anomalies.append((cat, text))

    def check(self, name, ok, detail=""):
        self.checks.append((name, "ok" if ok else "MISMATCH", detail))
        if not ok:
            self.anomaly("check failed", f"{name}: {detail}")


R = Report()


# ---------------------------------------------------------------------------------------------------------
# 1. Evaluation dirs
def scan_evals():
    evals = {}   # label -> dict
    seen = {}    # label -> list of (box, path) for duplicate reporting
    for study, roots in BENCH_ROOTS.items():
        for box, root in roots:
            if not os.path.isdir(root):
                continue   # reported in coverage.md "Inputs scanned"
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if not os.path.isdir(path):
                    continue
                m = LABEL_RE.match(name)
                if not m or m.group("model") not in MODEL_INFO:
                    R.excluded.append((rel(path), "unrecognised label (model not in the design)"))
                    continue
                if m.group("suffix"):
                    sfx = m.group("suffix")
                    reason = SUPERSEDED.get(sfx, f"superseded/variant attempt (suffix '{sfx}' not in the known list)")
                    files = sorted(os.listdir(path))
                    summ = os.path.join(path, "summary.json")
                    extra = ""
                    if os.path.isfile(summ):
                        try:
                            d = load_json(summ)
                            sc = {k: dstr(v.get("score")) for k, v in d.get("benchmarks", {}).items() if "score" in v}
                            extra = f"; its summary.json: backend={d.get('backend')}, finished={d.get('finished')}, scores={sc or 'none'}"
                        except Exception as e:  # noqa: BLE001
                            extra = f"; summary.json unreadable: {e}"
                    else:
                        extra = "; no summary.json"
                    R.excluded.append((rel(path), reason + extra + f"; files: {', '.join(files)}"))
                    continue
                label = name
                seen.setdefault(label, []).append((box, path))
                if label in evals:
                    continue   # earlier root wins (RTX before H200)
                evals[label] = read_eval(label, study, box, path)
    for label, lst in seen.items():
        if len(lst) > 1:
            R.anomaly("duplicate label", f"{label} exists in {[rel(p) for _, p in lst]}; used {rel(lst[0][1])} (RTX copy preferred)")
    return evals


def read_eval(label, study, box, path):
    m = LABEL_RE.match(label)
    ev = {"label": label, "model": m.group("model"), "method": m.group("method"), "study": study, "box": box,
          "path": path, "summary": None, "summary_path": None, "status": "no summary.json",
          "bench": {}, "tf": None, "tf_path": None, "tf_summary": None, "rg": None, "rg_path": None,
          "selection": None, "selection_path": None, "notes": []}
    sp = os.path.join(path, "summary.json")
    for f in sorted(os.listdir(path)):
        if f.startswith("summary") and f.endswith(".json") and f != "summary.json":
            R.ignored_files.append((rel(os.path.join(path, f)),
                                    "not read: only summary.json (and refusal_run/summary.json) are used"))
    if os.path.isfile(sp):
        s = load_json(sp)
        ev["summary"], ev["summary_path"] = s, sp
        ev["status"] = "complete" if s.get("finished") else "in progress (no 'finished' timestamp)"
        if s.get("method") != label:
            R.anomaly("label mismatch", f"{rel(sp)}: method field {s.get('method')!r} != dir label {label!r}")
        if s.get("finished"):
            for b, _ in BENCHES:
                v = s.get("benchmarks", {}).get(b)
                if v and "score" in v:
                    ev["bench"][b] = v
            if s.get("failed"):
                ev["notes"].append(f"failed: {s.get('failed')}")
            if s.get("teacher_forced") and "mean_logprob" in s["teacher_forced"]:
                ev["tf"], ev["tf_path"], ev["tf_summary"] = s["teacher_forced"], sp, s
            if s.get("refusal_generation"):
                ev["rg"], ev["rg_path"] = s["refusal_generation"], sp
    rr = os.path.join(path, "refusal_run", "summary.json")
    if os.path.isfile(rr):
        s2 = load_json(rr)
        if s2.get("finished") and ev["tf"] is None and s2.get("teacher_forced"):
            ev["tf"], ev["tf_path"], ev["tf_summary"] = s2["teacher_forced"], rr, s2
            ev["notes"].append("held-out generation + teacher forcing taken from refusal_run/summary.json")
        if s2.get("finished") and ev["rg"] is None and s2.get("refusal_generation"):
            ev["rg"], ev["rg_path"] = s2["refusal_generation"], rr
    sel = os.path.join(path, "selection.json")
    if os.path.isfile(sel):
        ev["selection"], ev["selection_path"] = load_json(sel), sel
    return ev


# ---------------------------------------------------------------------------------------------------------
# 2. Optimisation runs
def scan_runs():
    runs = {}   # run name -> {"copies": [(box, path)], "model", "method", "tag"}
    for box, root in RUN_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not os.path.isdir(p) or name.startswith("bench__") or name == "logs":
                continue
            m = RUN_RE.match(name)
            if not m:
                continue
            r = runs.setdefault(name, {"name": name, "model": m.group("model"), "method": m.group("method"),
                                       "tag": m.group("tag"), "suffix": m.group("suffix"), "copies": []})
            r["copies"].append((box, p))
    for r in runs.values():
        origin = next(((b, p) for b, p in r["copies"] if b in ("mc-h200-1", "local-4090")), r["copies"][0])
        r["origin_box"], r["origin_path"] = origin
        r["meta"] = read_meta(os.path.join(origin[1], "meta.txt"))
        end, ec = r["meta"].get("end"), r["meta"].get("exit_code")
        r["state"] = ("finished" if end and ec == "0" else f"failed (exit_code {ec})" if end and ec
                      else "ended, no exit_code in meta.txt" if end
                      else "running or interrupted (no end in meta.txt)" if r["meta"] else "no meta.txt")
        # what the run itself wrote: Heretic's headless save (adapter_t0/ = LoRA adapter, model/ = merged pilot
        # export) and merged re-exports of a selected trial (model_kl*_t*/, 4090 runs)
        op = origin[1]
        r["has_adapter_t0"] = os.path.isdir(os.path.join(op, "adapter_t0"))
        r["merged_dirs"] = sorted(n for n in os.listdir(op) if os.path.isdir(os.path.join(op, n))
                                  and (n == "model" or re.fullmatch(r"model_kl[0-9.]+_t\d+", n)))
        cmd = r["meta"].get("cmd", "")
        src = "meta.txt"
        if r["origin_box"] == "local-4090" and re.fullmatch(r"heretic --model \S+ \(\S+\)", cmd):
            # meta.txt line written by scripts/local_queue.sh, which holds the actual heretic command
            cmd, src = (open(LOCAL_RUN_QUEUE).read() if os.path.isfile(LOCAL_RUN_QUEUE) else ""), rel(LOCAL_RUN_QUEUE)
        es, sd = re.search(r"--export-strategy (\S+)", cmd), re.search(r"--save-directory (\S+)", cmd)
        r["save_cmd"] = (f"--export-strategy {es.group(1)}" + (f" --save-directory {sd.group(1)}" if sd else "")
                         + f" ({src})" if es else "")
    return runs


EVAL_SEL_UNFINISHED = {}   # run -> detail, when the evaluated selection.json predates the end of the study
UNFINISHED_NOTE = "evaluated selection.json written before the study finished; a finished copy exists"


def selection_key(d):
    s = d.get("selected") or {}
    pt = {k: ((v or {}).get("trial_index"), dstr((v or {}).get("refusals")), dstr((v or {}).get("kl")))
          for k, v in (d.get("per_threshold") or {}).items()}
    return (d.get("run"), d.get("n_trials_complete"), d.get("n_trials_planned"), d.get("finished"),
            dstr((d.get("baselines") or {}).get("Refusals")), s.get("trial_index"), dstr(s.get("refusals")),
            dstr(s.get("kl")), tuple(sorted(pt.items())))


def find_selection(run, runs, ev):
    """Return (selection dict, path) for a run. Preference: the eval-dir copy (what was evaluated), else the
    copy from a finished study with the most completed trials. Every other copy is compared; a copy written
    before the study finished is reported as stale, any other difference as a mismatch."""
    copies = []
    if ev is not None and ev.get("selection") is not None and ev["selection"].get("run") == run:
        copies.append((ev["selection_path"], ev["selection"], True))
    r = runs.get(run)
    if r:
        order = sorted(r["copies"], key=lambda bp: [b for b, _ in RUN_ROOTS].index(bp[0]))
        for _, p in order:
            sp = os.path.join(p, "selection.json")
            if os.path.isfile(sp):
                copies.append((sp, load_json(sp), False))
    if not copies:
        return None, None
    if copies[0][2]:
        primary = copies[0]
        # the evaluated model was chosen from this copy; it must come from the finished study
        fin = [c for c in copies[1:] if c[1].get("finished")]
        if primary[1].get("finished"):
            R.check(f"evaluated selection is from the finished study ({run})", True, rel(primary[0]))
        elif fin:
            fs = (fin[0][1].get("selected") or {}).get("trial_index")
            detail = (f"{rel(primary[0])} was written at {primary[1].get('n_trials_complete')}/"
                      f"{primary[1].get('n_trials_planned')} trials (selected "
                      f"t{(primary[1].get('selected') or {}).get('trial_index')}); finished copy {rel(fin[0][0])} "
                      f"selects t{fs}")
            R.check(f"evaluated selection is from the finished study ({run})", False, detail)
            EVAL_SEL_UNFINISHED[run] = detail
    else:
        primary = max(copies, key=lambda c: (bool(c[1].get("finished")), c[1].get("n_trials_complete") or 0,
                                             -copies.index(c)))
    k0 = selection_key(primary[1])
    for p, d, _ in copies:
        if p == primary[0]:
            continue
        if selection_key(d) == k0:
            R.check(f"selection copies identical ({run})", True, f"{rel(primary[0])} vs {rel(p)}")
        elif not d.get("finished") and primary[1].get("finished"):
            st = d.get("selected") or {}
            R.checks.append((f"selection copies identical ({run})", "stale copy",
                             f"{rel(p)} was written before the study finished ({d.get('n_trials_complete')}/"
                             f"{d.get('n_trials_planned')} trials; selected t{st.get('trial_index')}); "
                             f"used {rel(primary[0])} ({primary[1].get('n_trials_complete')}/{primary[1].get('n_trials_planned')})"))
            R.anomaly("stale selection copy", f"{rel(p)}: written before the study finished "
                      f"({d.get('n_trials_complete')}/{d.get('n_trials_planned')} trials); not used. Used "
                      f"{rel(primary[0])}. Selected trial in stale copy: t{st.get('trial_index')} "
                      f"(used copy: t{(primary[1].get('selected') or {}).get('trial_index')})")
        else:
            R.check(f"selection copies identical ({run})", False, f"{rel(primary[0])} vs {rel(p)}")
    if primary[1].get("finished") is not True:
        R.anomaly("unfinished selection", f"{rel(primary[0])}: study not finished when selection.json was written")
    return primary[1], primary[0]


def read_queues():
    q = {}  # label -> (run or None, queue file, repo)
    for box, path in QUEUES.items():
        if not os.path.isfile(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|")]
            if box == "mc-rtx-1" and len(parts) >= 3:
                toks = parts[2].split()
                if "eval_one.sh" in parts[2]:
                    i = next(i for i, t in enumerate(toks) if t.endswith("eval_one.sh"))
                    args = [t for t in toks[i + 1:] if not t.startswith("--")]
                    label, repo = args[0], args[1]
                    run = args[2] if len(args) > 2 else None
                    q[label] = (run, rel(path), repo)
            elif box == "local-4090" and len(parts) >= 3:
                label, repo = parts[0], parts[1]
                run = parts[2].split()[0] if parts[2].split() else None
                q[label] = (run, rel(path), repo)
    return q


def h200_queued_runs():
    names = set()
    if os.path.isfile(H200_RUN_QUEUE):
        for line in open(H200_RUN_QUEUE):
            if line.strip() and not line.startswith("#"):
                names.add(line.split("|")[0].strip())
    return names


def resolve_run(model, method, evals, runs, queues):
    """Which optimisation run a (model, method) cell uses, and by which rule."""
    label = f"{model}__{method}"
    ev = evals.get(label)
    if ev and ev.get("selection"):
        return ev["selection"]["run"], f"named by {rel(ev['selection_path'])} (run the evaluation used)"
    if label in queues and queues[label][0]:
        return queues[label][0], f"named on the eval queue line in {queues[label][1]}"
    cands = [r for r in runs.values() if r["model"] == model and r["method"] == method
             and r["tag"].startswith("core-s1") and not r["suffix"]]
    fin = [r for r in cands if r["state"] == "finished"]
    if len(fin) == 1:
        return fin[0]["name"], "only finished core-s1 run for this model x method (no eval yet)"
    if len(fin) > 1:
        return None, "AMBIGUOUS: several finished runs " + ", ".join(sorted(r["name"] for r in fin))
    run_ = [r for r in cands if r["state"].startswith("running")]
    if len(run_) == 1:
        return run_[0]["name"], "only core-s1 run for this model x method (still running)"
    return None, "no run directory found"


# ---------------------------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evals = scan_evals()
    runs = scan_runs()
    queues = read_queues()
    h200q = h200_queued_runs()

    # refusal-set composition (counts only; prompt text is never printed)
    rs = load_json(REFUSAL_SETS)
    rs_counts = {}
    for it in rs["items"]:
        rs_counts[(it["set"], it["kind"])] = rs_counts.get((it["set"], it["kind"]), 0) + 1
    n_benign = sum(v for (s, k), v in rs_counts.items() if k == "benign")
    benign_sets = sorted((s, v) for (s, k), v in rs_counts.items() if k == "benign")
    rs_sha = rs["meta"].get("items_sha256")

    # repo id per model, from every file that names it
    repos = {m: set() for m in MODEL_INFO}
    for ev in evals.values():
        if ev["summary"] and ev["summary"].get("base_model"):
            repos[ev["model"]].add(ev["summary"]["base_model"])
    for label, (run, qf, repo) in queues.items():
        m = LABEL_RE.match(label)
        if m and m.group("model") in repos:
            repos[m.group("model")].add(repo)
    REPOS.update(repos)

    # ------------------------------------------------------------------ cells
    cells = {}  # (model, method) -> dict
    for model, info in MODEL_INFO.items():
        for method in ["base"] + METHODS[info["study"]]:
            c = {"model": model, "method": method, "study": info["study"], "label": f"{model}__{method}"}
            ev = evals.get(c["label"])
            c["eval"] = ev
            # evaluation status (T2/T3)
            if ev and ev["status"] == "complete":
                c["eval_status"] = "present"
            elif ev:
                c["eval_status"] = f"pending: evaluation {ev['status']} ({rel(ev['path'])})"
            elif c["label"] in queues:
                c["eval_status"] = f"pending: evaluation queued ({queues[c['label']][1]})"
            else:
                c["eval_status"] = "pending: evaluation not queued yet"
            # dev selection (T1)
            if method != "base":
                run, rule = resolve_run(model, method, evals, runs, queues)
                c["run"], c["run_rule"] = run, rule
                sel, selp = (find_selection(run, runs, ev) if run else (None, None))
                c["selection"], c["selection_path"] = sel, selp
                if sel is not None:
                    c["sel_status"] = "present"
                elif run and run in runs:
                    st = runs[run]["state"]
                    c["sel_status"] = (f"pending: run {st}, selection.json not written yet (run "
                                       "scripts/select_trial.py on the run dir, then regenerate)" if st == "finished"
                                       else f"pending: run {st}")
                elif run:
                    c["sel_status"] = "pending: run dir not found"
                else:
                    queued = [n for n in h200q if n.startswith(f"{model}__{method}__core-s1")]
                    c["sel_status"] = ("pending: run queued on H200, not started (" + ", ".join(sorted(queued)) + ")"
                                       if queued else f"missing: {rule}")
                # alternative / superseded runs for this cell
                c["other_runs"] = sorted(
                    (r["name"], r["state"]) for r in runs.values()
                    if r["model"] == model and r["method"] == method and r["name"] != run)
                for name, st in c["other_runs"]:
                    if st != "finished" or not runs[name]["tag"].startswith("core-s1") or runs[name]["suffix"]:
                        continue
                    alt, altp = find_selection(name, runs, None)
                    if alt is not None:
                        t = (alt.get("per_threshold") or {}).get("0.1") or {}
                        R.anomaly("alternative complete run",
                                  f"{c['label']}: table uses `{run}`; another finished run `{name}` exists "
                                  f"({alt.get('n_trials_complete')}/{alt.get('n_trials_planned')} trials; KL<=0.1 pick "
                                  f"t{t.get('trial_index')} {t.get('refusals_display')} at KL {dstr(t.get('kl'))}; "
                                  f"{rel(altp)}) and is not used")
            cells[(model, method)] = c

    # ------------------------------------------------------------------ consistency checks
    # selection vs re-export (trial actually exported/evaluated)
    for c in cells.values():
        ev, sel = c["eval"], c.get("selection")
        if c["method"] == "base" or not ev or not sel:
            continue
        # evaluations that have not finished are checked under a separate name, so the counts for
        # finished evaluations are not inflated by runs still in progress
        tag = "" if ev["status"] == "complete" else " [evaluation not finished]"
        t_sel = (sel.get("selected") or {}).get("trial_index")
        # T2/T3 call the evaluated model the KL <= 0.1 trial: the selection must say so
        rule_kl = (sel.get("rule") or {}).get("kl_max")
        t01 = ((sel.get("per_threshold") or {}).get("0.1") or {}).get("trial_index")
        ok01 = (rule_kl is not None and Decimal(str(rule_kl)) == Decimal("0.1") and t_sel is not None
                and t_sel == t01)
        R.check(f"selected trial is the KL<=0.1 pick{tag} ({c['label']})", ok01,
                f"rule.kl_max {dstr(rule_kl)}, selected t{t_sel}, per_threshold['0.1'] t{t01} "
                f"({rel(c.get('selection_path'))})")
        if not ok01 and c["eval_status"] == "present":
            c["eval_status"] = ("excluded: evaluated trial is not the KL<=0.1 selection "
                                f"(rule.kl_max {dstr(rule_kl)}, selected t{t_sel}, per_threshold['0.1'] t{t01})")
        rx = os.path.join(ev["path"], "reexport.sh")
        if os.path.isfile(rx):
            txt = open(rx).read()
            m = re.search(r"--expect-trial (\d+)", txt)
            R.check(f"re-export trial == selected trial{tag} ({c['label']})", m and int(m.group(1)) == t_sel,
                    f"reexport.sh --expect-trial {m.group(1) if m else None}, selection t{t_sel}")
            R.check(f"re-export is a MERGE export{tag} ({c['label']})", "--export-strategy MERGE" in txt,
                    "reexport.sh contains '--export-strategy MERGE'")
        mp = (ev["summary"] or {}).get("model", "")
        m2 = re.search(r"model_kl[0-9.]+_t(\d+)$", str(mp))
        if m2:
            R.check(f"evaluated model dir trial == selected trial{tag} ({c['label']})", int(m2.group(1)) == t_sel,
                    f"summary.model {mp}")
    # baselines (original-model dev refusals) consistent across runs of a model
    base_ref = {}
    for model in MODEL_INFO:
        vals = {}
        for (mm, meth), c in cells.items():
            if mm == model and meth != "base" and c.get("selection"):
                b = (c["selection"].get("baselines") or {}).get("Refusals")
                vals.setdefault(dstr(b), []).append(rel(c["selection_path"]))
        if len(vals) > 1:
            R.anomaly("baseline mismatch", f"{model}: original-model dev refusals differ across runs: {vals}")
        base_ref[model] = vals

    # ------------------------------------------------------------------ T1
    t1_rows = []
    for model, info in MODEL_INFO.items():
        repo = "; ".join(sorted(repos[model]))
        # original-model baseline (from every selection.json of this model; first in method order is cited)
        bsrc = next((c for (mm, meth), c in cells.items() if mm == model and meth != "base" and c.get("selection")), None)
        row = {"model": model, "model_repo": repo, "study": info["study"], "family": info["family"],
               "generation": info["generation"], "method": "base", "kl_max": "", "status": "", "refusals_count": "",
               "refusals_denominator": "", "refusals_fraction": "", "kl": "", "trial_index": "",
               "pareto_position": "", "n_trials_complete": "", "n_trials_planned": "", "seed": "", "run": "",
               "run_resolution": "", "source_file": "", "note": ""}
        if bsrc:
            sel = bsrc["selection"]
            b = sel["baselines"]["Refusals"]
            den = next((parse_display(v.get("refusals_display"))[1] for v in sel["per_threshold"].values() if v), None)
            cnt = b * den if den else None
            row.update(status="present", refusals_fraction=dstr(b), refusals_denominator=den or "",
                       refusals_count=(dstr(cnt) if cnt is not None and cnt == cnt.to_integral_value() else ""),
                       kl=dstr((sel.get("baselines") or {}).get("KL divergence")),
                       source_file=rel(bsrc["selection_path"]), run=sel["run"],
                       note=("baseline = original model's keyword refusals recorded by Heretic in the run; "
                             f"identical in {sum(len(v) for v in base_ref[model].values())} selection.json file(s)"))
            if cnt is not None and cnt != cnt.to_integral_value():
                R.anomaly("non-integer count", f"{model} baseline {b} x {den} not integral")
        else:
            row.update(status="pending: no selection.json for any run of this model")
        cells[(model, "base")]["t1_base"] = row
        t1_rows.append(row)
        for method in METHODS[info["study"]]:
            c = cells[(model, method)]
            sel = c.get("selection")
            for thr in KL_ALL:
                r = dict(row)
                r.update(method=method, kl_max=thr, run=c.get("run") or "", run_resolution=c.get("run_rule", ""),
                         note="", source_file=rel(c.get("selection_path")) if sel else "",
                         refusals_count="", refusals_denominator="", refusals_fraction="", kl="",
                         trial_index="", pareto_position="", n_trials_complete="", n_trials_planned="", seed="")
                if sel is None:
                    r["status"] = c["sel_status"]
                else:
                    r["n_trials_complete"] = sel.get("n_trials_complete")
                    r["n_trials_planned"] = sel.get("n_trials_planned")
                    r["seed"] = sel.get("seed", "")
                    t = (sel.get("per_threshold") or {}).get(thr)
                    if t is None:
                        lk = sel.get("lowest_kl_trial") or {}
                        r["status"] = "none qualifies (no completed trial with KL <= threshold)"
                        r["note"] = f"lowest-KL trial t{lk.get('trial_index')} KL={dstr(lk.get('kl'))}"
                    else:
                        cnt, den = parse_display(t.get("refusals_display"))
                        r.update(status="present", refusals_count=cnt if cnt is not None else "",
                                 refusals_denominator=den if den is not None else "",
                                 refusals_fraction=dstr(t.get("refusals")), kl=dstr(t.get("kl")),
                                 trial_index=t.get("trial_index"), pareto_position=t.get("pareto_position"))
                        if cnt is not None and Decimal(cnt) != t.get("refusals") * den:
                            R.anomaly("display mismatch", f"{c['label']} KL<={thr}: display {t.get('refusals_display')} vs fraction {t.get('refusals')}")
                    if sel.get("n_trials_complete") != sel.get("n_trials_planned"):
                        r["note"] = (r["note"] + "; " if r["note"] else "") + \
                            f"{sel.get('n_trials_complete')}/{sel.get('n_trials_planned')} trials complete"
                    if c.get("run") in EVAL_SEL_UNFINISHED:
                        r["note"] = (r["note"] + "; " if r["note"] else "") + \
                            UNFINISHED_NOTE + ": " + EVAL_SEL_UNFINISHED[c["run"]]
                t1_rows.append(r)
    write_csv("t1_dev_refusals.csv", t1_rows)

    # ------------------------------------------------------------------ T2
    t2_rows = []
    bench_n = {}
    for model, info in MODEL_INFO.items():
        base = cells[(model, "base")]
        bev = base["eval"] if base["eval_status"] == "present" else None
        for method in ["base"] + METHODS[info["study"]]:
            c = cells[(model, method)]
            ev = c["eval"] if c["eval_status"] == "present" else None
            for b, bname in BENCHES:
                r = {"model": model, "model_repo": "; ".join(sorted(repos[model])), "study": info["study"],
                     "method": method, "benchmark": b, "metric": "", "n": "", "score_fraction": "",
                     "score_pct": "", "stderr_fraction": "", "original_score_pct": "", "delta_pct_points": "",
                     "status": "", "eval_label": c["label"], "source_file": "", "note": ""}
                v = ev["bench"].get(b) if ev else None
                if v is None:
                    r["status"] = c["eval_status"] if not ev else "missing: benchmark absent from finished summary"
                else:
                    sc = v["score"]
                    r.update(metric=v.get("metric", ""), n=v.get("n", ""), score_fraction=dstr(sc),
                             stderr_fraction=dstr(v.get("stderr")), source_file=rel(ev["summary_path"]))
                    bench_n.setdefault(b, {}).setdefault(v.get("n"), []).append(c["label"])
                    if not (Decimal(0) <= sc <= Decimal(1)):
                        r["status"] = "excluded: score outside [0,1] (not a fraction?)"
                        R.anomaly("fraction vs percent", f"{c['label']} {b} score {sc}")
                        t2_rows.append(r)
                        continue
                    r["score_pct"] = dstr(sc * 100)
                    r["status"] = "present"
                    notes = []
                    if v.get("rescored"):
                        notes.append(f"rescored: {v.get('rescored')}; unpatched-extractor score {dstr(v.get('score_unpatched_extractor'))}")
                        R.anomaly("rescored benchmark", f"{c['label']} {b}: score {dstr(sc)} is a rescore "
                                  f"({v.get('rescored')}); field score_unpatched_extractor = "
                                  f"{dstr(v.get('score_unpatched_extractor'))} ({rel(ev['summary_path'])})")
                    if method != "base":
                        bv = bev["bench"].get(b) if bev else None
                        if bv is None:
                            notes.append("original-model score missing -> no delta")
                        else:
                            r["original_score_pct"] = dstr(bv["score"] * 100)
                            r["delta_pct_points"] = dstr((sc - bv["score"]) * 100)
                            if bv.get("metric") != v.get("metric") or bv.get("n") != v.get("n"):
                                R.anomaly("benchmark protocol", f"{c['label']} {b}: metric/n differ from original")
                    r["note"] = "; ".join(notes)
                t2_rows.append(r)
    for b, d in bench_n.items():
        if len(d) > 1:
            R.anomaly("benchmark n", f"{b}: different n across cells: { {k: len(v) for k, v in d.items()} }")
    write_csv("t2_capability.csv", t2_rows)

    # ------------------------------------------------------------------ T3
    t3_rows = []
    tf_items_cache = {}

    def tf_items(ev):
        """id -> (n_tokens, sum_logprob) from refusal/teacher_forced.jsonl next to the summary that reports it."""
        p = os.path.join(os.path.dirname(ev["tf_path"]), "refusal", "teacher_forced.jsonl")
        if p in tf_items_cache:
            return tf_items_cache[p], p
        items = {}
        if os.path.isfile(p):
            for line in open(p):
                if line.strip():
                    d = json.loads(line, parse_float=Decimal)
                    items[d["id"]] = (d["n_tokens"], d["sum_logprob"])
        tf_items_cache[p] = items
        return items, p

    for model, info in MODEL_INFO.items():
        base = cells[(model, "base")]
        bev = base["eval"] if (base["eval_status"] == "present" and base["eval"]["tf"]) else None
        for method in ["base"] + METHODS[info["study"]]:
            c = cells[(model, method)]
            ev = c["eval"] if c["eval_status"] == "present" else None
            r = {"model": model, "model_repo": "; ".join(sorted(repos[model])), "study": info["study"],
                 "method": method, "status": "", "n_items": "", "n_tokens": "", "mean_logprob": "",
                 "original_mean_logprob": "", "shift_vs_original": "", "reference_label": "",
                 "jsonl_recomputed_check": "", "paired_with_original": "", "eval_label": c["label"],
                 "source_file": "", "note": ""}
            tf = ev["tf"] if ev else None
            if tf is None:
                r["status"] = c["eval_status"] if not ev else "missing: no teacher_forced block"
                t3_rows.append(r)
                continue
            mlp = tf["mean_logprob"]
            ref_label = label_from_path(tf.get("reference"))
            r.update(status="present", n_items=tf.get("n_items"), n_tokens=tf.get("n_tokens"),
                     mean_logprob=dstr(mlp), reference_label=ref_label or "", source_file=rel(ev["tf_path"]))
            R.check(f"teacher-forcing reference is the original model's answers ({c['label']})",
                    ref_label == f"{model}__base", f"reference={tf.get('reference')}")
            items, ip = tf_items(ev)
            if items:
                tot_n = sum(n for n, _ in items.values())
                tot_lp = sum((s for _, s in items.values()), Decimal(0))
                rec = tot_lp / tot_n if tot_n else None
                ok = rec is not None and abs(rec - mlp) <= Decimal("1e-9") and tot_n == tf.get("n_tokens") \
                    and len(items) == tf.get("n_items")
                r["jsonl_recomputed_check"] = "ok" if ok else f"MISMATCH (recomputed {dstr(rec)}, tokens {tot_n}, items {len(items)})"
                R.check(f"teacher_forced.jsonl reproduces summary mean/n_tokens/n_items ({c['label']})", ok,
                        f"{rel(ip)}")
                if not ok:
                    R.anomaly("teacher-forced jsonl", f"{c['label']}: {r['jsonl_recomputed_check']} ({rel(ip)})")
            else:
                r["jsonl_recomputed_check"] = "teacher_forced.jsonl not found"
            if method == "base":
                r["original_mean_logprob"] = dstr(mlp)
                r["note"] = "original model scored on its own greedy answers"
            elif bev is None:
                r["note"] = "original-model teacher-forced score missing -> no shift"
            else:
                bmlp = bev["tf"]["mean_logprob"]
                r["original_mean_logprob"] = dstr(bmlp)
                r["shift_vs_original"] = dstr(mlp - bmlp)
                bitems, _ = tf_items(bev)
                same = bool(items) and set(items) == set(bitems) and all(items[k][0] == bitems[k][0] for k in items)
                r["paired_with_original"] = "yes" if same else "NO"
                R.check(f"teacher-forced items/tokens paired with original ({c['label']})", same,
                        f"{len(items)} vs {len(bitems)} items; n_tokens {tf.get('n_tokens')} vs {bev['tf'].get('n_tokens')}")
            t3_rows.append(r)
    write_csv("t3_teacher_forced.csv", t3_rows)

    # ------------------------------------------------------------------ further checks (protocol, artefacts)
    protocol = []
    for c in cells.values():
        ev = c["eval"]
        if not ev or not ev["summary"]:
            continue
        s = ev["summary"]
        protocol.append(dict(
            {"label": c["label"], "status": ev["status"], "limit_mode": s.get("limit_mode")}, **proto_fields(s),
            rg_n=(ev["rg"] or {}).get("n"),
            rg_sha_ok=((ev["rg"] or {}).get("items_sha256") == rs_sha) if ev["rg"] else None))
        if ev["tf_summary"] is not None and ev["tf_path"] != ev["summary_path"]:
            # held-out generation + teacher forcing from a separate run (refusal_run/summary.json)
            s2 = ev["tf_summary"]
            protocol.append(dict(
                {"label": c["label"] + " (" + os.path.relpath(ev["tf_path"], ev["path"]) + ")",
                 "status": "complete" if s2.get("finished") else "in progress (no 'finished' timestamp)",
                 "limit_mode": s2.get("limit_mode")}, **proto_fields(s2),
                rg_n=(ev["rg"] or {}).get("n"),
                rg_sha_ok=((ev["rg"] or {}).get("items_sha256") == rs_sha) if ev["rg"] else None))
        if s.get("backend") != "vllm":
            R.anomaly("backend", f"{c['label']}: backend {s.get('backend')}")
        if s.get("limit_mode") != "paper":
            R.anomaly("limit mode", f"{c['label']}: limit_mode {s.get('limit_mode')}")
        if ev["rg"] and ev["rg"].get("items_sha256") != rs_sha:
            R.anomaly("refusal set", f"{c['label']}: held-out items_sha256 differs from scripts/bench/refusal_sets.json")
    # within-model protocol differences (original vs derived)
    for model in MODEL_INFO:
        pm = [p for p in protocol if p["label"].startswith(model + "__") and p["status"] == "complete"]
        for key in ("backend", "postprocess", "vllm_extra", "gpu_util", "gpu", "vllm", "lm_eval", "bench_py"):
            vals = {}
            for p in pm:
                vals.setdefault(str(p[key]), []).append(p["label"].split("__")[1])
            if len(vals) > 1:
                R.anomaly("protocol drift", f"{model}: {key} differs between cells: {vals}")
    # evaluated artefacts: merged weights, not adapters; bytes vs original; tensor headers where weights are local.
    # model_hash.json 'bytes' (bench.py tree_hash; copied to summary.json model_info.weights_bytes) is the total of
    # weight + config + tokenizer files, NOT weights only, and the non-weight file sets differ between Hugging Face
    # snapshots and Heretic exports. Weights-only sizes are computed here only where both dirs are on this machine.
    artefacts = []
    base_hash = {}
    for c in cells.values():
        if c["method"] == "base" and c["eval"] and c["eval_status"] == "present":
            mh = os.path.join(c["eval"]["path"], "model_hash.json")
            if os.path.isfile(mh):
                base_hash[c["model"]] = load_json(mh)
    pending_size = []   # export-size anomalies, written after every header check has run
    local_nw = []       # (label, safetensors diff, non-weight diff) measured on this machine
    for c in cells.values():
        ev = c["eval"]
        if not ev or c["eval_status"] != "present":
            continue
        mh = os.path.join(ev["path"], "model_hash.json")
        if not os.path.isfile(mh):
            artefacts.append((c["label"], "model_hash.json absent", "", "", "", "", "", "", "", ""))
            continue
        d = load_json(mh)
        files = d.get("files", {})
        shards = [f for f in files if f.endswith(".safetensors")]
        adapters = [f for f in files if "adapter" in f.lower()]
        bh = base_hash.get(c["model"]) or {}
        bb = bh.get("bytes")
        diff = "" if c["method"] == "base" or bb is None or d.get("bytes") is None else str(d["bytes"] - bb)
        hdr = "original" if c["method"] == "base" else header_compare(c, cells)
        # weights-only vs non-weight bytes, where the files are local
        own = local_sizes(eval_model_dir(ev))
        if own is not None:
            R.check(f"local file sizes reproduce model_hash.json bytes ({c['label']})", sum(own.values()) == d.get("bytes"),
                    f"{sum(own.values())} vs {d.get('bytes')} ({rel(mh)})")
        st_diff = nw_diff = ""
        if c["method"] != "base":
            bev = cells[(c["model"], "base")]["eval"]
            orig = local_sizes(eval_model_dir(bev)) if bev else None
            if own is not None and orig is not None:
                st_diff = str(sum(v for k, v in own.items() if k.endswith(".safetensors"))
                              - sum(v for k, v in orig.items() if k.endswith(".safetensors")))
                nw_diff = str(sum(v for k, v in own.items() if not k.endswith(".safetensors"))
                              - sum(v for k, v in orig.items() if not k.endswith(".safetensors")))
                local_nw.append((c["label"], st_diff, nw_diff))
            else:
                st_diff = nw_diff = "not on this machine"
        nwf = "" if c["method"] == "base" or not bh else nonweight_files_diff(bh.get("files", {}), files)
        artefacts.append((c["label"], str(len(shards)), ", ".join(adapters) or "none", str(d.get("bytes", "")),
                          diff, st_diff, nw_diff, nwf,
                          str((ev["summary"].get("model_info") or {}).get("source", "")), hdr))
        if c["method"] != "base" and adapters:
            R.anomaly("artefact", f"{c['label']}: adapter files in evaluated model: {adapters}")
        if diff and abs(int(diff)) > 1_000_000 and hdr.startswith("not checkable"):
            n_b = sum(1 for f in bh.get("files", {}) if f.endswith(".safetensors"))
            pending_size.append((c["label"], d.get("bytes"), bb, int(diff), n_b, len(shards), nwf))
    hdr_diff = {}   # model -> [method] of local exports whose tensor headers differ from the original
    for a in artefacts:
        if a[-1] not in ("original", "") and not a[-1].startswith(("same ", "not checkable")):
            mm, meth = a[0].split("__", 1)
            hdr_diff.setdefault(mm, []).append(meth)
    hdr_diff_txt = ", ".join(f"{mm}__{{{','.join(sorted(v))}}}" for mm, v in sorted(hdr_diff.items()))
    local_ex, seen_m = [], set()   # first locally measured export per model
    for lb, st, nw in local_nw:
        if lb.split("__")[0] not in seen_m:
            seen_m.add(lb.split("__")[0])
            local_ex.append((lb, st, nw))
    for label, tot, bb, diff, n_b, n_e, nwf in pending_size:
        txt = (f"{label}: model_hash.json 'bytes' = total of weight + config + tokenizer files (bench.py tree_hash; "
               f"not weights-only): {tot} vs {bb} for the original (difference {diff}). The evaluated weights are not on "
               f"this machine, so no weights-only size or tensor-header comparison is possible. safetensors shards: "
               f"{n_b} in the original, {n_e} in the export. Non-weight files vs the original: {nwf}")
        if nwf != "identical":
            txt += ("; their sizes are not recorded in model_hash.json, so this record cannot split the difference "
                    "between weights and non-weight files, and it is not by itself evidence of a weight difference")
            if local_ex:
                txt += (" (where measurable on this machine: "
                        + "; ".join(f"{lb} non-weight {int(nw):+,} B, safetensors {int(st):+,} B"
                                    for lb, st, nw in local_ex) + ")")
        if diff < 0 and hdr_diff:
            txt += (". Tensors dropped or re-typed at export would also reduce the size (found by the header check "
                    "on this machine for " + hdr_diff_txt + "); cause unverified for this model")
        R.anomaly("export size", txt + ".")

    # ------------------------------------------------------------------ LaTeX
    write_t1_tex(cells, t1_rows, now, benign_sets)
    write_t2_tex(cells, t2_rows, now)
    write_t3_tex(cells, t3_rows, now, n_benign, benign_sets)

    # ------------------------------------------------------------------ coverage.md
    write_coverage(now, evals, runs, cells, queues, protocol, artefacts, rs_counts, bench_n)
    print(f"wrote {len(os.listdir(OUT))} files to {rel(OUT)}/")
    for f in sorted(os.listdir(OUT)):
        print("  " + rel(os.path.join(OUT, f)))
    print(f"anomalies: {len(R.anomalies)}; failed checks: {sum(1 for c in R.checks if c[1] == 'MISMATCH')}")
    return 0


# ---------------------------------------------------------------------------------------------------------
def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


REPOS = {}   # model -> set of repo ids named in the files (filled in main)


def model_display(model, cells):
    names = sorted({r.split("/")[-1] for r in REPOS.get(model, set())})
    if len(names) == 1:
        return names[0]
    for c in cells.values():
        if c["model"] == model and c.get("selection") and c["selection"].get("model"):
            return c["selection"]["model"].split("/")[-1]
    return model


def safetensors_header(d):
    """tensor name -> (dtype, shape) from the JSON headers of every *.safetensors file in d (headers only)."""
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        with open(os.path.join(d, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], tuple(v["shape"]))
    return out


_HDR_CACHE = {}

# evaluation-harness settings compared between an original model's run and its modified models' runs
PROTO_KEYS = [("bench_py", "bench.py revision (sha256 prefix)"), ("postprocess", "output post-processing"),
              ("vllm_extra", "BENCH_VLLM_EXTRA"), ("backend", "backend"), ("gpu_util", "GPU memory utilisation"),
              ("gpu", "GPU"), ("vllm", "vLLM"), ("lm_eval", "lm-eval")]


def proto_fields(s):
    """Harness settings recorded in a bench.py summary.json."""
    s = s or {}
    v = s.get("versions", {})
    return {"backend": s.get("backend"), "bench_py": (v.get("sha256:bench.py") or "")[:10],
            "postprocess": "U+2581->space" if s.get("postprocess") else "none",
            "vllm_extra": (v.get("env") or {}).get("BENCH_VLLM_EXTRA"),
            "gpu_util": (s.get("lm_kwargs") or {}).get("gpu_memory_utilization"), "gpu": v.get("gpu"),
            "vllm": v.get("vllm"), "lm_eval": v.get("lm_eval")}


def base_protocol_diffs(cells, study, which):
    """model -> (diffs, separate_run, source) for original-model rows whose evaluation run differs from the runs
    of the same family's modified models. which='t2': the summary.json with the benchmark scores; 't3': the
    summary.json that reports teacher forcing (may be refusal_run/summary.json). diffs = [(setting, original
    value, sorted modified values)] for settings whose original value occurs in none of the modified runs."""
    out = {}
    for model, info in MODEL_INFO.items():
        if info["study"] != study:
            continue
        b = cells[(model, "base")]
        if b["eval_status"] != "present":
            continue
        bev = b["eval"]
        bs = bev["summary"] if which == "t2" else bev["tf_summary"]
        if not bs:
            continue
        mods = []
        for meth in METHODS[study]:
            c = cells[(model, meth)]
            if c["eval_status"] == "present":
                s = c["eval"]["summary"] if which == "t2" else c["eval"]["tf_summary"]
                if s:
                    mods.append(proto_fields(s))
        if not mods:
            continue
        bp = proto_fields(bs)
        diffs = []
        for k, name in PROTO_KEYS:
            mv = sorted({str(m[k]) for m in mods})
            if str(bp[k]) not in mv:
                diffs.append((name, str(bp[k]), mv))
        sep = which == "t3" and bev["tf_path"] != bev["summary_path"]
        if diffs or sep:
            out[model] = (diffs, sep, rel(bev["tf_path"] if which == "t3" else bev["summary_path"]))
    return out


def tex_setting(v):
    return {"None": "unset", "{}": "\\{\\}", "U+2581->space": "U+2581$\\rightarrow$space"}.get(v, tex_escape(v))


def protocol_footnote(diffs_by_model, cells, lead):
    parts = []
    for model, (diffs, sep, src) in diffs_by_model.items():
        items = []
        if sep:
            items.append("separate bench.py run (refusal\\_run/summary.json in its evaluation directory)")
        items += [f"{tex_escape(name)} {tex_setting(bv)} vs.\\ {', '.join(tex_setting(x) for x in mv)}"
                  for name, bv, mv in diffs]
        parts.append(f"{tex_escape(model_display(model, cells))}: " + "; ".join(items))
    if not parts:
        return ""
    return (f" $^\\S${lead} (original vs.\\ modified models; per-cell settings in paper/tables/coverage.md, "
            "evaluation protocol table): " + ". ".join(parts) + ".")

# file filter of scripts/bench/bench.py tree_hash (what model_hash.json 'bytes' sums)
TREE_HASH_SUFFIXES = (".safetensors", ".bin", ".pt", ".json", ".model", ".jinja", ".txt", ".tiktoken")


def eval_model_dir(ev):
    """Local directory of the model an evaluation scored (summary.json model_info.path, else 'model'), or None."""
    s = (ev or {}).get("summary") or {}
    for p in ((s.get("model_info") or {}).get("path"), s.get("model")):
        if p and os.path.isdir(str(p)):
            return str(p)
    return None


def local_sizes(d):
    """rel name -> size for the files bench.py tree_hash counts (same filter and exclusions); None if d is None."""
    if d is None:
        return None
    from pathlib import Path
    root, out = Path(d), {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in TREE_HASH_SUFFIXES:
            r = str(f.relative_to(root))
            if r.startswith((".cache", "reproduce/")):
                continue
            out[r] = f.resolve().stat().st_size
    return out


def nonweight_files_diff(bfiles, efiles):
    """Compare the non-safetensors files (config, tokenizer, index) of two model_hash.json 'files' maps by name
    and sha256."""
    b = {k: v for k, v in bfiles.items() if not k.endswith(".safetensors")}
    e = {k: v for k, v in efiles.items() if not k.endswith(".safetensors")}
    only_b, only_e = sorted(set(b) - set(e)), sorted(set(e) - set(b))
    chg = sorted(k for k in set(b) & set(e) if b[k] != e[k])
    if not (only_b or only_e or chg):
        return "identical"
    parts = []
    if only_b:
        parts.append("only in original: " + ", ".join(only_b))
    if only_e:
        parts.append("only in export: " + ", ".join(only_e))
    if chg:
        parts.append("content differs: " + ", ".join(chg))
    return "; ".join(parts)


def header_compare(c, cells):
    """Compare tensor names/dtypes/shapes of the evaluated merged model with the original snapshot, when both
    are on this machine (4090 runs keep model_kl*_t*/; the RTX-box exports are deleted after evaluation)."""
    ev = c["eval"]
    mp = str((ev["summary"] or {}).get("model", ""))
    base = cells[(c["model"], "base")]["eval"]
    bp = str(((base or {}).get("summary") or {}).get("model_info", {}).get("path", "")) if base else ""
    if not (mp and os.path.isdir(mp)):
        return "not checkable here (evaluated weights not on this machine)"
    if not (bp and os.path.isdir(bp)):
        return "not checkable here (original snapshot not on this machine)"
    for p in (mp, bp):
        if p not in _HDR_CACHE:
            _HDR_CACHE[p] = safetensors_header(p)
    a, b = _HDR_CACHE[bp], _HDR_CACHE[mp]
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    changed = {}
    for k in set(a) & set(b):
        if a[k] != b[k]:
            key = (a[k][0] + "->" + b[k][0]) if a[k][1] == b[k][1] else "shape change"
            changed[key] = changed.get(key, 0) + 1

    def groups(names):
        g = {}
        for k in names:
            gk = re.sub(r"\.\d+\.", ".N.", k)
            gk = re.sub(r"^(mtp)\..*", r"\1.*", gk)
            g[gk] = g.get(gk, 0) + 1
        return "; ".join(f"{k} x{v}" for k, v in sorted(g.items()))
    if not missing and not extra and not changed:
        return f"same {len(a)} tensor names, dtypes and shapes as the original"
    parts = [f"original {len(a)} tensors, evaluated {len(b)}"]
    if missing:
        parts.append(f"absent from evaluated model: {len(missing)} ({groups(missing)})")
    if extra:
        parts.append(f"only in evaluated model: {len(extra)} ({groups(extra)})")
    if changed:
        chg = [k for k in set(a) & set(b) if a[k] != b[k]]
        parts.append("dtype/shape changed: " + ", ".join(f"{k} x{v}" for k, v in sorted(changed.items()))
                     + f" ({groups(chg)})")
    txt = "; ".join(parts)
    R.anomaly("export differs from original beyond the edit", f"{c['label']}: {txt} "
              f"(evaluated {mp}; original {bp}; safetensors headers)")
    return txt


def header_rows(models):
    fams, i = [], 0
    while i < len(models):
        f = MODEL_INFO[models[i]]["family"]
        j = i
        while j < len(models) and MODEL_INFO[models[j]]["family"] == f:
            j += 1
        fams.append((f, i, j))
        i = j
    return fams


def missing_comment(missing):
    out = ["% Missing cells rendered as '--' (status from the CSV):"]
    out += [f"%   {m}" for m in missing] if missing else ["%   none"]
    return out


def todo_line(missing, what):
    if not missing:
        return []
    return [f"\\todo{{{what}: {len(missing)} cell(s) pending; regenerate with scripts/make_results_tables.py}}"]


def write_t1_tex(cells, t1_rows, now, benign_sets):
    idx = {(r["model"], r["method"], r["kl_max"]): r for r in t1_rows}
    for study in ("main", "small"):
        models = [m for m, i in MODEL_INFO.items() if i["study"] == study]
        fams = header_rows(models)
        ncol = len(models)
        missing, short, unfin = [], [], []
        present = [r for r in t1_rows if MODEL_INFO[r["model"]]["study"] == study and r["method"] != "base"
                   and r["status"] == "present"]
        seeds = sorted({str(r["seed"]) for r in present})
        planned = sorted({str(r["n_trials_planned"]) for r in present})

        def cell(m, meth, thr):
            r = idx.get((m, meth, thr))
            if r and r["status"] == "present" and r["refusals_count"] != "":
                s = str(r["refusals_count"])
                if meth != "base" and r["n_trials_complete"] != r["n_trials_planned"]:
                    short.append(f"{model_display(m, cells)} {METHOD_TEX[meth]}: {r['n_trials_complete']} of "
                                 f"{r['n_trials_planned']}")
                    s += "$^\\dagger$"
                if meth != "base" and UNFINISHED_NOTE in r["note"]:
                    unfin.append(f"{model_display(m, cells)} {METHOD_TEX[meth]}")
                    s += "$^\\ast$"
                return s
            missing.append(f"{m} x {METHOD_TEX[meth]}" + (f" @KL<={thr}" if thr else " (original)") +
                           f": {r['status'] if r else 'no row'}")
            return "--"
        body = ["Original & & " + " & ".join(cell(m, "base", "") for m in models) + " \\\\"]
        for meth in METHODS[study]:
            body.append("\\midrule")
            for k, thr in enumerate(KL_TEX):
                body.append(f"{METHOD_TEX[meth] if k == 0 else ''} & {thr} & " +
                            " & ".join(cell(m, meth, thr) for m in models) + " \\\\")
        short_runs = sorted(set(short))
        dag = ""
        if short_runs:
            dag = (" $^\\dagger$Run with fewer completed than planned trials ("
                   + "; ".join(tex_escape(x) for x in short_runs) + ").")
        if unfin:
            dag += (" $^\\ast$Selection taken from the copy used for evaluation, written before the study "
                    "finished; a finished copy exists (paper/tables/coverage.md, consistency checks): "
                    + "; ".join(tex_escape(x) for x in sorted(set(unfin))) + ".")
        L = [f"% Auto-generated by scripts/make_results_tables.py at {now} -- do not edit by hand.",
             "\\begin{table*}[t]" if study == "main" else "\\begin{table}[t]",
             "\\centering\\small",
             "\\caption{Keyword refusals on the development set (mlabonne/harmful\\_behaviors, test split, first 100 "
             "prompts; count out of 100, Heretic keyword scorer) for the trial selected at each first-token KL "
             "budget $\\tau$ (KL on mlabonne/harmless\\_alpaca, test split, first 100 prompts). Selection rule: among "
             "completed trials with KL $\\le\\tau$, fewest refusals; ties broken by lower KL, then lower trial index. "
             "``Original'' is the unmodified model as recorded by Heretic in the same runs. "
             + ("Main study" if study == "main" else "Small pair")
             + f"; seed {', '.join(seeds)}; {', '.join(planned)} trials planned per run." + dag
             + " --: not yet available.}",
             f"\\label{{tab:dev-refusals-{study}}}",
             "\\begin{tabular}{ll" + "r" * ncol + "}",
             "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{{j - i}}}{{c}}{{{tex_escape(f)}}}" for f, i, j in fams) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + i}-{2 + j}}}" for f, i, j in fams),
             "Method & $\\tau$ & " + " & ".join(MODEL_INFO[m]["col"] for m in models) + " \\\\",
             "\\midrule"] + body
        L += ["\\bottomrule", "\\end{tabular}",
              "\\end{table*}" if study == "main" else "\\end{table}",
              "% src: paper/tables/t1_dev_refusals.csv (built from selection.json files; per-cell source_file column)",
              "% src: " + ", ".join(sorted({r['source_file'] for r in t1_rows
                                             if r['source_file'] and MODEL_INFO[r['model']]['study'] == study})),
              "% src: ../heretic/config.default.toml (KeywordRate: mlabonne/harmful_behaviors test[:100]; "
              "KLDivergence: mlabonne/harmless_alpaca test[:100]); scripts/select_trial.py (selection rule)",
              "% src: seed, n_trials_planned, n_trials_complete: the selection.json files above (study settings "
              "stored in each Optuna journal); configs/core_{dim,mpoa,ara,som,jspace}.toml"]
        L += missing_comment(missing)
        L += todo_line(missing, f"Table tab:dev-refusals-{study}")
        open(os.path.join(OUT, f"t1_dev_refusals_{study}.tex"), "w").write("\n".join(L) + "\n")


def write_t2_tex(cells, t2_rows, now):
    idx = {(r["model"], r["method"], r["benchmark"]): r for r in t2_rows}
    for study in ("main", "small"):
        models = [m for m, i in MODEL_INFO.items() if i["study"] == study]
        ns = {}
        for (m, meth, b), r in idx.items():
            if MODEL_INFO[m]["study"] == study and r["n"] != "":
                ns.setdefault(b, set()).add(r["n"])
        hdr = []
        for b, name in BENCHES:
            nn = ns.get(b, set())
            hdr.append(f"{name} ({next(iter(nn))})" if len(nn) == 1 else name)
        resc = sorted({f"{model_display(r['model'], cells)} {METHOD_TEX[r['method']]} {dict(BENCHES)[r['benchmark']]}"
                       for r in t2_rows if MODEL_INFO[r["model"]]["study"] == study and r["status"] == "present"
                       and "rescored" in r["note"]})
        ddag = ""
        if resc:
            ddag = (" $^\\ddagger$Rescored from the logged samples with the patched MBPP answer extractor "
                    "(run predates the lm-eval patch; scripts/bench/LM\\_EVAL\\_PATCHES.md): "
                    + "; ".join(tex_escape(x) for x in resc) + ".")
        pdiff = base_protocol_diffs(cells, study, "t2")
        ddag += protocol_footnote(pdiff, cells, "Original model's capability run used evaluation-harness settings "
                                  "that differ from its modified models' runs")
        L = [f"% Auto-generated by scripts/make_results_tables.py at {now} -- do not edit by hand.",
             "\\begin{table*}[t]",
             "\\centering\\footnotesize" if study == "main" else "\\centering\\small",
             "\\caption{Capability of the original and the selected modified models (KL $\\le 0.1$ trial), in \\%. "
             "Modified models: score with the difference to the same family's original model in percentage points "
             "in parentheses (computed from unrounded scores). Number of items in the column header. Metrics: "
             "MMLU-Pro exact match (custom extract, 5-shot CoT); GSM8K exact match (flexible extract, 5-shot); "
             "IFEval prompt-level strict accuracy (0-shot); HumanEval pass@1 (0-shot); MBPP pass@1 (3-shot). "
             "Greedy decoding, chat template on, thinking disabled. "
             + ("Main study." if study == "main" else "Small pair.") + ddag + " --: not yet available.}",
             f"\\label{{tab:capability-{study}}}",
             "\\setlength{\\tabcolsep}{4pt}" if study == "small" else "",
             "\\begin{tabular}{ll" + "r" * len(BENCHES) + "}",
             "\\toprule",
             "Model & Method & " + " & ".join(hdr) + " \\\\",
             "\\midrule"]
        L = [x for x in L if x != ""]
        missing = []
        for gi, m in enumerate(models):
            disp = tex_escape(model_display(m, cells))
            if gi:
                L.append("\\midrule")
            for k, meth in enumerate(["base"] + METHODS[study]):
                vals = []
                for b, name in BENCHES:
                    r = idx.get((m, meth, b))
                    if not r or r["status"] != "present":
                        missing.append(f"{m} x {METHOD_TEX[meth]} x {name}: {r['status'] if r else 'no row'}")
                        vals.append("--")
                        continue
                    s = tex_num(Decimal(r["score_pct"]), 1)
                    if "rescored" in r["note"]:
                        s += "$^\\ddagger$"
                    if meth != "base":
                        s += (f"\\,({tex_delta(Decimal(r['delta_pct_points']), 1)})" if r["delta_pct_points"] != ""
                              else "\\,(--)")
                    vals.append(s)
                mlab = METHOD_TEX[meth] + ("$^\\S$" if meth == "base" and m in pdiff else "")
                L.append(f"{disp if k == 0 else ''} & {mlab} & " + " & ".join(vals) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}",
              "% src: paper/tables/t2_capability.csv (built from summary.json 'benchmarks'; per-cell source_file column)",
              "% src: " + ", ".join(sorted({r['source_file'] for r in t2_rows
                                             if r['source_file'] and MODEL_INFO[r['model']]['study'] == study})),
              "% src: scripts/bench/README.md (protocol: metrics, shots, decoding, chat template, thinking off); "
              "metric names and n: summary.json 'benchmarks' (metric, n)"]
        if pdiff:
            L.append("% src (section-sign footnote): summary.json versions['sha256:bench.py'], postprocess, "
                     "versions.env, backend, lm_kwargs, versions of " + ", ".join(v[2] for v in pdiff.values())
                     + " vs. the modified models' summary.json; paper/tables/coverage.md (evaluation protocol table)")
        L += missing_comment(missing)
        L += todo_line(missing, f"Table tab:capability-{study}")
        open(os.path.join(OUT, f"t2_capability_{study}.tex"), "w").write("\n".join(L) + "\n")


def write_t3_tex(cells, t3_rows, now, n_benign, benign_sets):
    idx = {(r["model"], r["method"]): r for r in t3_rows}
    sets_txt = ", ".join(f"{tex_escape(s)} {n}" for s, n in benign_sets)
    for study in ("main", "small"):
        models = [m for m, i in MODEL_INFO.items() if i["study"] == study]
        meths = METHODS[study]
        pdiff = base_protocol_diffs(cells, study, "t3")
        sect = protocol_footnote(pdiff, cells, "Original model's held-out generation and teacher forcing "
                                 "come from a run whose evaluation-harness settings differ from its modified "
                                 "models' runs")
        L = [f"% Auto-generated by scripts/make_results_tables.py at {now} -- do not edit by hand.",
             "\\begin{table*}[t]",
             "\\centering\\small",
             "\\caption{Teacher-forced log-likelihood of the original model's greedy answers to the "
             f"{n_benign} benign held-out prompts ({sets_txt}). ``Original'': mean log-probability per answer "
             "token under the original model (token-weighted over all items); method columns: the same quantity "
             "under the selected modified model (KL $\\le 0.1$ trial) minus the original's value (negative = the "
             "original answers became less likely). Tokens: number of scored answer tokens. "
             + ("Main study." if study == "main" else "Small pair.") + sect + " --: not yet available.}",
             f"\\label{{tab:teacher-forced-{study}}}",
             "\\begin{tabular}{lrr" + "r" * len(meths) + "}",
             "\\toprule",
             " & & & \\multicolumn{" + str(len(meths)) + "}{c}{Shift vs.\\ original} \\\\",
             f"\\cmidrule(lr){{4-{3 + len(meths)}}}",
             "Model & Tokens & Original & " + " & ".join(METHOD_TEX[m] for m in meths) + " \\\\",
             "\\midrule"]
        missing = []
        for m in models:
            disp = tex_escape(model_display(m, cells))
            b = idx.get((m, "base"))
            if b and b["status"] == "present":
                tok, orig = str(b["n_tokens"]), tex_num(Decimal(b["mean_logprob"]), 3)
                if m in pdiff:
                    orig += "$^\\S$"
            else:
                tok, orig = "--", "--"
                missing.append(f"{m} x Original: {b['status'] if b else 'no row'}")
            vals = []
            for meth in meths:
                r = idx.get((m, meth))
                if r and r["status"] == "present" and r["shift_vs_original"] != "":
                    vals.append(tex_delta(Decimal(r["shift_vs_original"]), 3))
                else:
                    missing.append(f"{m} x {METHOD_TEX[meth]}: {r['status'] if r else 'no row'}"
                                   + ("" if not r or r["status"] != "present" else " (no original -> no shift)"))
                    vals.append("--")
            L.append(f"{disp} & {tok} & {orig} & " + " & ".join(vals) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}",
              "% src: paper/tables/t3_teacher_forced.csv (summary.json 'teacher_forced'; per-cell source_file column)",
              "% src: " + ", ".join(sorted({r['source_file'] for r in t3_rows
                                             if r['source_file'] and MODEL_INFO[r['model']]['study'] == study})),
              "% src: scripts/bench/bench.py (teacher_forced_logprobs: token-weighted mean over benign items); "
              "scripts/bench/refusal_sets.json (benign item counts)"]
        if pdiff:
            L.append("% src (section-sign footnote): settings recorded in " + ", ".join(v[2] for v in pdiff.values())
                     + " vs. the modified models' summary.json; paper/tables/coverage.md (evaluation protocol table)")
        L += missing_comment(missing)
        L += todo_line(missing, f"Table tab:teacher-forced-{study}")
        open(os.path.join(OUT, f"t3_teacher_forced_{study}.tex"), "w").write("\n".join(L) + "\n")


def write_coverage(now, evals, runs, cells, queues, protocol, artefacts, rs_counts, bench_n):
    L = [f"# Coverage report — paper 33 results tables",
         "",
         f"Generated {now} by `scripts/make_results_tables.py` from a snapshot of the synced run directories "
         "(the boxes are still syncing; re-run the script to refresh). Read-only on all inputs.",
         "",
         "## Inputs scanned",
         ""]
    for study, roots in BENCH_ROOTS.items():
        for box, root in roots:
            n = len([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]) if os.path.isdir(root) else 0
            L.append(f"- {study}: `{rel(root)}` — " + (f"{n} dirs" if os.path.isdir(root) else "**absent** (nothing read)"))
    L.append("- Duplicate-label rule: if a label exists under both `runs/mc-rtx-1/results/bench` and "
             "`runs/mc-h200-1/results/bench`, the RTX copy is used. " +
             ("Duplicates found: see anomalies." if any(a[0] == "duplicate label" for a in R.anomalies)
              else "No duplicates found (the H200 copy has no `results/bench`)."))
    L.append(f"- Optimisation runs: {len(runs)} run dirs across " +
             ", ".join(f"`{rel(r)}`" for _, r in RUN_ROOTS) + ".")
    L.append("- Eval queues: " + ", ".join(f"`{rel(p)}`" for p in QUEUES.values()) + f"; H200 run queue `{rel(H200_RUN_QUEUE)}`.")
    L += ["", "## Status legend", "",
          "- **present**: value in the tables.",
          "- **pending**: artefact expected but not produced yet (run running/finished-but-unselected, eval queued/running/not queued).",
          "- **excluded**: superseded attempt directory, never used for numbers (listed below).", ""]

    for study in ("main", "small"):
        models = [m for m, i in MODEL_INFO.items() if i["study"] == study]
        L += [f"## Cell matrix — {'main study' if study == 'main' else 'small pair'}", "",
              "| model | method | run used (dev selection) | T1 dev refusals | T2 capability | T3 teacher-forced |",
              "|---|---|---|---|---|---|"]
        for m in models:
            for meth in ["base"] + METHODS[study]:
                c = cells[(m, meth)]
                if meth == "base":
                    t1 = c.get("t1_base", {}).get("status", "")
                    runcol = "— (baseline read from the method runs' selection.json)"
                else:
                    t1 = c.get("sel_status", "")
                    runcol = (f"`{c.get('run')}` — {c.get('run_rule')}" if c.get("run")
                              else f"none yet ({c.get('run_rule')})")
                ev = c["eval"]
                t2 = c["eval_status"]
                if t2 == "present":
                    t2 = f"present (`{rel(ev['summary_path'])}`)"
                t3 = c["eval_status"]
                if t3 == "present":
                    t3 = f"present (`{rel(ev['tf_path'])}`)" if ev["tf"] else "missing: no teacher_forced block"
                L.append(f"| {m} | {METHOD_TEX[meth]} | {runcol} | {t1} | {t2} | {t3} |")
        L.append("")

    # counts
    def count(study, key):
        n_p = n_t = 0
        for c in cells.values():
            if c["study"] != study:
                continue
            if key == "t1":
                st = c.get("t1_base", {}).get("status", "") if c["method"] == "base" else c.get("sel_status", "")
            else:
                st = c["eval_status"]
                if key == "t3" and st == "present" and not c["eval"]["tf"]:
                    st = "missing"
            n_t += 1
            n_p += st == "present"
        return n_p, n_t
    L += ["## Summary counts (cells = model x {Original + methods})", ""]
    for study in ("main", "small"):
        for key, name in (("t1", "T1 dev refusals"), ("t2", "T2 capability"), ("t3", "T3 teacher-forced")):
            p, t = count(study, key)
            L.append(f"- {study}: {name}: {p}/{t} present, {t - p} pending/missing")
    L.append("")

    L += ["## Excluded directories (superseded attempts) and ignored files", ""]
    for p, why in R.excluded:
        L.append(f"- `{p}` — {why}")
    for p, why in R.ignored_files:
        L.append(f"- `{p}` — {why}")
    for c in cells.values():
        ev = c["eval"]
        for n in (ev or {}).get("notes", []):
            L.append(f"- `{rel(ev['path'])}` — {n}")
    L.append("")

    L += ["## Non-canonical optimisation runs (not used for any cell)", ""]
    for c in cells.values():
        if c["method"] != "base" and c.get("other_runs"):
            L.append(f"- {c['label']}: used `{c.get('run')}`; others: " +
                     "; ".join(f"`{n}` ({st})" for n, st in c["other_runs"]))
    used = {c.get("run") for c in cells.values()}
    pilots = sorted(n for n, r in runs.items() if not r["tag"].startswith("core") and n not in used)
    if pilots:
        L.append(f"- pilot runs (tag not core-s1; not part of the tables): " + ", ".join(f"`{p}`" for p in pilots))
    L.append("")

    L += ["## Evaluated artefacts (model_hash.json of each present evaluation)", "",
          "| label | safetensors shards | adapter files | total bytes (weight + config + tokenizer files) | "
          "total minus original | safetensors bytes minus original | non-weight bytes minus original | "
          "non-weight files vs original (name, sha256) | model source | tensor headers vs original |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in sorted(artefacts):
        L.append("| " + " | ".join(a) + " |")
    L += ["", "Derived models were evaluated from merged exports (`--export-strategy MERGE` in each eval dir's "
          "`reexport.sh`; see checks), not from live LoRA adapters. 'total bytes' is model_hash.json `bytes` "
          "(scripts/bench/bench.py `tree_hash`, also copied to summary.json `model_info.weights_bytes` despite the "
          "name): the sum over safetensors, .json, .model, .jinja, .txt and .tiktoken files, not weights only. "
          "The non-weight file sets differ between Hugging Face snapshots and Heretic exports, so 'total minus "
          "original' is not a weight comparison. Weights-only ('safetensors bytes minus original') and non-weight "
          "byte differences, and tensor headers, are computed only where both the evaluated merged model and the "
          "original snapshot are on this machine (4090 runs).", ""]

    # outputs written by the optimisation runs themselves (adapter vs merged), and whether anything evaluated them
    eval_refs = []
    for ev in evals.values():
        rx = os.path.join(ev["path"], "reexport.sh")
        if os.path.isfile(rx) and "adapter_t0" in open(rx).read():
            eval_refs.append(rel(rx))
    for f in ("scripts/eval_one.sh", "scripts/eval_one_local.sh", "scripts/bench/run.sh", "scripts/bench/bench.py"):
        p = os.path.join(ROOT, f)
        if os.path.isfile(p) and "adapter_t0" in open(p, errors="replace").read():
            eval_refs.append(f)
    rows = []
    for name, r in sorted(runs.items()):
        if r["origin_box"] not in ("mc-h200-1", "local-4090"):
            continue
        sp = os.path.join(r["origin_path"], "selection.json")
        sel = load_json(sp) if os.path.isfile(sp) else None
        h0 = (sel or {}).get("headless_index0_trial") or {}
        s01 = ((sel or {}).get("per_threshold") or {}).get("0.1") or {}
        rows.append((name, r["origin_box"], r["state"], "yes" if r["has_adapter_t0"] else "no",
                     r["save_cmd"] or "not recorded",
                     (f"t{h0.get('trial_index')} {h0.get('refusals_display')} KL {dstr(h0.get('kl'))}" if h0
                      else "no selection.json" if sel is None else "none"),
                     (f"t{s01.get('trial_index')}" if s01 else "no selection.json" if sel is None else "none"),
                     ", ".join(r["merged_dirs"]) or "none"))
    n_ad = sum(1 for x in rows if x[3] == "yes")
    n_ad_box = {b: sum(1 for x in rows if x[3] == "yes" and x[1] == b) for b in ("mc-h200-1", "local-4090")}
    n_differs = sum(1 for x in rows if x[3] == "yes" and x[5].startswith("t") and x[6].startswith("t")
                    and x[5].split()[0] != x[6])
    L += ["## Optimisation-run outputs: LoRA adapters vs merged models", "",
          "Each queued run ends with Heretic's headless save (`--trial-index 0 --model-action save` plus the export "
          "strategy shown below, from the run's meta.txt or, for 4090 core runs, scripts/local_queue.sh). `--trial-index` is a position in Heretic's sorted "
          "Pareto front (`../heretic/src/heretic/config.py`, field `trial_index`), so `adapter_t0/` is the LoRA "
          "adapter of Pareto position 0, i.e. the completed trial with the fewest dev refusals whatever its KL "
          "(selection.json `headless_index0_trial`; scripts/select_trial.py docstring). It is not trial 0 and not "
          "the KL <= 0.1 selection. `model/` = merged export of the same position-0 trial (pilot runs, "
          "`--export-strategy MERGE`); `model_kl*_t*/` = merged re-export of the selected trial (4090 runs).", "",
          f"- run dirs holding `adapter_t0/`: {n_ad} ({', '.join(f'{b}: {n}' for b, n in n_ad_box.items())}); "
          f"in {n_differs} of them the position-0 trial differs from the KL <= 0.1 selection.",
          "- references to `adapter_t0` in eval-dir `reexport.sh` files and the evaluation scripts "
          "(scripts/eval_one.sh, scripts/eval_one_local.sh, scripts/bench/run.sh, scripts/bench/bench.py): "
          + (", ".join(eval_refs) if eval_refs else "none") + ".", "",
          "| run | box | state | adapter_t0/ | headless save (meta.txt) | Pareto position 0 trial (dev refusals, KL) "
          "| KL <= 0.1 selection | merged model dirs |",
          "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append("| " + " | ".join(x) + " |")
    L.append("")

    L += ["## Evaluation protocol per cell (summary.json)", "",
          "| label | status | backend | limit | bench.py sha256[:10] | postprocess | BENCH_VLLM_EXTRA | gpu util | GPU | held-out n | held-out set sha ok |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(protocol, key=lambda x: x["label"]):
        L.append(f"| {p['label']} | {p['status']} | {p['backend']} | {p['limit_mode']} | {p['bench_py']} | {p['postprocess']} | "
                 f"{p['vllm_extra']} | {p['gpu_util']} | {p['gpu']} | {p['rg_n']} | {p['rg_sha_ok']} |")
    L.append("")
    L += ["## Held-out refusal set composition (scripts/bench/refusal_sets.json; counts only)", ""]
    for (s, k), v in sorted(rs_counts.items()):
        L.append(f"- {s} ({k}): {v}")
    L.append(f"- total: {sum(rs_counts.values())}")
    L.append("")

    L += ["## Consistency checks", "", "| check | ok | not ok | cells checked |", "|---|---|---|---|"]
    groups = {}
    for name, res, det in R.checks:
        m = re.match(r"^(.*?) \((.*)\)$", name)
        g, who = (m.group(1), m.group(2)) if m else (name, "")
        e = groups.setdefault(g, {"ok": 0, "bad": [], "who": []})
        if res == "ok":
            e["ok"] += 1
        else:
            e["bad"].append(f"{who}: {res} — {det}")
        e["who"].append(who)
    for g, e in groups.items():
        L.append(f"| {g} | {e['ok']} | {len(e['bad'])} | {', '.join(sorted(set(w for w in e['who'] if w)))} |")
    bad = [(g, b) for g, e in groups.items() for b in e["bad"]]
    if bad:
        L += ["", "Checks that were not ok:", ""]
        for g, b in bad:
            L.append(f"- {g} — {b}")
    L.append("")
    L += ["## Anomalies", ""]
    if R.anomalies:
        for cat, t in R.anomalies:
            L.append(f"- **{cat}**: {t}")
    else:
        L.append("- none")
    L.append("")
    seed2 = sorted(n for n, r in runs.items() if re.search(r"-s([2-9]|\d\d+)\b", r["tag"]))
    judge_files = []
    for ev in evals.values():
        rd = os.path.join(ev["path"], "refusal")
        if os.path.isdir(rd):
            judge_files += [os.path.join(rel(rd), f) for f in os.listdir(rd)
                            if f not in ("responses.jsonl", "teacher_forced.jsonl")]
    L += ["## Conditions in the design (notes/EXECUTION_PLAN_v2.md) with no artefacts found in this scan", "",
          "- Seed-2 (or higher) runs: " + (", ".join(seed2) if seed2 else "no run directory with a seed >= 2 tag found."),
          "- gamma-corrected MPOA, OBLITERATUS presets, paper-faithful SOM / J-space reference hook, external "
          "community models: no run or result directories found (only methods dim/mpoa/ara/som/jspace exist).",
          "- Judges (scripts/bench/score_refusals.py): " +
          ("judge outputs found: " + ", ".join(judge_files) if judge_files else
           "no judge output files in any refusal/ dir (only responses.jsonl and teacher_forced.jsonl), so no "
           "held-out refusal rates exist yet; the held-out set enters these tables only through teacher forcing."),
          ""]
    open(os.path.join(OUT, "coverage.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
