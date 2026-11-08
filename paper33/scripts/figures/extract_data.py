#!/usr/bin/env python3
"""extract_data.py -- tidy plotting data for the paper-33 figures.

Run with the Heretic venv (needs optuna; plotting is NOT done here):
    /data/heretic_2/heretic/.venv/bin/python scripts/figures/extract_data.py

Read-only on every input. Writes ONLY paper/figures/data/:
    trials.csv   one row per COMPLETE optimisation trial of every run a cell uses (dev-set numbers)
    results.csv  one row per evaluation cell (model x method, incl. the original and planned conditions)
    runs.csv     the run mapping: which optimisation run each model x method x seed uses, and why
Column definitions and sources: paper/figures/data/README.md.

Reuse
  scripts/select_trial.py         journal reading (find_journal, load_study, score, display, objective_names)
                                  and THE selection rule (select): fewest keyword refusals among COMPLETE
                                  trials with KL <= 0.1; ties -> lower KL, then lower trial index.
  scripts/make_results_tables.py  study design (MODELS, METHODS, BENCHES) and the superseded-dir reasons,
                                  so figures and tables share one definition.

Run mapping (per model x method x seed), first rule that applies:
  1. an evaluation dir <label>/selection.json names the run      -> the run that was actually evaluated
  2. a line on an evaluation queue names the run                 -> the run that will be evaluated
  3. exactly one FINISHED run core-s<seed>[-variant] without a '.' suffix
  4. exactly one core-s<seed>[-variant] run still running        -> trials of a partial study (flagged)
  anything else -> no run (reported). Pilot runs (pilot-s*) and '.'-suffixed attempt dirs are never used.
Journals are read from the box that ran the optimisation (runs/mc-h200-1/runs, runs/local-4090); the
runs/mc-rtx-1/runs copies are mirrors used for re-export and are not read.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import sys
from decimal import Decimal, getcontext

getcontext().prec = 50

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

import select_trial as st            # noqa: E402  (needs optuna -> Heretic venv)
import make_results_tables as mrt    # noqa: E402  (stdlib only; import has no side effects)
from optuna.trial import TrialState  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures", "data")
TABLES = os.path.join(ROOT, "paper", "tables")

# ---------------------------------------------------------------------------------------------------------
# Design
MODELS = mrt.MODELS                                   # (model, family, generation, study, column header)
MODEL_INFO = mrt.MODEL_INFO
CORE_METHODS = mrt.METHODS                            # study -> core methods
BENCHES = [b for b, _ in mrt.BENCHES]                 # mmlu_pro, gsm8k, ifeval, humaneval, mbpp
MODEL_DISPLAY = {
    "qwen2.5-32b": "Qwen2.5-32B", "qwen3.8-27b": "Qwen3.8-27B",
    "gemma-2-27b": "Gemma-2-27B", "gemma-4-31b": "Gemma-4-31B",
    "olmo-2-32b": "OLMo-2-32B", "olmo-3.1-32b": "Olmo-3.1-32B",
    "mistral-small-2409": "Mistral-Small-2409", "mistral-small-3.2": "Mistral-Small-3.2",
    "qwen2.5-7b": "Qwen2.5-7B", "qwen3.5-4b": "Qwen3.5-4B",
}
METHOD_DISPLAY = {"base": "Original", "dim": "DiM", "mpoa": "MPOA", "ara": "ARA", "som": "SOM",
                  "jspace": "J-space", "gamma_mpoa": "gamma-MPOA", "community": "Community"}
# label/run method tokens that mean gamma-corrected MPOA (no such dirs exist yet; accepted spellings)
GAMMA_TOKENS = {"gmpoa", "gammampoa", "mpoagamma", "gamma_mpoa", "gamma-mpoa"}

# Planned conditions without artefacts yet (task brief + notes/EXECUTION_PLAN_v2.md "Design").
# A placeholder row is written only while no evaluation dir for the condition exists.
PLANNED_GAMMA = ["gemma-2-27b", "gemma-4-31b", "olmo-2-32b", "olmo-3.1-32b"]
# Community reference models are NOT hard-coded: every `eval_community.sh <model>__community-<variant> <repo> ...`
# line on an evaluation queue (and every existing <model>__community-<variant> evaluation dir) is one cell.

EVAL_ROOTS = [  # (box, dir); on a duplicate label the earlier root wins (same rule as make_results_tables.py)
    ("mc-rtx-1", os.path.join(ROOT, "runs", "mc-rtx-1", "results", "bench")),
    ("mc-h200-1", os.path.join(ROOT, "runs", "mc-h200-1", "results", "bench")),
    ("local-4090", os.path.join(ROOT, "results", "bench-4090")),
]
RUN_ROOTS = [  # boxes that ran the optimisation (journal origin)
    ("mc-h200-1", os.path.join(ROOT, "runs", "mc-h200-1", "runs")),
    ("local-4090", os.path.join(ROOT, "runs", "local-4090")),
]
EVAL_QUEUES = [
    ("mc-rtx-1", os.path.join(ROOT, "runs", "mc-rtx-1", "bench_queue.txt")),
    ("mc-h200-1", os.path.join(ROOT, "runs", "mc-h200-1", "bench_queue.txt")),
    ("local-4090", os.path.join(ROOT, "runs", "local-4090", "bench_queue.txt")),
]
SEED_QUEUES = [os.path.join(ROOT, "runs", "local-4090", "queue_s2.txt")]   # planned seed >= 2 runs
BOX_GPU = {"mc-rtx-1": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "mc-h200-1": "NVIDIA H200 NVL",
           "local-4090": "NVIDIA GeForce RTX 4090"}
DEFAULT_EVAL_BOX = {"Qwen": "mc-rtx-1", "Gemma": "mc-rtx-1", "OLMo": "mc-h200-1", "Mistral Small": "mc-h200-1",
                    "Qwen (small pair)": "local-4090"}
KL_MAX = st.KL_MAX            # 0.1
REF, KL = st.REFUSAL_OBJECTIVE, st.KL_OBJECTIVE

WARN = []                     # printed at the end and written to README's "last run" block


def warn(msg):
    WARN.append(msg)


def rel(p):
    return os.path.relpath(p, ROOT) if p else ""


def generation(model):
    g = MODEL_INFO[model]["generation"]
    return g if MODEL_INFO[model]["study"] == "main" else f"small-{g}"


def dstr(x):
    """Exact decimal text (no exponent, no trailing zeros); '' for None."""
    if x is None:
        return ""
    if isinstance(x, Decimal):
        s = format(x, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    return repr(x) if isinstance(x, float) else str(x)


def load_json(path):
    with open(path) as f:
        return json.load(f, parse_float=Decimal)


def read_meta(path):
    meta = {}
    if os.path.isfile(path):
        for line in open(path, errors="replace"):
            if ":" in line:
                k, v = line.split(":", 1)
                meta.setdefault(k.strip(), v.strip())
    return meta


def method_token(tok):
    """Label/run method token -> canonical method key (None = not a recognised method)."""
    if tok in ("base", "dim", "mpoa", "ara", "som", "jspace"):
        return tok
    if tok in GAMMA_TOKENS:
        return "gamma_mpoa"
    if tok.startswith("community-"):
        return "community"
    return None


# ---------------------------------------------------------------------------------------------------------
# 1. Evaluation dirs
def superseded_reason(suffix):
    for k, v in mrt.SUPERSEDED.items():
        if suffix.startswith(k):
            return v
    return f"attempt dir with suffix '{suffix}' (never used for numbers)"


def scan_evals():
    evals, excluded = {}, []
    for box, root in EVAL_ROOTS:
        if not os.path.isdir(root):
            warn(f"evaluation root absent: {rel(root)}")
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            model, _, rest = name.partition("__")
            if model not in MODEL_INFO or not rest:
                excluded.append((rel(path), "model not in the design"))
                continue
            if "." in rest:
                excluded.append((rel(path), superseded_reason(rest[rest.index("."):])))
                continue
            method = method_token(rest)
            if method is None:
                excluded.append((rel(path), f"unrecognised method token '{rest}'"))
                warn(f"unrecognised evaluation dir {rel(path)} (not used)")
                continue
            if name in evals:
                warn(f"duplicate evaluation label {name}: used {rel(evals[name]['path'])}, ignored {rel(path)}")
                continue
            evals[name] = read_eval(name, model, method, rest, box, path)
    return evals, excluded


def read_eval(label, model, method, rest, box, path):
    ev = {"label": label, "model": model, "method": method, "variant": rest[len("community-"):]
          if method == "community" else "", "box": box, "path": path, "summary_path": None, "finished": False,
          "status": "pending: evaluation started (no summary.json yet)", "bench": {}, "tf": None, "tf_path": None,
          "gpu": "", "selection": None, "notes": [], "n_bench_done": 0}
    sp = os.path.join(path, "summary.json")
    if os.path.isfile(sp):
        s = load_json(sp)
        ev["summary_path"] = sp
        ev["gpu"] = (s.get("versions") or {}).get("gpu", "")
        done = [b for b in BENCHES if "score" in ((s.get("benchmarks") or {}).get(b) or {})]
        ev["n_bench_done"] = len(done)
        if s.get("method") != label:
            warn(f"{rel(sp)}: method field {s.get('method')!r} != dir label {label!r}")
        if s.get("finished"):
            ev["finished"] = True
            ev["status"] = "present"
            for b in done:
                ev["bench"][b] = s["benchmarks"][b]
                if s["benchmarks"][b].get("rescored"):
                    ev["notes"].append(f"{b} rescored ({s['benchmarks'][b]['rescored']})")
            if s.get("failed"):
                ev["notes"].append(f"failed: {s['failed']}")
                warn(f"{label}: summary.json lists failed steps {s['failed']}")
            tf = s.get("teacher_forced") or {}
            if "mean_logprob" in tf:
                ev["tf"], ev["tf_path"] = tf, sp
        else:
            ev["status"] = f"pending: evaluation in progress ({len(done)}/{len(BENCHES)} benchmarks done)"
    rr = os.path.join(path, "refusal_run", "summary.json")
    if ev["tf"] is None and os.path.isfile(rr):
        s2 = load_json(rr)
        if s2.get("finished") and "mean_logprob" in (s2.get("teacher_forced") or {}):
            ev["tf"], ev["tf_path"] = s2["teacher_forced"], rr
            ev["notes"].append("teacher forcing from refusal_run/summary.json (held-out re-run of the original)")
    selp = os.path.join(path, "selection.json")
    if os.path.isfile(selp):
        ev["selection"] = load_json(selp)
    return ev


def read_eval_queues():
    """label -> (box, run or '', queue file, repo or ''). Later lines override earlier ones for the same label.
    repo = the Hugging Face repo that eval_community.sh evaluates (community lines only)."""
    q = {}
    for box, path in EVAL_QUEUES:
        if not os.path.isfile(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 3:
                continue
            toks = parts[2].split()
            script = next((i for i, t in enumerate(toks) if t.endswith(("eval_one.sh", "eval_community.sh"))), None)
            if script is not None:
                args = [t for t in toks[script + 1:] if not t.startswith("--") and t != ";"]
                if not args:
                    continue
                label = args[0].strip(";")
                community = toks[script].endswith("eval_community.sh")
                run = args[2] if not community and len(args) > 2 else ""
                repo = args[1].strip(";") if community and len(args) > 1 else ""
            else:                       # 4090 format: label | repo | run [--flags]
                label, run, repo = parts[0], (toks[0] if toks else ""), ""
            q[label] = (box, run.strip(";"), rel(path), repo)
    return q


# ---------------------------------------------------------------------------------------------------------
# 2. Optimisation runs
RUN_RE = re.compile(r"^(?P<model>.+?)__(?P<method>[^_]+(?:_[^_]+)*?)__(?P<tag>[^.]+)(?P<suffix>\..+)?$")
TAG_RE = re.compile(r"^(?P<kind>core|pilot)-s(?P<seed>\d+)(?P<variant>-.+)?$")


def scan_runs():
    runs = {}
    for box, root in RUN_ROOTS:
        if not os.path.isdir(root):
            warn(f"run root absent: {rel(root)}")
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            m = RUN_RE.match(name)
            if not os.path.isdir(p) or name.startswith("bench__") or not m:
                continue
            tm = TAG_RE.match(m.group("tag"))
            meth = method_token(m.group("method"))
            if m.group("model") not in MODEL_INFO or meth is None or not tm:
                continue
            meta = read_meta(os.path.join(p, "meta.txt"))
            end, ec = meta.get("end"), meta.get("exit_code")
            state = ("finished" if end and ec == "0" else f"failed (exit_code {ec})" if end and ec
                     else "ended without exit_code" if end else "running" if meta else "no meta.txt")
            runs[name] = {"name": name, "box": box, "path": p, "model": m.group("model"), "method": meth,
                          "kind": tm.group("kind"), "seed": int(tm.group("seed")), "variant": tm.group("variant") or "",
                          "suffix": m.group("suffix") or "", "state": state}
    return runs


def resolve(model, method, seed, evals, queues, runs):
    """-> (run name or None, rule text)."""
    label = f"{model}__{method}"
    if seed == 1:
        ev = evals.get(label)
        if ev and ev["selection"] and ev["selection"].get("run"):
            return ev["selection"]["run"], f"named by {rel(os.path.join(ev['path'], 'selection.json'))} (evaluated run)"
        if label in queues and queues[label][1]:
            return queues[label][1], f"named on the evaluation queue {queues[label][2]}"
    cands = [r for r in runs.values() if r["model"] == model and r["method"] == method and r["kind"] == "core"
             and r["seed"] == seed and not r["suffix"]]
    fin = [r for r in cands if r["state"] == "finished"]
    if len(fin) == 1:
        return fin[0]["name"], f"only finished core-s{seed} run (not evaluated or queued yet)"
    if len(fin) > 1:
        return None, "AMBIGUOUS: several finished runs " + ", ".join(sorted(r["name"] for r in fin))
    running = [r for r in cands if r["state"] == "running"]
    if len(running) == 1:
        return running[0]["name"], f"only core-s{seed} run, still running (partial study)"
    return None, "no run directory found"


def journal_empty(run_path):
    try:
        return os.path.getsize(st.find_journal(run_path)) == 0
    except FileNotFoundError:
        return True


def planned_trials(run_path):
    """n_trials from the run's config.toml (None if absent)."""
    m = re.search(r"^n_trials\s*=\s*(\d+)", open(os.path.join(run_path, "config.toml")).read(), re.M) \
        if os.path.isfile(os.path.join(run_path, "config.toml")) else None
    return int(m.group(1)) if m else None


def nondominated(complete):
    pts = [(st.score(t, REF), st.score(t, KL)) for t in complete]
    out = set()
    for i, (r, k) in enumerate(pts):
        if not any(r2 <= r and k2 <= k and (r2 < r or k2 < k) for j, (r2, k2) in enumerate(pts) if j != i):
            out.add(complete[i].number)
    return out


def read_run(r, eval_sel):
    """Load a journal; return (run summary dict, trial rows)."""
    journal = st.find_journal(r["path"])
    trials, best, uattrs = st.load_study(journal)
    settings = json.loads(uattrs.get("settings", "{}"))
    names = st.objective_names(trials)
    if names[:2] != [REF, KL]:
        raise ValueError(f"{r['name']}: unexpected objectives {names}")
    complete = sorted((t for t in trials if t.state == TrialState.COMPLETE), key=lambda t: t.user_attrs["index"])
    front = {t.number for t in best}
    if front != nondominated(complete):
        warn(f"{r['name']}: Optuna best_trials differs from the recomputed non-dominated set")
    sel = st.select(complete, KL_MAX)
    sel_idx = sel.user_attrs["index"] if sel is not None else None
    ev_idx = None
    if eval_sel is not None and eval_sel.get("run") == r["name"]:
        ev_idx = (eval_sel.get("selected") or {}).get("trial_index")
    base_vals = set()
    rows = []
    for t in complete:
        frac = st.score(t, REF)
        disp = st.display(t, REF) or ""
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", disp)
        den = int(m.group(2)) if m else 100
        cnt = int(round(frac * den))
        if m and int(m.group(1)) != cnt:
            warn(f"{r['name']} t{t.user_attrs['index']}: display {disp} != fraction {frac}")
        for s in t.user_attrs.get("scores", []):
            if s["name"] == REF:
                base_vals.add((s["baseline"]["value"], den))
        rows.append({"trial": t.user_attrs["index"], "optuna_number": t.number, "refusals": cnt, "n_prompts": den,
                     "kl": st.score(t, KL), "on_pareto_front": t.number in front,
                     "is_selected_kl0.1": t.number == (sel.number if sel is not None else None),
                     "is_evaluated": ev_idx is not None and t.user_attrs["index"] == ev_idx})
    if len(base_vals) > 1:
        warn(f"{r['name']}: original-model refusal baseline differs between trials: {sorted(base_vals)}")
    bfrac, bden = next(iter(base_vals)) if base_vals else (None, 100)
    sel_row = next((x for x in rows if x["is_selected_kl0.1"]), None)
    lowest = min(rows, key=lambda x: (x["kl"], x["refusals"], x["trial"])) if rows else None
    info = {"journal": journal, "finished_attr": bool(uattrs.get("finished")), "seed_settings": settings.get("seed"),
            "n_trials_planned": settings.get("n_trials"), "n_trials_complete": len(complete),
            "n_trials_failed": sum(1 for t in trials if t.state == TrialState.FAIL), "n_pareto": len(front),
            "baseline_refusals": int(round(bfrac * bden)) if bfrac is not None else None,
            "selected_trial": sel_idx, "selected_refusals": sel_row["refusals"] if sel_row else None,
            "selected_kl": sel_row["kl"] if sel_row else None, "evaluated_trial": ev_idx,
            "lowest_kl_trial": lowest["trial"] if lowest else None, "lowest_kl": lowest["kl"] if lowest else None}
    return info, rows


# ---------------------------------------------------------------------------------------------------------
def write_csv(name, rows, cols):
    path = os.path.join(OUT, name)
    extra = {k for r in rows for k in r} - set(cols)
    if extra:
        raise KeyError(f"{name}: keys not in the column list: {sorted(extra)}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: dstr(r.get(k)) if not isinstance(r.get(k), bool) else str(r.get(k)) for k in cols})
    return path


def model_cols(model):
    i = MODEL_INFO[model]
    return {"model": model, "model_display": MODEL_DISPLAY[model], "family": i["family"], "study": i["study"],
            "generation": generation(model)}


def main():
    os.makedirs(OUT, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evals, excluded = scan_evals()
    queues = read_eval_queues()
    runs = scan_runs()

    # ---------------------------------------------------------------- run mapping
    cells = []   # (model, method, seed)
    for model, info in MODEL_INFO.items():
        for method in CORE_METHODS[info["study"]]:
            cells.append((model, method, 1))
    # seed >= 2 and gamma-MPOA core runs that exist on disk join automatically
    for r in runs.values():
        if r["kind"] == "core" and not r["suffix"] and (r["seed"] >= 2 or r["method"] == "gamma_mpoa"):
            if (r["model"], r["method"], r["seed"]) not in cells:
                cells.append((r["model"], r["method"], r["seed"]))
    order = {m: i for i, m in enumerate(MODEL_INFO)}
    morder = {m: i for i, m in enumerate(METHOD_DISPLAY)}
    cells.sort(key=lambda c: (c[2], order[c[0]], morder[c[1]]))

    run_rows, trial_rows, used_runs, runinfo = [], [], set(), {}
    for model, method, seed in cells:
        label = f"{model}__{method}"
        run, rule = resolve(model, method, seed, evals, queues, runs)
        ev = evals.get(label) if seed == 1 else None
        row = dict(model_cols(model), method=method, method_display=METHOD_DISPLAY[method], seed=seed,
                   run=run or "", resolution=rule, opt_machine="", journal="", run_state="",
                   eval_label=label if seed == 1 else "", status="")
        if run is None:
            row["status"] = f"missing: {rule}"
            warn(f"{label} seed {seed}: no run ({rule})")
        elif run not in runs:
            row["status"] = "missing: run named by the evaluation/queue is not in the journal-origin dirs"
            warn(f"{label} seed {seed}: run {run} not found under {[rel(p) for _, p in RUN_ROOTS]}")
        else:
            r = runs[run]
            used_runs.add(run)
            if r["state"] == "running" and journal_empty(r["path"]):
                # started a moment ago: the journal has no record yet -> 0 completed trials (not an error)
                row.update(opt_machine=r["box"], run_state=r["state"], run_finished=False, n_trials_complete=0,
                           n_trials_failed=0, n_trials_planned=planned_trials(r["path"]),
                           status=f"partial: run running (0/{planned_trials(r['path']) or '?'} trials)")
                run_rows.append(row)
                continue
            try:
                info, rows = read_run(r, ev["selection"] if ev else None)
            except Exception as e:  # noqa: BLE001 - report, keep going
                row["status"] = f"missing: journal unreadable ({e})"
                warn(f"{run}: journal unreadable: {e}")
                run_rows.append(row)
                continue
            if info["seed_settings"] != seed:
                warn(f"{run}: seed in journal settings {info['seed_settings']} != seed in run name {seed}")
            finished = r["state"] == "finished" and info["finished_attr"]
            row.update(opt_machine=r["box"], journal=rel(info["journal"]), run_state=r["state"],
                       run_finished=finished, **{k: v for k, v in info.items()
                                                 if k not in ("journal", "finished_attr", "seed_settings")})
            # cross-check the selection rule against every selection.json written for this run
            checks = []
            if ev and ev["selection"] is not None:
                es = (ev["selection"].get("selected") or {}).get("trial_index")
                checks.append(("eval", es))
                if es != info["selected_trial"]:
                    warn(f"{label}: evaluated trial t{es} (eval-dir selection.json, written at "
                         f"{ev['selection'].get('n_trials_complete')}/{ev['selection'].get('n_trials_planned')} trials) "
                         f"!= KL<=0.1 pick on the full journal t{info['selected_trial']}")
            rs = os.path.join(r["path"], "selection.json")
            if os.path.isfile(rs):
                rsel = (load_json(rs).get("selected") or {}).get("trial_index")
                checks.append(("run", rsel))
                if rsel != info["selected_trial"]:
                    warn(f"{run}: {rel(rs)} selects t{rsel}, recomputed KL<=0.1 pick is t{info['selected_trial']}")
            row["selection_check"] = ("no selection.json to compare" if not checks else
                                      "agrees with " + " + ".join(f"{k} selection.json" for k, _ in checks)
                                      if all(v == info["selected_trial"] for _, v in checks) else
                                      "DIFFERS: " + ", ".join(f"{k} t{v}" for k, v in checks))
            row["status"] = ("present" if finished else
                             f"partial: run {r['state']} ({info['n_trials_complete']}/{info['n_trials_planned']} trials)")
            if info["selected_trial"] is None:
                row["status"] += "; no trial with KL <= 0.1"
                warn(f"{run}: no completed trial with KL <= {KL_MAX}")
            runinfo[(model, method, seed)] = (run, info)
            for t in rows:
                trial_rows.append(dict(model_cols(model), method=method, seed=seed, run=run, opt_machine=r["box"],
                                       run_finished=finished, **t))
        others = sorted(n for n, x in runs.items() if x["model"] == model and x["method"] == method
                        and x["seed"] == seed and n != run)
        row["other_runs"] = "; ".join(f"{n} ({runs[n]['kind']}, {runs[n]['state']})" for n in others)
        run_rows.append(row)

    # planned optimisation runs that have no directory yet
    for qf in SEED_QUEUES:
        if not os.path.isfile(qf):
            continue
        for line in open(qf):
            name = line.split("|")[0].strip()
            m = RUN_RE.match(name) if name and not name.startswith("#") else None
            tm = TAG_RE.match(m.group("tag")) if m else None
            if not m or not tm or name in runs or m.group("model") not in MODEL_INFO:
                continue
            meth = method_token(m.group("method"))
            run_rows.append(dict(model_cols(m.group("model")), method=meth, method_display=METHOD_DISPLAY[meth],
                                 seed=int(tm.group("seed")), run=name, resolution=f"listed on {rel(qf)}",
                                 status="pending: run queued, not started"))
    for model in PLANNED_GAMMA:
        if not any(c[0] == model and c[1] == "gamma_mpoa" for c in cells):
            run_rows.append(dict(model_cols(model), method="gamma_mpoa", method_display=METHOD_DISPLAY["gamma_mpoa"],
                                 seed=1, status="planned: not started (no run directory)"))

    # ---------------------------------------------------------------- results (evaluation cells)
    res_rows = []
    base_tf, base_bench = {}, {}
    for label, ev in evals.items():
        if ev["method"] == "base" and ev["finished"]:
            base_bench[ev["model"]] = ev["bench"]
            if ev["tf"]:
                base_tf[ev["model"]] = ev["tf"]
    result_cells = []
    for model, info in MODEL_INFO.items():
        for method in ["base"] + CORE_METHODS[info["study"]]:
            result_cells.append((model, method, f"{model}__{method}", ""))
    for label, ev in evals.items():          # gamma-MPOA / community evaluations that exist
        if ev["method"] in ("gamma_mpoa", "community"):
            result_cells.append((ev["model"], ev["method"], label, ev["variant"]))
    for label in queues:                     # community models on an evaluation queue, not started yet
        model, _, rest = label.partition("__")
        if model in MODEL_INFO and method_token(rest) == "community" and label not in evals:
            result_cells.append((model, "community", label, rest[len("community-"):]))
    for model in PLANNED_GAMMA:
        if not any(c[0] == model and c[1] == "gamma_mpoa" for c in result_cells):
            result_cells.append((model, "gamma_mpoa", "", ""))
    result_cells.sort(key=lambda c: (order[c[0]], morder[c[1]], c[3]))

    for model, method, label, variant in result_cells:
        info = MODEL_INFO[model]
        ev = evals.get(label) if label else None
        row = dict(model_cols(model), method=method, method_display=METHOD_DISPLAY[method], variant=variant,
                   repo=queues[label][3] if label in queues else "", label=label,
                   finished=bool(ev and ev["finished"]), notes="")
        if ev:
            row.update(machine=ev["box"], gpu=ev["gpu"] or BOX_GPU[ev["box"]], status=ev["status"],
                       source_summary=rel(ev["summary_path"]), source_teacher_forced=rel(ev["tf_path"]))
        elif not label:
            row.update(machine="", gpu="", status="planned: not evaluated yet (no evaluation dir)")
        elif label in queues:
            box = queues[label][0]
            row.update(machine=box, gpu=BOX_GPU[box], status=f"pending: evaluation queued ({queues[label][2]})")
        else:
            box = DEFAULT_EVAL_BOX[info["family"]]
            row.update(machine=box, gpu=BOX_GPU[box], status="pending: evaluation not queued yet")
        notes = list(ev["notes"]) if ev else []
        # dev-set numbers of the selected trial (methods) / of the original (base)
        if method == "base":
            bl = {x[1]["baseline_refusals"] for k, x in runinfo.items() if k[0] == model and k[2] == 1}
            bl.discard(None)
            if len(bl) > 1:
                warn(f"{model}: original-model dev refusals differ across runs {sorted(bl)}")
            if bl:
                row.update(dev_refusals=min(bl), dev_kl=0.0)
        elif (model, method, 1) in runinfo:
            run, ri = runinfo[(model, method, 1)]
            row.update(run=run, selected_trial=ri["selected_trial"], evaluated_trial=ri["evaluated_trial"],
                       dev_refusals=ri["selected_refusals"], dev_kl=ri["selected_kl"])
            if ri["evaluated_trial"] is not None and ri["evaluated_trial"] != ri["selected_trial"]:
                notes.append(f"evaluated t{ri['evaluated_trial']} != KL<=0.1 pick t{ri['selected_trial']}")
        # capability
        if ev and ev["finished"]:
            bb = base_bench.get(model) if method != "base" else None
            pcts, deltas = [], []
            for b in BENCHES:
                v = ev["bench"].get(b)
                if v is None:
                    notes.append(f"{b} missing from finished summary")
                    continue
                pct = v["score"] * 100
                row[f"{b}_pct"] = pct
                row[f"{b}_se_pct"] = v["stderr"] * 100 if v.get("stderr") is not None else None
                row[f"{b}_n"] = v.get("n")
                pcts.append(pct)
                if bb and b in bb:
                    if bb[b].get("n") != v.get("n") or bb[b].get("metric") != v.get("metric"):
                        warn(f"{label} {b}: n/metric differ from the original's")
                    d = pct - bb[b]["score"] * 100
                    row[f"{b}_delta_pp"] = d
                    deltas.append(d)
            if len(pcts) == len(BENCHES):
                row["capability_mean_pct"] = sum(pcts) / len(pcts)
            if len(deltas) == len(BENCHES):
                row["capability_mean_delta_pp"] = sum(deltas) / len(deltas)
            elif method != "base":
                notes.append("original-model scores incomplete -> no deltas" if model not in base_bench
                             else "some deltas missing")
        # teacher forcing (held-out benign answers of the original)
        if ev and ev["finished"] and ev["tf"]:
            tf = ev["tf"]
            row.update(tf_mean_logprob=tf["mean_logprob"], tf_n_items=tf.get("n_items"), tf_n_tokens=tf.get("n_tokens"))
            ref = str(tf.get("reference", ""))
            if f"/{model}__base/" not in ref:
                warn(f"{label}: teacher-forcing reference is not {model}__base ({ref})")
            if method != "base":
                btf = base_tf.get(model)
                if btf is None:
                    notes.append("original-model teacher forcing missing -> no shift")
                else:
                    row["tf_shift"] = tf["mean_logprob"] - btf["mean_logprob"]
                    # the paper's divergence (04_methods.tex, bench.py): token-averaged log p_orig - log p_mod
                    row["tf_divergence"] = btf["mean_logprob"] - tf["mean_logprob"]
                    row["tf_paired"] = (tf.get("n_items") == btf.get("n_items")
                                        and tf.get("n_tokens") == btf.get("n_tokens"))
                    if not row["tf_paired"]:
                        warn(f"{label}: teacher-forced items/tokens differ from the original's")
        elif ev and ev["finished"]:
            notes.append("finished summary has no teacher_forced block")
        row["notes"] = "; ".join(notes)
        res_rows.append(row)

    # ---------------------------------------------------------------- write
    trial_cols = ["model", "model_display", "family", "study", "generation", "method", "seed", "run", "opt_machine",
                  "run_finished", "trial", "optuna_number", "refusals", "n_prompts", "kl", "on_pareto_front",
                  "is_selected_kl0.1", "is_evaluated"]
    run_cols = ["model", "model_display", "family", "study", "generation", "method", "method_display", "seed", "run",
                "resolution", "status", "opt_machine", "run_state", "run_finished", "journal", "n_trials_planned",
                "n_trials_complete", "n_trials_failed", "n_pareto", "baseline_refusals", "selected_trial",
                "selected_refusals", "selected_kl", "lowest_kl_trial", "lowest_kl", "eval_label", "evaluated_trial",
                "selection_check", "other_runs"]
    res_cols = (["model", "model_display", "family", "study", "generation", "method", "method_display", "variant",
                 "repo", "label", "machine", "gpu", "status", "finished", "run", "selected_trial", "evaluated_trial",
                 "dev_refusals", "dev_kl"]
                + [f"{b}_pct" for b in BENCHES] + [f"{b}_delta_pp" for b in BENCHES]
                + [f"{b}_se_pct" for b in BENCHES] + [f"{b}_n" for b in BENCHES]
                + ["capability_mean_pct", "capability_mean_delta_pp", "tf_mean_logprob", "tf_shift", "tf_divergence",
                   "tf_n_items",
                   "tf_n_tokens", "tf_paired", "source_summary", "source_teacher_forced", "notes"])
    write_csv("trials.csv", trial_rows, trial_cols)
    write_csv("runs.csv", run_rows, run_cols)
    write_csv("results.csv", res_rows, res_cols)

    verify_against_tables(res_rows, runinfo)
    report(now, trial_rows, run_rows, res_rows, excluded, runs, used_runs)
    return 0


# ---------------------------------------------------------------------------------------------------------
def verify_against_tables(res_rows, runinfo):
    """Compare with paper/tables/*.csv (make_results_tables.py). Tables may be older than this snapshot:
    a cell present here but pending there is 'newer', not an error; any value difference is reported."""
    print("\n== cross-check against paper/tables (make_results_tables.py output)")
    res = {(r["model"], r["method"]): r for r in res_rows}
    p = os.path.join(TABLES, "t2_capability.csv")
    if os.path.isfile(p):
        agree = differ = newer = 0
        for t in csv.DictReader(open(p)):
            r = res.get((t["model"], t["method"]))
            mine = r.get(f"{t['benchmark']}_pct") if r else None
            if t["status"] != "present":
                newer += mine is not None
                continue
            if mine is None:
                warn(f"T2 {t['model']}__{t['method']} {t['benchmark']}: in tables, not in results.csv"); differ += 1
            elif Decimal(t["score_pct"]) != mine or (t["delta_pct_points"] and
                                                     Decimal(t["delta_pct_points"]) != r.get(f"{t['benchmark']}_delta_pp")):
                warn(f"T2 {t['model']}__{t['method']} {t['benchmark']}: tables {t['score_pct']}/{t['delta_pct_points']} "
                     f"vs results {dstr(mine)}/{dstr(r.get(t['benchmark'] + '_delta_pp'))}"); differ += 1
            else:
                agree += 1
        print(f"  T2 capability: {agree} values agree, {differ} differ, {newer} present here but pending in tables")
    p = os.path.join(TABLES, "t3_teacher_forced.csv")
    if os.path.isfile(p):
        agree = differ = newer = 0
        for t in csv.DictReader(open(p)):
            r = res.get((t["model"], t["method"]))
            mine = r.get("tf_mean_logprob") if r else None
            if t["status"] != "present":
                newer += mine is not None
                continue
            ok = mine is not None and Decimal(t["mean_logprob"]) == mine and \
                (not t["shift_vs_original"] or Decimal(t["shift_vs_original"]) == r.get("tf_shift"))
            agree += ok
            differ += not ok
            if not ok:
                warn(f"T3 {t['model']}__{t['method']}: tables {t['mean_logprob']}/{t['shift_vs_original']} vs "
                     f"results {dstr(mine)}/{dstr(r.get('tf_shift') if r else None)}")
        print(f"  T3 teacher-forced: {agree} agree, {differ} differ, {newer} present here but pending in tables")
    p = os.path.join(TABLES, "t1_dev_refusals.csv")
    if os.path.isfile(p):
        agree = differ = newer = 0
        for t in csv.DictReader(open(p)):
            if t["method"] == "base" or t["kl_max"] != "0.1":
                continue
            ri = runinfo.get((t["model"], t["method"], 1))
            if t["status"] != "present":
                newer += ri is not None and ri[1]["selected_trial"] is not None
                continue
            ok = (ri is not None and ri[0] == t["run"] and str(ri[1]["selected_trial"]) == t["trial_index"]
                  and str(ri[1]["selected_refusals"]) == t["refusals_count"])
            agree += ok
            differ += not ok
            if not ok:
                warn(f"T1 {t['model']}__{t['method']}: tables run {t['run']} t{t['trial_index']} {t['refusals_count']}/100 "
                     f"vs trials {ri[0] if ri else None} t{ri[1]['selected_trial'] if ri else None} "
                     f"{ri[1]['selected_refusals'] if ri else None}/100")
        print(f"  T1 dev refusals @KL<=0.1: {agree} agree, {differ} differ, {newer} present here but pending in tables")


def report(now, trial_rows, run_rows, res_rows, excluded, runs, used_runs):
    print(f"\n== wrote {rel(OUT)}/trials.csv ({len(trial_rows)} rows), runs.csv ({len(run_rows)}), "
          f"results.csv ({len(res_rows)})  [{now}]")
    print("\n== trials.csv rows per model x method (seed): complete trials | run")
    for r in run_rows:
        if r.get("journal"):
            n = sum(1 for t in trial_rows if t["run"] == r["run"])
            print(f"  {r['model']:<20} {r['method']:<6} s{r['seed']}  {n:>4}  {r['run']:<40} {r['opt_machine']:<10} "
                  f"sel t{r.get('selected_trial')} ({r.get('selected_refusals')}/100, KL {r.get('selected_kl') or 0:.4f}) "
                  f"| {r['resolution']}")
    print("\n== runs not complete (partial runs ARE in trials.csv with run_finished=False; pending/planned are not)")
    for r in run_rows:
        if not r.get("journal") or not str(r["status"]).startswith("present"):
            print(f"  {r['model']:<20} {r['method']:<10} s{r['seed']}  {r.get('run') or '-':<36} {r['status']}")
    print("\n== results.csv cells not present")
    for r in res_rows:
        if r["status"] != "present":
            print(f"  {r['model']:<20} {r['method']:<10} {r['variant'] or '':<24} {r['status']}")
    print(f"\n== evaluation dirs excluded ({len(excluded)}): " + "; ".join(p for p, _ in excluded))
    unused = sorted(n for n in runs if n not in used_runs)
    print(f"== optimisation runs not used ({len(unused)}): " + ", ".join(f"{n} [{runs[n]['state']}]" for n in unused))
    print(f"\n== warnings ({len(WARN)})")
    for w in WARN:
        print("  - " + w)


if __name__ == "__main__":
    sys.exit(main())
