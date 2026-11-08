#!/usr/bin/env python3
"""Headless, method-independent trial selection for heretic 2.0 runs (paper 33).

Why: headless runs use `--trial-index 0`, i.e. position 0 of heretic's sorted Pareto
front = the trial with the FEWEST refusals, whatever its KL (an ARA pilot exported
refusals 0/100 at KL 10.8 = a broken model). Every run in the paper is instead
selected with ONE fixed rule:

    SELECTION RULE (applied identically to every model / method / seed)
      1. candidates = COMPLETE trials with KL divergence <= KL_MAX
      2. pick the candidate with the fewest refusals
      3. ties -> lower KL, then lower trial index (heretic's 1-based "index")
      4. no candidate -> select NOTHING; report "none" plus the lowest-KL trial

The pick is always on the Pareto front (anything dominating it would itself have
been picked), so it has a position in heretic's sorted front, which is what
`--trial-index` takes.

Usage (local, read-only on the journal; writes <run>/selection.json):
    python select_trial.py RUN_DIR [RUN_DIR ...] [--kl-max 0.1]
        [--thresholds 0.05,0.1,0.2,0.5] [--emit-command] [--no-write]

Re-export (IMPORTANT): `heretic --checkpoint-action continue` replaces ALL settings with
the ones stored in the journal (Settings.model_validate_init_only), so a CLI
`--trial-index N` is silently overridden by the stored `trial_index = 0`. The emitted
command therefore works on a COPY of the run's journal whose stored settings are patched
(`--patch-journal`, done on the box). The original checkpoint is never modified.
    python select_trial.py --patch-journal COPY.jsonl --trial-index N
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import warnings

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

# ---- documented constants -------------------------------------------------------
KL_MAX = 0.1                           # primary KL budget for the paper's headline numbers
REPORT_THRESHOLDS = (0.05, 0.1, 0.2, 0.5)  # sensitivity analysis thresholds
REFUSAL_OBJECTIVE = "Refusals"         # heretic KeywordRate scorer name (fraction 0..1)
KL_OBJECTIVE = "KL divergence"         # heretic KLDivergence scorer name
STUDY_NAME = "heretic"
BOX_ROOT = "/workspace/paper33"        # $W on the rented box (see scripts/box_run.sh)
RULE_TEXT = ("among COMPLETE trials with KL <= kl_max: fewest refusals; "
             "ties -> lower KL, then lower trial index; none qualifies -> no selection")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# ---- journal reading ------------------------------------------------------------
def find_journal(run_dir: str) -> str:
    ck = os.path.join(run_dir, "checkpoints")
    files = sorted(f for f in os.listdir(ck) if f.endswith(".jsonl")) if os.path.isdir(ck) else []
    if len(files) != 1:
        raise FileNotFoundError(f"{run_dir}: expected exactly one checkpoints/*.jsonl, found {files}")
    return os.path.join(ck, files[0])


def load_study(journal: str):
    """Return (trials, best_trials, study_user_attrs), read from a temp copy so the live
    checkpoint (possibly still being written on the box) is never touched."""
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "j.jsonl")
        shutil.copyfile(journal, tmp)
        study = optuna.load_study(study_name=STUDY_NAME, storage=JournalStorage(JournalFileBackend(tmp)))
        trials = list(study.trials)
        best = list(study.best_trials) if any(t.state == TrialState.COMPLETE for t in trials) else []
        return trials, best, dict(study.user_attrs)


def score(trial, name):
    for s in trial.user_attrs.get("scores", []):
        if s["name"] == name:
            return s["score"]["value"]
    return None


def display(trial, name):
    for s in trial.user_attrs.get("scores", []):
        if s["name"] == name:
            return s["score"].get("md_display")
    return None


def objective_names(trials) -> list[str]:
    for t in trials:
        if t.user_attrs.get("scores"):
            return [s["name"] for s in t.user_attrs["scores"]]
    return []


def heretic_sorted_front(best_trials, names):
    """Exactly heretic main.py: sorted(study.best_trials, key=tuple of score values in objective order)."""
    return sorted(best_trials, key=lambda t: tuple(score(t, n) for n in names))


# ---- selection rule ------------------------------------------------------------
def select(complete, kl_max, ref=REFUSAL_OBJECTIVE, kl=KL_OBJECTIVE):
    cands = [t for t in complete if score(t, kl) is not None and score(t, kl) <= kl_max]
    if not cands:
        return None
    return min(cands, key=lambda t: (score(t, ref), score(t, kl), t.user_attrs["index"]))


def trial_record(t, names, front_pos=None, ref=REFUSAL_OBJECTIVE, kl=KL_OBJECTIVE):
    if t is None:
        return None
    return {
        "trial_index": t.user_attrs["index"],
        "optuna_number": t.number,
        "refusals": score(t, ref),
        "refusals_display": display(t, ref),
        "kl": score(t, kl),
        "scores": {n: score(t, n) for n in names},
        "pareto_position": front_pos.get(t.number) if front_pos else None,
        "parameters": t.user_attrs.get("parameters"),
    }


def parse_run_name(name):
    # <model>__<method>__<tag>-s<seed>, e.g. qwen3.5-4b__ara__pilot-s1
    parts = name.split("__")
    m = re.search(r"-s(\d+)$", parts[-1]) if parts else None
    return {"model_short": parts[0] if parts else name,
            "method": parts[1] if len(parts) > 1 else None,
            "tag": parts[2] if len(parts) > 2 else None,
            "seed_from_name": int(m.group(1)) if m else None}


def analyse(run_dir, kl_max=KL_MAX, thresholds=REPORT_THRESHOLDS, box_root=BOX_ROOT):
    run_dir = os.path.abspath(run_dir.rstrip("/"))
    run = os.path.basename(run_dir)
    journal = find_journal(run_dir)
    trials, best, uattrs = load_study(journal)
    settings = json.loads(uattrs.get("settings", "{}"))
    names = objective_names(trials)
    for need in (REFUSAL_OBJECTIVE, KL_OBJECTIVE):
        if need not in names:
            raise ValueError(f"{run}: objective {need!r} not in {names}")
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    front = heretic_sorted_front(best, names)
    pos = {t.number: i for i, t in enumerate(front)}

    sel = select(complete, kl_max)
    lowest_kl = min(complete, key=lambda t: (score(t, KL_OBJECTIVE), score(t, REFUSAL_OBJECTIVE),
                                             t.user_attrs["index"])) if complete else None
    if sel is not None and sel.number not in pos:   # cannot happen by construction; guard anyway
        raise RuntimeError(f"{run}: selected trial {sel.user_attrs['index']} not on Pareto front")

    per_thr = {}
    for thr in sorted(set(thresholds) | {kl_max}):
        t = select(complete, thr)
        per_thr[f"{thr:g}"] = trial_record(t, names, pos)

    rec = {
        "run": run,
        "run_dir": run_dir,
        "journal": journal,
        **parse_run_name(run),
        "model": settings.get("model"),
        "seed": settings.get("seed"),
        "modifiers": [m.get("plugin") for m in settings.get("modifiers", [])] if settings.get("modifiers") else None,
        "finished": uattrs.get("finished"),
        "n_trials_planned": settings.get("n_trials"),
        "n_trials_total": len(trials),
        "n_trials_complete": len(complete),
        "objectives": names,
        "baselines": {x["name"]: x["baseline"]["value"] for x in complete[0].user_attrs["scores"]} if complete else {},
        "rule": {"text": RULE_TEXT, "kl_max": kl_max},
        "selected": trial_record(sel, names, pos),
        "lowest_kl_trial": trial_record(lowest_kl, names, pos),
        "headless_index0_trial": trial_record(front[0], names, pos) if front else None,
        "per_threshold": per_thr,
        "pareto_front": [trial_record(t, names, pos) for t in front],
    }
    if sel is not None:
        rec["reexport"] = reexport_commands(rec, box_root)
    return rec


# ---- re-export --------------------------------------------------------------------
def reexport_commands(rec, box_root=BOX_ROOT):
    p = rec["selected"]["pareto_position"]
    t = rec["selected"]["trial_index"]
    jname = os.path.basename(rec["journal"])
    box_run = f"{box_root}/runs/{rec['run']}"
    work = f"{box_run}/reexport_t{t}"
    q = shlex.quote
    heretic = [f"{box_root}/heretic/.venv/bin/heretic", "--model", rec["model"],
               "--checkpoint-action", "continue", "--trial-index", str(p),
               "--model-action", "save", "--export-strategy", "MERGE",
               "--save-directory", f"{box_run}/model_kl{rec['rule']['kl_max']:g}_t{t}"]
    lines = [
        f"cd {q(box_run)}",
        f"mkdir -p reexport_t{t}/checkpoints",
        f"cp config.toml reexport_t{t}/config.toml",
        f"cp checkpoints/{q(jname)} reexport_t{t}/checkpoints/{q(jname)}",
        f"{box_root}/heretic/.venv/bin/python {box_root}/scripts/select_trial.py "
        f"--patch-journal reexport_t{t}/checkpoints/{q(jname)} --trial-index {p} --expect-trial {t}",
        f"cd reexport_t{t}",
        " ".join(q(a) for a in heretic) + " 2>&1 | tee reexport.log",
    ]
    return {"pareto_position": p, "trial_index": t, "workdir": work,
            "heretic_command": " ".join(q(a) for a in heretic), "script": " && \\\n  ".join(lines)}


def patch_journal(path, trial_index, expect_trial=None, upload=False, repro="full"):
    """Append a study user_attr op to a journal COPY so `continue` uses trial_index (and save/merge)."""
    storage = JournalStorage(JournalFileBackend(path))
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    if expect_trial is not None:
        names = objective_names(study.trials)
        front = heretic_sorted_front(study.best_trials, names)
        got = front[trial_index].user_attrs["index"]
        if got != expect_trial:
            sys.exit(f"position {trial_index} is trial {got}, expected {expect_trial}; aborting")
    s = json.loads(study.user_attrs["settings"])
    s.update(trial_index=trial_index, model_action="upload" if upload else "save", export_strategy="merge",
             checkpoint_action="continue", n_additional_trials=None)
    if upload:
        # public repo + full reproduce manifest (Heretic Grimoire); repo id comes from the CLI
        # repro="none" for methods upstream Heretic cannot reproduce (SOM / J-space need the paper33-plugins fork;
        # J-space also reads a local direction file) -> no reproduce/ folder, no "reproducible" tag
        s.update(upload_repo_private=False, upload_reproducibility_information=repro)
    study.set_user_attr("settings", json.dumps(s))
    print(f"patched {path}: trial_index={trial_index} model_action={s['model_action']} export_strategy=merge")


# ---- output ---------------------------------------------------------------------
def fmt(t):
    if t is None:
        return "none"
    return f"t{t['trial_index']:<3} ref={t['refusals_display'] or t['refusals']:<7} KL={t['kl']:.4f} pos={t['pareto_position']}"


def print_table(recs, kl_max, thresholds):
    thr = sorted(set(thresholds) | {kl_max})
    print(f"rule: {RULE_TEXT}  (primary kl_max={kl_max:g})")
    hdr = f"{'run':<34} {'done':>5}  {'selected @KL<=' + format(kl_max, 'g'):<40} {'index0 (headless default)':<40}"
    print(hdr); print("-" * len(hdr))
    for r in recs:
        done = f"{r['n_trials_complete']}/{r['n_trials_planned']}"
        sel = fmt(r["selected"]) if r["selected"] else f"NONE (lowest KL: {fmt(r['lowest_kl_trial'])})"
        print(f"{r['run']:<34} {done:>5}  {sel:<40} {fmt(r['headless_index0_trial']):<40}")
    print()
    print(f"{'run':<34} " + " ".join(f"{'KL<=' + format(x, 'g'):>16}" for x in thr))
    for r in recs:
        cells = []
        for x in thr:
            t = r["per_threshold"][f"{x:g}"]
            cell = "-" if t is None else f"t{t['trial_index']} {t['refusals_display']}"
            cells.append(f"{cell:>16}")
        print(f"{r['run']:<34} " + " ".join(cells))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (each with checkpoints/<model>.jsonl)")
    ap.add_argument("--kl-max", type=float, default=KL_MAX, help=f"primary KL budget (default {KL_MAX})")
    ap.add_argument("--thresholds", default=",".join(f"{x:g}" for x in REPORT_THRESHOLDS),
                    help="comma-separated KL thresholds to report (default %(default)s)")
    ap.add_argument("--emit-command", action="store_true", help="print the box re-export commands")
    ap.add_argument("--box-root", default=BOX_ROOT, help="paper root on the box (default %(default)s)")
    ap.add_argument("--no-write", action="store_true", help="do not write <run>/selection.json")
    ap.add_argument("--patch-journal", metavar="JSONL", help="(box) patch a journal COPY for re-export")
    ap.add_argument("--trial-index", type=int, help="with --patch-journal: Pareto position")
    ap.add_argument("--expect-trial", type=int, help="with --patch-journal: sanity-check trial index")
    ap.add_argument("--upload", action="store_true", help="with --patch-journal: model_action=upload (public, full reproduce info)")
    ap.add_argument("--upload-repro", choices=["full", "basic", "none"], default="full", help="with --upload: reproducibility information (default %(default)s)")
    a = ap.parse_args(argv)

    if a.patch_journal:
        if a.trial_index is None:
            ap.error("--patch-journal needs --trial-index")
        patch_journal(a.patch_journal, a.trial_index, a.expect_trial, a.upload, a.upload_repro)
        return 0
    if not a.runs:
        ap.error("give at least one run dir")

    thresholds = [float(x) for x in a.thresholds.split(",") if x.strip()]
    recs, rc = [], 0
    for d in a.runs:
        try:
            r = analyse(d, a.kl_max, thresholds, a.box_root)
        except Exception as e:  # noqa: BLE001 - report and continue with other runs
            print(f"[skip] {d}: {e}", file=sys.stderr); rc = 1; continue
        recs.append(r)
        if not a.no_write:
            with open(os.path.join(r["run_dir"], "selection.json"), "w") as f:
                json.dump(r, f, indent=2)
    print_table(recs, a.kl_max, thresholds)
    if a.emit_command:
        for r in recs:
            print(f"\n# --- {r['run']}: re-export trial selected at KL <= {a.kl_max:g}")
            if not r["finished"]:
                print("# WARNING: study not finished; `continue` would RUN the remaining trials first")
            if r.get("reexport"):
                print(r["reexport"]["script"])
            else:
                print("# no trial qualifies -> nothing to export")
    return rc


if __name__ == "__main__":
    sys.exit(main())
