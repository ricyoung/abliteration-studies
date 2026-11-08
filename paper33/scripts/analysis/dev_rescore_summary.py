#!/usr/bin/env python3
"""Paper 33 -- dev-set re-score of the 46 released merged exports vs the selected trials (no GPU, read-only inputs).

    scripts/figures/.venv/bin/python scripts/analysis/dev_rescore_summary.py

Background
  The selected trial's dev numbers (keyword refusals on mlabonne/harmful_behaviors test[:100], first-token KL on
  mlabonne/harmless_alpaca test[:100]) come from the Optuna search, where the trial's adapter is applied to the base
  model. The released model is a separate merge/export of that trial. scripts/rescore_dev.py ran Heretic's own evaluator
  (`heretic --model <pinned base> --evaluate-model <export>`) on every released export with the run's own config,
  code version (PYTHONPATH) and batch size. The search of the 36 main-study runs ran on the H200 box (mc-h200-1); their
  re-score ran on the RTX PRO 6000 box (massed-rtx6000). The 10 small-pair runs were searched and re-scored on the
  RTX 4090 workstation.

  Gemma-2-27B: the five exports were saved by transformers 5.18, whose config serialisation drops
  `_attn_implementation` (PretrainedConfig._remove_keys_not_serialized pops `_attn_implementation_internal`). The base
  config (google/gemma-2-27b-it@aaf20e6b) sets "_attn_implementation": "eager"; without it transformers loads the
  export with SDPA, which does not apply Gemma-2's attention-logit soft-capping, so the first re-score (rtx TSV) gave
  KL 0.40-0.49. scripts/rescore_gemma2_eager.sh re-scored the same pinned exports with "_attn_implementation": "eager"
  written into a LOCAL copy of config.json (weights untouched). rescore_dev_all.csv carries the eager rows; this
  script checks that, and keeps the SDPA values only as the contrast columns sdpa_*.

Inputs (read only)
  results/rescore_dev/rescore_dev_all.csv                 46 rows (combined; Gemma-2 = eager re-score)
  results/rescore_dev/{local,rtx,rtx_gemma2_eager}/rescore_dev.tsv   per-host results (rtx holds the SDPA Gemma-2 rows)
  results/rescore_dev/jobs_gemma2_eager.tsv              eager jobs (local model dirs, pinned export shas)
  notes/gemma2_config_patch_20261106.log                 public Gemma-2 config patch: old -> new commit (12 hex)
  paper/figures/data/trials.csv                          every completed trial (refusals, KL, selected flag)
  results/analysis/final_kl_vs_capability_cells.csv      teacher-forced divergence (vLLM) per cell, for context
  runs/mc-rtx-1/results/bench/gemma-2-27b__*/bench.log, model_hash.json   vLLM engine/backend lines, config.json sha256
  runs/mc-rtx-1/scripts/bench/.venv/.../vllm/model_executor/models/gemma2.py   vLLM 0.33.0 Gemma-2 source (bench env)
  /data/heretic_2/heretic/.venv/.../transformers/{integrations/sdpa_attention.py,configuration_utils.py}  (5.18.0)
  /data/heretic_2/models/paper33/{base,ours}/*/config.json   local copies of base and export configs

Outputs (written only here)
  results/analysis/dev_rescore_cells.csv     one row per released export (46)
  results/analysis/dev_rescore_groups.csv    summary per group (all, role, study/host, method)
  results/analysis/dev_rescore_tests.csv     all 46 exports: Wilcoxon signed-rank and sign test, re-scored vs selected
                                             (refusals, KL; Holm over the two), and Spearman of selected vs re-scored
                                             refusals with bootstrap CI -- helpers imported from final.py
  results/analysis/dev_rescore_summary.md    summary tables and checks

Columns of dev_rescore_cells.csv (selected = the trial chosen by the KL <= 0.1 rule, logged during the search;
rescore = the released export scored by Heretic's evaluator)
  d_refusals = rescore_refusals - sel_refusals (count of 100); d_kl = rescore_kl - sel_kl
  rescore_lo/hi           Wilson 95% interval of the re-scored refusal proportion (k/100), as in h1h2.wilson
  sel_in_rescore_ci       selected-trial proportion lies inside that interval (descriptive; the per-prompt outcomes of
                          the search-time trial were not logged -- the journal stores only the count -- so no paired
                          McNemar / Newcombe comparison is possible)
  over_kl_cap             rescore_kl > 0.1 (the selection budget)
  large_refusal_drift     |d_refusals| > 2
  gemma2_eager            Gemma-2 export re-scored with eager attention set in a local config copy
  sdpa_*                  first (SDPA) Gemma-2 re-score, invalid, kept for the contrast
  runnerup_*              the rule applied to the logged trials without the selected trial (the trial the rule would
                          return if the selected trial were ineligible)
  drift_rule_*            sensitivity: the rule applied with every trial's logged KL shifted by this export's d_kl
                          (assumes the export's KL drift applies equally to every trial of the run; untested)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
RS = ROOT / "results" / "rescore_dev"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1h2 import holm, wilson  # noqa: E402  (Wilson 95% score interval, Holm; alpha 0.05)
from final import spearman_ci, wilcoxon_paired  # noqa: E402  (as in final.py analyses (d), (e), (f))

KL_MAX = 0.1
DRIFT = 2
N_DEV = 100
MODELS_DIR = Path("/data/heretic_2/models/paper33")
TF_VENV = Path("/data/heretic_2/heretic/.venv/lib/python3.12/site-packages")
BENCH_VENV = ROOT / "runs/mc-rtx-1/scripts/bench/.venv/lib/python3.12/site-packages"
BENCH_RTX = ROOT / "runs/mc-rtx-1/results/bench"
MODEL_ORDER = ["qwen2.5-32b", "qwen3.8-27b", "gemma-2-27b", "gemma-4-31b", "olmo-2-32b", "olmo-3.1-32b",
               "mistral-small-2409", "mistral-small-3.2", "qwen2.5-7b", "qwen3.5-4b"]
METHOD_ORDER = ["dim", "mpoa", "som", "jspace", "ara", "gmpoa"]
METHOD_DISPLAY = {"dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space", "ara": "ARA",
                  "gmpoa": "gamma-MPOA"}
REPO_METHOD = {"dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "JSpace", "ara": "ARA", "gmpoa": "GMPOA"}
LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def read_tsv(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, sep="\t", dtype={"merged_rev": str, "base_rev": str})
    d["method"] = d["method"].replace({"gamma_mpoa": "gmpoa"})
    return d


def select(g: pd.DataFrame, kl_col: str = "kl", kl_max: float = KL_MAX) -> pd.Series | None:
    """scripts/select_trial.py rule: KL <= kl_max, fewest refusals, then lower KL, then lower trial index."""
    f = g[g[kl_col] <= kl_max]
    if not len(f):
        return None
    return f.sort_values(["refusals", kl_col, "trial"]).iloc[0]


def fmt_d(x, d=0):
    return f"{x:+.{d}f}" if d else f"{int(x):+d}"


# ------------------------------------------------------------------------------------------------------------ load
def load():
    allc = pd.read_csv(RS / "rescore_dev_all.csv", dtype={"merged_rev": str, "base_rev": str})
    allc["method"] = allc["method"].replace({"gamma_mpoa": "gmpoa"})
    local = read_tsv(RS / "local/rescore_dev.tsv")
    rtx = read_tsv(RS / "rtx/rescore_dev.tsv")
    eager = read_tsv(RS / "rtx_gemma2_eager/rescore_dev.tsv")
    jobs_e = read_tsv(RS / "jobs_gemma2_eager.tsv")
    trials = pd.read_csv(ROOT / "paper/figures/data/trials.csv")
    trials["method"] = trials["method"].replace({"gamma_mpoa": "gmpoa"})
    tf = pd.read_csv(OUT / "final_kl_vs_capability_cells.csv")
    return allc, local, rtx, eager, jobs_e, trials, tf


def patch_log() -> pd.DataFrame:
    rows = []
    p = ROOT / "notes/gemma2_config_patch_20261106.log"
    for line in p.read_text().splitlines():
        m = re.match(r"(\S+): ([0-9a-f]+) -> ([0-9a-f]+) eager=(\S+)", line.strip())
        if m:
            rows.append(dict(repo=m.group(1), old_rev12=m.group(2), new_rev12=m.group(3), eager_after=m.group(4)))
    return pd.DataFrame(rows)


def repo_index() -> dict:
    """rev12 -> HF repo id from the local mirror dirs richardyoung__<name>@<rev12>."""
    idx = {}
    for d in (MODELS_DIR / "ours").glob("richardyoung__*@*"):
        name, rev = d.name.split("@")
        idx[rev] = name.replace("richardyoung__", "richardyoung/")
    return idx


def attn_impl(cfg: Path):
    if not cfg.exists():
        return "no local copy", ""
    c = json.loads(cfg.read_text())
    tc = c.get("text_config", {}) if isinstance(c.get("text_config"), dict) else {}
    v = c.get("_attn_implementation", tc.get("_attn_implementation"))
    return (v if v is not None else "absent"), str(c.get("transformers_version", ""))


# ------------------------------------------------------------------------------------------------------------ main
def main():
    allc, local, rtx, eager, jobs_e, trials, tf = load()
    log(f"# dev_rescore_summary.py  {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    assert len(allc) == 46 and allc.run.is_unique, len(allc)
    assert (allc.rc == 0).all()

    # --- consistency of the combined file with the per-host files; Gemma-2 rows must be the eager re-score
    g2 = allc.model == "gemma-2-27b"
    e = eager.set_index("run")
    for _, r in allc[g2].iterrows():
        assert "gemma2_eager" in r.merged, r.run
        assert int(e.loc[r.run, "rescore_refusals"]) == r.rescore_refusals, r.run
        assert abs(float(e.loc[r.run, "rescore_kl"]) - r.rescore_kl) < 1e-9, r.run
    sdpa = rtx[rtx.model == "gemma-2-27b"].set_index("run")
    assert len(sdpa) == 5 and len(e) == 5
    other = pd.concat([local, rtx[rtx.model != "gemma-2-27b"]]).set_index("run")
    for _, r in allc[~g2].iterrows():
        assert int(other.loc[r.run, "rescore_refusals"]) == r.rescore_refusals, r.run
        assert abs(float(other.loc[r.run, "rescore_kl"]) - r.rescore_kl) < 1e-9, r.run
    je = jobs_e.set_index("run")
    for run in e.index:  # eager jobs scored the same pinned export revisions as the SDPA run
        assert je.loc[run, "merged_rev"] == sdpa.loc[run, "merged_rev"] == e.loc[run, "merged_rev"], run
    log("combined file = local (10) + rtx non-Gemma-2 (31) + rtx_gemma2_eager (5): consistent; "
        "Gemma-2 rows are the eager re-score of the same pinned export revisions as the SDPA run")
    # original model's dev refusals in the re-score vs the value Heretic recorded during the search
    t1 = pd.read_csv(ROOT / "paper/tables/t1_dev_refusals.csv")
    t1b = t1[t1.method == "base"].drop_duplicates("model").set_index("model").refusals_count
    base_ok = allc.apply(lambda r: int(r.baseline_refusals) == int(t1b[r.model]), axis=1)
    globals()["BASE_OK"] = int(base_ok.sum())
    log(f"baseline (original) dev refusals equal to the search's recorded value: {int(base_ok.sum())}/46")

    # --- per-export table
    pl = patch_log().set_index("old_rev12")
    ridx = repo_index()
    sel = trials[trials["is_selected_kl0.1"].astype(str) == "True"].set_index("run")
    tfi = tf.set_index("label")
    base_cfg = {}
    for d in (MODELS_DIR / "base").glob("*@*"):
        base_cfg[d.name.split("@")[1]] = attn_impl(d / "config.json")[0]
    rows = []
    for _, r in allc.iterrows():
        g = trials[trials.run == r.run].copy()
        s = sel.loc[r.run]
        ok = (int(s.trial) == r.sel_trial and int(s.refusals) == r.sel_refusals and abs(s.kl - r.sel_kl) < 5e-5)
        assert ok, (r.run, s.trial, s.refusals, s.kl)
        assert int(g.n_prompts.iloc[0]) == N_DEV
        if r.model == "gemma-2-27b":
            repo = sdpa.loc[r.run, "merged"]
        elif r.merged.startswith("richardyoung/"):
            repo = r.merged
        else:
            repo = ridx.get(r.merged_rev[:12], "")
        rev12 = r.merged_rev[:12]
        patched = pl.loc[rev12, "new_rev12"] if rev12 in pl.index else ""
        lo, hi = wilson(int(r.rescore_refusals), N_DEV)
        ru = select(g[g.trial != r.sel_trial])
        g["kl_shift"] = g.kl + r.d_kl
        dr = select(g, "kl_shift")
        exp_cfg = MODELS_DIR / "ours" / f"{repo.replace('/', '__')}@{rev12}" / "config.json" if repo else Path("/none")
        e_attn, e_tv = attn_impl(exp_cfg)
        label = f"{r.model}__{r.method}"
        row = dict(
            run=r.run, model=r.model, model_display=s.model_display, family=s.family, study=s.study,
            generation=s.generation, method=r.method, method_display=METHOD_DISPLAY[r.method],
            role="control" if r.method == "gmpoa" else "primary", search_machine=s.opt_machine,
            rescore_host=r.host, batch_size=int(r.batch_size),
            code_pythonpath=("heretic-gamma/src" if "gamma" in str(r.pythonpath) else "heretic-p33/src"
                             if "p33" in str(r.pythonpath) else "none (host's default Heretic install)"),
            sel_trial=int(r.sel_trial), sel_refusals=int(r.sel_refusals), sel_kl=float(s.kl),
            rescore_refusals=int(r.rescore_refusals), rescore_kl=float(r.rescore_kl),
            baseline_refusals=int(r.baseline_refusals),
            d_refusals=int(r.rescore_refusals - r.sel_refusals), d_kl=float(r.rescore_kl - s.kl),
            rescore_rate=r.rescore_refusals / N_DEV, rescore_lo=lo, rescore_hi=hi,
            sel_in_rescore_ci=bool(lo <= r.sel_refusals / N_DEV <= hi),
            over_kl_cap=bool(r.rescore_kl > KL_MAX), large_refusal_drift=bool(abs(r.rescore_refusals - r.sel_refusals)
                                                                              > DRIFT),
            gemma2_eager=bool(r.model == "gemma-2-27b"),
            sdpa_refusals=int(sdpa.loc[r.run, "rescore_refusals"]) if r.run in sdpa.index else np.nan,
            sdpa_kl=float(sdpa.loc[r.run, "rescore_kl"]) if r.run in sdpa.index else np.nan,
            merged_repo=repo, evaluated_rev=r.merged_rev, patched_rev12=patched,
            base_config_attn_impl=base_cfg.get(r.base_rev[:12], "n/a"), export_config_attn_impl=e_attn,
            export_config_transformers=e_tv,
            runnerup_trial=int(ru.trial) if ru is not None else np.nan,
            runnerup_refusals=int(ru.refusals) if ru is not None else np.nan,
            runnerup_kl=float(ru.kl) if ru is not None else np.nan,
            drift_rule_trial=int(dr.trial) if dr is not None else np.nan,
            drift_rule_refusals=int(dr.refusals) if dr is not None else np.nan,
            drift_rule_kl_logged=float(dr.kl) if dr is not None else np.nan,
            tf_divergence_vllm=float(tfi.loc[label, "tf_divergence"]) if label in tfi.index else np.nan,
        )
        row["sdpa_d_refusals"] = row["sdpa_refusals"] - row["sel_refusals"] if row["gemma2_eager"] else np.nan
        row["sdpa_d_kl"] = row["sdpa_kl"] - row["sel_kl"] if row["gemma2_eager"] else np.nan
        rows.append(row)
    cells = pd.DataFrame(rows)
    cells["_m"] = cells.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    cells["_k"] = cells.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    cells = cells.sort_values(["_m", "_k"]).drop(columns=["_m", "_k"]).reset_index(drop=True)
    cells.to_csv(OUT / "dev_rescore_cells.csv", index=False)
    log(f"wrote {OUT / 'dev_rescore_cells.csv'} ({len(cells)} rows)")

    # --- summaries
    def summ(name, d):
        return dict(group=name, n=len(d), median_d_ref=float(d.d_refusals.median()),
                    mean_d_ref=round(float(d.d_refusals.mean()), 2),
                    n_up=int((d.d_refusals > 0).sum()), n_same=int((d.d_refusals == 0).sum()),
                    n_down=int((d.d_refusals < 0).sum()), n_within_2=int((d.d_refusals.abs() <= DRIFT).sum()),
                    min_d_ref=int(d.d_refusals.min()), max_d_ref=int(d.d_refusals.max()),
                    max_abs_d_ref=int(d.d_refusals.abs().max()),
                    median_d_kl=round(float(d.d_kl.median()), 4), mean_d_kl=round(float(d.d_kl.mean()), 4),
                    max_abs_d_kl=round(float(d.d_kl.abs().max()), 4), n_over_kl_cap=int(d.over_kl_cap.sum()),
                    n_sel_in_ci=int(d.sel_in_rescore_ci.sum()))
    S = [summ("all 46 released exports", cells),
         summ("primary (42)", cells[cells.role == "primary"]),
         summ("gamma-MPOA controls (4)", cells[cells.role == "control"]),
         summ("main study: searched H200, re-scored RTX PRO 6000 (36)", cells[cells.study == "main"]),
         summ("small pair: searched and re-scored RTX 4090 (10)", cells[cells.study == "small"])]
    for m in METHOD_ORDER:
        S.append(summ(f"method {METHOD_DISPLAY[m]}", cells[cells.method == m]))
    S.append(summ("all methods except ARA (36)", cells[cells.method != "ara"]))
    S.append(summ("ARA, main study (8)", cells[(cells.method == "ara") & (cells.study == "main")]))
    S.append(summ("non-ARA, main study (28)", cells[(cells.method != "ara") & (cells.study == "main")]))
    summ_df = pd.DataFrame(S)
    log(summ_df.to_string())
    summ_df.to_csv(OUT / "dev_rescore_groups.csv", index=False)
    # tests over all 46 exports (re-scored vs selected); Holm over the two Wilcoxon tests
    T = []
    for name, a, b in (("refusals (count of 100)", cells.sel_refusals, cells.rescore_refusals),
                       ("first-token KL", cells.sel_kl, cells.rescore_kl)):
        w = wilcoxon_paired(a.values, b.values)
        T.append(dict(test=f"Wilcoxon signed-rank, re-scored vs selected {name}, all 46 exports", **w))
    T = pd.DataFrame(T)
    T["wilcoxon_p_holm"] = holm(T["wilcoxon_p"]).values
    sp = spearman_ci(cells.sel_refusals.values, cells.rescore_refusals.values)
    T = pd.concat([T, pd.DataFrame([dict(test="Spearman, selected vs re-scored refusals, all 46 exports "
                                          "(bootstrap 95% CI, 4000 resamples)", **sp)])], ignore_index=True)
    T.to_csv(OUT / "dev_rescore_tests.csv", index=False)
    log(T.to_string())

    # --- vLLM / transformers evidence for the Gemma-2 config issue
    ev = []
    for lab in ["base", "dim", "mpoa", "ara", "som", "gmpoa"]:
        bd = BENCH_RTX / f"gemma-2-27b__{lab}"
        txt = (bd / "bench.log").read_text(errors="replace")
        eng = re.search(r"Initializing a V1 LLM engine \(v([0-9.]+)\)", txt)
        be = re.search(r"Using (\w+) attention backend", txt)
        fa = re.search(r"Using FlashAttention version (\d+)", txt)
        arch = re.search(r"Resolved architecture: (\w+)", txt)
        mh = json.loads((bd / "model_hash.json").read_text())["files"].get("config.json", "")
        bk = json.loads((bd / "summary.json").read_text()).get("backend", "")
        ev.append(dict(cell=f"gemma-2-27b__{lab}", bench_backend=bk, vllm=eng.group(1) if eng else "",
                       backend=be.group(1) if be else "",
                       flash_attn=fa.group(1) if fa else "", architecture=arch.group(1) if arch else "",
                       config_sha256=mh[:12],
                       tf_divergence=(round(float(tfi.loc[f"gemma-2-27b__{lab}", "tf_divergence"]), 4)
                                      if f"gemma-2-27b__{lab}" in tfi.index else "")))
    ev = pd.DataFrame(ev)
    base_cfg_p = next((MODELS_DIR / "base").glob("google__gemma-2-27b-it@*")) / "config.json"
    exp_cfgs = sorted((MODELS_DIR / "ours").glob("richardyoung__gemma-2-27b-it-*@*/config.json"))
    base_sha = sha256(base_cfg_p)
    exp_shas = {p.parent.name: sha256(p) for p in exp_cfgs}
    vg = BENCH_VENV / "vllm/model_executor/models/gemma2.py"
    vsrc = vg.read_text().splitlines()
    vlines = [f"{i + 1}: {l.strip()}" for i, l in enumerate(vsrc) if re.search(r"soft_?cap", l)]
    v_attn_impl = sum("_attn_implementation" in l for l in vsrc)
    fa_src = (BENCH_VENV / "vllm/v1/attention/backends/flash_attn.py").read_text()
    fa_softcap = fa_src.count("softcap=self.logits_soft_cap")
    vver = [p.name for p in BENCH_VENV.glob("vllm-*.dist-info")]
    sdpa_src = (TF_VENV / "transformers/integrations/sdpa_attention.py").read_text()
    fl_soft = sum((TF_VENV / f).read_text().count("softcap") > 0 for f in
                  ("transformers/modeling_flash_attention_utils.py", "transformers/integrations/flex_attention.py"))
    tf_ver = [p.name for p in TF_VENV.glob("transformers-*.dist-info")]
    cu = (TF_VENV / "transformers/configuration_utils.py").read_text()
    drop = "_attn_implementation_internal" in cu[cu.find("def _remove_keys_not_serialized"):][:1500]
    log(ev.to_string())
    log(f"vLLM gemma2.py softcap lines: {vlines}; _attn_implementation references: {v_attn_impl}")

    # tf divergence context (vLLM, primary cells)
    prim = tf[tf.role == "primary"]
    g2tf = prim[prim.label.str.startswith("gemma-2-27b__")].tf_divergence
    oth = prim[~prim.label.str.startswith("gemma-2-27b__")].tf_divergence
    g2cap = prim[prim.label.str.startswith("gemma-2-27b__")].capability_mean_delta_pp
    othcap = prim[~prim.label.str.startswith("gemma-2-27b__")].capability_mean_delta_pp

    # ------------------------------------------------------------------------------------------------ markdown
    A = summ_df.iloc[0]
    L = []
    L.append("# Dev-set re-score of the released exports (paper 33)\n")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} by "
             "`scripts/analysis/dev_rescore_summary.py` from `results/rescore_dev/rescore_dev_all.csv` (46 rows), "
             "the per-host TSVs, `paper/figures/data/trials.csv` and the Gemma-2 config/vLLM artefacts listed in the "
             "script docstring. Per-export rows: `results/analysis/dev_rescore_cells.csv`.\n")
    L.append("Selected = the trial chosen by the KL <= 0.1 rule, values logged during the search (adapter applied to "
             "the base). Re-scored = the released merged export scored by Heretic's evaluator (`heretic --model <pinned "
             "base> --evaluate-model <export>`) with the run's own config, code version and batch size, on the same 100 "
             "dev prompts (refusals) and 100 harmless prompts (first-token KL). Delta = re-scored - selected. Every "
             "selected trial/refusal/KL in the re-score table matches `trials.csv` (is_selected_kl0.1).\n")
    L.append("## Headline\n")
    L.append(f"- 46 released exports (42 primary + 4 gamma-MPOA controls; the 10 seed-2 replicates are not released). "
             f"All 46 re-scores rc = 0; the original model's dev refusals in the re-score equal the value Heretic "
             f"recorded in the search (`paper/tables/t1_dev_refusals.csv`, method base) in {BASE_OK}/46 runs.")
    L.append(f"- Delta refusals: median {A.median_d_ref:+.0f}, mean {A.mean_d_ref:+.2f}; {A.n_up} up, {A.n_same} "
             f"unchanged, {A.n_down} down; {A.n_within_2}/46 within +-2; range {A.min_d_ref:+d} to {A.max_d_ref:+d}.")
    L.append(f"- Delta KL: median {A.median_d_kl:+.4f}, mean {A.mean_d_kl:+.4f}, max |delta| {A.max_abs_d_kl:.4f}.")
    L.append(f"- Selected-trial refusal proportion inside the Wilson 95% interval of the re-scored proportion: "
             f"{A.n_sel_in_ci}/46.")
    oc = cells[cells.over_kl_cap]
    L.append(f"- Re-scored KL > 0.1: {len(oc)} exports: " + "; ".join(
        f"{r.model_display} {r.method_display} {r.sel_kl:.4f} -> {r.rescore_kl:.4f}" for _, r in oc.iterrows()) + ".")
    big = cells[cells.large_refusal_drift]
    L.append(f"- |Delta refusals| > 2: {len(big)} exports: " + "; ".join(
        f"{r.model_display} {r.method_display} {r.sel_refusals} -> {r.rescore_refusals} ({r.d_refusals:+d})"
        for _, r in big.iterrows()) + ".\n")

    L.append("## Tests (all 46 exports)\n")
    tt = T.astype(object).map(lambda x: "" if (isinstance(x, float) and np.isnan(x)) else
                              (f"{x:.4g}" if isinstance(x, float) else x))
    L.append(md_table(tt) + "\n")
    w0, w1, s2 = T.iloc[0], T.iloc[1], T.iloc[2]
    L.append(f"Re-scored refusal counts are higher than the selected trial's more often than lower ({int(w0.n_pos)} "
             f"vs {int(w0.n_neg)}, {int(w0.n_zero)} ties; Wilcoxon p = {w0.wilcoxon_p:.4f}, Holm {w0.wilcoxon_p_holm:.4f}; "
             f"sign test p = {w0.sign_test_p:.4f}); KL shows no significant shift (Wilcoxon p = {w1.wilcoxon_p:.3f}). Selected and "
             f"re-scored counts rank the exports almost identically (Spearman rho = {s2.rho:.3f} "
             f"[{s2.ci_lo:.3f}, {s2.ci_hi:.3f}]).\n")
    L.append("## By group\n")
    show = summ_df.rename(columns={"median_d_ref": "median d_ref", "mean_d_ref": "mean d_ref",
                                   "n_within_2": "within +-2", "max_abs_d_ref": "max |d_ref|",
                                   "median_d_kl": "median d_KL", "mean_d_kl": "mean d_KL",
                                   "max_abs_d_kl": "max |d_KL|", "n_over_kl_cap": "KL>0.1"})
    show = show[["group", "n", "median d_ref", "mean d_ref", "n_up", "n_same", "n_down", "within +-2", "min_d_ref",
                 "max_d_ref", "max |d_ref|", "median d_KL", "mean d_KL", "max |d_KL|", "KL>0.1"]]
    L.append(md_table(show) + "\n")
    ara = cells[cells.method == "ara"]
    L.append(f"ARA drifts most: {int((ara.d_refusals.abs() > DRIFT).sum())}/10 ARA exports move by more than 2 "
             f"refusals (vs {int((cells[cells.method != 'ara'].d_refusals.abs() > DRIFT).sum())}/36 for the other "
             f"methods), and ARA has the largest KL changes (mean {ara.d_kl.mean():+.4f}, max {ara.d_kl.max():+.4f}; "
             f"other methods max |d_KL| {cells[cells.method != 'ara'].d_kl.abs().max():.4f}). The large ARA drifts "
             f"are all main-study exports (searched on the H200, re-scored on the RTX PRO 6000); the two small-pair "
             f"ARA exports (same 4090 for search and re-score) move by "
             f"{', '.join(f'{x:+d}' for x in ara[ara.study == 'small'].d_refusals)}.\n")

    L.append("## Exports whose re-scored KL exceeds 0.1, and the selection rule\n")
    L.append("Runner-up = the rule applied to the logged trials of the run without the selected trial. Drift rule = "
             "the rule with every trial's logged KL shifted by this export's d_KL (assumes equal drift for every "
             "trial; untested).\n")
    t = oc[["model_display", "method_display", "role", "sel_trial", "sel_refusals", "sel_kl", "rescore_refusals",
            "rescore_kl", "d_kl", "runnerup_trial", "runnerup_refusals", "runnerup_kl", "drift_rule_trial",
            "drift_rule_refusals", "drift_rule_kl_logged"]].copy()
    for c in ["sel_kl", "rescore_kl", "runnerup_kl", "drift_rule_kl_logged"]:
        t[c] = t[c].map(lambda x: f"{x:.4f}")
    t["d_kl"] = t["d_kl"].map(lambda x: f"{x:+.4f}")
    L.append(md_table(t) + "\n")
    L.append("All 46 selected trials met KL <= 0.1 as logged. Held-out refusal, capability and teacher-forced "
             "divergence were measured on the released weights themselves (main study: downloaded back from the "
             "published repository; small pair: SHA-256-identical shards), so they describe the released models "
             "whatever the dev re-score. Only the 4 exports above lose eligibility if the re-scored KL replaced the "
             "logged KL; for the other 42 the selected trial stays within the budget on re-score (other trials were "
             "not re-scored, so their eligibility under re-measurement is unknown).\n")

    L.append("## Gemma-2-27B: SDPA vs eager re-score\n")
    gt = cells[cells.gemma2_eager][["method_display", "sel_refusals", "sel_kl", "sdpa_refusals", "sdpa_kl",
                                    "sdpa_d_refusals", "rescore_refusals", "rescore_kl", "d_refusals", "d_kl",
                                    "merged_repo", "evaluated_rev", "patched_rev12"]].copy()
    for c in ["sel_kl", "sdpa_kl", "rescore_kl"]:
        gt[c] = gt[c].map(lambda x: f"{x:.4f}")
    gt["d_kl"] = gt["d_kl"].map(lambda x: f"{x:+.4f}")
    gt["sdpa_d_refusals"] = gt["sdpa_d_refusals"].map(lambda x: f"{int(x):+d}")
    gt["sdpa_refusals"] = gt["sdpa_refusals"].map(lambda x: f"{int(x)}")
    gt["d_refusals"] = gt["d_refusals"].map(lambda x: f"{int(x):+d}")
    gt["evaluated_rev"] = gt["evaluated_rev"].str[:12]
    L.append(md_table(gt) + "\n")
    gs = cells[cells.gemma2_eager]
    L.append(f"- SDPA (config without `_attn_implementation`): KL {gs.sdpa_kl.min():.4f}-{gs.sdpa_kl.max():.4f}, "
             f"delta refusals {int(gs.sdpa_d_refusals.min()):+d} to {int(gs.sdpa_d_refusals.max()):+d}. Eager "
             f"(local config copy with `_attn_implementation: eager`, same pinned weights): KL "
             f"{gs.rescore_kl.min():.4f}-{gs.rescore_kl.max():.4f}, delta refusals {gs.d_refusals.min():+d} to "
             f"{gs.d_refusals.max():+d}. `rescore_dev_all.csv` carries the eager rows (merged path "
             f"`.../gemma2_eager/models/...`; values identical to `rtx_gemma2_eager/rescore_dev.tsv`); the SDPA rows "
             f"exist only in `rtx/rescore_dev.tsv`.")
    L.append(f"- Base config `{base_cfg_p.parent.name}/config.json` (sha256 {base_sha[:12]}): "
             f"`_attn_implementation` = {attn_impl(base_cfg_p)[0]}, transformers_version {attn_impl(base_cfg_p)[1]}. "
             f"Local copies of the pre-patch export configs: " + "; ".join(
                 f"{k} sha256 {v[:12]} `_attn_implementation` {attn_impl(MODELS_DIR / 'ours' / k / 'config.json')[0]}"
                 f" (transformers {attn_impl(MODELS_DIR / 'ours' / k / 'config.json')[1]})"
                 for k, v in exp_shas.items()) + ". No other base config in `models/paper33/base` sets "
             f"`_attn_implementation` or attention-logit soft-capping (Gemma-4 sets only final-logit soft-capping).")
    L.append(f"- Mechanism: in {', '.join(tf_ver)} (Heretic env; the RTX capture `pip-freeze_heretic_.venv.txt` lists "
             f"the same version) `PretrainedConfig._remove_keys_not_serialized` drops `_attn_implementation_internal` "
             f"when saving (string found in that function's source: {drop}). `integrations/sdpa_attention.py` contains 'softcap' "
             f"{sdpa_src.count('softcap')} times: the SDPA path ignores the soft-cap argument that `modeling_gemma2.py` "
             f"passes to the attention function; the eager function applies it (the flash-attention and flex paths, "
             f"not used here, also accept it: 'softcap' in flash/flex sources = {fl_soft}).")
    L.append("- Public configs patched 2026-11-06 (`notes/gemma2_config_patch_20261106.log`; one commit per repo adding "
             "only `_attn_implementation: eager`; old -> new, 12 hex): " + "; ".join(
                 f"{r.repo} {o} -> {r.new_rev12}" for o, r in pl.iterrows()) + ".\n")

    L.append("## vLLM (held-out generations, benchmarks, teacher-forced divergence) is unaffected\n")
    L.append(md_table(ev) + "\n")
    L.append(f"- All six Gemma-2 vLLM runs (original and five exports) used vLLM {ev.vllm.iloc[0]} with the same "
             f"attention backend ({ev.backend.iloc[0]}, FlashAttention {ev.flash_attn.iloc[0]}) and architecture "
             f"{ev.architecture.iloc[0]}. The five exports were benchmarked from configs with sha256 "
             f"{ev.config_sha256.iloc[1]} = the pre-patch export config (no `_attn_implementation`).")
    L.append(f"- vLLM source in the bench environment ({', '.join(vver)}; "
             f"`runs/mc-rtx-1/scripts/bench/.venv/.../vllm/model_executor/models/gemma2.py`): " + "; ".join(vlines)
             + f". `_attn_implementation` occurs {v_attn_impl} times in that file: vLLM takes the soft-cap values "
             "from `attn_logit_softcapping` / `final_logit_softcapping` in config.json (preserved in the exports) and "
             "passes them to its own attention backend; the transformers attention-implementation key is not read. "
             f"The FLASH_ATTN backend (`vllm/v1/attention/backends/flash_attn.py`) passes "
             f"`softcap=self.logits_soft_cap` to the kernel in {fa_softcap} call sites.")
    L.append(f"- Empirical check: teacher-forced divergence (vLLM, 350 benign held-out prompts) of the four primary "
             f"Gemma-2 exports {g2tf.min():.4f}-{g2tf.max():.4f} nats/token vs {oth.min():.4f}-{oth.max():.4f} for "
             f"the other 38 primary cells (inside their range); the Gemma-2 "
             f"capability changes are {g2cap.min():+.2f} to {g2cap.max():+.2f} pp (mean of five benchmarks; other 38 "
             f"primary cells {othcap.min():+.2f} to {othcap.max():+.2f} pp). This is "
             f"consistent with, but on its own does not prove, intact soft-capping; the basis is the source and "
             f"backend lines above.\n")
    L.append("## Notes\n")
    L.append("- The per-prompt dev outcomes of the search-time trial are not stored (the Optuna journal keeps the count "
             "and KL only), so no per-prompt paired test (McNemar, Newcombe) is possible; the counts are compared across exports with "
             "the Wilcoxon signed-rank and Spearman helpers of final.py.")
    L.append("- Differences between the trial and the export combine: merged bf16 weights W+BA vs the adapter applied "
             "at run time; GPU and kernel differences for the 36 main-study exports (H200 search, RTX PRO 6000 "
             "re-score); and greedy decoding on borderline prompts. They are not separated here.")
    (OUT / "dev_rescore_summary.md").write_text("\n".join(L) + "\n")
    log(f"wrote {OUT / 'dev_rescore_summary.md'}")


if __name__ == "__main__":
    main()
