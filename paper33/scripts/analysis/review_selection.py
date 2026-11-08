#!/usr/bin/env python3
"""Paper 33 -- reviewer-response analyses (review v5, 2026-11-06): KL-cap binding, the pair as the unit of
inference, baseline adjustment on two scales, the seed-2 batch-size confound and ARA export drift.

Purpose
  Answers, from data already on disk, concerns A1, A2, A3, B1 (data part), C1 and C5 of
  notes/REVIEW_v5_reviewer_agent_20261106.md. Analysis only: no GPU, no network, no ssh. Every input is read-only;
  no existing script, CSV or paper file is changed. Run by run_all.sh (step 9, after dev_rescore_summary.py; added 2026-11-06).

    (1) A2  KL-cap binding. Per optimisation run (trials.csv, seed 1 and seed 2): trials, trials at KL <= 0.1 and
            > 0.1, maximum KL, selected trial, and whether the cap bound. "Cap binding" = some trial with KL > 0.1 has
            fewer dev keyword refusals than the selected trial (equivalently: the selected trial is not the
            minimum-refusal trial of the run). "Left the cap region <= 1 time" = at most one trial above 0.1.
            The selected trial's parameters are read from the run's Optuna journal and placed in Heretic's search
            ranges (top/bottom 5% of a range = "at the bound"). H1 (direction class) and H2 are re-fitted, with the
            pooled GEE of h1h2.py, on the pair x method units where the cap bound in both cells and where it did not.
    (2) A1, C5  Pair as the unit. (a) Wald heterogeneity tests copied from h1h2_models.csv. (b) Per-pair generation
            effects of the direction class and ARA, raw and baseline-adjusted, in pp (paired item differences) and
            log-odds (per-pair GEE, as h1h2.py), pooled over the 4 main pairs with fixed-effect, DerSimonian-Laird
            and Hartung-Knapp-Sidik-Jonkman random-effects models (tau^2, I^2, 95% prediction interval).
            (c) The same pooling with run-to-run (seed) variance added to every per-pair effect. (d) Leave-one-family-
            out: final.analysis_g re-run and checked against final_pooled_robustness.csv, extended to the
            baseline-adjusted terms.
    (3) C1  Generation effect, originals' difference and baseline-adjusted effect per pair x method on the pp and
            the log-odds scale side by side; sign and significance differences flagged.
    (4) B1  Batch size of every small-pair seed run (output.log), sigma and the 2 x SD_diff threshold with and without
            the batch-64 seed-2 runs, and the flagged main-study effects under both.
    (5) A3  Dev re-score of the released exports (dev_rescore_cells.csv) with the GPU that rebuilt each export
            (reexport.log / publish.log) next to the search machine (trials.csv opt_machine).

Statistical conventions (reused from h1h2.py / final.py)
  Outcome: WildGuard refusal (ref_wg) on the 972 harmful held-out items; cells with role original / primary.
  GEE: logistic, exchangeable working correlation, clusters = prompt, robust SEs (h1h2.gee, h1h2._lc).
  pp effects per pair: mean of paired item differences, SE = SD(item differences)/sqrt(n) (Wald, paired); for a single
  method the paper's Newcombe interval (h1h2_generation_effects.csv) is carried along.
  Log-odds per pair x method (part 3): logit(p_new) - logit(p_old) of the marginal rates, delta-method SE with the
  item-level covariance of the paired 0/1 vectors (equals the saturated GEE with robust SEs); 0.5 added to k and 1 to
  n only if a rate is 0 or 1 (not needed in the current data).
  Random effects: DL tau^2 = max(0, (Q - (k-1)) / (sum w - sum w^2 / sum w)); DL CI = mu +/- 1.96 SE; HKSJ CI =
  mu +/- t(k-1) * sqrt(sum w*(y-mu)^2 / ((k-1) sum w*)) with the DL tau^2 (no truncation of the HK factor);
  prediction interval = mu +/- t(k-2) * sqrt(tau^2 + SE_DL^2) (Higgins, Thompson & Spiegelhalter 2009). k = 4.
  Seed variance (part 2c): sigma from the 10 small-pair seed pairs, sqrt(mean(d^2)/2) as in seednoise_flags.py,
  computed on the pp scale and, separately, on the logit scale (d = logit(rate s2) - logit(rate s1)). Each optimised
  cell gets an extra variance sigma^2; a single-method generation effect gets 2 sigma^2, the direction-class mean of
  m methods 2 sigma^2 / m, the H2 contrast 2 sigma^2 + 2 sigma^2 / m; originals are not optimised and add none.
  Approximations: sigma measured on 4-7B models is assumed for 22-32B models and for every method and refusal level;
  the log-odds class effect of the GEE is treated as the mean of the method log-odds effects. A delta-method variant
  (pp sigma mapped to logit at each cell's rate) is reported as a sensitivity; it is unstable near 0% refusal.

Outputs (results/analysis/, all overwritten)
  review_sel_kl_runs.csv          one row per optimisation run (56): KL-cap binding, selected trial, parameter flags
  review_sel_kl_params.csv        one row per run x parameter of the selected trial: value, range, position, flags
  review_sel_kl_summary.csv       binding counts by study x method class x generation (and role)
  review_sel_kl_units.csv         one row per pair x method (5 pairs, seed 1): binding in each cell, unit category
  review_sel_kl_h1h2.csv          pooled GEE H1/H2 terms re-fitted per cell subset (all, cap bound, not bound, ...)
  review_sel_pair_heterogeneity.csv  Wald generation x family / class x family tests (h1h2_models.csv)
  review_sel_pair_effects.csv     per pair x estimand x scale: estimate, prompt SE, seed SE, intervals
  review_sel_pair_meta.csv        pooled 4 main pairs: FE, DL, HKSJ, tau^2, I^2, Q, prediction interval per variance
  review_sel_pair_lofo.csv        leave-one-family-out GEE terms (reproduction of final_pooled_robustness.csv + adj.)
  review_sel_scales.csv           per pair x method/class: pp and log-odds, raw / originals / adjusted, sign flags
  review_sel_seed_runs.csv        per small-pair seed run: chosen batch size, OOM fallback
  review_sel_seed_sigma.csv       sigma, SD_diff, threshold: all 10 seed pairs vs without the batch-64 pairs
  review_sel_seed_flags.csv       16 main-study effects (raw and adjusted) flagged under both thresholds
  review_sel_export_drift.csv     46 released exports: search vs export GPU, dev re-score drift, KL > 0.1
  review_sel_export_tests.csv     drift by method x export host; Fisher exact tests
  review_sel_summary.md           key numbers and a verdict per reviewer concern

Run (from the paper root):
    scripts/figures/.venv/bin/python scripts/analysis/review_selection.py
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1h2 import ALPHA, Z, _lc, gee, md_table  # noqa: E402
import final  # noqa: E402  (final.load, final.analysis_g)

KL_MAX = 0.1
N_STARTUP = 30  # n_startup_trials in configs/core_*.toml (random TPE startup trials)
EDGE = 0.05  # a parameter value is "at a bound" when it lies in the top / bottom 5% of its search range
MAIN = ["qwen", "gemma", "olmo", "mistral"]
PAIRS = MAIN + ["qwen-small"]
PAIR_DISPLAY = {"qwen": "Qwen", "gemma": "Gemma", "olmo": "OLMo", "mistral": "Mistral", "qwen-small": "Small pair"}
FAMILY_TO_PAIR = {"Qwen": "qwen", "Gemma": "gemma", "OLMo": "olmo", "Mistral Small": "mistral",
                  "Qwen (small pair)": "qwen-small"}
DIR_METHODS = ["dim", "mpoa", "som", "jspace"]
METHOD_DISPLAY = {"original": "Original", "dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space",
                  "ara": "ARA", "gamma_mpoa": "gamma-MPOA", "direction": "direction class", "h2": "H2 contrast"}
RUN_ROOTS = {"mc-h200-1": ROOT / "runs/mc-h200-1/runs", "local-4090": ROOT / "runs/local-4090",
             "mc-rtx-1": ROOT / "runs/mc-rtx-1/runs"}
GPU_TO_MACHINE = {"H200": "mc-h200-1", "RTX PRO 6000": "mc-rtx-1", "RTX 4090": "local-4090"}
LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def logit(p):
    p = np.asarray(p, float)
    return np.log(p / (1 - p))


def fpp(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"


def flo(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{d}f}"


def fp(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ================================================================================================ (1) KL-cap binding
def cell_label(model: str, method: str, seed: int) -> str:
    m = "gmpoa" if method == "gamma_mpoa" else method
    return f"{model}__{m}" + (f"-s{seed}" if seed != 1 else "")


def find_journal(run: str) -> tuple[Path | None, str]:
    for mach, root in RUN_ROOTS.items():
        ck = root / run / "checkpoints"
        if ck.is_dir():
            js = sorted(ck.glob("*.jsonl"))
            if len(js) == 1:
                return js[0], mach
    return None, ""


def parse_journal(path: Path) -> dict:
    """Optuna JournalStorage replay (one study per file): trial_id -> params (internal values), distributions,
    state, values, Heretic's 1-based index. op codes: 4 create trial, 5 set param, 6 set state/values, 8 user attr."""
    tr: dict[int, dict] = {}
    n = 0
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            op = j["op_code"]
            if op == 4:
                tr[n] = dict(params={}, dists={}, state=None, values=None, index=None)
                n += 1
            elif op == 5:
                t = tr[j["trial_id"]]
                t["params"][j["param_name"]] = j["param_value_internal"]
                t["dists"][j["param_name"]] = json.loads(j["distribution"])
            elif op == 6:
                tr[j["trial_id"]]["state"] = j.get("state")
                if j.get("values") is not None:
                    tr[j["trial_id"]]["values"] = j["values"]
            elif op == 8:
                ua = j.get("user_attr", {})
                if "index" in ua:
                    tr[j["trial_id"]]["index"] = ua["index"]
    return tr


def rel_position(v: float, dist: dict) -> tuple[float, float, float, bool]:
    a = dist["attributes"]
    lo, hi, lg = float(a["low"]), float(a["high"]), bool(a.get("log", False))
    if hi == lo:
        return np.nan, lo, hi, lg
    if lg:
        return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)), lo, hi, lg
    return (v - lo) / (hi - lo), lo, hi, lg


def strength_direction(pname: str) -> str:
    """Which end of a parameter's range makes the edit stronger (more layers / larger weights): 'upper', 'lower' or
    '' (position-type parameters such as layer positions and direction index have no strength direction)."""
    if pname.endswith((".max_weight", ".min_weight", ".min_weight_distance")) or pname == "direction_decay":
        return "upper"
    if pname in ("end_layer_index", "steer_bad_behavior_weight", "overcorrect_relative_weight", "neighbor_count"):
        return "upper"
    if pname == "start_layer_index":
        return "lower"
    if pname == "preserve_good_behavior_weight":
        return "lower"
    return ""


def analysis_kl(trials: pd.DataFrame):
    run_rows, prm_rows = [], []
    for (model, method, seed, run), g in trials.groupby(["model", "method", "seed", "run"], sort=False):
        g = g.sort_values("trial")
        sel = g[g["is_selected_kl0.1"].astype(str) == "True"]
        assert len(sel) == 1, run
        s = sel.iloc[0]
        above = g[g.kl > KL_MAX]
        min_all = int(g.refusals.min())
        cls = "optimisation" if method == "ara" else "direction"
        role = "control" if method == "gamma_mpoa" else ("replicate" if seed != 1 else "primary")
        r = dict(run=run, label=cell_label(model, method, int(seed)), model=model, family=s["family"],
                 pair=FAMILY_TO_PAIR.get(s["family"], s["family"]), study=s["study"], generation=s["generation"],
                 gen_new=int(s["generation"] in ("new", "small-new")), method=method, method_class=cls, role=role,
                 seed=int(seed), opt_machine=s["opt_machine"], n_trials=len(g), n_kl_le_cap=int((g.kl <= KL_MAX).sum()),
                 n_kl_gt_cap=len(above), max_kl=float(g.kl.max()), min_kl=float(g.kl.min()),
                 sel_trial=int(s.trial), sel_optuna_number=int(s.optuna_number), sel_refusals=int(s.refusals),
                 sel_kl=float(s.kl), min_refusals_all=min_all,
                 min_refusals_above_cap=int(above.refusals.min()) if len(above) else np.nan,
                 kl_of_best_above_cap=float(above.sort_values(["refusals", "kl"]).kl.iloc[0]) if len(above) else np.nan)
        r["cap_binding"] = bool(len(above) and above.refusals.min() < s.refusals)
        r["sel_is_min_refusal_overall"] = bool(s.refusals == min_all)
        assert r["cap_binding"] == (not r["sel_is_min_refusal_overall"]), run
        r["refusals_gained_above_cap"] = (int(s.refusals) - r["min_refusals_above_cap"]) if r["cap_binding"] else 0
        r["never_left_cap"] = bool(len(above) == 0)
        r["left_cap_at_most_once"] = bool(len(above) <= 1)
        r["sel_kl_within_0.01_of_cap"] = bool(s.kl > KL_MAX - 0.01)
        if r["cap_binding"] and not r["left_cap_at_most_once"]:
            r["cap_status"] = "binding"
        elif r["cap_binding"]:
            r["cap_status"] = "binding (only 1 trial above cap)"
        elif r["left_cap_at_most_once"]:
            r["cap_status"] = "not binding: search stayed below the cap (<= 1 trial above)"
        else:
            r["cap_status"] = "not binding: trials above the cap were not better"
        # ---- journal: parameters of the selected trial and their position in the search ranges
        jp, jm = find_journal(run)
        r["journal"] = str(jp.relative_to(ROOT)) if jp else ""
        r["journal_machine"] = jm
        if jp is not None:
            tr = parse_journal(jp)
            mism = 0
            for row in g.itertuples():
                t = tr.get(int(row.optuna_number))
                if t is None or t["values"] is None or abs(t["values"][0] * 100 - row.refusals) > 0.5 \
                        or abs(t["values"][1] - row.kl) > 1e-6 or (t["index"] is not None and t["index"] != row.trial):
                    mism += 1
            r["journal_rows_mismatching_trials_csv"] = mism
            ts = tr[int(s.optuna_number)]
            n_upper = n_lower = n_strength_edge = 0
            strength_edge = []
            for pname, v in ts["params"].items():
                dist = ts["dists"][pname]
                if dist["name"] == "CategoricalDistribution":
                    prm_rows.append(dict(run=run, label=r["label"], method=method, method_class=cls, param=pname,
                                         value=dist["attributes"]["choices"][int(v)], low=np.nan, high=np.nan,
                                         log_scale=False, rel_position=np.nan, at_upper=False, at_lower=False,
                                         stronger_end="", at_strength_end=False))
                    continue
                pos, lo, hi, lg = rel_position(float(v), dist)
                up, lw = bool(pos >= 1 - EDGE), bool(pos <= EDGE)
                sd = strength_direction(pname)
                at_s = (sd == "upper" and up) or (sd == "lower" and lw)
                n_upper += up
                n_lower += lw
                if at_s:
                    n_strength_edge += 1
                    strength_edge.append(pname)
                prm_rows.append(dict(run=run, label=r["label"], method=method, method_class=cls, param=pname,
                                     value=float(v), low=lo, high=hi, log_scale=lg, rel_position=pos, at_upper=up,
                                     at_lower=lw, stronger_end=sd, at_strength_end=at_s))
                if pname in ("attn.o_proj.max_weight", "mlp.down_proj.max_weight"):
                    r[f"sel_{pname.split('.')[0]}_max_weight"] = float(v)
                    r[f"sel_{pname.split('.')[0]}_max_weight_pos"] = pos
            r["sel_n_params_at_upper"] = n_upper
            r["sel_n_params_at_lower"] = n_lower
            r["sel_n_strength_params_at_strong_end"] = n_strength_edge
            r["sel_strength_params_at_strong_end"] = ";".join(strength_edge)
            if cls == "direction":
                r["sel_max_weight_at_ceiling"] = bool(r.get("sel_attn_max_weight_pos", 0) >= 1 - EDGE
                                                      or r.get("sel_mlp_max_weight_pos", 0) >= 1 - EDGE)
                # how often the search tried the strongest weights, and what it found there
                ceil = []
                for row in g.itertuples():
                    t = tr[int(row.optuna_number)]
                    pa = rel_position(float(t["params"]["attn.o_proj.max_weight"]),
                                      t["dists"]["attn.o_proj.max_weight"])[0]
                    pm = rel_position(float(t["params"]["mlp.down_proj.max_weight"]),
                                      t["dists"]["mlp.down_proj.max_weight"])[0]
                    if pa >= 1 - EDGE or pm >= 1 - EDGE:
                        ceil.append((row.refusals, row.kl))
                r["n_trials_max_weight_at_ceiling"] = len(ceil)
                r["min_refusals_max_weight_at_ceiling"] = min(c[0] for c in ceil) if ceil else np.nan
                r["max_kl_max_weight_at_ceiling"] = max(c[1] for c in ceil) if ceil else np.nan
        r["sel_is_startup_trial"] = bool(int(s.trial) <= N_STARTUP)
        run_rows.append(r)
    runs = pd.DataFrame(run_rows)
    prm = pd.DataFrame(prm_rows)
    # identical selected parameter vectors across runs (same sampler seed + same search space = same startup draws)
    num = prm[prm.rel_position.notna()]
    sig_ = num.assign(v=num.value.astype(float).round(10).astype(str)).groupby("run").apply(
        lambda d: "|".join(f"{a}={b}" for a, b in sorted(zip(d.param, d.v))), include_groups=False)
    runs["sel_params_identical_to"] = [";".join(sorted(x for x in sig_[sig_ == sig_.get(run)].index if x != run))
                                       if run in sig_ else "" for run in runs.run]

    # ---- summary by study x class x generation (role split)
    srows = []

    def summ(name, sub):
        return dict(group=name, n_runs=len(sub), n_cap_binding=int(sub.cap_binding.sum()),
                    n_binding_more_than_one_above=int((sub.cap_binding & ~sub.left_cap_at_most_once).sum()),
                    n_never_left_cap=int(sub.never_left_cap.sum()),
                    n_left_cap_at_most_once=int(sub.left_cap_at_most_once.sum()),
                    n_not_binding_and_le1_above=int((~sub.cap_binding & sub.left_cap_at_most_once).sum()),
                    median_max_kl=float(sub.max_kl.median()), min_max_kl=float(sub.max_kl.min()),
                    max_max_kl=float(sub.max_kl.max()), median_sel_kl=float(sub.sel_kl.median()),
                    n_sel_kl_within_0p01_of_cap=int(sub["sel_kl_within_0.01_of_cap"].sum()),
                    median_refusals_gained_above_cap=float(sub.loc[sub.cap_binding, "refusals_gained_above_cap"].median())
                    if sub.cap_binding.any() else np.nan,
                    n_sel_max_weight_at_ceiling=int(sub.get("sel_max_weight_at_ceiling", pd.Series(dtype=bool))
                                                    .fillna(False).astype(bool).sum()),
                    n_sel_any_strength_param_at_strong_end=int((sub.sel_n_strength_params_at_strong_end > 0).sum()),
                    runs_le1_above=";".join(sub[sub.left_cap_at_most_once].label))
    prim = runs[runs.role == "primary"]
    for st in ["main", "small"]:
        for cls in ["direction", "optimisation"]:
            sub = prim[(prim.study == st) & (prim.method_class == cls)]
            srows.append(summ(f"{st} seed 1, {cls}", sub))
            for gen in sorted(sub.generation.unique()):
                srows.append(summ(f"{st} seed 1, {cls}, {gen}", sub[sub.generation == gen]))
    srows.append(summ("seed-2 replicates (small pair), direction", runs[(runs.role == "replicate")
                                                                      & (runs.method_class == "direction")]))
    srows.append(summ("seed-2 replicates (small pair), optimisation", runs[(runs.role == "replicate")
                                                                         & (runs.method_class == "optimisation")]))
    srows.append(summ("gamma-MPOA controls", runs[runs.role == "control"]))
    summary = pd.DataFrame(srows)
    return runs, prm, summary


def kl_units(runs: pd.DataFrame, gen_eff: pd.DataFrame) -> pd.DataFrame:
    """Main-study and small-pair seed-1 pair x method units with the cap status of both cells."""
    p = runs[runs.role == "primary"]
    rows = []
    ge = gen_eff[(gen_eff.judge == "wg") & (gen_eff.subset == "all")].set_index(["pair", "method"])
    for (pair, method), g in p.groupby(["pair", "method"], sort=False):
        if set(g.gen_new) != {0, 1}:
            continue
        o, n = g[g.gen_new == 0].iloc[0], g[g.gen_new == 1].iloc[0]
        nb = int(o.cap_binding) + int(n.cap_binding)
        cat = {2: "cap bound in both cells", 0: "cap bound in neither cell", 1: "cap bound in one cell"}[nb]
        le1 = bool(o.left_cap_at_most_once or n.left_cap_at_most_once)
        e = ge.loc[(pair, method)]
        rows.append(dict(pair=pair, study=o.study, method=method, method_class=o.method_class, old_label=o.label,
                         new_label=n.label, old_cap_binding=bool(o.cap_binding), new_cap_binding=bool(n.cap_binding),
                         old_n_kl_gt_cap=int(o.n_kl_gt_cap), new_n_kl_gt_cap=int(n.n_kl_gt_cap),
                         old_sel_kl=o.sel_kl, new_sel_kl=n.sel_kl, unit_category=cat,
                         any_cell_left_cap_at_most_once=le1, gen_effect_pp=100 * e["delta"],
                         gen_effect_lo_pp=100 * e["delta_lo"], gen_effect_hi_pp=100 * e["delta_hi"],
                         baseline_adj_pp=100 * e["did_baseline_adj"]))
    return pd.DataFrame(rows)


def harmful_main(rj: pd.DataFrame) -> pd.DataFrame:
    h = rj[(rj.kind == "harmful") & rj.ref_wg.notna() & rj.role.isin(["original", "primary"])].copy()
    h["y"] = h["ref_wg"].astype(float)
    h["gen_new"] = h["gen_new"].astype(int)
    h["opt"] = (h["method_class"] == "optimisation").astype(int)
    h["cls3"] = h["method_class"]
    return h


def fit_h1h2_terms(df_mod: pd.DataFrame, df_all: pd.DataFrame) -> list[dict]:
    """Pooled main-study GEE of h1h2.py on a cell subset: planned model (or direction-only model when the subset has
    no ARA units) and the with-originals model (baseline-adjusted effects)."""
    out = []
    fe = " + C(family)" if df_mod.family.nunique() > 1 else ""
    has_opt = df_mod.opt.nunique() == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if has_opt:
            res, names, note = gee(df_mod, "y ~ gen_new * opt" + fe)
            terms = (("gen effect | direction class (log-odds)", {"gen_new": 1}),
                     ("gen effect | ARA (log-odds)", {"gen_new": 1, "gen_new:opt": 1}),
                     ("generation x class, H2 (log-odds)", {"gen_new:opt": 1}))
        else:
            res, names, note = gee(df_mod, "y ~ gen_new" + fe)
            terms = (("gen effect | direction class (log-odds)", {"gen_new": 1}),)
        for t, w in terms:
            out.append(dict(term=t, model="gee_planned" if has_opt else "gee_direction_only", **_lc(res, names, w),
                            note=note))
        if df_all is not None and (df_all.cls3 == "original").any():
            res, names, note = gee(df_all, "y ~ C(cls3, Treatment('original')) * gen_new" + fe)
            for nm in names:
                if nm.endswith(":gen_new") and "[T." in nm:
                    c = nm.split("[T.")[1].split("]")[0]
                    out.append(dict(term=f"baseline-adjusted gen effect | {'ARA' if c == 'optimisation' else c} "
                                         f"(log-odds)", model="gee_with_originals", **_lc(res, names, {nm: 1}),
                                    note=note))
            out.append(dict(term="originals' generation difference (log-odds)", model="gee_with_originals",
                            **_lc(res, names, {"gen_new": 1}), note=note))
    return out


def analysis_kl_h1h2(h: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    hm = h[h.study == "main"]
    um = units[units.study == "main"]

    def cells_of(u):
        return set(u.old_label) | set(u.new_label)

    subsets = {
        "S0 all 16 units (= h1h2.py)": um,
        "S1 cap bound in both cells": um[um.unit_category == "cap bound in both cells"],
        "S2 cap not bound in at least one cell": um[um.unit_category != "cap bound in both cells"],
        "S3 cap bound in neither cell": um[um.unit_category == "cap bound in neither cell"],
        "S4 both cells had >= 2 trials above the cap": um[~um.any_cell_left_cap_at_most_once],
        "S5 a cell had <= 1 trial above the cap": um[um.any_cell_left_cap_at_most_once],
    }
    rows = []
    for sname, u in subsets.items():
        labs = cells_of(u)
        dm = hm[hm.label.isin(labs) & (hm.role == "primary")]
        fams = sorted(dm.family.unique())
        da = hm[(hm.label.isin(labs) | ((hm.role == "original") & hm.pair.isin(u.pair.unique())))]
        base = dict(cell_subset=sname, n_units=len(u), n_direction_units=int((u.method_class == "direction").sum()),
                    n_ara_units=int((u.method_class == "optimisation").sum()),
                    units=";".join(f"{a}:{b}" for a, b in zip(u.pair, u.method)), families="+".join(fams),
                    n_obs=len(dm))
        ud, ua = u[u.method_class == "direction"], u[u.method_class == "optimisation"]
        base.update(mean_gen_effect_direction_pp=float(ud.gen_effect_pp.mean()) if len(ud) else np.nan,
                    mean_baseline_adj_direction_pp=float(ud.baseline_adj_pp.mean()) if len(ud) else np.nan,
                    mean_gen_effect_ara_pp=float(ua.gen_effect_pp.mean()) if len(ua) else np.nan,
                    mean_baseline_adj_ara_pp=float(ua.baseline_adj_pp.mean()) if len(ua) else np.nan)
        if len(ud) == 0:
            rows.append(dict(**base, term="(no direction units)", model=""))
            continue
        for t in fit_h1h2_terms(dm, da):
            rows.append(dict(**base, **t))
    return pd.DataFrame(rows)


# ================================================================================= (4) seed noise and batch size
def batch_size_of(logfile: Path) -> tuple[float, bool]:
    if not logfile.is_file():
        return np.nan, False
    txt = logfile.read_text(errors="replace")
    m = re.findall(r"Chosen batch size: (\d+)", txt)
    return (float(m[0]) if m else np.nan), ("out of memory" in txt)


def sigma_from(d):
    d = np.asarray(d, float)
    return float(np.sqrt(np.mean(d ** 2) / 2.0))


def chi2_ci(d):
    d = np.asarray(d, float)
    n, ss = len(d), float(np.sum(d ** 2))
    return float(np.sqrt(ss / (2 * stats.chi2.ppf(0.975, n)))), float(np.sqrt(ss / (2 * stats.chi2.ppf(0.025, n))))


def analysis_seed(trials: pd.DataFrame, seed_cells: pd.DataFrame, gen_eff: pd.DataFrame, sig_csv: pd.DataFrame):
    # ---- batch size of every small-pair run (seed 1 and 2)
    rr = []
    for run in sorted(trials[trials.study == "small"].run.unique()):
        bs, oom = batch_size_of(RUN_ROOTS["local-4090"] / run / "output.log")
        model, method, tag = run.split("__")
        rr.append(dict(run=run, model=model, method=method, seed=int(tag[-1]), chosen_batch_size=bs,
                       oom_fallback=oom, output_log=f"runs/local-4090/{run}/output.log"))
    sr = pd.DataFrame(rr)
    bs2 = sr[sr.seed == 2].set_index(["model", "method"]).chosen_batch_size
    bs1 = sr[sr.seed == 1].set_index(["model", "method"]).chosen_batch_size
    sc = seed_cells.copy()
    sc["d_pp"] = 100 * (sc.s2_wg_rate - sc.s1_wg_rate)
    sc["d_logit"] = logit(sc.s2_wg_rate) - logit(sc.s1_wg_rate)
    sc["s1_batch"] = [bs1.get((m, me), np.nan) for m, me in zip(sc.model, sc.method)]
    sc["s2_batch"] = [bs2.get((m, me), np.nan) for m, me in zip(sc.model, sc.method)]
    sc["batch_changed"] = sc.s1_batch != sc.s2_batch
    # ---- sigma variants
    sig_ref = float(sig_csv.loc[sig_csv.quantity.str.startswith("sigma (per-cell"), "estimate"].iloc[0])
    variants = {"all 10 seed pairs": sc, "without the batch-64 seed-2 runs": sc[~sc.batch_changed]}
    srows, sig = [], {}
    for name, sub in variants.items():
        for scale, col in (("pp", "d_pp"), ("logit", "d_logit")):
            s = sigma_from(sub[col])
            lo, hi = chi2_ci(sub[col])
            med = float(np.median(np.abs(sub[col])))
            s_mad = med / (stats.norm.ppf(0.75) * np.sqrt(2))
            sig[(name, scale)] = s
            srows.append(dict(seed_pairs=name, scale=scale, n_pairs=len(sub), sigma=s, sigma_lo=lo, sigma_hi=hi,
                              sd_diff=s * np.sqrt(2), threshold=2 * np.sqrt(2) * s,
                              threshold_lo=2 * np.sqrt(2) * lo, threshold_hi=2 * np.sqrt(2) * hi, sigma_mad=s_mad,
                              mean_d=float(sub[col].mean()), max_abs_d=float(np.abs(sub[col]).max()),
                              cells="; ".join(f"{m} {me}" for m, me in zip(sub.model, sub.method))))
    sigma = pd.DataFrame(srows)
    assert abs(sig[("all 10 seed pairs", "pp")] - sig_ref) < 1e-3, (sig[("all 10 seed pairs", "pp")], sig_ref)
    # ---- flags of the 16 main-study effects under both thresholds
    e = gen_eff[(gen_eff.judge == "wg") & (gen_eff.subset == "all") & (gen_eff.method != "original")].copy()
    e = e[e.pair.isin(MAIN)]
    frows = []
    thr = {n: 2 * np.sqrt(2) * sig[(n, "pp")] for n in variants}
    for r in e.itertuples():
        d = dict(pair=r.pair, method=r.method, effect_pp=100 * r.delta, effect_lo_pp=100 * r.delta_lo,
                 effect_hi_pp=100 * r.delta_hi, baseline_adj_pp=100 * r.did_baseline_adj, p_holm_H1=r.p_holm_H1)
        for n, t in thr.items():
            k = "all10" if n.startswith("all") else "excl_b64"
            d[f"threshold_{k}"] = t
            d[f"exceeds_{k}"] = bool(abs(100 * r.delta) > t)
            d[f"adj_exceeds_{k}"] = bool(abs(100 * r.did_baseline_adj) > t)
        frows.append(d)
    flags = pd.DataFrame(frows)
    return sr, sc, sigma, flags, sig


# ======================================================================================= (2) pair as the unit
def pair_cells(h: pd.DataFrame, pair: str) -> dict:
    m = h[h.pair == pair].drop_duplicates("label")
    return {(r.method, int(r.gen_new)): r.label for r in m.itertuples()}


def item_mean_se(d: np.ndarray) -> tuple[float, float]:
    d = d[~np.isnan(d)]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def analysis_pair_effects(h: pd.DataFrame, sig: dict) -> pd.DataFrame:
    wide = h.pivot(index="id", columns="label", values="y")
    rows = []
    for pair in PAIRS:
        C = pair_cells(h, pair)
        dms = [m for m in DIR_METHODS if (m, 0) in C and (m, 1) in C]
        nd = len(dms)
        ok_needed = [C[(m, g)] for m in dms + ["ara", "original"] for g in (0, 1)]
        W = wide[ok_needed]
        dd = np.mean([W[C[(m, 1)]].values - W[C[(m, 0)]].values for m in dms], axis=0)
        da = W[C[("ara", 1)]].values - W[C[("ara", 0)]].values
        do = W[C[("original", 1)]].values - W[C[("original", 0)]].values
        rates = W.mean()
        p_dir_old = float(np.mean([rates[C[(m, 0)]] for m in dms]))
        p_dir_new = float(np.mean([rates[C[(m, 1)]] for m in dms]))
        p_ara_old, p_ara_new = float(rates[C[("ara", 0)]]), float(rates[C[("ara", 1)]])
        # ---- pp: paired item differences
        pp = {"direction raw": dd, "direction adjusted": dd - do, "ARA raw": da, "ARA adjusted": da - do,
              "H2 contrast (ARA - direction)": da - dd}
        # ---- log-odds: per-pair GEE (as h1h2.py)
        mod = h[(h.pair == pair) & (h.role == "primary") & h.method.isin(dms + ["ara"])]
        alc = h[(h.pair == pair) & h.method.isin(dms + ["ara", "original"])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1, n1, _ = gee(mod, "y ~ gen_new * opt")
            r2, n2, _ = gee(alc, "y ~ C(cls3, Treatment('original')) * gen_new")
        lo_terms = {"direction raw": _lc(r1, n1, {"gen_new": 1}),
                    "ARA raw": _lc(r1, n1, {"gen_new": 1, "gen_new:opt": 1}),
                    "H2 contrast (ARA - direction)": _lc(r1, n1, {"gen_new:opt": 1}),
                    "direction adjusted": _lc(r2, n2, {"C(cls3, Treatment('original'))[T.direction]:gen_new": 1}),
                    "ARA adjusted": _lc(r2, n2, {"C(cls3, Treatment('original'))[T.optimisation]:gen_new": 1})}
        # seed-variance multipliers (units of sigma^2) and delta-method logit variances
        mult = {"direction raw": 2 / nd, "direction adjusted": 2 / nd, "ARA raw": 2.0, "ARA adjusted": 2.0,
                "H2 contrast (ARA - direction)": 2.0 + 2 / nd}

        def dvar(p):
            return 1.0 / (p * (1 - p)) ** 2

        sp = sig[("all 10 seed pairs", "pp")] / 100
        delta_lo = {"direction raw": sp ** 2 / nd * (dvar(p_dir_new) + dvar(p_dir_old)),
                    "ARA raw": sp ** 2 * (dvar(p_ara_new) + dvar(p_ara_old))}
        delta_lo["direction adjusted"] = delta_lo["direction raw"]
        delta_lo["ARA adjusted"] = delta_lo["ARA raw"]
        delta_lo["H2 contrast (ARA - direction)"] = delta_lo["direction raw"] + delta_lo["ARA raw"]
        for est in pp:
            for scale in ("pp", "log-odds"):
                if scale == "pp":
                    e, se = item_mean_se(pp[est])
                    s_all = sig[("all 10 seed pairs", "pp")] / 100
                    s_b64 = sig[("without the batch-64 seed-2 runs", "pp")] / 100
                    dl = np.nan
                else:
                    e, se = lo_terms[est]["estimate"], lo_terms[est]["se"]
                    s_all = sig[("all 10 seed pairs", "logit")]
                    s_b64 = sig[("without the batch-64 seed-2 runs", "logit")]
                    dl = delta_lo[est]
                seed_se = np.sqrt(mult[est]) * s_all
                seed_se_b64 = np.sqrt(mult[est]) * s_b64
                tot = np.sqrt(se ** 2 + seed_se ** 2)
                rows.append(dict(pair=pair, study="small" if pair == "qwen-small" else "main", estimand=est,
                                 scale=scale, n_direction_methods=nd, direction_methods="+".join(dms),
                                 estimate=e, se_prompt=se, lo_prompt=e - Z * se, hi_prompt=e + Z * se,
                                 p_prompt=float(2 * stats.norm.sf(abs(e / se))),
                                 seed_var_multiplier=mult[est], se_seed=seed_se, se_total=tot,
                                 lo_total=e - Z * tot, hi_total=e + Z * tot,
                                 p_total=float(2 * stats.norm.sf(abs(e / tot))),
                                 se_seed_excl_b64=seed_se_b64,
                                 se_total_excl_b64=np.sqrt(se ** 2 + seed_se_b64 ** 2),
                                 se_seed_delta_method=np.sqrt(dl) if np.isfinite(dl) else np.nan,
                                 p_dir_old=p_dir_old, p_dir_new=p_dir_new, p_ara_old=p_ara_old, p_ara_new=p_ara_new,
                                 n_items=int((~np.isnan(pp[est])).sum())))
    return pd.DataFrame(rows)


def meta(y, v) -> dict:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1 / v
    mu_fe = float(np.sum(w * y) / np.sum(w))
    se_fe = float(np.sqrt(1 / np.sum(w)))
    Q = float(np.sum(w * (y - mu_fe) ** 2))
    Cc = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / Cc)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se_dl = float(np.sqrt(1 / np.sum(ws)))
    q_hk = float(np.sum(ws * (y - mu) ** 2) / (k - 1))
    se_hk = float(np.sqrt(q_hk / np.sum(ws)))
    t1, t2 = stats.t.ppf(0.975, k - 1), stats.t.ppf(0.975, k - 2)
    pi_half = t2 * np.sqrt(tau2 + se_dl ** 2)
    return dict(k=k, fe_estimate=mu_fe, fe_lo=mu_fe - Z * se_fe, fe_hi=mu_fe + Z * se_fe,
                fe_p=float(2 * stats.norm.sf(abs(mu_fe / se_fe))), Q=Q, Q_df=k - 1,
                Q_p=float(stats.chi2.sf(Q, k - 1)), I2=I2, tau2=tau2, tau=np.sqrt(tau2), re_estimate=mu,
                dl_se=se_dl, dl_lo=mu - Z * se_dl, dl_hi=mu + Z * se_dl, dl_p=float(2 * stats.norm.sf(abs(mu / se_dl))),
                hksj_se=se_hk, hksj_q=q_hk, hksj_lo=mu - t1 * se_hk, hksj_hi=mu + t1 * se_hk,
                hksj_p=float(2 * stats.t.sf(abs(mu / se_hk), k - 1)) if se_hk > 0 else np.nan,
                pi_lo=mu - pi_half, pi_hi=mu + pi_half,
                n_pairs_positive=int(np.sum(y > 0)), n_pairs_negative=int(np.sum(y < 0)),
                min_pair=float(np.min(y)), max_pair=float(np.max(y)))


def analysis_meta(eff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    m = eff[eff.pair.isin(MAIN)]
    for (est, scale), g in m.groupby(["estimand", "scale"], sort=False):
        g = g.set_index("pair").loc[MAIN]
        variants = {"prompt sampling only": g.se_prompt ** 2,
                    "prompt + seed (sigma, 10 seed pairs)": g.se_total ** 2,
                    "prompt + seed (sigma without batch-64 pairs)": g.se_total_excl_b64 ** 2}
        if scale == "log-odds":
            variants["prompt + seed (pp sigma, delta method; unstable near 0%)"] = \
                g.se_prompt ** 2 + g.se_seed_delta_method ** 2
        for vname, v in variants.items():
            rows.append(dict(estimand=est, scale=scale, variance=vname,
                             pair_estimates="; ".join(f"{p} {x:+.4f}" for p, x in zip(MAIN, g.estimate)),
                             **meta(g.estimate.values, v.values)))
    return pd.DataFrame(rows)


def analysis_lofo(rj: pd.DataFrame, meta_df: pd.DataFrame, h: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    g = final.analysis_g(rj, meta_df)
    ref = pd.read_csv(OUT / "final_pooled_robustness.csv")
    mm = g.merge(ref, on=["cell_set", "term"], suffixes=("", "_ref"))
    maxdiff = float(np.max(np.abs(mm.estimate - mm.estimate_ref)))
    g = g.assign(source="final.analysis_g (reproduces final_pooled_robustness.csv)", scale="log-odds")
    rows = []
    hm = h[h.study == "main"]
    sets = {"all main-study cells": MAIN}
    for f in MAIN:
        sets[f"leave out {f}"] = [p for p in MAIN if p != f]
    for sname, fams in sets.items():
        da = hm[hm.pair.isin(fams)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, names, _ = gee(da, "y ~ C(cls3, Treatment('original')) * gen_new + C(family)")
        for nm, lab in (("C(cls3, Treatment('original'))[T.direction]:gen_new",
                         "baseline-adjusted gen effect | direction class"),
                        ("C(cls3, Treatment('original'))[T.optimisation]:gen_new", "baseline-adjusted gen effect | ARA"),
                        ("gen_new", "originals' generation difference")):
            e = _lc(res, names, {nm: 1})
            rows.append(dict(cell_set=sname, term=lab, estimate=e["estimate"], se=e["se"], lo=e["lo"], hi=e["hi"],
                             p=e["p"], n_obs=len(da), n_cells=da.label.nunique(), families="+".join(fams),
                             source="new (gee_with_originals of h1h2.py, family fixed effects)", scale="log-odds"))
    out = pd.concat([g, pd.DataFrame(rows)], ignore_index=True)
    return out, maxdiff


# ================================================================================ (3) both scales, per method
def delta_logit(Y: np.ndarray, signs: np.ndarray) -> tuple[float, float]:
    """sum_j signs_j * logit(mean(Y[:, j])) and its delta-method SE (item-level covariance of the columns)."""
    n = Y.shape[0]
    k = Y.sum(axis=0)
    p = np.where((k == 0) | (k == n), (k + 0.5) / (n + 1), k / n)
    est = float(np.sum(signs * logit(p)))
    S = np.cov(Y.T, ddof=1) / n
    gvec = signs / (p * (1 - p))
    return est, float(np.sqrt(gvec @ S @ gvec))


def analysis_scales(h: pd.DataFrame, gen_eff: pd.DataFrame, h2c: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    wide = h.pivot(index="id", columns="label", values="y")
    ge = gen_eff[(gen_eff.judge == "wg") & (gen_eff.subset == "all")].set_index(["pair", "method"])
    rows = []
    for pair in PAIRS:
        C = pair_cells(h, pair)
        oo, on = C[("original", 0)], C[("original", 1)]
        for m in DIR_METHODS + ["ara"]:
            if (m, 0) not in C:
                continue
            lo_, ln_ = C[(m, 0)], C[(m, 1)]
            Y2 = wide[[lo_, ln_]].dropna().values
            Yo = wide[[oo, on]].dropna().values
            Y4 = wide[[lo_, ln_, oo, on]].dropna().values
            raw_lo, raw_se = delta_logit(Y2, np.array([-1.0, 1.0]))
            org_lo, org_se = delta_logit(Yo, np.array([-1.0, 1.0]))
            adj_lo, adj_se = delta_logit(Y4, np.array([-1.0, 1.0, 1.0, -1.0]))
            e = ge.loc[(pair, m)]
            rows.append(dict(pair=pair, level="method", method=m, p_old=e.p_old, p_new=e.p_new,
                             p_orig_old=float(Yo[:, 0].mean()), p_orig_new=float(Yo[:, 1].mean()),
                             raw_pp=100 * e.delta, raw_lo_pp=100 * e.delta_lo, raw_hi_pp=100 * e.delta_hi,
                             orig_diff_pp=100 * e.delta_orig, adj_pp=100 * e.did_baseline_adj,
                             adj_lo_pp=100 * e.did_lo, adj_hi_pp=100 * e.did_hi,
                             raw_lo=raw_lo, raw_lo_lo=raw_lo - Z * raw_se, raw_lo_hi=raw_lo + Z * raw_se,
                             orig_diff_lo=org_lo, orig_diff_lo_lo=org_lo - Z * org_se,
                             orig_diff_lo_hi=org_lo + Z * org_se,
                             adj_lo=adj_lo, adj_lo_lo=adj_lo - Z * adj_se, adj_lo_hi=adj_lo + Z * adj_se,
                             pp_source="h1h2_generation_effects.csv (Newcombe CI raw; item-bootstrap CI adjusted)",
                             lo_source="marginal log-odds, delta method with item covariance"))
        # class-level rows (from part 2) and the H2 contrast
        E = eff[eff.pair == pair].set_index(["estimand", "scale"])
        for lev, raw, adj in (("direction class", "direction raw", "direction adjusted"),
                              ("ARA (class model)", "ARA raw", "ARA adjusted"),
                              ("H2 contrast", "H2 contrast (ARA - direction)", None)):
            r = dict(pair=pair, level=lev, method=lev)
            a, b = E.loc[(raw, "pp")], E.loc[(raw, "log-odds")]
            r.update(raw_pp=100 * a.estimate, raw_lo_pp=100 * a.lo_prompt, raw_hi_pp=100 * a.hi_prompt,
                     raw_lo=b.estimate, raw_lo_lo=b.lo_prompt, raw_lo_hi=b.hi_prompt)
            if adj:
                a, b = E.loc[(adj, "pp")], E.loc[(adj, "log-odds")]
                r.update(adj_pp=100 * a.estimate, adj_lo_pp=100 * a.lo_prompt, adj_hi_pp=100 * a.hi_prompt,
                         adj_lo=b.estimate, adj_lo_lo=b.lo_prompt, adj_lo_hi=b.hi_prompt)
            r.update(pp_source="paired item differences (Wald)", lo_source="per-pair GEE as h1h2.py")
            rows.append(r)
    s = pd.DataFrame(rows)

    def sgn(x):
        return np.sign(np.round(x, 12))

    def sig_dir(lo, hi):
        return np.where(lo > 0, 1, np.where(hi < 0, -1, 0))

    s["raw_sign_differs"] = sgn(s.raw_pp) != sgn(s.raw_lo)
    s["adj_sign_differs"] = s.adj_pp.notna() & (sgn(s.adj_pp) != sgn(s.adj_lo))
    s["raw_significance_differs"] = sig_dir(s.raw_lo_pp, s.raw_hi_pp) != sig_dir(s.raw_lo_lo, s.raw_lo_hi)
    s["adj_significance_differs"] = s.adj_pp.notna() & (sig_dir(s.adj_lo_pp, s.adj_hi_pp)
                                                        != sig_dir(s.adj_lo_lo, s.adj_lo_hi))
    # GEE H2 interaction of h1h2.py next to the RD contrast
    hc = h2c[(h2c.judge == "wg") & (h2c.subset == "all")].set_index("pair")
    s["h1h2_gee_interaction"] = [hc.gee_interaction.get(p) if lv == "H2 contrast" else np.nan
                                 for p, lv in zip(s.pair, s.level)]
    s["h1h2_rd_contrast_pp"] = [100 * hc.rd_contrast.get(p) if lv == "H2 contrast" else np.nan
                                for p, lv in zip(s.pair, s.level)]
    return s


# ====================================================================================== (5) ARA export drift
def export_gpu(run: str, trial: int) -> tuple[str, str, str]:
    for mach, root in RUN_ROOTS.items():
        for sub, fn in ((f"reexport_t{trial}", "reexport.log"), (f"publish_t{trial}", "publish.log")):
            f = root / run / sub / fn
            if f.is_file():
                txt = f.read_text(errors="replace").replace("\r", "\n")
                gpu = re.findall(r"\* CUDA 0: (NVIDIA [^\n(]+)", txt)
                restored = re.findall(r"Restoring model from trial (\d+)", txt)
                g = gpu[0].strip() if gpu else ""
                host = next((v for k, v in GPU_TO_MACHINE.items() if k in g), "")
                if restored and int(restored[0]) != trial:
                    host += f" (log restored trial {restored[0]})"
                return host, g, str(f.relative_to(ROOT))
    return "", "", ""


def analysis_export(dr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = dr.copy()
    eh = [export_gpu(r, int(t)) for r, t in zip(d.run, d.sel_trial)]
    d["export_machine"] = [x[0] for x in eh]
    d["export_gpu"] = [x[1] for x in eh]
    d["export_log"] = [x[2] for x in eh]
    d["export_on_other_machine_than_search"] = d.export_machine != d.search_machine
    d["is_ara"] = d.method == "ara"
    d["abs_d_refusals"] = d.d_refusals.abs()
    d["drift_gt2"] = d.abs_d_refusals > 2
    d["rescore_kl_gt_cap"] = d.rescore_kl > KL_MAX
    keep = ["run", "model_display", "study", "generation", "method", "role", "search_machine", "export_machine",
            "export_gpu", "export_log", "rescore_host", "export_on_other_machine_than_search", "sel_trial",
            "sel_refusals", "sel_kl", "rescore_refusals", "rescore_kl", "d_refusals", "d_kl", "abs_d_refusals",
            "drift_gt2", "rescore_kl_gt_cap", "sel_in_rescore_ci", "runnerup_trial", "runnerup_refusals",
            "runnerup_kl", "tf_divergence_vllm", "is_ara"]
    d = d[keep]
    trows = []
    main = d[d.study == "main"]
    for name, sub in (("main study, ARA", main[main.is_ara]), ("main study, other methods", main[~main.is_ara]),
                      ("all 46, ARA", d[d.is_ara]), ("all 46, other methods", d[~d.is_ara])):
        for other in (True, False):
            s = sub[sub.export_on_other_machine_than_search == other]
            if not len(s):
                continue
            trows.append(dict(group=name, export_on_other_machine=other, n=len(s), n_drift_gt2=int(s.drift_gt2.sum()),
                              median_abs_d_refusals=float(s.abs_d_refusals.median()),
                              max_abs_d_refusals=float(s.abs_d_refusals.max()),
                              mean_d_kl=float(s.d_kl.mean()), max_abs_d_kl=float(s.d_kl.abs().max()),
                              n_rescore_kl_gt_cap=int(s.rescore_kl_gt_cap.sum()),
                              exports=";".join(s.run)))
        t = pd.crosstab(sub.export_on_other_machine_than_search, sub.drift_gt2).reindex(
            index=[False, True], columns=[False, True], fill_value=0)
        orr, p = stats.fisher_exact(t.values)
        trows.append(dict(group=name, export_on_other_machine="Fisher exact: drift > 2 vs export on other machine",
                          n=len(sub), fisher_p=p, odds_ratio=orr,
                          table=f"same machine: {t.loc[False, True]}/{t.loc[False].sum()} drift>2; other machine: "
                                f"{t.loc[True, True]}/{t.loc[True].sum()}"))
        if len(sub) >= 4 and sub.export_on_other_machine_than_search.nunique() == 2:
            a = sub[sub.export_on_other_machine_than_search].abs_d_refusals
            b = sub[~sub.export_on_other_machine_than_search].abs_d_refusals
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            trows.append(dict(group=name, export_on_other_machine="Mann-Whitney: |d refusals| other vs same machine",
                              n=len(sub), mw_U=float(mw.statistic), mw_p=float(mw.pvalue)))
    return d, pd.DataFrame(trows)


# ============================================================================================== summary (md)
def rng(se: pd.Series, scale: float = 100.0) -> str:
    """'a-b' range of 95% half-widths (Z * SE) in pp; a single value when all are equal to one decimal."""
    lo, hi = scale * Z * float(se.min()), scale * Z * float(se.max())
    return f"{lo:.1f}" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}-{hi:.1f}"


def write_summary(runs, ksum, units, kh, het, eff, mt, lofo, lofo_diff, scales, sruns, sc, sigma, flags, exp, expt,
                  tests, t0):
    L = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    L.append("# Reviewer-response analyses: selection, pair-level inference, scales, seed noise, ARA exports "
             "(paper 33)")
    L.append("")
    L.append(f"Generated by `scripts/analysis/review_selection.py` at {now} ({(datetime.now() - t0).seconds} s). "
             "Answers concerns A1, A2, A3, B1 (data part), C1 and C5 of `notes/REVIEW_v5_reviewer_agent_20261106.md` "
             "from data on disk. Regenerated on every run; do not edit by hand. Outcome everywhere: WildGuard refusal "
             "on the 972 harmful held-out items, seed-1 cells (role original / primary) unless stated. pp = "
             "percentage points; intervals are 95%.")
    L.append("")

    # ------------------------------------------------------------------ (1)
    L.append("## 1. Did the KL cap bind? (A2)")
    L.append("")
    L.append("Definition: the cap **binds** in a run when some trial with dev KL > 0.1 had fewer dev keyword refusals "
             "than the selected trial (equivalently, the selected trial is not the run's minimum-refusal trial). "
             "A run **stayed below the cap** when at most one of its trials had KL > 0.1.")
    L.append("")
    pm = runs[(runs.role == "primary") & (runs.study == "main")]
    pdm, pam = pm[pm.method_class == "direction"], pm[pm.method_class == "optimisation"]
    le1 = pdm[pdm.left_cap_at_most_once]
    L.append(f"- Main study, seed 1, direction methods (24 runs): the cap bound in **{int(pdm.cap_binding.sum())}/24**; "
             f"it did not bind in {int((~pdm.cap_binding).sum())}. "
             f"{int(pdm.never_left_cap.sum())} runs never sampled a trial above 0.1 and "
             f"{int((pdm.n_kl_gt_cap == 1).sum())} sampled exactly one "
             f"({', '.join(f'{r.label} ({r.n_kl_gt_cap}, max KL {r.max_kl:.3f})' for r in le1.itertuples())}). "
             + ("This confirms the reviewer's count of 4 + 4."
                if (int(pdm.never_left_cap.sum()), int((pdm.n_kl_gt_cap == 1).sum())) == (4, 4)
                else "This differs from the reviewer's count of 4 + 4."))
    L.append(f"- Main study, ARA (8 runs): the cap bound in **{int(pam.cap_binding.sum())}/8**; every ARA run reached "
             f"0 dev refusals above the cap (maximum KL {pam.max_kl.min():.1f}-{pam.max_kl.max():.1f}). The median ARA "
             f"run gave up {pam.refusals_gained_above_cap.median():.0f} dev refusals to stay under the cap "
             f"(range {pam.refusals_gained_above_cap.min():.0f}-{pam.refusals_gained_above_cap.max():.0f}); direction "
             f"runs where it bound gave up a median "
             f"{pdm.loc[pdm.cap_binding, 'refusals_gained_above_cap'].median():.0f}.")
    for gen in ("old", "new"):
        s = pdm[pdm.generation == gen]
        L.append(f"  - direction, {gen}er models: bound in {int(s.cap_binding.sum())}/12, stayed below the cap in "
                 f"{int(s.left_cap_at_most_once.sum())}/12.")
    ps = runs[(runs.study == "small")]
    psd = ps[ps.method_class == "direction"]
    L.append(f"- Small pair (seed 1 and 2, 20 runs): direction runs bound in {int(psd.cap_binding.sum())}/16, "
             f"{int(psd.left_cap_at_most_once.sum())}/16 stayed below the cap (all four J-space runs never left it); "
             f"ARA bound in {int(ps[ps.method_class == 'optimisation'].cap_binding.sum())}/4.")
    if "sel_max_weight_at_ceiling" in pdm:
        ceil = pdm[pdm.sel_max_weight_at_ceiling.fillna(False).astype(bool)]
        le1c = le1[le1.sel_max_weight_at_ceiling.fillna(False).astype(bool)]
        L.append(f"- Parameter ranges (Optuna journals; 'at the ceiling' = attention or MLP maximum weight in the top "
                 f"5% of its range, i.e. >= 1.465 or >= 1.4125 of 1.5): the selected trial sits at the ceiling in "
                 f"{len(ceil)}/24 main direction runs and in **{len(le1c)}/{len(le1)}** of the runs that stayed below "
                 f"the cap ({', '.join(f'{r.label}' for r in le1c.itertuples()) or 'none'}). "
                 f"Selected trials with any strength-type parameter (maximum weight, minimum weight, minimum-weight "
                 f"distance, SOM decay) at its strong end: {int((pdm.sel_n_strength_params_at_strong_end > 0).sum())}/24 "
                 f"(stayed-below runs: {int((le1.sel_n_strength_params_at_strong_end > 0).sum())}/{len(le1)}). "
                 "Per-parameter positions: `review_sel_kl_params.csv`.")
        rest = pdm[~pdm.left_cap_at_most_once]
        L.append(f"  - Position of the selected maximum weights in their range (0 = lower bound, 1 = 1.5): runs that "
                 f"stayed below the cap, attention median {le1.sel_attn_max_weight_pos.median():.2f} "
                 f"({le1.sel_attn_max_weight_pos.min():.2f}-{le1.sel_attn_max_weight_pos.max():.2f}), MLP "
                 f"{le1.sel_mlp_max_weight_pos.median():.2f} ({le1.sel_mlp_max_weight_pos.min():.2f}-"
                 f"{le1.sel_mlp_max_weight_pos.max():.2f}); other direction runs attention "
                 f"{rest.sel_attn_max_weight_pos.median():.2f}, MLP {rest.sel_mlp_max_weight_pos.median():.2f}. "
                 "The selected edits lean to the strong half of the range but are not at its bound.")
        if len(le1):
            worse = le1[le1.min_refusals_max_weight_at_ceiling >= le1.sel_refusals]
            better = le1[le1.min_refusals_max_weight_at_ceiling < le1.sel_refusals]
            L.append(f"  - In the stayed-below runs the search did sample the strongest maximum weights; those trials "
                     f"had no fewer refusals than the selected one in {len(worse)}/{len(le1)} runs"
                     + (" (" + ", ".join(f"{r.label}: the better ceiling trial had KL {r.max_kl_max_weight_at_ceiling:.3f}"
                                         for r in better.itertuples()) + ")" if len(better) else "") + ": "
                     + "; ".join(f"{r.label}: {int(r.n_trials_max_weight_at_ceiling)} trials at the ceiling, best "
                                 f"{r.min_refusals_max_weight_at_ceiling:.0f} refusals (selected {r.sel_refusals}), "
                                 f"their max KL {r.max_kl_max_weight_at_ceiling:.3f}"
                                 for r in le1.itertuples()) + ".")
        st = pdm[pdm.sel_is_startup_trial]
        L.append(f"  - The selected trial is one of the 30 random TPE startup trials in {len(st)}/24 main direction "
                 f"runs ({int(le1.sel_is_startup_trial.sum())}/{len(le1)} of the stayed-below runs; "
                 f"{int(pam.sel_is_startup_trial.sum())}/8 ARA runs). With seed 1 and the same search space (same "
                 "number of layers) the startup draws are identical across models, so some selected parameter vectors "
                 "coincide: "
                 + "; ".join(sorted({" = ".join(sorted([r.label] + [runs.set_index('run').label[x] for x in
                                                                     r.sel_params_identical_to.split(';')]))
                                     for r in pdm.itertuples() if r.sel_params_identical_to})) + ".")
    L.append(f"- Selected dev KL: direction {pdm.sel_kl.min():.3f}-{pdm.sel_kl.max():.3f} (median "
             f"{pdm.sel_kl.median():.3f}), ARA {pam.sel_kl.min():.3f}-{pam.sel_kl.max():.3f}. "
             f"Within 0.01 of the cap: {int(pdm['sel_kl_within_0.01_of_cap'].sum())}/24 direction, "
             f"{int(pam['sel_kl_within_0.01_of_cap'].sum())}/8 ARA.")
    L.append("")
    um = units[units.study == "main"]
    L.append("Pair x method units (main study): "
             + "; ".join(f"{c}: {', '.join(f'{PAIR_DISPLAY[a]} {METHOD_DISPLAY[b]}' for a, b in zip(g.pair, g.method))}"
                         for c, g in um.groupby("unit_category")) + ".")
    L.append("")
    L.append("H1 and H2 re-fitted on unit subsets (pooled GEE of h1h2.py, family fixed effects; log-odds, prompt "
             "clusters; mean pp over units for orientation):")
    L.append("")
    tb = []
    for sname, g in kh.groupby("cell_subset", sort=False):
        def term(t):
            x = g[g.term == t]
            return f"{flo(x.estimate.iloc[0])} [{flo(x.lo.iloc[0])}, {flo(x.hi.iloc[0])}]" if len(x) else "n/a"
        r0 = g.iloc[0]
        tb.append({"subset": sname, "units (dir/ARA)": f"{r0.n_direction_units}/{r0.n_ara_units}",
                   "direction gen. effect": term("gen effect | direction class (log-odds)"),
                   "direction adjusted": term("baseline-adjusted gen effect | direction (log-odds)"),
                   "ARA gen. effect": term("gen effect | ARA (log-odds)"),
                   "H2 interaction": term("generation x class, H2 (log-odds)"),
                   "mean dir. effect, pp": fpp(r0.mean_gen_effect_direction_pp / 100),
                   "mean dir. adjusted, pp": fpp(r0.mean_baseline_adj_direction_pp / 100)})
    L.append(md_table(pd.DataFrame(tb)))
    L.append("")
    s1 = kh[kh.cell_subset.str.startswith("S1")]
    s2 = kh[kh.cell_subset.str.startswith("S2")]

    def get(df, t, c="estimate"):
        x = df[df.term == t]
        return float(x[c].iloc[0]) if len(x) else np.nan
    def comp(u):
        return ", ".join(f"{PAIR_DISPLAY[p]} {n}" for p, n in u.pair.value_counts().reindex(MAIN).dropna()
                         .astype(int).items())

    def ci(df, t):
        return (f"{flo(get(df, t))} [{flo(get(df, t, 'lo'))}, {flo(get(df, t, 'hi'))}]"
                if np.isfinite(get(df, t)) else "n/a")
    L.append(f"- The raw direction effect is positive in both groups: {ci(s1, 'gen effect | direction class (log-odds)')} "
             f"where the cap bound in both cells, {ci(s2, 'gen effect | direction class (log-odds)')} where it did not "
             f"bind in at least one. The baseline-adjusted direction effect changes sign: "
             f"{ci(s1, 'baseline-adjusted gen effect | direction (log-odds)')} vs "
             f"{ci(s2, 'baseline-adjusted gen effect | direction (log-odds)')}; the paper's +0.03 is an average of the "
             "two. The groups also differ in family (cap-bound direction units are mostly Gemma; the others are mostly "
             "OLMo and Mistral), so this split cannot separate the cap from the family.")
    L[-1] = L[-1].replace("(cap-bound direction units are mostly Gemma; the others are mostly OLMo and Mistral)",
                          "(cap-bound direction units: " + comp(um[(um.unit_category == "cap bound in both cells")
                                                                  & (um.method_class == "direction")])
                          + "; the others: " + comp(um[(um.unit_category != "cap bound in both cells")
                                                       & (um.method_class == "direction")]) + ")")
    L.append(f"- H2 can only be estimated where the cap bound, because it bound in every ARA cell: within the "
             f"cap-bound units the interaction is {ci(s1, 'generation x class, H2 (log-odds)')} (all units: "
             f"{ci(kh[kh.cell_subset.str.startswith('S0')], 'generation x class, H2 (log-odds)')}). "
             "H2 thus compares a class whose residual refusal is always set by the KL budget with a class whose "
             f"residual refusal is set by the budget in {int(pdm.cap_binding.sum())}/24 cells and by what the search "
             "space reached in the rest.")
    L.append("")

    # ------------------------------------------------------------------ (2)
    L.append("## 2. The pair as the unit of inference (A1, C5)")
    L.append("")
    hw = het[(het.judge == "wg") & (het.subset == "all")]
    L.append("(a) Heterogeneity across families (GEE Wald tests, `h1h2_models.csv`, WildGuard, 972 items): "
             + "; ".join(f"{r.scope}: {r.term} chi2 = {r.estimate:.1f} ({r.note.split(';')[0].replace('Wald chi2, ', '')}), "
                         f"p = {r.p:.1e}" for r in hw.itertuples())
             + f". The same tests have p <= {het.p.max():.1e} for every judge and item subset (chi2 "
               f"{het.estimate.min():.0f}-{het.estimate.max():.0f}; `review_sel_pair_heterogeneity.csv`).")
    L.append("")
    L.append("(b, c) Per-pair effects and their pooling over the four main pairs (k = 4). Prompt-only intervals are "
             "the paper's kind; 'with seed' adds run-to-run variance (sigma = 9.15 pp per cell on the pp scale, "
             f"{sigma[(sigma.seed_pairs == 'all 10 seed pairs') & (sigma.scale == 'logit')].sigma.iloc[0]:.2f} on the "
             "logit scale, both from the 10 small-pair seed pairs; a single-method effect gets 2 sigma^2, the "
             "direction-class mean of 3 methods 2 sigma^2/3).")
    L.append("")
    tb = []
    E = eff.set_index(["pair", "estimand", "scale"])
    for est in ["direction raw", "direction adjusted", "ARA raw", "ARA adjusted", "H2 contrast (ARA - direction)"]:
        for scale in ("pp", "log-odds"):
            f = (lambda x: fpp(x)) if scale == "pp" else (lambda x: flo(x))
            row = {"estimand": est, "scale": scale}
            for p in MAIN:
                e = E.loc[(p, est, scale)]
                row[PAIR_DISPLAY[p]] = f"{f(e.estimate)} [{f(e.lo_total)}, {f(e.hi_total)}]"
            e = E.loc[("qwen-small", est, scale)]
            row["Small pair"] = f"{f(e.estimate)} [{f(e.lo_total)}, {f(e.hi_total)}]"
            tb.append(row)
    L.append("Per-pair estimates with prompt + seed 95% intervals:")
    L.append("")
    L.append(md_table(pd.DataFrame(tb)))
    L.append("")
    tb = []
    for est in ["direction raw", "direction adjusted", "ARA raw", "ARA adjusted", "H2 contrast (ARA - direction)"]:
        for scale in ("pp", "log-odds"):
            f = (lambda x: fpp(x)) if scale == "pp" else (lambda x: flo(x))
            g = mt[(mt.estimand == est) & (mt.scale == scale)].set_index("variance")
            a, b = g.loc["prompt sampling only"], g.loc["prompt + seed (sigma, 10 seed pairs)"]
            t2 = (f"{100 * np.sqrt(a.tau2):.1f} pp" if scale == "pp" else f"{np.sqrt(a.tau2):.2f}")
            tb.append({"estimand": est, "scale": scale,
                       "fixed effect (prompt only)": f"{f(a.fe_estimate)} [{f(a.fe_lo)}, {f(a.fe_hi)}]",
                       "RE estimate": f(a.re_estimate), "DL 95% CI": f"[{f(a.dl_lo)}, {f(a.dl_hi)}]",
                       "HKSJ 95% CI": f"[{f(a.hksj_lo)}, {f(a.hksj_hi)}]",
                       "95% prediction": f"[{f(a.pi_lo)}, {f(a.pi_hi)}]", "tau": t2, "I^2": f"{100 * a.I2:.0f}%",
                       "HKSJ CI with seed": f"[{f(b.hksj_lo)}, {f(b.hksj_hi)}]",
                       "pairs +/-": f"{int(a.n_pairs_positive)}/{int(a.n_pairs_negative)}"})
    L.append("Pooled over the 4 main pairs (`review_sel_pair_meta.csv`; DL = DerSimonian-Laird, HKSJ = "
             "Hartung-Knapp-Sidik-Jonkman with t(3), prediction interval with t(2)):")
    L.append("")
    L.append(md_table(pd.DataFrame(tb)))
    L.append("")
    a = mt[(mt.estimand == "direction adjusted") & (mt.scale == "log-odds") & (mt.variance == "prompt sampling only")].iloc[0]
    b = mt[(mt.estimand == "direction raw") & (mt.scale == "log-odds") & (mt.variance == "prompt sampling only")].iloc[0]
    c = mt[(mt.estimand == "H2 contrast (ARA - direction)") & (mt.scale == "log-odds")
           & (mt.variance == "prompt sampling only")].iloc[0]
    s0 = kh[kh.cell_subset.str.startswith("S0")].set_index("term")
    g0 = s0.loc["baseline-adjusted gen effect | direction (log-odds)"]
    re_ok = mt[(mt.variance == "prompt sampling only")]
    n_excl0 = int(((re_ok.hksj_lo > 0) | (re_ok.hksj_hi < 0)).sum())
    L.append(f"- The paper's baseline-adjusted direction effect ({flo(g0.estimate)} [{flo(g0.lo)}, {flo(g0.hi)}], "
             f"pooled GEE) becomes {flo(a.re_estimate)} with HKSJ interval [{flo(a.hksj_lo)}, {flo(a.hksj_hi)}] and "
             f"prediction interval [{flo(a.pi_lo)}, {flo(a.pi_hi)}] when the pair is the unit (I^2 = "
             f"{100 * a.I2:.0f}%). The raw direction effect: {flo(b.re_estimate)} [{flo(b.hksj_lo)}, {flo(b.hksj_hi)}]; "
             f"H2 interaction {flo(c.re_estimate)} [{flo(c.hksj_lo)}, {flo(c.hksj_hi)}]. HKSJ intervals that exclude "
             f"zero: {n_excl0} of {len(re_ok)} pooled estimands x scales (prompt-only variance).")
    da_ = mt[(mt.estimand == "direction adjusted") & (mt.scale == "pp")].set_index("variance")
    e_da = eff[(eff.estimand == "direction adjusted") & (eff.scale == "pp") & eff.pair.isin(MAIN)]
    e_ar = eff[(eff.estimand == "ARA raw") & (eff.scale == "pp") & eff.pair.isin(MAIN)]
    L.append(f"- Seed variance: for the adjusted direction effect (pp) the between-pair SD tau is "
             f"{100 * da_.loc['prompt sampling only'].tau:.1f} pp against a seed SE of "
             f"{100 * e_da.se_seed.iloc[0]:.1f} pp and a prompt SE of {100 * e_da.se_prompt.min():.1f}-"
             f"{100 * e_da.se_prompt.max():.1f} pp per pair, so adding seed variance mainly lowers the DL tau^2 "
             f"({100 * da_.loc['prompt sampling only'].tau:.1f} -> "
             f"{100 * da_.loc['prompt + seed (sigma, 10 seed pairs)'].tau:.1f} pp) and leaves the pooled HKSJ "
             f"interval almost unchanged. Per pair it matters: half-widths grow from "
             f"{rng(e_da.se_prompt)} pp to {rng(e_da.se_total)} pp (direction class, adjusted) and from "
             f"{rng(e_ar.se_prompt)} pp to {rng(e_ar.se_total)} pp (ARA, one run per cell). Several per-pair "
             "effects that are 'significant' with prompt-only intervals are not once seed variance is added: of the "
             f"{int(eff.pair.isin(MAIN).sum())} main-pair estimand x scale cells, "
             f"{int(((eff.p_prompt < ALPHA) & eff.pair.isin(MAIN)).sum())} exclude zero with prompt-only intervals and "
             f"{int(((eff.p_total < ALPHA) & eff.pair.isin(MAIN)).sum())} with seed variance added "
             "(`review_sel_pair_effects.csv`, p_prompt / p_total).")
    dm_ = mt[mt.variance.str.contains("delta method")]
    L.append(f"- Sensitivity: mapping the pp sigma to the logit scale by the delta method at each cell's rate (instead of "
             f"a logit sigma) gives seed SEs up to "
             f"{eff[eff.scale == 'log-odds'].se_seed_delta_method.max():.1f} log-odds for cells near 0% refusal "
             f"(Gemma-4 ARA, 0.6%); pooled HKSJ intervals under that variant: "
             + "; ".join(f"{r.estimand} [{flo(r.hksj_lo)}, {flo(r.hksj_hi)}]" for r in dm_.itertuples()) + ".")
    L.append("")
    lf = lofo
    L.append("(d) Leave one family out (pooled GEE, log-odds; `review_sel_pair_lofo.csv`; final.analysis_g reproduces "
             f"`final_pooled_robustness.csv`, max |difference| {lofo_diff:.1e}):")
    L.append("")
    tb = []
    for term in ["gen effect | direction class", "gen effect | optimisation class", "generation x class (H2)",
                 "baseline-adjusted gen effect | direction class", "baseline-adjusted gen effect | ARA",
                 "originals' generation difference"]:
        g = lf[lf.term == term]
        allr = g[g.cell_set.str.startswith("all main")]
        lo_ = g[g.cell_set.str.startswith("leave out")]
        tb.append({"term": term.replace(" | ", ", "), "all four": f"{flo(allr.estimate.iloc[0])}",
                   **{f"without {f_}": flo(lo_[lo_.cell_set == f'leave out {f_}'].estimate.iloc[0]) for f_ in MAIN},
                   "range": f"{flo(lo_.estimate.min())} to {flo(lo_.estimate.max())}"})
    L.append(md_table(pd.DataFrame(tb)))
    L.append("")
    # C5: conjunction probability
    sdd = np.sqrt(2) * sigma[(sigma.seed_pairs == "all 10 seed pairs") & (sigma.scale == "pp")].sigma.iloc[0]
    L.append(f"C5, conjunctive H1: if every method in a pair had the same true generation effect delta, each observed "
             f"per-method effect has run-to-run SD {sdd:.1f} pp (SD_diff), so P(all four methods positive) = "
             f"Phi(delta/SD_diff)^4 = {stats.norm.cdf(5 / sdd) ** 4:.2f} for delta = +5 pp, "
             f"{stats.norm.cdf(10 / sdd) ** 4:.2f} for +10 pp, {stats.norm.cdf(20 / sdd) ** 4:.2f} for +20 pp "
             "(prompt noise ignored). The 'every method' wording of H1 fails often even when H1 is true in the mean.")
    t1 = tests[(tests.judge == "wg") & (tests.family == "H1") & tests.pair.isin(MAIN)]
    ta = t1[t1.subset == "all"].set_index("test_id")
    tx = t1[t1.subset == "excl_advbench"].set_index("test_id")
    lost_all = [i for i in ta.index if ta.loc[i, "reject_holm"] and not tx.loc[i, "reject_holm"]]
    neg = [i for i in ta.index if ta.loc[i, "reject_holm"] and ta.loc[i, "estimate"] < 0]
    lost = [i for i in lost_all if i in neg]
    L.append(f"- Holm-significant decreases (WildGuard, 972 items): {len(neg)} ({', '.join(neg)}). Losing Holm "
             f"significance without the 44 AdvBench-overlap items: "
             + ", ".join(f"{i} (p_Holm {ta.loc[i, 'p_holm']:.3f} -> {tx.loc[i, 'p_holm']:.3f})" for i in lost_all)
             + f". H1 Holm family size {int(ta.n_in_family.iloc[0])} per judge x subset; judges and subsets are "
               "not corrected across each other.")
    L.append("")

    # ------------------------------------------------------------------ (3)
    L.append("## 3. Baseline adjustment on both scales (C1)")
    L.append("")
    sm_ = scales[scales.level == "method"]
    tb = []
    for r in sm_.itertuples():
        tb.append({"pair": PAIR_DISPLAY[r.pair], "method": METHOD_DISPLAY[r.method],
                   "effect, pp": f"{fpp(r.raw_pp / 100)} [{fpp(r.raw_lo_pp / 100)}, {fpp(r.raw_hi_pp / 100)}]",
                   "originals, pp": fpp(r.orig_diff_pp / 100),
                   "adjusted, pp": f"{fpp(r.adj_pp / 100)} [{fpp(r.adj_lo_pp / 100)}, {fpp(r.adj_hi_pp / 100)}]",
                   "effect, log-odds": f"{flo(r.raw_lo)} [{flo(r.raw_lo_lo)}, {flo(r.raw_lo_hi)}]",
                   "originals, log-odds": flo(r.orig_diff_lo),
                   "adjusted, log-odds": f"{flo(r.adj_lo)} [{flo(r.adj_lo_lo)}, {flo(r.adj_lo_hi)}]",
                   "sign differs (adj.)": "YES" if r.adj_sign_differs else "",
                   "significance differs (adj.)": "YES" if r.adj_significance_differs else ""})
    L.append(md_table(pd.DataFrame(tb)))
    L.append("")
    nd = int(sm_.adj_sign_differs.sum())
    nsd = int(sm_.adj_significance_differs.sum())
    L.append(f"- Raw generation effects of a single method always have the same sign on both scales (a monotone "
             f"transform of two rates); {int(sm_.raw_significance_differs.sum())} of {len(sm_)} differ in "
             f"significance. For the baseline-adjusted effects the sign differs between scales in **{nd} of {len(sm_)}** "
             f"pair x method cells ({', '.join(f'{PAIR_DISPLAY[r.pair]} {METHOD_DISPLAY[r.method]}' for r in sm_[sm_.adj_sign_differs].itertuples()) or 'none'}) "
             f"and the significance verdict differs in {nsd} "
             f"({', '.join(f'{PAIR_DISPLAY[r.pair]} {METHOD_DISPLAY[r.method]}' for r in sm_[sm_.adj_significance_differs & ~sm_.adj_sign_differs].itertuples()) or 'none'} besides the sign changes).")
    cl = scales[scales.level != "method"]
    cld = cl[cl.raw_sign_differs | cl.adj_sign_differs]
    L.append("- Class level and H2 contrast (pp from paired item differences, log-odds from per-pair GEE), cells where "
             "the two scales disagree in sign: "
             + ("; ".join(f"{PAIR_DISPLAY[r.pair]} {r.level}"
                          + (f" {'adjusted' if r.adj_sign_differs else 'effect'} "
                             f"{fpp((r.adj_pp if r.adj_sign_differs else r.raw_pp) / 100)} pp vs "
                             f"{flo(r.adj_lo if r.adj_sign_differs else r.raw_lo)} log-odds")
                          for r in cld.itertuples()) if len(cld) else "none") + ".")
    mi = sm_[sm_.pair == "mistral"]
    ga = sm_[(sm_.pair == "gemma") & (sm_.method == "ara")].iloc[0]
    big = sm_[sm_.pair.isin(MAIN)].assign(a=lambda x: x.raw_lo.abs()).sort_values("a", ascending=False).iloc[0]
    L.append(f"- Mistral: adjusted effects {fpp(mi.adj_pp.min() / 100)} to {fpp(mi.adj_pp.max() / 100)} pp "
             f"(log-odds {flo(mi.adj_lo.min())} to {flo(mi.adj_lo.max())}). Gemma ARA: raw {fpp(ga.raw_pp / 100)} pp "
             f"but {flo(ga.raw_lo)} log-odds ({100 * ga.p_old:.1f}% -> {100 * ga.p_new:.1f}%). Largest |log-odds| "
             f"effect in the main study: {PAIR_DISPLAY[big.pair]} {METHOD_DISPLAY[big.method]} {flo(big.raw_lo)} "
             f"({fpp(big.raw_pp / 100)} pp).")
    L.append("")

    # ------------------------------------------------------------------ (4)
    L.append("## 4. Seed-noise band without the batch-64 runs (B1, data part)")
    L.append("")
    b64 = sruns[(sruns.chosen_batch_size == 64)]
    L.append(f"- `output.log` of the 20 small-pair runs: batch size 64 (after a CUDA out-of-memory at 128) in "
             f"{len(b64)} runs: {', '.join(b64.run)}; all other small-pair runs used 128. Every seed-1 run used 128, "
             "so the three seed pairs Qwen3.5-4B DiM, SOM and ARA differ in batch size as well as seed. "
             "Confirmed.")
    sa = sigma[(sigma.seed_pairs == "all 10 seed pairs") & (sigma.scale == "pp")].iloc[0]
    sb = sigma[(sigma.seed_pairs != "all 10 seed pairs") & (sigma.scale == "pp")].iloc[0]
    L.append(f"- |seed 2 - seed 1| of those three: {', '.join(f'{abs(x):.1f}' for x in sc[sc.batch_changed].d_pp)} pp; "
             f"of the other seven: median {np.median(np.abs(sc[~sc.batch_changed].d_pp)):.1f} pp, max "
             f"{np.abs(sc[~sc.batch_changed].d_pp).max():.1f} pp.")
    L.append(f"- sigma: all 10 pairs {sa.sigma:.2f} pp [{sa.sigma_lo:.2f}, {sa.sigma_hi:.2f}], threshold 2 x SD_diff = "
             f"{sa.threshold:.1f} pp; without the batch-64 pairs (n = 7) **{sb.sigma:.2f} pp [{sb.sigma_lo:.2f}, "
             f"{sb.sigma_hi:.2f}], threshold {sb.threshold:.1f} pp** [{sb.threshold_lo:.1f}, {sb.threshold_hi:.1f}].")
    fa, fb = flags[flags.exceeds_all10], flags[flags.exceeds_excl_b64]
    same = set(zip(fa.pair, fa.method)) == set(zip(fb.pair, fb.method))
    L.append(f"- Main-study effects beyond the band: {len(fa)}/16 with all pairs, {len(fb)}/16 without the batch-64 "
             f"pairs; the set is {'the same' if same else 'different'} "
             f"({', '.join(f'{PAIR_DISPLAY[r.pair]} {METHOD_DISPLAY[r.method]} {r.effect_pp:+.1f}' for r in fb.itertuples())}). "
             f"Baseline-adjusted: {int(flags.adj_exceeds_all10.sum())}/16 vs {int(flags.adj_exceeds_excl_b64.sum())}/16"
             + (": " + ", ".join(f"{PAIR_DISPLAY[r.pair]} {METHOD_DISPLAY[r.method]} {r.baseline_adj_pp:+.1f}"
                                 for r in flags[flags.adj_exceeds_excl_b64 & ~flags.adj_exceeds_all10].itertuples())
                + " join" if (flags.adj_exceeds_excl_b64 & ~flags.adj_exceeds_all10).any() else "")
             + ". Closest calls without the batch-64 pairs: "
             + ", ".join(f"{PAIR_DISPLAY[r.pair]} {METHOD_DISPLAY[r.method]} {r.effect_pp:+.1f}"
                         for r in flags.assign(gap=(flags.effect_pp.abs() - sb.threshold).abs())
                         .sort_values("gap").head(3).itertuples()) + ".")
    sl = sigma[(sigma.scale == "logit")]
    L.append(f"- On the logit scale sigma is {sl.iloc[0].sigma:.2f} (10 pairs) and {sl.iloc[1].sigma:.2f} (7 pairs); "
             "these feed the log-odds seed intervals of part 2.")
    L.append("")

    # ------------------------------------------------------------------ (5)
    L.append("## 5. ARA export drift (A3, data part)")
    L.append("")
    ea = exp[exp.is_ara]
    tb = []
    for r in ea.itertuples():
        tb.append({"export": r.model_display, "search": r.search_machine, "export built on": r.export_machine,
                   "re-scored on": r.rescore_host, "dev refusals sel -> re-score": f"{r.sel_refusals} -> "
                   f"{r.rescore_refusals} ({r.d_refusals:+d})",
                   "dev KL sel -> re-score": f"{r.sel_kl:.4f} -> {r.rescore_kl:.4f}",
                   "KL > 0.1": "YES" if r.rescore_kl_gt_cap else ""})
    L.append(md_table(pd.DataFrame(tb)))
    L.append("")
    mo = exp[(exp.study == "main")]
    L.append(f"- Search ran on the H200 for every main-study cell. Exports (L-BFGS refit for ARA, `reexport.log`: "
             f"'Obtaining module I/O ... Modifying model using ARA') were rebuilt on the RTX PRO 6000 for all Qwen and "
             f"Gemma cells ({int(mo.export_on_other_machine_than_search.sum())}/36 main-study exports, "
             f"{int(mo[mo.is_ara].export_on_other_machine_than_search.sum())}/8 ARA) and on the H200 for OLMo and "
             "Mistral. Small-pair search, export and re-score all ran on the RTX 4090.")
    ft = expt[expt.export_on_other_machine.astype(str).str.startswith("Fisher")].set_index("group")
    sub = expt[expt.export_on_other_machine.isin([True, False])].set_index(["group", "export_on_other_machine"])
    L.append(f"- ARA, main study: |drift| > 2 dev refusals in "
             f"{int(sub.loc[('main study, ARA', True)].n_drift_gt2)}/{int(sub.loc[('main study, ARA', True)].n)} exports "
             f"rebuilt on the other machine vs {int(sub.loc[('main study, ARA', False)].n_drift_gt2)}/"
             f"{int(sub.loc[('main study, ARA', False)].n)} rebuilt on the search machine (Fisher p = "
             f"{ft.loc['main study, ARA'].fisher_p:.3f}); max |d KL| {sub.loc[('main study, ARA', True)].max_abs_d_kl:.4f} "
             f"vs {sub.loc[('main study, ARA', False)].max_abs_d_kl:.4f}. Other methods, main study: "
             f"{int(sub.loc[('main study, other methods', True)].n_drift_gt2)}/"
             f"{int(sub.loc[('main study, other methods', True)].n)} vs "
             f"{int(sub.loc[('main study, other methods', False)].n_drift_gt2)}/"
             f"{int(sub.loc[('main study, other methods', False)].n)} (p = "
             f"{ft.loc['main study, other methods'].fisher_p:.3f}). All main-study re-scores ran on the RTX PRO 6000, "
             "so the OLMo/Mistral ARA exports (built on the H200, re-scored on the RTX) show that re-scoring on other "
             "hardware alone moves dev refusals by at most 2; the large ARA drifts coincide with refitting on other "
             "hardware.")
    L.append(f"- ARA exports re-scoring above KL 0.1: **{int(ea.rescore_kl_gt_cap.sum())} of {len(ea)}** "
             f"({', '.join(ea[ea.rescore_kl_gt_cap].model_display)}, {', '.join(f'{x:.4f}' for x in ea[ea.rescore_kl_gt_cap].rescore_kl)}); "
             f"all exports: {int(exp.rescore_kl_gt_cap.sum())} of {len(exp)} "
             f"({', '.join(f'{a} {b}' for a, b in zip(exp[exp.rescore_kl_gt_cap].model_display, exp[exp.rescore_kl_gt_cap].method))}). "
             "Held-out results were measured on the released (refitted) exports, so they describe those models; what "
             "drifts is the link between the searched trial and the released model.")
    L.append("")

    # ------------------------------------------------------------------ verdicts
    L.append("## Verdicts per reviewer concern")
    L.append("")
    L.append("| concern | verdict | basis |")
    L.append("|---|---|---|")
    h0 = hw[hw.term.str.startswith("generation x family")].iloc[0]
    lh = lf[(lf.term == "generation x class (H2)") & lf.cell_set.str.startswith("leave out")]
    la = lf[(lf.term == "baseline-adjusted gen effect | direction class") & lf.cell_set.str.startswith("leave out")]
    L.append(f"| A1 pair is the unit; precise null is a cancellation | **supported** | Wald chi2 {h0.estimate:.0f} "
             f"(df 3) for generation x family; per-pair adjusted direction effects range "
             f"{flo(a.min_pair)} to {flo(a.max_pair)} log-odds; RE pooling over 4 pairs gives HKSJ "
             f"[{flo(a.hksj_lo)}, {flo(a.hksj_hi)}] and a prediction interval [{flo(a.pi_lo)}, {flo(a.pi_hi)}] "
             f"instead of [{flo(g0.lo)}, {flo(g0.hi)}]; leaving one family out moves H2 from {flo(lh.estimate.min())} "
             f"to {flo(lh.estimate.max())} and the adjusted direction effect from {flo(la.estimate.min())} to "
             f"{flo(la.estimate.max())}. A pooled estimate near 0 is a summary of large opposite per-pair effects, "
             "not a precise null. |")
    nb = int(pdm.cap_binding.sum())
    L.append(f"| A2 matched shift was not matched | **partly supported** | Supported: {int(pdm.never_left_cap.sum())} "
             f"main direction runs never and {int((pdm.n_kl_gt_cap == 1).sum())} only once went above KL 0.1; the cap "
             f"bound in {int(pam.cap_binding.sum())}/8 ARA runs but only {nb}/24 direction runs; the baseline-adjusted "
             f"direction effect is {flo(get(s1, 'baseline-adjusted gen effect | direction (log-odds)'))} where the cap "
             f"bound and {flo(get(s2, 'baseline-adjusted gen effect | direction (log-odds)'))} where it did not, and H2 "
             f"is only estimable where it bound ({flo(get(s1, 'generation x class, H2 (log-odds)'))}). Not supported in "
             f"its specific form: residual refusal in the stayed-below cells is the best the default search space "
             f"reached, but not at its maximum-weight bound -- the selected maximum weights are at the 1.5 bound in "
             f"{len(le1c)}/{len(le1)} of them, and trials at the bound did not do better within the cap. "
             "'Matched distribution shift' should read 'a common KL cap that bound in some cells'. |")
    xo = ea[(ea.study == "main") & ea.export_on_other_machine_than_search]
    xs = ea[(ea.study == "main") & ~ea.export_on_other_machine_than_search]
    L.append(f"| A3 ARA instantiation / export refit | **supported (data part)** | "
             f"{', '.join(xo.model_display)} ARA exports were refitted on another GPU than the search and drift "
             f"{', '.join(f'{x:+d}' for x in xo.d_refusals)} dev refusals; "
             f"{', '.join(xs.model_display)} ARA exports refitted on the search GPU drift "
             f"{', '.join(f'{x:+d}' for x in xs.d_refusals)}. "
             f"{', '.join(f'{a} at KL {b:.4f}' for a, b in zip(ea[ea.rescore_kl_gt_cap].model_display, ea[ea.rescore_kl_gt_cap].rescore_kl))} "
             f"is the only ARA export above the cap on re-score. The heretic-org 0/100 discrepancy needs GPU work. |")
    L.append(f"| B1 batch-size confound (data part) | **supported; raw-effect flags robust** | {len(b64)} seed-2 runs used "
             f"batch 64 (OOM at 128). sigma {sa.sigma:.1f} -> {sb.sigma:.1f} pp, threshold {sa.threshold:.1f} -> "
             f"{sb.threshold:.1f} pp; flagged raw effects {'unchanged' if same else 'change'} ({len(fa)} -> {len(fb)} "
             f"of 16); flagged baseline-adjusted effects {int(flags.adj_exceeds_all10.sum())} -> "
             f"{int(flags.adj_exceeds_excl_b64.sum())} of 16. The heuristic band itself (one sigma for all levels, "
             "n = 7-10) remains the weaker point. |")
    sh = cl[(cl.pair == "qwen-small") & (cl.level == "H2 contrast")].iloc[0]
    L.append(f"| C1 scale dependence of the adjustment | **supported** | Adjusted effects change sign between pp and "
             f"log-odds in {nd}/{len(sm_)} pair x method cells (and {len(cld)} class-level cells); the small-pair H2 "
             f"contrast is {fpp(sh.raw_pp / 100)} pp vs {flo(sh.raw_lo)} log-odds. Report both scales. Whether the "
             "adjustment was pre-specified cannot be checked from data. |")
    L.append(f"| C5 conjunctive H1, fragile significance | **supported** | Under seed noise alone 'positive for all 4 "
             f"methods' has probability {stats.norm.cdf(10 / sdd) ** 4:.2f} at a true +10 pp effect; {len(lost)} of "
             f"{len(neg)} Holm-significant decreases lose significance without the overlap items. Per-pair class "
             "effects with seed-inclusive intervals (part 2) are the estimation-style replacement. |")
    L.append("")
    L.append("## Files")
    L.append("")
    for f in sorted(OUT.glob("review_sel_*")):
        L.append(f"- `results/analysis/{f.name}`")
    L.append("")
    L.append("Caveats: sigma comes from 10 (or 7) small-model cells and is assumed constant across models, methods and "
             "refusal levels; the HKSJ interval with k = 4 uses t(3) and the prediction interval t(2), so both are wide "
             "by construction (pp prediction intervals run past +/-100 pp, i.e. a normal approximation on a bounded "
             "scale with k = 4 says nothing beyond 'any sign is plausible for a new pair'); the KL-cap analysis uses logged dev values (keyword refusals of 100, 100 tokens), not "
             "held-out rates; parameter positions refer to Heretic's default ranges (Table 8).")
    (OUT / "review_sel_summary.md").write_text("\n".join(L) + "\n")


# ===================================================================================================== main
def main():
    t0 = datetime.now()
    trials = pd.read_csv(ROOT / "paper/figures/data/trials.csv")
    gen_eff = pd.read_csv(OUT / "h1h2_generation_effects.csv")
    h2c = pd.read_csv(OUT / "h1h2_h2_contrasts.csv")
    models = pd.read_csv(OUT / "h1h2_models.csv")
    tests = pd.read_csv(OUT / "h1h2_tests.csv")
    seed_cells = pd.read_csv(OUT / "final_seed_cells.csv")
    sig_csv = pd.read_csv(OUT / "seednoise_sigma.csv")
    dr = pd.read_csv(OUT / "dev_rescore_cells.csv")
    rj, cr, ds, _, results, meta_df = final.load()
    h = harmful_main(rj)

    log("(1) KL-cap binding")
    runs, prm, ksum = analysis_kl(trials)
    assert int(runs.journal_rows_mismatching_trials_csv.sum()) == 0, "journal values differ from trials.csv"
    units = kl_units(runs, gen_eff)
    kh = analysis_kl_h1h2(h, units)
    s0 = kh[kh.cell_subset.str.startswith("S0")].set_index("term")
    ref = models[(models.judge == "wg") & (models.subset == "all") & (models.scope == "main")]
    chk = {("gen effect | direction class (log-odds)", "gee_planned", "gen effect | direction (= gen_new)"),
           ("generation x class, H2 (log-odds)", "gee_planned", "gen_new:opt"),
           ("baseline-adjusted gen effect | direction (log-odds)", "gee_with_originals",
            "baseline-adjusted gen effect | direction")}
    for mine, mdl, theirs in chk:
        a = float(s0.loc[mine, "estimate"])
        b = float(ref[(ref.model == mdl) & (ref.term == theirs)].estimate.iloc[0])
        assert abs(a - b) < 1e-4, (mine, a, b)
    log("  S0 reproduces h1h2_models.csv (direction, H2, adjusted direction)")

    log("(4) seed noise / batch size")
    sruns, sc, sigma, flags, sig = analysis_seed(trials, seed_cells, gen_eff, sig_csv)

    log("(2) pair as the unit")
    het = models[models.model == "gee_heterogeneity"][["judge", "subset", "scope", "term", "estimate", "p", "note",
                                                       "n_obs", "n_prompts", "pairs_included"]]
    het = het.rename(columns={"estimate": "wald_chi2"}).assign(estimate=lambda x: x.wald_chi2)
    eff = analysis_pair_effects(h, sig)
    # check: per-pair GEE equals h1h2_models.csv pair scopes; pp H2 contrast equals h1h2_h2_contrasts.csv
    for p in PAIRS:
        a = eff[(eff.pair == p) & (eff.estimand == "H2 contrast (ARA - direction)") & (eff.scale == "log-odds")]
        b = models[(models.judge == "wg") & (models.subset == "all") & (models.scope == f"pair:{p}")
                   & (models.model == "gee_planned") & (models.term == "gen_new:opt")]
        assert abs(float(a.estimate.iloc[0]) - float(b.estimate.iloc[0])) < 1e-4, p
        a = eff[(eff.pair == p) & (eff.estimand == "H2 contrast (ARA - direction)") & (eff.scale == "pp")]
        b = h2c[(h2c.judge == "wg") & (h2c.subset == "all") & (h2c.pair == p)].rd_contrast
        assert abs(float(a.estimate.iloc[0]) - float(b.iloc[0])) < 2e-3, (p, a.estimate.iloc[0], b.iloc[0])
    log("  per-pair GEE and RD contrasts reproduce h1h2_models.csv / h1h2_h2_contrasts.csv")
    mt = analysis_meta(eff)
    lofo, lofo_diff = analysis_lofo(rj, meta_df, h)
    assert lofo_diff < 1e-4, lofo_diff

    log("(3) both scales")
    scales = analysis_scales(h, gen_eff, h2c, eff)

    log("(5) export drift")
    exp, expt = analysis_export(dr)

    # ---- write
    runs.to_csv(OUT / "review_sel_kl_runs.csv", index=False)
    prm.to_csv(OUT / "review_sel_kl_params.csv", index=False)
    ksum.to_csv(OUT / "review_sel_kl_summary.csv", index=False)
    units.to_csv(OUT / "review_sel_kl_units.csv", index=False)
    kh.to_csv(OUT / "review_sel_kl_h1h2.csv", index=False)
    het.drop(columns=["estimate"]).to_csv(OUT / "review_sel_pair_heterogeneity.csv", index=False)
    eff.to_csv(OUT / "review_sel_pair_effects.csv", index=False)
    mt.to_csv(OUT / "review_sel_pair_meta.csv", index=False)
    lofo.to_csv(OUT / "review_sel_pair_lofo.csv", index=False)
    scales.to_csv(OUT / "review_sel_scales.csv", index=False)
    sruns.to_csv(OUT / "review_sel_seed_runs.csv", index=False)
    sigma.to_csv(OUT / "review_sel_seed_sigma.csv", index=False)
    flags.to_csv(OUT / "review_sel_seed_flags.csv", index=False)
    exp.to_csv(OUT / "review_sel_export_drift.csv", index=False)
    expt.to_csv(OUT / "review_sel_export_tests.csv", index=False)
    write_summary(runs, ksum, units, kh, het, eff, mt, lofo, lofo_diff, scales, sruns, sc, sigma, flags, exp, expt,
                  tests, t0)
    log(f"wrote results/analysis/review_sel_*.csv and review_sel_summary.md ({(datetime.now() - t0).seconds} s)")


if __name__ == "__main__":
    main()
