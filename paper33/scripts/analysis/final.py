#!/usr/bin/env python3
"""Paper 33 -- final additional analyses after judge batch 2 (step 6 of run_all.sh; no GPU, read-only inputs).

    scripts/figures/.venv/bin/python scripts/analysis/final.py

Runs after assemble.py, h1h2.py, h3.py and secondary.py. Every number goes to a CSV in results/analysis/ (prefix
`final_`) and the console report results/analysis/final_log.txt; results/analysis/FINAL_NOTES.md (hand-written) cites
these files. Nothing here changes H1/H2/H3 (those are h1h2.py / h3.py); these are the control, replicate, external and
selection checks of notes/atlas_sweep.md D2-D6.

(cap) final_capability.csv            capability (5 benchmarks) and teacher-forced divergence of EVERY judged cell, read
                                      from the evaluation dirs' summary.json (results.csv has seed-1 cells only; seed-2
                                      replicates are read here). Cross-checked against paper/figures/data/results.csv.
(a) final_gmpoa_control.csv           gamma-MPOA vs MPOA (and vs the original) per model: held-out refusal (WildGuard,
                                      keyword, Qwen3Guard), over-refusal, degenerate/capped share, capability, dev
                                      KL/refusals, teacher-forced divergence; exact McNemar gamma-MPOA vs MPOA, Holm (4).
    final_gmpoa_benchmarks.csv        per model x benchmark: gamma-MPOA vs original and vs MPOA, paired exact McNemar on
                                      the lm-eval per-sample scores (when they reproduce the summary score), Holm.
    final_gmpoa_generation.csv        generation effect (new - old) of gamma-MPOA vs that of MPOA, per family.
    final_kl_vs_capability.csv        does first-token KL <= 0.1 bound capability loss? Spearman of dev KL and
                                      teacher-forced divergence with the capability change over all modified cells.
(b) final_community_vs_selected.csv   each community checkpoint vs each of our selected (seed-1) cells of the same
                                      model: refusal, McNemar, Holm (per community model), capability, divergence.
(c) final_seed_cells.csv              seed 2 vs seed 1 per small-pair cell: held-out refusal, paired difference,
                                      McNemar, Holm (10), Cohen's kappa between seeds, dev point, capability.
    final_seed_generation.csv         generation effect (Qwen3.5-4B - Qwen2.5-7B) per method under seed 1, seed 2, all
                                      four seed combinations and the seed average; H2 risk-difference contrast per seed.
    final_seed_method_vs_seed.csv     per model: method spread vs seed spread (one-way ANOVA of cell rates on method,
                                      seeds as replicates; eta^2, variance-component share, MS ratio) and pairwise
                                      method orderings that hold under both seeds.
(d) final_dev_vs_heldout.csv          Spearman of dev keyword refusals (selected trial) with held-out refusal over every
                                      cell with a dev point (primary, gamma-MPOA, seed-2, originals), per cell set.
    final_dev_vs_heldout_within_model.csv  per model x seed: within-model Spearman over methods; best dev = best held-out?
                                      (dev ties reported; broken by lower dev KL as in scripts/select_trial.py)
(e) final_search_runs.csv             per optimisation run: best-so-far feasible keyword count at trials 30/50/75/100,
                                      trial that first reached the selected count, late improvement (after trial 75).
    final_search_summary.csv          late-improving runs by study x generation x method; Fisher / Wilcoxon tests.
(f) final_selection_bias_pairs.csv    matched old/new cells (pair x method, seed 1): selected KL, excess KL spent on the
                                      last keyword refusals, teacher-forced divergence, capability change, and the
                                      held-out keyword-only refusal share (keyword yes / WildGuard no among WildGuard
                                      compliance) -- the D6 check.
    final_selection_bias_tests.csv    paired Wilcoxon tests new vs old and cross-cell Spearman correlations.
(g) final_pooled_robustness.csv       pooled main-study GEE of h1h2.py re-fitted without the failed Gemma-4 DiM edit,
                                      leaving one family out, and with gamma-MPOA in place of MPOA (Gemma, OLMo).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1h2 import holm, mcnemar_exact, newcombe_paired, to01, wilson  # noqa: E402

SEED = 20261106
N_BOOT = 4000
KL_MAX = 0.1
BENCHES = ["mmlu_pro", "gsm8k", "ifeval", "humaneval", "mbpp"]
SAMPLE_METRIC = {"mmlu_pro": "exact_match", "gsm8k": "exact_match", "ifeval": "prompt_level_strict_acc",
                 "humaneval": "pass@1", "mbpp": "pass_at_1"}
EVAL_ROOTS = [ROOT / "runs/mc-rtx-1/results/bench", ROOT / "runs/mc-h200-1/results/bench", ROOT / "results/bench-4090"]
PAIR_ORDER = ["qwen", "gemma", "olmo", "mistral", "qwen-small"]
METHOD_ORDER = ["original", "dim", "mpoa", "som", "jspace", "ara", "gamma_mpoa", "community"]
METHOD_DISPLAY = {"original": "Original", "dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space",
                  "ara": "ARA", "gamma_mpoa": "gamma-MPOA", "community": "community"}
DIR_METHODS = {"dim", "mpoa", "som", "jspace"}
LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def spearman_ci(x, y, rng=None) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    d = dict(n=n, rho=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, tau_b=np.nan, tau_p=np.nan)
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return d
    r, t = stats.spearmanr(x, y), stats.kendalltau(x, y)
    d.update(rho=float(r.statistic), p=float(r.pvalue), tau_b=float(t.statistic), tau_p=float(t.pvalue))
    rng = rng or np.random.default_rng(SEED)
    idx = rng.integers(0, n, (N_BOOT, n))
    rx, ry = stats.rankdata(x[idx], axis=1), stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (rx * ry).sum(1) / den
    rb = rb[np.isfinite(rb)]
    if len(rb):
        d["ci_lo"], d["ci_hi"] = (float(v) for v in np.percentile(rb, [2.5, 97.5]))
    return d


def wilcoxon_paired(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    out = dict(n_pairs=len(d), median_diff=float(np.median(d)) if len(d) else np.nan,
               mean_diff=float(np.mean(d)) if len(d) else np.nan, n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
               n_zero=int((d == 0).sum()), wilcoxon_W=np.nan, wilcoxon_p=np.nan, sign_test_p=np.nan)
    if len(d) >= 3 and np.any(d != 0):
        w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        out.update(wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue))
        npos, nneg = out["n_pos"], out["n_neg"]
        out["sign_test_p"] = float(stats.binomtest(npos, npos + nneg, 0.5).pvalue) if npos + nneg else np.nan
    return out


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    po = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


# ------------------------------------------------------------------------------------------------------------ inputs
def load():
    cols = ["label", "model", "model_display", "family", "pair", "study", "generation", "gen_new", "method", "variant",
            "method_class", "role", "seed", "id", "set", "kind", "adv_overlap", "ref_wg", "ref_kw", "ref_kw256",
            "ref_q3g", "harm_hb", "harm_wg", "is_degenerate", "incomplete", "outcome"]
    rj = pd.read_csv(OUT / "responses_judged.csv.gz", usecols=cols, low_memory=False)
    for c in ["ref_wg", "ref_kw", "ref_kw256", "ref_q3g", "harm_hb", "harm_wg", "is_degenerate", "incomplete",
              "adv_overlap"]:
        rj[c] = to01(rj[c])
    cr = pd.read_csv(OUT / "cell_rates.csv", low_memory=False)
    ds = pd.read_csv(OUT / "dir_status.csv", low_memory=False)
    trials = pd.read_csv(ROOT / "paper/figures/data/trials.csv")
    results = pd.read_csv(ROOT / "paper/figures/data/results.csv")
    meta = rj.drop_duplicates("label")[["label", "model", "model_display", "family", "pair", "study", "generation",
                                        "gen_new", "method", "variant", "method_class", "role", "seed"]].copy()
    meta["seed"] = pd.to_numeric(meta["seed"], errors="coerce")
    meta = meta.set_index("label")
    return rj, cr, ds, trials, results, meta


def eval_dir(label: str) -> Path | None:
    for r in EVAL_ROOTS:
        p = r / label / "summary.json"
        if p.is_file():
            return p.parent
    return None


def sample_scores(edir: Path, bench: str) -> pd.Series | None:
    """Per-item score keyed by (task file, doc_id) from lm-eval samples_*.jsonl; None if absent."""
    bd = edir / "lmeval" / bench
    files = sorted(bd.glob("samples_*.jsonl")) if bd.is_dir() else []
    if not files:
        return None
    key = SAMPLE_METRIC[bench]
    vals = {}
    for f in files:
        task = f.stem.replace("samples_", "")
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                v = d.get(key)
                if isinstance(v, list):
                    v = v[0] if v else np.nan
                vals[(task, int(d["doc_id"]))] = float(v)
    return pd.Series(vals)


# ------------------------------------------------------------------------------------------------- (cap) capability
def capability_table(meta: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, m in meta.iterrows():
        ed = eval_dir(lab)
        r = {"label": lab, "model": m["model"], "pair": m["pair"], "generation": m["generation"], "method": m["method"],
             "variant": m["variant"], "role": m["role"], "seed": m["seed"],
             "eval_dir": str(ed.relative_to(ROOT)) if ed else ""}
        if ed is not None:
            s = json.load(open(ed / "summary.json"))
            r["finished"] = bool(s.get("finished"))
            r["failed_steps"] = ";".join(map(str, s.get("failed") or []))
            for b in BENCHES:
                v = (s.get("benchmarks") or {}).get(b) or {}
                r[f"{b}_pct"] = 100 * v["score"] if "score" in v else np.nan
                r[f"{b}_se_pct"] = 100 * v["stderr"] if v.get("stderr") is not None else np.nan
                r[f"{b}_n"] = v.get("n", np.nan)
            tf = s.get("teacher_forced") or {}
            r["tf_mean_logprob"] = tf.get("mean_logprob", np.nan)
            r["tf_n_tokens"] = tf.get("n_tokens", np.nan)
            r["tf_reference"] = str(tf.get("reference", "")).split("/results/")[-1]
            if "mean_logprob" not in tf and (ed / "refusal_run" / "summary.json").is_file():
                s2 = json.load(open(ed / "refusal_run" / "summary.json"))
                tf2 = s2.get("teacher_forced") or {}
                r["tf_mean_logprob"] = tf2.get("mean_logprob", np.nan)
                r["tf_n_tokens"] = tf2.get("n_tokens", np.nan)
                r["tf_reference"] = str(tf2.get("reference", "")).split("/results/")[-1] + " (refusal_run)"
        rows.append(r)
    cap = pd.DataFrame(rows)
    cap["capability_mean_pct"] = cap[[f"{b}_pct" for b in BENCHES]].mean(axis=1, skipna=False)
    base = cap[cap["method"] == "original"].set_index("model")
    for b in BENCHES + ["capability_mean"]:
        cap[f"{b}_delta_pp"] = cap[f"{b}_pct"] - cap["model"].map(base[f"{b}_pct"])
    cap["tf_divergence"] = cap["model"].map(base["tf_mean_logprob"]) - cap["tf_mean_logprob"]
    cap["tf_paired_tokens"] = cap["tf_n_tokens"] == cap["model"].map(base["tf_n_tokens"])
    cap.loc[cap["method"] == "original", ["tf_divergence"]] = 0.0
    # cross-check with paper/figures/data/results.csv where it has the cell
    rr = results.dropna(subset=["label"]).set_index("label")
    chk = []
    for _, r in cap.iterrows():
        if r["label"] in rr.index:
            a, b = r["capability_mean_pct"], rr.loc[r["label"], "capability_mean_pct"]
            ta, tb = r["tf_divergence"], rr.loc[r["label"], "tf_divergence"]
            ok_c = (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-6
            ok_t = (np.isnan(tb) and r["method"] == "original") or (np.isnan(ta) and np.isnan(tb)) or abs(ta - tb) < 1e-9
            chk.append("agrees with results.csv" if ok_c and ok_t else f"DIFFERS from results.csv ({a} vs {b}; {ta} vs {tb})")
        else:
            chk.append("not in results.csv (read here only)")
    cap["check_results_csv"] = chk
    return cap


def rates_from_items(d: pd.DataFrame) -> dict:
    h, b = d[d["kind"] == "harmful"], d[d["kind"] == "benign"]
    out = {}
    for col, name in (("ref_wg", "wg"), ("ref_kw", "kw"), ("ref_kw256", "kw256"), ("ref_q3g", "q3g")):
        x = h[col].dropna()
        out[f"{name}_k"], out[f"{name}_n"] = int(x.sum()), int(len(x))
        out[f"{name}_rate"] = x.mean()
        out[f"{name}_lo"], out[f"{name}_hi"] = wilson(int(x.sum()), int(len(x)))
    x = b["ref_wg"].dropna()
    out["overrefusal_wg_k"], out["overrefusal_wg_n"], out["overrefusal_wg_rate"] = int(x.sum()), len(x), x.mean()
    out["overrefusal_kw_rate"] = b["ref_kw"].mean()
    out["harm_hb_rate"] = h["harm_hb"].mean()
    out["harm_wg_rate"] = h["harm_wg"].dropna().mean()
    out["degenerate_harmful_k"] = int(h["is_degenerate"].sum())
    out["degenerate_all_k"] = int(d["is_degenerate"].sum())
    out["capped_harmful_rate"] = h["incomplete"].mean()
    return out


def paired(rj: pd.DataFrame, la: str, lb: str, col: str = "ref_wg", kind: str = "harmful"):
    """Paired comparison of cell lb against cell la on the same items: diff = p_b - p_a, McNemar b = a-only, c = b-only."""
    a = rj[(rj.label == la) & (rj.kind == kind)].set_index("id")[col]
    b = rj[(rj.label == lb) & (rj.kind == kind)].set_index("id")[col]
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    D, lo, hi = newcombe_paired(j["b"].values, j["a"].values)
    nb, nc, p = mcnemar_exact(j["a"].values, j["b"].values)
    return dict(n_paired=len(j), diff=D, diff_lo=lo, diff_hi=hi, a_only=nb, b_only=nc, p_mcnemar=p,
                kappa=kappa(j["a"].values, j["b"].values))


# --------------------------------------------------------------------------------------------------- (a) gamma-MPOA
def analysis_a(rj, meta, cap, cr):
    capi = cap.set_index("label")
    devi = cr[cr.set_group == "harmful_all"].drop_duplicates("label").set_index("label")
    g_labels = [l for l in meta.index if meta.loc[l, "method"] == "gamma_mpoa"]
    rows, brows, grows = [], [], []
    for gl in sorted(g_labels, key=lambda l: (PAIR_ORDER.index(meta.loc[l, "pair"]), meta.loc[l, "gen_new"])):
        model = meta.loc[gl, "model"]
        ml, ol = f"{model}__mpoa", f"{model}__base"
        for lab, comp in ((gl, "gamma_mpoa"), (ml, "mpoa"), (ol, "original")):
            r = {"model": model, "pair": meta.loc[gl, "pair"], "generation": meta.loc[gl, "generation"],
                 "cell": lab, "method": comp}
            r.update(rates_from_items(rj[rj.label == lab]))
            for c in ["dev_refusals", "dev_kl", "dev_selected_trial"]:
                r[c] = devi.loc[lab, c] if lab in devi.index else np.nan
            for c in [f"{b}_pct" for b in BENCHES] + ["capability_mean_pct", "capability_mean_delta_pp",
                                                       "tf_divergence"]:
                r[c] = capi.loc[lab, c]
            if comp != "original":
                po = paired(rj, ol, lab)
                r.update({f"vs_original_{k}": v for k, v in po.items()})
            if comp == "gamma_mpoa":
                pm = paired(rj, ml, gl)
                r.update({f"vs_mpoa_{k}": v for k, v in pm.items()})
                for col, nm in (("ref_kw", "kw"), ("ref_q3g", "q3g")):
                    pk = paired(rj, ml, gl, col)
                    r.update({f"vs_mpoa_{nm}_diff": pk["diff"], f"vs_mpoa_{nm}_p": pk["p_mcnemar"]})
                pb = paired(rj, ml, gl, "ref_wg", "benign")
                r.update(vs_mpoa_overrefusal_diff=pb["diff"], vs_mpoa_overrefusal_p=pb["p_mcnemar"])
            rows.append(r)
        # per-benchmark paired comparisons
        ed = {lab: eval_dir(lab) for lab in (gl, ml, ol)}
        for b in BENCHES:
            sc = {lab: (sample_scores(ed[lab], b) if ed[lab] else None) for lab in (gl, ml, ol)}
            for ref_lab, ref_name in ((ol, "original"), (ml, "mpoa")):
                pg, pr = capi.loc[gl, f"{b}_pct"], capi.loc[ref_lab, f"{b}_pct"]
                se = np.sqrt(capi.loc[gl, f"{b}_se_pct"] ** 2 + capi.loc[ref_lab, f"{b}_se_pct"] ** 2)
                row = dict(model=model, benchmark=b, reference=ref_name, gamma_mpoa_pct=pg, reference_pct=pr,
                           diff_pp=pg - pr, diff_unpaired_lo=pg - pr - 1.96 * se, diff_unpaired_hi=pg - pr + 1.96 * se,
                           p_unpaired_z=float(2 * stats.norm.sf(abs(pg - pr) / se)) if se > 0 else np.nan)
                sg, sr = sc[gl], sc[ref_lab]
                if sg is not None and sr is not None:
                    j = pd.concat([sr, sg], axis=1, keys=["r", "g"]).dropna()
                    rep_g = abs(100 * sg.mean() - pg) < 1e-6
                    rep_r = abs(100 * sr.mean() - pr) < 1e-6
                    binary = set(np.unique(j.values)) <= {0.0, 1.0}
                    row.update(n_items_paired=len(j), samples_reproduce_summary=bool(rep_g and rep_r),
                               samples_binary=bool(binary))
                    if rep_g and rep_r and binary:
                        nb, nc, p = mcnemar_exact(j["r"].values, j["g"].values)
                        D, lo, hi = newcombe_paired(j["g"].values, j["r"].values)
                        row.update(ref_only_correct=nb, gamma_only_correct=nc, p_mcnemar=p, diff_paired_lo=100 * lo,
                                   diff_paired_hi=100 * hi)
                brows.append(row)
    a = pd.DataFrame(rows)
    gm = a[a.method == "gamma_mpoa"].copy()
    a.loc[a.method == "gamma_mpoa", "vs_mpoa_p_holm"] = holm(gm["vs_mpoa_p_mcnemar"]).values
    bt = pd.DataFrame(brows)
    if "p_mcnemar" in bt:
        bt["p_holm_within_reference"] = np.nan
        for ref in bt.reference.unique():
            m = bt.reference == ref
            bt.loc[m, "p_holm_within_reference"] = holm(bt.loc[m, "p_mcnemar"]).values
    # generation effects gamma-MPOA vs MPOA per family
    for pair in [p for p in PAIR_ORDER if ((meta.method == "gamma_mpoa") & (meta.pair == p)).any()]:
        lab = {(meta.loc[l, "method"], int(meta.loc[l, "gen_new"])): l for l in meta.index if meta.loc[l, "pair"] == pair
               and meta.loc[l, "role"] in ("original", "primary", "control")}
        for m in ("original", "mpoa", "gamma_mpoa"):
            if (m, 0) in lab and (m, 1) in lab:
                pr = paired(rj, lab[(m, 0)], lab[(m, 1)])
                grows.append(dict(pair=pair, method=m, old_label=lab[(m, 0)], new_label=lab[(m, 1)],
                                  p_old=rj[(rj.label == lab[(m, 0)]) & (rj.kind == "harmful")].ref_wg.mean(),
                                  p_new=rj[(rj.label == lab[(m, 1)]) & (rj.kind == "harmful")].ref_wg.mean(),
                                  **{k: v for k, v in pr.items() if k != "kappa"}))
    gen = pd.DataFrame(grows)
    # KL vs capability across modified cells
    mod = cap[cap.method != "original"].merge(
        devi[["dev_kl", "dev_refusals"]], left_on="label", right_index=True, how="left")
    krows = []
    sets = {"all modified with dev KL (primary + gamma-MPOA + seed 2)": mod.role.isin(["primary", "control", "replicate"]),
            "primary seed 1 only": mod.role == "primary",
            "primary + gamma-MPOA": mod.role.isin(["primary", "control"]),
            "all modified incl. community (teacher-forced divergence only)": mod.role.isin(
                ["primary", "control", "replicate", "external"])}
    for sname, sm in sets.items():
        t = mod[sm]
        for x in ("dev_kl", "tf_divergence"):
            for y in ("capability_mean_delta_pp", "humaneval_delta_pp", "mbpp_delta_pp"):
                if x == "dev_kl" and "community" in sname:
                    continue
                krows.append(dict(cell_set=sname, x=x, y=y, **spearman_ci(t[x], t[y])))
    kt = pd.DataFrame(krows)
    # outlier context: capability change of the other cells at KL <= 0.1
    ctx = mod[mod.role.isin(["primary", "control", "replicate"])][
        ["label", "role", "dev_kl", "tf_divergence", "capability_mean_delta_pp", "humaneval_delta_pp",
         "mbpp_delta_pp"]].sort_values("capability_mean_delta_pp")
    return a, bt, gen, kt, ctx


# ------------------------------------------------------------------------------------------------- (b) community
def analysis_b(rj, meta, cap):
    capi = cap.set_index("label")
    rows = []
    for cl in [l for l in meta.index if meta.loc[l, "method"] == "community"]:
        model = meta.loc[cl, "model"]
        ours = [l for l in meta.index if meta.loc[l, "model"] == model and meta.loc[l, "role"] == "primary"]
        cr_ = rates_from_items(rj[rj.label == cl])
        our_rates = {l: rates_from_items(rj[rj.label == l]) for l in ours}
        best = min(ours, key=lambda l: our_rates[l]["wg_rate"])
        for ol in sorted(ours, key=lambda l: METHOD_ORDER.index(meta.loc[l, "method"])):
            pr = paired(rj, ol, cl)
            r = dict(community=cl, variant=meta.loc[cl, "variant"], model=model, pair=meta.loc[cl, "pair"],
                     generation=meta.loc[cl, "generation"], ours=ol, our_method=meta.loc[ol, "method"],
                     ours_is_best_wg=(ol == best),
                     community_wg_rate=cr_["wg_rate"], community_wg_lo=cr_["wg_lo"], community_wg_hi=cr_["wg_hi"],
                     ours_wg_rate=our_rates[ol]["wg_rate"], ours_wg_lo=our_rates[ol]["wg_lo"],
                     ours_wg_hi=our_rates[ol]["wg_hi"],
                     diff_community_minus_ours=pr["diff"], diff_lo=pr["diff_lo"], diff_hi=pr["diff_hi"],
                     ours_only_refused=pr["a_only"], community_only_refused=pr["b_only"], p_mcnemar=pr["p_mcnemar"],
                     community_kw_rate=cr_["kw_rate"], ours_kw_rate=our_rates[ol]["kw_rate"],
                     community_q3g_rate=cr_["q3g_rate"], ours_q3g_rate=our_rates[ol]["q3g_rate"],
                     community_harm_hb_rate=cr_["harm_hb_rate"], ours_harm_hb_rate=our_rates[ol]["harm_hb_rate"],
                     community_overrefusal_wg=cr_["overrefusal_wg_rate"],
                     ours_overrefusal_wg=our_rates[ol]["overrefusal_wg_rate"],
                     community_capped=cr_["capped_harmful_rate"], ours_capped=our_rates[ol]["capped_harmful_rate"],
                     community_degenerate_k=cr_["degenerate_all_k"], ours_degenerate_k=our_rates[ol]["degenerate_all_k"],
                     community_capability_delta_pp=capi.loc[cl, "capability_mean_delta_pp"],
                     ours_capability_delta_pp=capi.loc[ol, "capability_mean_delta_pp"],
                     community_humaneval_delta_pp=capi.loc[cl, "humaneval_delta_pp"],
                     ours_humaneval_delta_pp=capi.loc[ol, "humaneval_delta_pp"],
                     community_tf_divergence=capi.loc[cl, "tf_divergence"],
                     ours_tf_divergence=capi.loc[ol, "tf_divergence"])
            rows.append(r)
    b = pd.DataFrame(rows)
    b["p_holm_within_community"] = b.groupby("community")["p_mcnemar"].transform(holm)
    b["p_holm_all_32"] = holm(b["p_mcnemar"]).values
    return b


# --------------------------------------------------------------------------------------------------------- (c) seeds
def analysis_c(rj, meta, cap, cr):
    capi = cap.set_index("label")
    devi = cr[cr.set_group == "harmful_all"].drop_duplicates("label").set_index("label")
    reps = [l for l in meta.index if meta.loc[l, "role"] == "replicate"]
    rows = []
    for l2 in sorted(reps, key=lambda l: (meta.loc[l, "gen_new"], METHOD_ORDER.index(meta.loc[l, "method"]))):
        l1 = l2.rsplit("-s", 1)[0]
        pr = paired(rj, l1, l2)
        r = dict(model=meta.loc[l2, "model"], generation=meta.loc[l2, "generation"], method=meta.loc[l2, "method"],
                 seed1_label=l1, seed2_label=l2)
        for lab, s in ((l1, "s1"), (l2, "s2")):
            rr = rates_from_items(rj[rj.label == lab])
            r.update({f"{s}_wg_rate": rr["wg_rate"], f"{s}_wg_lo": rr["wg_lo"], f"{s}_wg_hi": rr["wg_hi"],
                      f"{s}_kw_rate": rr["kw_rate"], f"{s}_q3g_rate": rr["q3g_rate"],
                      f"{s}_overrefusal_wg": rr["overrefusal_wg_rate"],
                      f"{s}_dev_refusals": devi.loc[lab, "dev_refusals"], f"{s}_dev_kl": devi.loc[lab, "dev_kl"],
                      f"{s}_capability_mean_pct": capi.loc[lab, "capability_mean_pct"],
                      f"{s}_capability_delta_pp": capi.loc[lab, "capability_mean_delta_pp"],
                      f"{s}_tf_divergence": capi.loc[lab, "tf_divergence"]})
        r.update(diff_s2_minus_s1=pr["diff"], diff_lo=pr["diff_lo"], diff_hi=pr["diff_hi"], s1_only=pr["a_only"],
                 s2_only=pr["b_only"], p_mcnemar=pr["p_mcnemar"], kappa_items=pr["kappa"],
                 discordant_share=(pr["a_only"] + pr["b_only"]) / pr["n_paired"],
                 diff_kw=paired(rj, l1, l2, "ref_kw")["diff"], diff_q3g=paired(rj, l1, l2, "ref_q3g")["diff"])
        rows.append(r)
    sc = pd.DataFrame(rows)
    sc["p_holm"] = holm(sc["p_mcnemar"]).values

    # generation effects per seed (small pair) + H2 contrast per seed
    h = rj[(rj.kind == "harmful") & (rj.study == "small")]
    wide = h.pivot(index="id", columns="label", values="ref_wg")
    methods = [m for m in METHOD_ORDER if m in set(meta.loc[reps, "method"])]
    old_m, new_m = "qwen2.5-7b", "qwen3.5-4b"
    rng = np.random.default_rng(SEED)
    items = wide.index.values
    W = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=N_BOOT)
    grows = []
    lab = lambda mdl, m, s: f"{mdl}__{m}" + ("-s2" if s == 2 else "")  # noqa: E731
    combos = {"seed1": (1, 1), "seed2": (2, 2), "old s1 / new s2": (1, 2), "old s2 / new s1": (2, 1)}
    eff = {}
    for m in methods:
        for cname, (so, sn) in combos.items():
            lo_, ln_ = lab(old_m, m, so), lab(new_m, m, sn)
            pr = paired(rj, lo_, ln_)
            eff[(m, cname)] = pr["diff"]
            grows.append(dict(method=m, seeds=cname, old_label=lo_, new_label=ln_, p_old=wide[lo_].mean(),
                              p_new=wide[ln_].mean(), delta=pr["diff"], delta_lo=pr["diff_lo"], delta_hi=pr["diff_hi"],
                              old_only=pr["a_only"], new_only=pr["b_only"], p_mcnemar=pr["p_mcnemar"]))
        # seed average (item-level mean of the two seeds)
        yo = (wide[lab(old_m, m, 1)] + wide[lab(old_m, m, 2)]) / 2
        yn = (wide[lab(new_m, m, 1)] + wide[lab(new_m, m, 2)]) / 2
        ok = (yo.notna() & yn.notna()).values
        dif = (yn - yo).values[ok]
        Wm = W[:, ok].astype(float)
        bs = (Wm @ dif) / Wm.sum(1)
        grows.append(dict(method=m, seeds="seed average", old_label=f"{lab(old_m, m, 1)}+s2",
                          new_label=f"{lab(new_m, m, 1)}+s2", p_old=float(yo[ok].mean()), p_new=float(yn[ok].mean()),
                          delta=float(dif.mean()), delta_lo=float(np.percentile(bs, 2.5)),
                          delta_hi=float(np.percentile(bs, 97.5)), old_only=np.nan, new_only=np.nan,
                          p_mcnemar=np.nan))
        eff[(m, "seed average")] = float(dif.mean())
    g = pd.DataFrame(grows)
    g["p_holm_within_seeds"] = g.groupby("seeds")["p_mcnemar"].transform(holm)
    g["range_over_4_combos_pp"] = g["method"].map(
        {m: 100 * (max(eff[(m, c)] for c in combos) - min(eff[(m, c)] for c in combos)) for m in methods})
    # H2 RD contrast per seed setting
    dm = [m for m in methods if m in DIR_METHODS]
    h2rows = []
    for cname in list(combos) + ["seed average"]:
        if "ara" not in methods:
            break
        cval = eff[("ara", cname)] - np.mean([eff[(m, cname)] for m in dm])
        h2rows.append(dict(method="H2 RD contrast (ARA - mean direction)", seeds=cname, delta=cval,
                           direction_methods="+".join(dm)))
    # item-bootstrap CI for the H2 contrast under every seed setting
    for cname in list(combos) + ["seed average"]:
        def colvec(mdl, m, cname=cname):
            if cname == "seed average":
                return ((wide[lab(mdl, m, 1)] + wide[lab(mdl, m, 2)]) / 2).values
            so, sn = combos[cname]
            return wide[lab(mdl, m, sn if mdl == new_m else so)].values
        Y = np.column_stack([colvec(new_m, m) - colvec(old_m, m) for m in ["ara"] + dm])
        ok = ~np.isnan(Y).any(axis=1)
        Wm = W[:, ok].astype(float)
        means = (Wm @ Y[ok]) / Wm.sum(1, keepdims=True)
        bs = means[:, 0] - means[:, 1:].mean(1)
        for r in h2rows:
            if r["seeds"] == cname:
                r["delta_lo"], r["delta_hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    g = pd.concat([g, pd.DataFrame(h2rows)], ignore_index=True)

    # method spread vs seed spread, per model
    mrows = []
    hh = rj[rj.kind == "harmful"]
    for mdl in (old_m, new_m):
        R = {(m, s): hh[hh.label == lab(mdl, m, s)].ref_wg.mean() for m in methods for s in (1, 2)}
        vals = [[R[(m, 1)], R[(m, 2)]] for m in methods]
        f = stats.f_oneway(*vals)
        mean_m = {m: (R[(m, 1)] + R[(m, 2)]) / 2 for m in methods}
        seed_abs = [abs(R[(m, 1)] - R[(m, 2)]) for m in methods]
        pairs_ = list(combinations(methods, 2))
        same_sign = sum(np.sign(R[(a, 1)] - R[(b, 1)]) == np.sign(R[(a, 2)] - R[(b, 2)]) for a, b in pairs_)
        robust = sum(len({np.sign(R[(a, sa)] - R[(b, sb)]) for sa in (1, 2) for sb in (1, 2)}) == 1 for a, b in pairs_)
        ms_method = 2 * np.var(list(mean_m.values()), ddof=1)
        ms_seed = np.mean([np.var([R[(m, 1)], R[(m, 2)]], ddof=1) for m in methods])
        # effect sizes of the same one-way ANOVA (k methods, n = 2 seeds per method)
        grand = np.mean([v for pair in vals for v in pair])
        ss_method = sum(2 * (mean_m[m] - grand) ** 2 for m in methods)
        ss_within = sum((R[(m, s)] - mean_m[m]) ** 2 for m in methods for s in (1, 2))
        var_method = max(0.0, (ms_method - ms_seed) / 2)  # method variance component (ANOVA estimator)
        mrows.append(dict(model=mdl, n_methods=len(methods), methods="+".join(methods),
                          method_means_range_pp=100 * (max(mean_m.values()) - min(mean_m.values())),
                          method_means_sd_pp=100 * np.std(list(mean_m.values()), ddof=1),
                          seed_abs_diff_median_pp=100 * np.median(seed_abs), seed_abs_diff_max_pp=100 * np.max(seed_abs),
                          seed_abs_diff_mean_pp=100 * np.mean(seed_abs),
                          ms_method=ms_method, ms_seed_within=ms_seed,
                          anova_F=float(f.statistic), anova_df=f"{len(methods) - 1},{len(methods)}",
                          anova_p=float(f.pvalue),
                          # eta^2 = SS_method / SS_total; variance-component share = s2_method / (s2_method + MS_within);
                          # ms_ratio_method = MS_method / (MS_method + MS_within) = F / (F + 1), a ratio of mean squares
                          # (NOT a variance share; reported as "method share of variance" before 2026-11-06)
                          eta_sq_method=float(ss_method / (ss_method + ss_within)) if ss_method + ss_within else np.nan,
                          var_component_share_method=float(var_method / (var_method + ms_seed))
                          if var_method + ms_seed else np.nan,
                          ms_ratio_method=float(ms_method / (ms_method + ms_seed)) if ms_method + ms_seed else np.nan,
                          n_method_pairs=len(pairs_), n_pairs_same_sign_s1_s2=int(same_sign),
                          n_pairs_same_sign_all_4_combos=int(robust),
                          **{f"wg_{m}_s{s}": R[(m, s)] for m in methods for s in (1, 2)}))
    mv = pd.DataFrame(mrows)
    return sc, g, mv


# ---------------------------------------------------------------------------------------------- (d) dev vs held-out
def analysis_d(rj, meta, cr):
    c = cr[cr.set_group == "harmful_all"].drop_duplicates("label").set_index("label")
    t = meta.join(c[["dev_refusals", "dev_kl", "dev_selected_trial", "wg_refusal_rate", "kw_refusal_rate",
                     "kw256_refusal_rate", "q3g_refusal_rate"]])
    t["dev_rate"] = t["dev_refusals"] / 100
    sets = {
        "all cells with a dev point (originals + primary + gamma-MPOA + seed 2)": t.role.isin(
            ["original", "primary", "control", "replicate"]),
        "all modified with a dev point (primary + gamma-MPOA + seed 2)": t.role.isin(["primary", "control", "replicate"]),
        "primary (seed 1, all pairs)": t.role == "primary",
        "primary main study": (t.role == "primary") & (t.study == "main"),
        "primary small pair": (t.role == "primary") & (t.study == "small"),
        "primary, older models": (t.role == "primary") & t.generation.isin(["old", "small-old"]),
        "primary, newer models": (t.role == "primary") & t.generation.isin(["new", "small-new"]),
        "gamma-MPOA (4)": t.role == "control",
        "seed 2 (10)": t.role == "replicate",
        "small pair, seed 1 + seed 2 (20)": (t.study == "small") & t.role.isin(["primary", "replicate"]),
    }
    for fam in ["qwen", "gemma", "olmo", "mistral"]:
        sets[f"primary, {fam} pair"] = (t.role == "primary") & (t.pair == fam)
    rows = []
    rng = np.random.default_rng(SEED)
    for sname, sm in sets.items():
        tt = t[sm & t.dev_rate.notna()]
        for y in ("wg_refusal_rate", "kw_refusal_rate", "kw256_refusal_rate", "q3g_refusal_rate"):
            r = spearman_ci(tt.dev_rate, tt[y], rng)
            rows.append(dict(cell_set=sname, x="dev keyword refusals of the selected trial (of 100)",
                             y=f"held-out {y.replace('_refusal_rate', '')} refusal, 972 harmful items",
                             median_heldout_minus_dev_pp=float(np.median(100 * (tt[y] - tt.dev_rate))) if len(tt) else np.nan,
                             cells=";".join(tt.index), **r))
    d = pd.DataFrame(rows)
    # within model
    wrows = []
    tm = t[t.role.isin(["primary", "replicate"]) & t.dev_rate.notna()].copy()
    tm["seed"] = tm["seed"].fillna(1)
    for (mdl, s), g in tm.groupby(["model", "seed"], sort=False):
        if len(g) < 3:
            continue
        rho = stats.spearmanr(g.dev_rate, g.wg_refusal_rate).statistic
        rho_kw = stats.spearmanr(g.dev_rate, g.kw_refusal_rate).statistic
        bd = set(g.index[g.dev_rate == g.dev_rate.min()].map(lambda l: meta.loc[l, "method"]))
        bw = set(g.index[g.wg_refusal_rate == g.wg_refusal_rate.min()].map(lambda l: meta.loc[l, "method"]))
        # dev ties: break them like the within-run selection rule (scripts/select_trial.py: min refusals, then min KL)
        gk = g.sort_values(["dev_rate", "dev_kl"])
        bd_kl = meta.loc[gk.index[0], "method"]
        wrows.append(dict(model=mdl, seed=int(s), n_methods=len(g), spearman_dev_vs_wg=rho, spearman_dev_vs_kw=rho_kw,
                          best_dev="/".join(sorted(bd)), best_heldout_wg="/".join(sorted(bw)),
                          best_dev_tied=len(bd) > 1,
                          best_dev_tie_kl=";".join(f"{meta.loc[l, 'method']} {g.loc[l, 'dev_kl']:.3f}"
                                                   for l in gk.index if meta.loc[l, "method"] in bd)
                          if len(bd) > 1 else "",
                          best_dev_is_best_wg=bool(bd & bw),  # any tied dev-best method matches
                          best_dev_unique_is_best_wg=bool(len(bd) == 1 and bd <= bw),
                          best_dev_kl_tiebreak=bd_kl, best_dev_kl_tiebreak_is_best_wg=bool(bd_kl in bw)))
    return d, pd.DataFrame(wrows)


# ------------------------------------------------------------------------------------------------ (e) convergence
def run_curves(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, method, seed, run), g in trials.groupby(["model", "method", "seed", "run"], sort=False):
        g = g.sort_values("trial")
        feas = g.kl <= KL_MAX
        ref = np.where(feas, g.refusals, np.inf)
        best = np.minimum.accumulate(ref)
        trial = g.trial.values
        sel = g[g["is_selected_kl0.1"].astype(str) == "True"]
        sel_ref = float(sel.refusals.iloc[0]) if len(sel) else np.nan
        sel_tr = int(sel.trial.iloc[0]) if len(sel) else np.nan
        sel_kl = float(sel.kl.iloc[0]) if len(sel) else np.nan
        assert np.isnan(sel_ref) or best[-1] == sel_ref, (run, best[-1], sel_ref)
        first_reach = int(trial[np.argmax(best == best[-1])]) if np.isfinite(best[-1]) else np.nan

        def at(t):
            m = trial <= t
            v = best[m][-1] if m.any() else np.inf
            return float(v) if np.isfinite(v) else np.nan
        r0 = g.iloc[0]
        rows.append(dict(model=model, method=method, seed=int(seed), run=run, family=r0["family"], study=r0["study"],
                         generation=r0["generation"], n_trials=len(g), n_feasible=int(feas.sum()),
                         first_feasible_trial=int(trial[np.argmax(feas.values)]) if feas.any() else np.nan,
                         best_at_30=at(30), best_at_50=at(50), best_at_75=at(75), best_at_100=at(10 ** 6),
                         selected_refusals=sel_ref, selected_trial=sel_tr, selected_kl=sel_kl,
                         first_reach_trial=first_reach, late_after_75=bool(first_reach > 75),
                         gain_after_75=at(75) - at(10 ** 6), gain_after_50=at(50) - at(10 ** 6)))
    return pd.DataFrame(rows)


def analysis_e(trials):
    runs = run_curves(trials)
    core = runs[(runs.seed == 1) & (runs.method != "gamma_mpoa")]
    srows = []

    def summ(name, sub):
        return dict(group=name, n_runs=len(sub), n_late_after_75=int(sub.late_after_75.sum()),
                    share_late=float(sub.late_after_75.mean()) if len(sub) else np.nan,
                    median_first_reach=float(sub.first_reach_trial.median()) if len(sub) else np.nan,
                    median_gain_after_50=float(sub.gain_after_50.median()) if len(sub) else np.nan,
                    mean_gain_after_75=float(sub.gain_after_75.mean()) if len(sub) else np.nan,
                    max_gain_after_75=float(sub.gain_after_75.max()) if len(sub) else np.nan,
                    late_runs=";".join(sub[sub.late_after_75].run))
    srows.append(summ("core seed-1 runs (all)", core))
    for st in ["main", "small"]:
        srows.append(summ(f"core seed-1, {st}", core[core.study == st]))
    for gname, gens in (("older", ["old", "small-old"]), ("newer", ["new", "small-new"])):
        srows.append(summ(f"core seed-1, {gname} models", core[core.generation.isin(gens)]))
    for m in ["dim", "mpoa", "som", "jspace", "ara"]:
        srows.append(summ(f"core seed-1, method {m}", core[core.method == m]))
    for fam in core.family.unique():
        srows.append(summ(f"core seed-1, family {fam}", core[core.family == fam]))
    srows.append(summ("seed-2 runs (small pair)", runs[runs.seed == 2]))
    srows.append(summ("seed-1 runs of the same 10 small-pair cells", core[core.study == "small"]))
    srows.append(summ("gamma-MPOA runs", runs[runs.method == "gamma_mpoa"]))
    s = pd.DataFrame(srows)
    # tests
    trows = []
    newer = core.generation.isin(["new", "small-new"])
    tab = pd.crosstab(newer, core.late_after_75).reindex(index=[False, True], columns=[False, True], fill_value=0)
    orr, p = stats.fisher_exact(tab.values)
    trows.append(dict(test="Fisher exact: late (first reach after trial 75) vs newer model, core seed-1 runs",
                      table=f"older: {tab.loc[False, True]} late / {tab.loc[False].sum()}; newer: {tab.loc[True, True]} "
                            f"late / {tab.loc[True].sum()}", statistic=orr, p=p))
    tab2 = pd.crosstab(core.method, core.late_after_75)
    chi = stats.chi2_contingency(tab2.values) if tab2.shape[1] == 2 else None
    if chi is not None:
        trows.append(dict(test="chi-square: late vs method, core seed-1 runs (small expected counts; descriptive)",
                          table="; ".join(f"{m}: {int(tab2.loc[m, True])}/{int(tab2.loc[m].sum())}" for m in tab2.index),
                          statistic=float(chi.statistic), p=float(chi.pvalue)))
    # paired first-reach new vs old (pair x method)
    pairs_map = {"Qwen": "qwen", "Gemma": "gemma", "OLMo": "olmo", "Mistral Small": "mistral",
                 "Qwen (small pair)": "qwen-small"}
    core = core.assign(pair=core.family.map(pairs_map).fillna(core.family),
                       gen_new=newer.astype(int))
    pv = core.pivot_table(index=["pair", "method"], columns="gen_new", values="first_reach_trial")
    w = wilcoxon_paired(pv[0], pv[1])
    trows.append(dict(test="Wilcoxon signed-rank: first-reach trial new vs old (pair x method)",
                      table=f"n = {w['n_pairs']}, new later in {w['n_pos']}, earlier in {w['n_neg']}, tie {w['n_zero']}",
                      statistic=w["wilcoxon_W"], p=w["wilcoxon_p"], median_diff=w["median_diff"]))
    pv2 = core.pivot_table(index=["pair", "method"], columns="gen_new", values="gain_after_50")
    w2 = wilcoxon_paired(pv2[0], pv2[1])
    trows.append(dict(test="Wilcoxon signed-rank: gain after trial 50 new vs old (pair x method)",
                      table=f"n = {w2['n_pairs']}, larger on new in {w2['n_pos']}, smaller in {w2['n_neg']}, "
                            f"tie {w2['n_zero']}", statistic=w2["wilcoxon_W"], p=w2["wilcoxon_p"],
                      median_diff=w2["median_diff"]))
    return runs, s, pd.DataFrame(trows)


# ------------------------------------------------------------------------------------------- (f) selection bias D6
def excess_kl(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, method, seed, run), g in trials.groupby(["model", "method", "seed", "run"], sort=False):
        f = g[g.kl <= KL_MAX]
        sel = g[g["is_selected_kl0.1"].astype(str) == "True"]
        if not len(sel):
            continue
        sr, sk = float(sel.refusals.iloc[0]), float(sel.kl.iloc[0])
        r = dict(model=model, method=method, seed=int(seed), run=run, selected_refusals=sr, selected_kl=sk,
                 at_kl_cap=bool(sk > 0.09))
        for k in (2, 5, 10):
            m = f[f.refusals <= sr + k]
            r[f"min_kl_within_{k}"] = float(m.kl.min())
            r[f"excess_kl_{k}"] = sk - float(m.kl.min())
            r[f"kl_ratio_{k}"] = sk / float(m.kl.min()) if m.kl.min() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def analysis_f(rj, meta, cap, cr, trials):
    ex = excess_kl(trials)
    capi = cap.set_index("label")
    h = rj[rj.kind == "harmful"]
    # keyword-only refusal share among WildGuard compliance, per cell
    kw_only = h[h.ref_wg == 0].groupby("label").ref_kw.mean()
    kw_only_n = h[h.ref_wg == 0].groupby("label").ref_kw.size()
    gap = h.dropna(subset=["ref_wg"]).groupby("label").apply(lambda d: (d.ref_kw - d.ref_wg).mean(),
                                                            include_groups=False)
    cells = meta[meta.role.isin(["original", "primary"])].copy()
    cells["seed"] = cells["seed"].fillna(1).astype(int)
    cells = cells.reset_index()
    cells = cells.merge(ex[["model", "method", "seed", "selected_kl", "at_kl_cap", "excess_kl_2", "excess_kl_5",
                            "excess_kl_10", "kl_ratio_5"]].assign(method=lambda x: x.method),
                        on=["model", "method", "seed"], how="left")
    cells["kw_only_share_wg_compliant"] = cells.label.map(kw_only)
    cells["n_wg_compliant"] = cells.label.map(kw_only_n)
    cells["kw_minus_wg_gap"] = cells.label.map(gap)
    for c in ["tf_divergence", "capability_mean_delta_pp", "humaneval_delta_pp"]:
        cells[c] = cells.label.map(capi[c])
    cells["dev_refusals"] = cells.label.map(cr[cr.set_group == "harmful_all"].drop_duplicates("label")
                                            .set_index("label").dev_refusals)
    cells["wg_rate"] = cells.label.map(h.groupby("label").ref_wg.mean())
    # matched pairs (pair x method)
    prow = []
    for (pair, method), g in cells.groupby(["pair", "method"], sort=False):
        if set(g.gen_new) != {0, 1}:
            continue
        o, n = g[g.gen_new == 0].iloc[0], g[g.gen_new == 1].iloc[0]
        r = dict(pair=pair, method=method, old_label=o.label, new_label=n.label)
        for c in ["selected_kl", "at_kl_cap", "excess_kl_5", "excess_kl_10", "kl_ratio_5", "tf_divergence",
                  "capability_mean_delta_pp", "humaneval_delta_pp", "kw_only_share_wg_compliant", "kw_minus_wg_gap",
                  "dev_refusals", "wg_rate"]:
            r[f"{c}_old"], r[f"{c}_new"] = o[c], n[c]
        prow.append(r)
    pairs_ = pd.DataFrame(prow)
    pairs_["pair_order"] = pairs_.pair.map({p: i for i, p in enumerate(PAIR_ORDER)})
    pairs_["method_order"] = pairs_.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    pairs_ = pairs_.sort_values(["pair_order", "method_order"]).drop(columns=["pair_order", "method_order"])
    mod = pairs_[pairs_.method != "original"]
    trows = []
    for scope, sub in (("all pairs (main + small), seed 1", mod), ("main study", mod[mod.pair != "qwen-small"])):
        for c, desc in (("selected_kl", "selected dev first-token KL"),
                        ("excess_kl_5", "excess KL: selected KL minus lowest KL within 5 dev refusals of it"),
                        ("excess_kl_10", "excess KL within 10 dev refusals"),
                        ("tf_divergence", "teacher-forced divergence on 350 benign held-out items"),
                        ("capability_mean_delta_pp", "capability change vs original, mean of 5 benchmarks (pp)"),
                        ("humaneval_delta_pp", "HumanEval change vs original (pp)"),
                        ("kw_only_share_wg_compliant", "held-out keyword-yes share among WildGuard-compliant answers"),
                        ("kw_minus_wg_gap", "held-out keyword minus WildGuard refusal (harmful)")):
            w = wilcoxon_paired(sub[f"{c}_old"], sub[f"{c}_new"])
            trows.append(dict(test="paired new vs old (Wilcoxon signed-rank)", scope=scope, variable=c,
                              description=desc, median_old=float(sub[f"{c}_old"].median()),
                              median_new=float(sub[f"{c}_new"].median()), **w))
        n_cap_old, n_cap_new = int(sub.at_kl_cap_old.sum()), int(sub.at_kl_cap_new.sum())
        trows.append(dict(test="selected KL > 0.09 (at the cap): count old vs new, exact sign test on discordant pairs",
                          scope=scope, variable="at_kl_cap",
                          description="selected trial within 0.01 of the KL cap", n_pairs=len(sub),
                          count_old=n_cap_old, count_new=n_cap_new,
                          n_pos=int(((sub.at_kl_cap_new) & (~sub.at_kl_cap_old)).sum()),
                          n_neg=int(((~sub.at_kl_cap_new) & (sub.at_kl_cap_old)).sum()),
                          sign_test_p=float(stats.binomtest(
                              int(((sub.at_kl_cap_new) & (~sub.at_kl_cap_old)).sum()),
                              int((sub.at_kl_cap_new != sub.at_kl_cap_old).sum()), 0.5).pvalue)
                          if (sub.at_kl_cap_new != sub.at_kl_cap_old).any() else np.nan))
    # originals: keyword-only share on the unedited models (do newer models comply with caveat words?)
    o = pairs_[pairs_.method == "original"]
    w = wilcoxon_paired(o.kw_only_share_wg_compliant_old, o.kw_only_share_wg_compliant_new)
    trows.append(dict(test="originals: keyword-yes share among WildGuard-compliant answers, new vs old", scope="5 pairs",
                      variable="kw_only_share_wg_compliant", median_old=float(o.kw_only_share_wg_compliant_old.median()),
                      median_new=float(o.kw_only_share_wg_compliant_new.median()), **w))
    # cross-cell correlations among modified primary cells
    mc = cells[cells.role == "primary"]
    for x, y in (("kw_only_share_wg_compliant", "selected_kl"), ("kw_only_share_wg_compliant", "excess_kl_5"),
                 ("kw_only_share_wg_compliant", "tf_divergence"),
                 ("kw_only_share_wg_compliant", "capability_mean_delta_pp"),
                 ("selected_kl", "capability_mean_delta_pp"), ("selected_kl", "tf_divergence"),
                 ("excess_kl_5", "capability_mean_delta_pp")):
        trows.append(dict(test="Spearman across modified primary cells", scope=f"{len(mc)} cells", variable=f"{x} vs {y}",
                          **{k: v for k, v in spearman_ci(mc[x], mc[y]).items()}))
    return cells, pairs_, pd.DataFrame(trows)


# ------------------------------------------------------------------------------------- (g) pooled H1/H2 robustness
def analysis_g(rj, meta):
    """Pooled main-study GEE of h1h2.py (y ~ gen_new * opt + C(family), logistic, exchangeable, clusters = prompt;
    WildGuard, 972 harmful items) re-fitted on modified cell sets: all (reproduces h1h2_models.csv), without the failed
    Gemma-4 DiM edit (both Gemma DiM cells dropped), leaving one family out, and with gamma-MPOA in place of MPOA for
    the post-norm families (Gemma, OLMo)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import warnings
    h = rj[(rj.kind == "harmful") & (rj.study == "main") & rj.ref_wg.notna()].copy()
    h["y"] = h["ref_wg"]
    h["opt"] = (h["method_class"] == "optimisation").astype(int)
    h["gen_new"] = h["gen_new"].astype(int)
    prim = h[h.role == "primary"]
    ctrl = h[h.role == "control"].copy()
    ctrl["method"] = "mpoa"
    sets = {"all main-study modified cells (= h1h2.py)": prim,
            "without Gemma DiM (failed Gemma-4 edit; both generations dropped)":
                prim[~((prim.pair == "gemma") & (prim.method == "dim"))]}
    for fam in ["qwen", "gemma", "olmo", "mistral"]:
        sets[f"leave out {fam}"] = prim[prim.pair != fam]
    sets["gamma-MPOA replaces MPOA for Gemma and OLMo"] = pd.concat(
        [prim[~((prim.pair.isin(["gemma", "olmo"])) & (prim.method == "mpoa"))], ctrl])
    sets["gamma-MPOA replaces MPOA, without Gemma DiM"] = sets["gamma-MPOA replaces MPOA for Gemma and OLMo"].pipe(
        lambda d: d[~((d.pair == "gemma") & (d.method == "dim"))])
    rows = []
    for sname, df in sets.items():
        fe = " + C(family)" if df.family.nunique() > 1 else ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.gee("y ~ gen_new * opt" + fe, groups="id", data=df, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
        names = list(res.params.index)
        cov = res.cov_params()
        for term, w in (("gen effect | direction class", {"gen_new": 1}),
                        ("gen effect | optimisation class", {"gen_new": 1, "gen_new:opt": 1}),
                        ("generation x class (H2)", {"gen_new:opt": 1})):
            v = np.array([w.get(n, 0.0) for n in names])
            est = float(v @ res.params.values)
            se = float(np.sqrt(v @ cov.values @ v))
            rows.append(dict(cell_set=sname, term=term, estimate=est, se=se, lo=est - 1.959964 * se,
                             hi=est + 1.959964 * se, p=float(2 * stats.norm.sf(abs(est / se))), n_obs=len(df),
                             n_cells=df.label.nunique(), families="+".join(sorted(df.family.unique()))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    log(f"# final.py run {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    rj, cr, ds, trials, results, meta = load()
    used = ds[ds.used.astype(str) == "True"]
    log(f"cells in responses_judged: {meta.shape[0]} ({', '.join(f'{k} {v}' for k, v in meta.role.value_counts().items())});"
        f" judge status of used dirs: {used.judge_status.value_counts().to_dict()}")

    cap = capability_table(meta, results)
    cap.to_csv(OUT / "final_capability.csv", index=False, float_format="%.6g")
    log(f"(cap) capability for {cap.capability_mean_pct.notna().sum()} / {len(cap)} cells; "
        f"check vs results.csv: {cap.check_results_csv.value_counts().to_dict()}")

    a, ab, ag, ak, actx = analysis_a(rj, meta, cap, cr)
    a.to_csv(OUT / "final_gmpoa_control.csv", index=False, float_format="%.6g")
    ab.to_csv(OUT / "final_gmpoa_benchmarks.csv", index=False, float_format="%.6g")
    ag.to_csv(OUT / "final_gmpoa_generation.csv", index=False, float_format="%.6g")
    ak.to_csv(OUT / "final_kl_vs_capability.csv", index=False, float_format="%.6g")
    actx.to_csv(OUT / "final_kl_vs_capability_cells.csv", index=False, float_format="%.6g")
    log("(a) gamma-MPOA control")
    for r in a.itertuples():
        log(f"  {r.cell:28s} wg {100 * r.wg_rate:5.1f} [{100 * r.wg_lo:.1f}, {100 * r.wg_hi:.1f}] kw {100 * r.kw_rate:5.1f} "
            f"q3g {100 * r.q3g_rate:5.1f} over-ref {100 * r.overrefusal_wg_rate:4.1f} cap {r.capability_mean_pct:5.1f} "
            f"(d {r.capability_mean_delta_pp:+.2f}) HE {r.humaneval_pct:5.1f} MBPP {r.mbpp_pct:5.1f} tf {r.tf_divergence:.3f} "
            f"devKL {r.dev_kl:.3f} dev {r.dev_refusals} degen {r.degenerate_all_k} capped {100 * r.capped_harmful_rate:.0f}%"
            + (f" | vs MPOA {100 * r.vs_mpoa_diff:+.1f} [{100 * r.vs_mpoa_diff_lo:+.1f}, {100 * r.vs_mpoa_diff_hi:+.1f}] "
               f"b/c {r.vs_mpoa_a_only}/{r.vs_mpoa_b_only} p {r.vs_mpoa_p_mcnemar:.2g} Holm {r.vs_mpoa_p_holm:.2g}"
               if r.method == "gamma_mpoa" else ""))
    for r in ag.itertuples():
        log(f"  gen effect {r.pair} {r.method}: {100 * r.p_old:.1f} -> {100 * r.p_new:.1f} = {100 * r.diff:+.1f} "
            f"[{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}] b/c {r.a_only}/{r.b_only} p {r.p_mcnemar:.2g}")
    for r in ak.itertuples():
        log(f"  KL/cap: {r.cell_set} | {r.x} vs {r.y}: rho {r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] p {r.p:.2g} n {r.n}")

    b = analysis_b(rj, meta, cap)
    b.to_csv(OUT / "final_community_vs_selected.csv", index=False, float_format="%.6g")
    log("(b) community vs our selected cells (best = lowest held-out WildGuard among our 4 methods)")
    for r in b[b.ours_is_best_wg].itertuples():
        log(f"  {r.community:40s} wg {100 * r.community_wg_rate:4.1f} [{100 * r.community_wg_lo:.1f}, "
            f"{100 * r.community_wg_hi:.1f}] vs best ours {r.ours} {100 * r.ours_wg_rate:4.1f}: "
            f"{100 * r.diff_community_minus_ours:+.1f} [{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}] "
            f"b/c {r.ours_only_refused}/{r.community_only_refused} pHolm4 {r.p_holm_within_community:.2g} | "
            f"tf {r.community_tf_divergence:.3f} vs {r.ours_tf_divergence:.3f} | cap {r.community_capability_delta_pp:+.2f} "
            f"vs {r.ours_capability_delta_pp:+.2f} | kw {100 * r.community_kw_rate:.1f} vs {100 * r.ours_kw_rate:.1f}")

    sc, sg, sm = analysis_c(rj, meta, cap, cr)
    sc.to_csv(OUT / "final_seed_cells.csv", index=False, float_format="%.6g")
    sg.to_csv(OUT / "final_seed_generation.csv", index=False, float_format="%.6g")
    sm.to_csv(OUT / "final_seed_method_vs_seed.csv", index=False, float_format="%.6g")
    log("(c) seed 2 vs seed 1 (WildGuard, 972 harmful items)")
    for r in sc.itertuples():
        log(f"  {r.seed1_label:22s} s1 {100 * r.s1_wg_rate:5.1f} s2 {100 * r.s2_wg_rate:5.1f} d {100 * r.diff_s2_minus_s1:+5.1f} "
            f"[{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}] b/c {r.s1_only}/{r.s2_only} p {r.p_mcnemar:.2g} Holm "
            f"{r.p_holm:.2g} kappa {r.kappa_items:.2f} dev {r.s1_dev_refusals:.0f}/{r.s2_dev_refusals:.0f} KL "
            f"{r.s1_dev_kl:.3f}/{r.s2_dev_kl:.3f} cap d {r.s1_capability_delta_pp:+.2f}/{r.s2_capability_delta_pp:+.2f}")
    for r in sg.itertuples():
        log(f"  gen {r.method} {r.seeds}: {100 * r.delta:+.1f} [{100 * r.delta_lo:+.1f}, {100 * r.delta_hi:+.1f}]"
            + (f" p {r.p_mcnemar:.2g}" if pd.notna(r.p_mcnemar) else ""))
    for r in sm.itertuples():
        log(f"  {r.model}: method-mean range {r.method_means_range_pp:.1f} pp (SD {r.method_means_sd_pp:.1f}); seed |d| "
            f"median {r.seed_abs_diff_median_pp:.1f}, max {r.seed_abs_diff_max_pp:.1f}; ANOVA F({r.anova_df}) = "
            f"{r.anova_F:.2f}, p = {r.anova_p:.3g}; eta^2 (method) {r.eta_sq_method:.2f}, variance-component share "
            f"{r.var_component_share_method:.2f} (MS ratio {r.ms_ratio_method:.2f}); orderings same sign "
            f"{r.n_pairs_same_sign_s1_s2}/{r.n_method_pairs}, robust to all combos {r.n_pairs_same_sign_all_4_combos}")

    d, dw = analysis_d(rj, meta, cr)
    d.to_csv(OUT / "final_dev_vs_heldout.csv", index=False, float_format="%.6g")
    dw.to_csv(OUT / "final_dev_vs_heldout_within_model.csv", index=False, float_format="%.6g")
    log("(d) dev vs held-out Spearman")
    for r in d.itertuples():
        log(f"  {r.cell_set:75s} {r.y[:22]:22s} n {r.n:2d} rho {r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] p {r.p:.2g}")
    ties = dw[dw.best_dev_tied]
    log(f"  within model: best dev = best held-out WG in {int(dw.best_dev_kl_tiebreak_is_best_wg.sum())}/{len(dw)} "
        f"(dev ties broken by lower dev KL, the selection rule's tie-break); "
        f"{int(dw.best_dev_unique_is_best_wg.sum())}/{len(dw)} counting only a unique dev best; "
        f"{int(dw.best_dev_is_best_wg.sum())}/{len(dw)} if any tied dev-best method may match; median "
        f"within-model rho {dw.spearman_dev_vs_wg.median():.2f}")
    for r in ties.itertuples():
        log(f"    dev tie: {r.model} seed {r.seed}: {r.best_dev} (dev KL {r.best_dev_tie_kl}); KL tie-break -> "
            f"{r.best_dev_kl_tiebreak}; best held-out WG {r.best_heldout_wg}")

    runs, es, et = analysis_e(trials)
    runs.to_csv(OUT / "final_search_runs.csv", index=False, float_format="%.6g")
    es.to_csv(OUT / "final_search_summary.csv", index=False, float_format="%.6g")
    et.to_csv(OUT / "final_search_tests.csv", index=False, float_format="%.6g")
    log("(e) search convergence")
    for r in es.itertuples():
        log(f"  {r.group:45s} n {r.n_runs:2d} late {r.n_late_after_75} ({100 * r.share_late:.0f}%) median first reach "
            f"{r.median_first_reach}")
    for r in et.itertuples():
        log(f"  {r.test}: {r.table}; stat {r.statistic:.3g} p {r.p:.3g}")

    fc, fp, ft = analysis_f(rj, meta, cap, cr, trials)
    fp.to_csv(OUT / "final_selection_bias_pairs.csv", index=False, float_format="%.6g")
    fc.to_csv(OUT / "final_selection_bias_cells.csv", index=False, float_format="%.6g")
    ft.to_csv(OUT / "final_selection_bias_tests.csv", index=False, float_format="%.6g")
    log("(f) selection bias (D6)")
    for r in ft.itertuples():
        if r.test.startswith("Spearman"):
            log(f"  {r.test} {r.variable}: rho {r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] p {r.p:.2g} n {r.n}")
        elif r.variable == "at_kl_cap":
            log(f"  {r.test} | {r.scope}: {r.count_old:.0f} old vs {r.count_new:.0f} new of {r.n_pairs:.0f}; "
                f"discordant new-only {r.n_pos:.0f} / old-only {r.n_neg:.0f}, p {r.sign_test_p:.3g}")
        else:
            log(f"  {r.test} | {r.scope} | {r.variable}: median old {r.median_old:.4g} new {r.median_new:.4g}; "
                f"n {r.n_pairs} +{r.n_pos}/-{getattr(r, 'n_neg', np.nan)} p {r.wilcoxon_p if pd.notna(r.wilcoxon_p) else r.sign_test_p:.3g}")
    gt = analysis_g(rj, meta)
    gt.to_csv(OUT / "final_pooled_robustness.csv", index=False, float_format="%.6g")
    log("(g) pooled main-study GEE (WildGuard), robustness of the H1 class effects and the H2 interaction")
    for r in gt.itertuples():
        log(f"  {r.cell_set:70s} {r.term:32s} {r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] p {r.p:.2g} "
            f"({r.n_cells} cells)")
    log(f"done in {time.time() - t0:.0f} s")
    (OUT / "final_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
