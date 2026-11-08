#!/usr/bin/env python3
"""Paper 33 -- H1 and H2 on the harmful held-out prompts (step 2 of the held-out analysis).

Run after scripts/analysis/assemble.py (re-run both after every judge batch; no GPU, read-only inputs):
    scripts/figures/.venv/bin/python scripts/analysis/assemble.py
    scripts/figures/.venv/bin/python scripts/analysis/h1h2.py            # ~3-7 min (VB fits for wg and kw)
    scripts/figures/.venv/bin/python scripts/analysis/h1h2.py --no-vb    # skip the variational-Bayes fits

Hypotheses (paper/sections/02_introduction.tex)
  H1  at matched first-token KL (<= 0.1, the selection rule), residual refusal after ablation is higher for the newer
      model of each pair than for its predecessor, for every method.
  H2  the generation effect interacts with method class: optimisation-based ablation (ARA) loses less effectiveness on
      newer models than direction-based methods (DiM, MPOA, SOM; J-space on the small pair).

Data: results/analysis/responses_judged.{parquet,csv.gz} (assemble.py), harmful items only (972; 928 without the 44
AdvBench-overlap items), cells with role 'original' or 'primary' (seed-1 study methods). gamma-MPOA (control), community
models (external) and seed-2 replicates are not part of H1/H2. Cells that are not judged yet are listed as pending; they
enter automatically once assemble.py finds their refusal/judgments.jsonl. A pair x method comparison needs both
generations judged; a pair enters the class models with the methods judged in both generations.

Outcomes ("judges"), each 0/1 per response, missing labels dropped:
  wg        WildGuard refusal label ref_wg (PRIMARY; Methods "Refusal and outcome classes")
  wg_class  outcome class == refusal (assemble.py rule: degenerate takes precedence, so degenerate = not refused)
  kw        Heretic's 33 keyword markers on the full (<= 512-token) response (ref_kw)
  kw256     the same markers on the first 256 response tokens (ref_kw256)
  q3g       Qwen3Guard-Gen-8B refusal label (ref_q3g; second LLM judge, sensitivity; added 2026-11-06)

Estimates and tests
  generation effect (pair, method) = p_new - p_old of residual refusal on the same items; 95% CI: Newcombe's hybrid
      score interval for paired proportions (Wilson limits + within-item correlation phi, phi numerator reduced by n/2
      when positive); item-bootstrap percentile CI as a check. Test: exact McNemar (binomial on discordant items).
  baseline-adjusted generation effect = (p_new,mod - p_new,orig) - (p_old,mod - p_old,orig), item-bootstrap CI.
  modified vs original (per model): exact McNemar, Newcombe CI ("ablation" family; descriptive, not H1/H2).
  H2 risk-difference contrast (per pair) = gen effect of ARA - mean gen effect of the direction-based methods,
      item-bootstrap CI.
  Planned model (Methods sec:setup-stats): refusal ~ generation x method_class + (1|prompt) + (1|family), fitted with
      statsmodels BinomialBayesMixedGLM.fit_vb (R/lme4 is not installed). Variational Bayes, mean-field Gaussian
      posterior: posterior mean, SD and mean +/- 1.96 SD intervals; no p-values; the derived simple effect for the
      optimisation class uses SD = sqrt(sum of SD^2) (mean-field independence). Fitted for the main study (family
      intercept), main study + small pair (pair intercept) and per pair (prompt intercept only).
  Robustness: GEE logistic, exchangeable working correlation clustered by prompt, robust (sandwich) SEs, family as a
      fixed effect: same fixed effects as the planned model; per-method generation effects; a version with the
      originals as reference class (baseline-adjusted generation effects); and Wald tests of generation x family and
      generation x class x family (heterogeneity across families, main study only).
  Coefficients of the mixed model are conditional (prompt-specific) log-odds; GEE coefficients are population-averaged
  log-odds and therefore smaller in magnitude. Neither equals the risk difference.

Holm families (alpha = 0.05, two-sided; applied separately for every judge x item subset):
  H1        exact McNemar new vs old modified model for every judged pair x method (main study and small pair), plus
            the pooled main-study GEE generation effect within each method class (2 tests)
  H2        per-pair GEE generation x class interaction, plus the pooled main-study GEE interaction
  ablation  exact McNemar modified vs original for every judged modified model (descriptive)
  The VB mixed-model results are reported next to these tests and are not Holm-adjusted (no p-values).

Outputs (results/analysis/; all overwritten)
  h1h2_status.csv              every planned cell: judge status, used in H1/H2 or why not
  h1h2_cells.csv               per cell x judge x subset: residual refusal + Wilson CI; modified vs original (McNemar)
  h1h2_generation_effects.csv  per pair x method x judge x subset: generation effect, CI, McNemar, Holm, baseline-adjusted
  h1h2_h2_contrasts.csv        per pair x judge x subset: risk-difference H2 contrast + GEE/VB interaction
  h1h2_models.csv              every model term (VB and GEE) per scope x judge x subset
  h1h2_tests.csv               every test in a Holm family with raw and Holm-adjusted p
  h1h2_summary.md              key numbers and caveats (regenerated from the tables above)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import statsmodels  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import binomtest, norm  # noqa: E402
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
ALPHA = 0.05
Z = float(norm.ppf(0.975))
SEED = 20261105

JUDGES = {
    "wg": "WildGuard refusal label (primary)",
    "wg_class": "WildGuard outcome class (degenerate counted as not refused)",
    "kw": "keyword markers, full response (<= 512 tokens)",
    "kw256": "keyword markers, first 256 tokens",
    "q3g": "Qwen3Guard-Gen-8B refusal label (second LLM judge)",
}
SUBSETS = {"all": "all 972 harmful items", "excl_advbench": "928 harmful items without the 44 AdvBench-overlap items"}
PRIMARY = ("wg", "all")
MAIN_PAIRS_PLANNED = ["qwen", "gemma", "olmo", "mistral"]
PAIR_ORDER = ["qwen", "gemma", "olmo", "mistral", "qwen-small"]
PAIR_DISPLAY = {"qwen": "Qwen (2.5-32B -> 3.8-27B)", "gemma": "Gemma (2-27B -> 4-31B)", "olmo": "OLMo (2-32B -> 3.1-32B)",
                "mistral": "Mistral (Small-2409 -> Small-3.2)", "qwen-small": "Small pair (Qwen2.5-7B -> Qwen3.5-4B)"}
METHOD_ORDER = ["original", "dim", "mpoa", "som", "jspace", "ara"]
METHOD_DISPLAY = {"original": "Original", "dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space", "ara": "ARA"}
USED_ROLES = ("original", "primary")
STALE: list[str] = []  # set by load_inputs (freshness.check)


# ---------------------------------------------------------------------------------------------------------- helpers
def to01(s: pd.Series) -> pd.Series:
    m = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0, 1: 1.0, 0: 0.0}
    return s.map(lambda v: m.get(v, np.nan) if not (isinstance(v, float) and np.isnan(v)) else np.nan).astype(float)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, alpha=ALPHA, method="wilson")
    return float(lo), float(hi)


def newcombe_paired(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Difference mean(x) - mean(y) of paired 0/1 vectors with Newcombe's (1998) hybrid score interval."""
    x = x.astype(bool)
    y = y.astype(bool)
    n = len(x)
    a = int(np.sum(x & y)); b = int(np.sum(x & ~y)); c = int(np.sum(~x & y)); d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    denom = float(a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2, 0.0)
        phi = num / np.sqrt(denom)
    D = p1 - p2
    dl = np.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    du = np.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return D, D - dl, D + du


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    """b = x refuses & y does not, c = y refuses & x does not; exact two-sided binomial test on b of b + c."""
    x = x.astype(bool); y = y.astype(bool)
    b = int(np.sum(x & ~y)); c = int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def holm(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        out[ok] = multipletests(p[ok].values, alpha=ALPHA, method="holm")[1]
    return out


def boot_ci(W: np.ndarray, Y: np.ndarray, mask: np.ndarray, fn) -> tuple[float, float]:
    """Percentile CI of fn(column means) under the item bootstrap given by multinomial weights W (B x n_items)."""
    Wm = W[:, mask].astype(float)
    means = (Wm @ Y[mask]) / Wm.sum(axis=1, keepdims=True)
    stat = fn(means)
    lo, hi = np.nanpercentile(stat, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def lincom(res, names: list[str], weights: dict[str, float]):
    L = np.zeros(len(names))
    for k, w in weights.items():
        L[names.index(k)] = w
    est = float(L @ np.asarray(res.params))
    se = float(np.sqrt(L @ np.asarray(res.cov_params()) @ L))
    z = est / se if se > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if se > 0 else np.nan
    return est, se, est - Z * se, est + Z * se, z, p


def pp(x: float, digits: int = 1, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * x:+.{digits}f}" if sign else f"{100 * x:.{digits}f}"


def fp(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fnum(x: float, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------- data
def load_inputs():
    pq, csv = OUT / "responses_judged.parquet", OUT / "responses_judged.csv.gz"
    cands = [p for p in (pq, csv) if p.exists()]
    if not cands:
        sys.exit("responses_judged.* not found -- run scripts/analysis/assemble.py first")
    src = max(cands, key=lambda p: p.stat().st_mtime)
    cols = ["label", "model", "family", "pair", "study", "generation", "gen_new", "method", "method_class", "role",
            "id", "set", "kind", "adv_overlap", "ref_wg", "ref_kw", "ref_kw256", "ref_q3g", "outcome", "is_degenerate",
            "incomplete"]
    d = pd.read_parquet(src, columns=cols) if src.suffix == ".parquet" else pd.read_csv(src, usecols=cols,
                                                                                        low_memory=False)
    d = d[(d["kind"] == "harmful") & d["role"].isin(USED_ROLES)].copy()
    d["adv_overlap"] = to01(d["adv_overlap"]).fillna(0).astype(bool)
    d["incomplete"] = to01(d["incomplete"]).fillna(0).astype(bool)
    d["is_degenerate"] = to01(d["is_degenerate"]).fillna(0).astype(bool)
    d["y_wg"] = to01(d["ref_wg"])
    d["y_kw"] = to01(d["ref_kw"])
    d["y_kw256"] = to01(d["ref_kw256"])
    d["y_q3g"] = to01(d["ref_q3g"])
    oc = d["outcome"].astype(str)
    d["y_wg_class"] = np.where(oc == "refusal", 1.0, np.where(oc.isin(["compliance", "degenerate"]), 0.0, np.nan))
    d["gen_new"] = d["gen_new"].astype(int)
    rates = pd.read_csv(OUT / "cell_rates.csv", low_memory=False)
    log_head = ""
    lp = OUT / "assemble_log.txt"
    if lp.exists():
        log_head = lp.read_text().splitlines()[0].lstrip("# ").strip()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import freshness  # scripts/analysis/freshness.py (content-based check that the assembly is current)
    global STALE
    STALE = freshness.check()
    if STALE:
        print(freshness.summary_line(STALE), flush=True)
    return d, rates, src, log_head


def build_status(rates: pd.DataFrame, judged_labels: set[str]) -> pd.DataFrame:
    r = rates[rates["set_group"] == "harmful_all"].copy()
    keep = ["label", "model", "pair", "study", "generation", "method", "role", "seed", "status", "source_dir",
            "dev_run", "dev_selected_trial", "dev_refusals", "dev_kl"]
    r = r[keep].drop_duplicates("label")

    def why(row):
        if row["role"] not in USED_ROLES:
            return f"not in H1/H2 (role {row['role']})"
        if row["status"] != "judged":
            return f"pending: {row['status']}"
        if row["label"] not in judged_labels:
            return "judged but absent from responses_judged (re-run assemble.py)"
        return "used"

    r["h1h2_use"] = r.apply(why, axis=1)
    r["pair_order"] = r["pair"].map({p: i for i, p in enumerate(PAIR_ORDER)})
    r["method_order"] = r["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    r = r.sort_values(["pair_order", "method_order", "generation", "label"]).drop(columns=["pair_order", "method_order"])
    return r


# --------------------------------------------------------------------------------------------------------- analysis
class Pair:
    def __init__(self, name: str, meta: pd.DataFrame):
        self.name = name
        self.study = meta["study"].iloc[0]
        self.cells = {}  # (method, gen_new) -> label
        for _, r in meta.iterrows():
            self.cells[(r["method"], int(r["gen_new"]))] = r["label"]
        self.methods_both = [m for m in METHOD_ORDER if (m, 0) in self.cells and (m, 1) in self.cells]
        self.modified_both = [m for m in self.methods_both if m != "original"]
        self.has_orig = "original" in self.methods_both
        self.dir_methods = [m for m in self.modified_both if m != "ara"]
        self.has_ara = "ara" in self.modified_both


def analyse(d: pd.DataFrame, n_boot: int, vb_judges: set[str]):
    meta = d.drop_duplicates("label")[["label", "model", "family", "pair", "study", "generation", "gen_new", "method",
                                       "method_class", "role"]].set_index("label")
    pairs = {p: Pair(p, meta[meta["pair"] == p].reset_index()) for p in PAIR_ORDER if (meta["pair"] == p).any()}
    items_all = np.sort(d["id"].unique())
    adv_ids = set(d.loc[d["adv_overlap"], "id"].unique())
    subset_items = {"all": items_all, "excl_advbench": np.array([i for i in items_all if i not in adv_ids])}
    rng = np.random.default_rng(SEED)

    cell_rows, gen_rows, h2_rows, model_rows, test_rows = [], [], [], [], []
    ctx = d.groupby("label").agg(capped_share=("incomplete", "mean"), n_degenerate=("is_degenerate", "sum"))

    for judge in JUDGES:
        wide_all = d.pivot(index="id", columns="label", values=f"y_{judge}")
        for subset, items in subset_items.items():
            wide = wide_all.loc[items]
            n_items = len(items)
            W = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n_boot)
            tag = dict(judge=judge, subset=subset)

            # ---- cells: residual refusal + modified vs original
            for lab in wide.columns:
                y = wide[lab].values
                ok = ~np.isnan(y)
                k, n = int(np.nansum(y)), int(ok.sum())
                lo, hi = wilson(k, n)
                m = meta.loc[lab]
                row = dict(**tag, label=lab, pair=m["pair"], study=m["study"], generation=m["generation"],
                           method=m["method"], method_class=m["method_class"], n=n, k=k, rate=k / n if n else np.nan,
                           lo=lo, hi=hi, capped_share=float(ctx.loc[lab, "capped_share"]),
                           n_degenerate=int(ctx.loc[lab, "n_degenerate"]))
                P = pairs[m["pair"]]
                olab = P.cells.get(("original", int(m["gen_new"])))
                if m["method"] != "original" and olab is not None and olab in wide.columns:
                    yo = wide[olab].values
                    okp = ok & ~np.isnan(yo)
                    D, dlo, dhi = newcombe_paired(y[okp], yo[okp])
                    b, c, p = mcnemar_exact(yo[okp], y[okp])  # b: original refused, modified did not
                    row.update(orig_label=olab, n_paired=int(okp.sum()), orig_rate=float(np.mean(yo[okp])),
                               diff_vs_orig=D, diff_lo=dlo, diff_hi=dhi, removed_b=b, introduced_c=c, p_mcnemar=p,
                               removal_fraction=1 - float(np.mean(y[okp])) / float(np.mean(yo[okp]))
                               if np.mean(yo[okp]) > 0 else np.nan)
                    test_rows.append(dict(**tag, family="ablation", test_id=f"{lab} vs original", pair=m["pair"],
                                          method=m["method"], scale="risk difference (modified - original)",
                                          estimate=D, lo=dlo, hi=dhi, p=p, hypothesis_direction="negative (removal)",
                                          source="exact McNemar"))
                cell_rows.append(row)

            # ---- generation effects per pair x method
            for pname, P in pairs.items():
                labs = [P.cells[(m, g)] for m in P.methods_both for g in (0, 1)]
                Yp = wide[labs].values
                col = {lab: i for i, lab in enumerate(labs)}
                for m in P.methods_both:
                    lo_lab, ln_lab = P.cells[(m, 0)], P.cells[(m, 1)]
                    yo, yn = wide[lo_lab].values, wide[ln_lab].values
                    okp = ~np.isnan(yo) & ~np.isnan(yn)
                    D, dlo, dhi = newcombe_paired(yn[okp], yo[okp])
                    b, c, p = mcnemar_exact(yo[okp], yn[okp])  # b: old refused, new did not; c: new refused, old not
                    io, inew = col[lo_lab], col[ln_lab]
                    blo, bhi = boot_ci(W, Yp, okp, lambda M, io=io, inew=inew: M[:, inew] - M[:, io])
                    row = dict(**tag, pair=pname, study=P.study, method=m, old_label=lo_lab, new_label=ln_lab,
                               n_paired=int(okp.sum()), p_old=float(yo[okp].mean()), p_new=float(yn[okp].mean()),
                               delta=D, delta_lo=dlo, delta_hi=dhi, delta_boot_lo=blo, delta_boot_hi=bhi,
                               old_only_b=b, new_only_c=c, p_mcnemar=p)
                    if m != "original" and P.has_orig:
                        oo, on = P.cells[("original", 0)], P.cells[("original", 1)]
                        y4 = wide[[lo_lab, ln_lab, oo, on]].values
                        ok4 = ~np.isnan(y4).any(axis=1)
                        do_ = float(np.mean(y4[ok4, 3] - y4[ok4, 2]))
                        did = float(np.mean((y4[ok4, 1] - y4[ok4, 0]) - (y4[ok4, 3] - y4[ok4, 2])))
                        io_, in_ = col[oo], col[on]
                        dlo2, dhi2 = boot_ci(W, Yp, ok4, lambda M, io=io, inew=inew, io_=io_, in_=in_:
                                             (M[:, inew] - M[:, io]) - (M[:, in_] - M[:, io_]))
                        row.update(delta_orig=do_, did_baseline_adj=did, did_lo=dlo2, did_hi=dhi2, n_did=int(ok4.sum()))
                    gen_rows.append(row)
                    if m != "original":
                        test_rows.append(dict(**tag, family="H1", test_id=f"{pname} {m}: new vs old", pair=pname,
                                              method=m, scale="risk difference (new - old)", estimate=D, lo=dlo,
                                              hi=dhi, p=p, hypothesis_direction="positive", source="exact McNemar"))

                # ---- H2 risk-difference contrast per pair
                if P.has_ara and P.dir_methods:
                    ok = ~np.isnan(wide[[P.cells[(m, g)] for m in P.modified_both for g in (0, 1)]].values).any(axis=1)

                    def contrast(M, P=P, col=col):
                        da = M[:, col[P.cells[("ara", 1)]]] - M[:, col[P.cells[("ara", 0)]]]
                        dd = np.mean([M[:, col[P.cells[(m, 1)]]] - M[:, col[P.cells[(m, 0)]]] for m in P.dir_methods],
                                     axis=0)
                        return da - dd

                    pt = contrast(np.nanmean(Yp[ok], axis=0)[None, :])[0]
                    clo, chi = boot_ci(W, Yp, ok, contrast)
                    means = np.nanmean(Yp[ok], axis=0)
                    d_ara = means[col[P.cells[("ara", 1)]]] - means[col[P.cells[("ara", 0)]]]
                    d_dir = {m: means[col[P.cells[(m, 1)]]] - means[col[P.cells[(m, 0)]]] for m in P.dir_methods}
                    h2_rows.append(dict(**tag, pair=pname, study=P.study, direction_methods="+".join(P.dir_methods),
                                        n_items=int(ok.sum()), delta_ara=d_ara,
                                        delta_direction_mean=float(np.mean(list(d_dir.values()))),
                                        **{f"delta_{m}": v for m, v in d_dir.items()},
                                        rd_contrast=pt, rd_lo=clo, rd_hi=chi))

            # ---- models
            long = d[d["id"].isin(items)].copy()
            long["y"] = long[f"y_{judge}"]
            long = long[long["y"].notna()]
            mr, tr = fit_models(long, pairs, judge in vb_judges, tag)
            model_rows += mr
            test_rows += tr
        print(f"  judge {judge}: done ({time.strftime('%H:%M:%S')})", flush=True)

    cells = pd.DataFrame(cell_rows)
    gen = pd.DataFrame(gen_rows)
    h2 = pd.DataFrame(h2_rows)
    models = pd.DataFrame(model_rows)
    tests = pd.DataFrame(test_rows)
    tests["p_holm"] = tests.groupby(["judge", "subset", "family"])["p"].transform(holm)
    tests["reject_holm"] = tests["p_holm"] < ALPHA
    tests["n_in_family"] = tests.groupby(["judge", "subset", "family"])["p"].transform("count")

    def direction(r):
        if np.isnan(r["estimate"]):
            return ""
        pos = r["estimate"] > 0
        if r["hypothesis_direction"].startswith("positive"):
            return "as hypothesised" if pos else "opposite"
        return "as hypothesised" if not pos else "opposite"

    tests["observed_direction"] = tests.apply(direction, axis=1)
    # attach Holm results to the generation-effect, cell and H2 tables
    h1t = tests[tests["family"] == "H1"].set_index(["judge", "subset", "pair", "method"])
    if len(gen):
        key = pd.MultiIndex.from_frame(gen[["judge", "subset", "pair", "method"]])
        gen["p_holm_H1"] = h1t["p_holm"].reindex(key).values
        gen["n_H1_family"] = h1t["n_in_family"].reindex(key).values
    abt = tests[tests["family"] == "ablation"].copy()
    abt["label"] = abt["test_id"].str.replace(" vs original", "", regex=False)
    abt = abt.set_index(["judge", "subset", "label"])
    key = pd.MultiIndex.from_frame(cells[["judge", "subset", "label"]])
    cells["p_holm_ablation"] = abt["p_holm"].reindex(key).values
    if len(h2):
        h2t = tests[(tests["family"] == "H2")].set_index(["judge", "subset", "pair"])
        key = pd.MultiIndex.from_frame(h2[["judge", "subset", "pair"]])
        for c_src, c_dst in [("estimate", "gee_interaction"), ("lo", "gee_lo"), ("hi", "gee_hi"), ("p", "gee_p"),
                             ("p_holm", "p_holm_H2")]:
            h2[c_dst] = h2t[c_src].reindex(key).values
        mi = models[(models["model"] == "gee_method_intercepts") & (models["term"] == "gen_new:opt")]
        mi = mi.assign(pair=mi["scope"].str.replace("pair:", "", regex=False)).set_index(["judge", "subset", "pair"])
        for c_src, c_dst in [("estimate", "gee_mi_interaction"), ("lo", "gee_mi_lo"), ("hi", "gee_mi_hi"),
                             ("p", "gee_mi_p")]:
            h2[c_dst] = mi[c_src].reindex(key).values
        vb = models[(models["model"] == "vb_planned") & (models["term"] == "gen_new:opt")]
        if len(vb):
            vb = vb.assign(pair=vb["scope"].str.replace("pair:", "", regex=False)).set_index(["judge", "subset", "pair"])
            for c_src, c_dst in [("estimate", "vb_interaction"), ("lo", "vb_lo"), ("hi", "vb_hi")]:
                h2[c_dst] = vb[c_src].reindex(key).values
    return cells, gen, h2, models, tests, pairs


def fit_models(long: pd.DataFrame, pairs: dict, do_vb: bool, tag: dict):
    """Planned class model (VB + GEE), per-method / with-originals / heterogeneity GEEs; per scope."""
    rows, tests = [], []
    # balanced cell sets: per pair, methods judged in both generations
    keep_mod, keep_all = [], []
    for P in pairs.values():
        for m in P.modified_both:
            keep_mod += [P.cells[(m, 0)], P.cells[(m, 1)]]
        for m in P.methods_both:
            keep_all += [P.cells[(m, 0)], P.cells[(m, 1)]]
    mod = long[long["label"].isin(keep_mod)].copy()
    mod["opt"] = (mod["method_class"] == "optimisation").astype(int)
    mod["pm"] = mod["pair"] + ":" + mod["method"]
    allc = long[long["label"].isin(keep_all)].copy()
    allc["cls3"] = allc["method_class"]
    main_pairs = [p for p, P in pairs.items() if P.study == "main" and P.has_ara and P.dir_methods]
    small_pairs = [p for p, P in pairs.items() if P.study == "small" and P.has_ara and P.dir_methods]

    scopes = []
    if main_pairs:
        scopes.append(("main", mod[mod["pair"].isin(main_pairs)], "family", main_pairs))
    if main_pairs and small_pairs:
        scopes.append(("main+small", mod[mod["pair"].isin(main_pairs + small_pairs)], "pair", main_pairs + small_pairs))
    for p in main_pairs + small_pairs:
        scopes.append((f"pair:{p}", mod[mod["pair"] == p], None, [p]))

    def base(scope, model, df, grp, plist):
        return dict(**tag, scope=scope, model=model, n_obs=len(df), n_prompts=df["id"].nunique(),
                    n_groups=df[grp].nunique() if grp else 1, pairs_included="+".join(plist))

    for scope, df, grp, plist in scopes:
        multi = grp is not None and df[grp].nunique() > 1
        fe = f" + C({grp})" if multi else ""
        # ---- planned model, VB
        if do_vb:
            vc = {"prompt": "0 + C(id)"}
            if multi:
                vc[grp] = f"0 + C({grp})"
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                mdl = BinomialBayesMixedGLM.from_formula("y ~ gen_new * opt", vc, df)
                res = mdl.fit_vb()
            note = "; ".join(sorted({str(w.message)[:120] for w in wlist}))
            names = list(mdl.exog_names)
            fm, fs = np.asarray(res.fe_mean), np.asarray(res.fe_sd)
            for i, nme in enumerate(names):
                rows.append(dict(**base(scope, "vb_planned", df, grp, plist), term=nme, estimate=fm[i], se=fs[i],
                                 lo=fm[i] - Z * fs[i], hi=fm[i] + Z * fs[i], z=fm[i] / fs[i], p=np.nan,
                                 note="posterior mean/SD (VB); interval = mean +/- 1.96 SD; " + note))
            i_g, i_x = names.index("gen_new"), names.index("gen_new:opt")
            est, sd = fm[i_g] + fm[i_x], float(np.sqrt(fs[i_g] ** 2 + fs[i_x] ** 2))
            rows.append(dict(**base(scope, "vb_planned", df, grp, plist), term="gen effect | optimisation (derived)",
                             estimate=est, se=sd, lo=est - Z * sd, hi=est + Z * sd, z=est / sd, p=np.nan,
                             note="gen_new + gen_new:opt; SD assumes mean-field independence"))
            rows.append(dict(**base(scope, "vb_planned", df, grp, plist), term="gen effect | direction (= gen_new)",
                             estimate=fm[i_g], se=fs[i_g], lo=fm[i_g] - Z * fs[i_g], hi=fm[i_g] + Z * fs[i_g],
                             z=fm[i_g] / fs[i_g], p=np.nan, note="alias of gen_new"))
            vnames = list(getattr(mdl, "vcp_names", vc.keys()))
            vm, vs = np.asarray(res.vcp_mean), np.asarray(res.vcp_sd)
            for i, nme in enumerate(vnames):
                rows.append(dict(**base(scope, "vb_planned", df, grp, plist), term=f"SD({nme}) random intercept",
                                 estimate=float(np.exp(vm[i])), se=np.nan, lo=float(np.exp(vm[i] - Z * vs[i])),
                                 hi=float(np.exp(vm[i] + Z * vs[i])), z=np.nan, p=np.nan,
                                 note="exp(posterior of log SD)" + ("; only 2 levels: prior-dominated"
                                                                   if nme == grp and df[grp].nunique() <= 2 else "")))

        # ---- planned model, GEE robustness
        res, names, note = gee(df, "y ~ gen_new * opt" + fe)
        b = base(scope, "gee_planned", df, grp, plist)
        for nme in names:
            rows.append(dict(**b, term=nme, **_lc(res, names, {nme: 1}), note=note))
        e_dir = _lc(res, names, {"gen_new": 1})
        e_opt = _lc(res, names, {"gen_new": 1, "gen_new:opt": 1})
        e_int = _lc(res, names, {"gen_new:opt": 1})
        rows.append(dict(**b, term="gen effect | direction (= gen_new)", **e_dir, note=note))
        rows.append(dict(**b, term="gen effect | optimisation (derived)", **e_opt, note=note))
        # robustness: one intercept per pair x method (method-specific baselines), class-level generation effects
        res2, names2, note2 = gee(df, "y ~ C(pm) + gen_new + gen_new:opt")
        b2 = base(scope, "gee_method_intercepts", df, grp, plist)
        for term, w in (("gen effect | direction (= gen_new)", {"gen_new": 1}),
                        ("gen effect | optimisation (derived)", {"gen_new": 1, "gen_new:opt": 1}),
                        ("gen_new:opt", {"gen_new:opt": 1})):
            rows.append(dict(**b2, term=term, **_lc(res2, names2, w), note=note2))
        if scope == "main":
            for t_id, e in (("pooled main: gen effect | direction class", e_dir),
                            ("pooled main: gen effect | optimisation class", e_opt)):
                tests.append(dict(**tag, family="H1", test_id=t_id, pair="main (pooled)",
                                  method=t_id.split("| ")[1], scale="log-odds (GEE, population-averaged)",
                                  estimate=e["estimate"], lo=e["lo"], hi=e["hi"], p=e["p"],
                                  hypothesis_direction="positive", source="GEE Wald"))
            tests.append(dict(**tag, family="H2", test_id="pooled main: generation x class", pair="main (pooled)",
                              method="ara vs direction", scale="log-odds (GEE)", estimate=e_int["estimate"],
                              lo=e_int["lo"], hi=e_int["hi"], p=e_int["p"], hypothesis_direction="negative",
                              source="GEE Wald"))
        elif scope.startswith("pair:"):
            p = scope.split(":", 1)[1]
            tests.append(dict(**tag, family="H2", test_id=f"{p}: generation x class", pair=p,
                              method="ara vs direction", scale="log-odds (GEE)", estimate=e_int["estimate"],
                              lo=e_int["lo"], hi=e_int["hi"], p=e_int["p"], hypothesis_direction="negative",
                              source="GEE Wald"))

        if scope.startswith("pair:"):
            continue
        # ---- per-method generation effects (GEE)
        res, names, note = gee(df, "y ~ C(method) + C(method):gen_new" + fe)
        b = base(scope, "gee_per_method", df, grp, plist)
        for nme in names:
            if nme.endswith(":gen_new"):
                m = nme.split("[")[1].split("]")[0].replace("T.", "")
                rows.append(dict(**b, term=f"gen effect | {m}", **_lc(res, names, {nme: 1}), note=note))
        # ---- with originals as reference class: baseline-adjusted generation effects (GEE)
        da = allc[allc["pair"].isin(plist)].copy()
        if (da["method_class"] == "original").any():
            res, names, note = gee(da, "y ~ C(cls3, Treatment('original')) * gen_new" + fe)
            b = base(scope, "gee_with_originals", da, grp, plist)
            for nme in names:
                if "gen_new" in nme:
                    lab = nme.replace("C(cls3, Treatment('original'))", "class")
                    if nme == "gen_new":
                        lab = "gen effect | original (baseline difference)"
                    else:
                        lab = "baseline-adjusted gen effect | " + nme.split("[T.")[1].split("]")[0]
                    rows.append(dict(**b, term=lab, **_lc(res, names, {nme: 1}), note=note))
        # ---- heterogeneity across families/pairs (GEE)
        if multi:
            res, names, note = gee(df, f"y ~ gen_new * opt * C({grp})")
            b = base(scope, "gee_heterogeneity", df, grp, plist)
            for label, sel in (("generation x family (direction class)",
                                lambda n: "gen_new" in n and f"C({grp})" in n and "opt" not in n),
                               ("generation x class x family (H2 heterogeneity)",
                                lambda n: "gen_new" in n and "opt" in n and f"C({grp})" in n)):
                idx = [i for i, n in enumerate(names) if sel(n)]
                R = np.zeros((len(idx), len(names)))
                for r_i, c_i in enumerate(idx):
                    R[r_i, c_i] = 1
                wt = res.wald_test(R, scalar=True)
                rows.append(dict(**b, term=label.replace("family", grp), estimate=float(np.squeeze(wt.statistic)),
                                 se=np.nan, lo=np.nan, hi=np.nan, z=np.nan, p=float(np.squeeze(wt.pvalue)),
                                 note=f"Wald chi2, df={len(idx)}; {note}"))
    return rows, tests


def _lc(res, names, w):
    est, se, lo, hi, z, p = lincom(res, names, w)
    return dict(estimate=est, se=se, lo=lo, hi=hi, z=z, p=p)


def gee(df, formula):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        mdl = smf.gee(formula, groups="id", data=df, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable())
        res = mdl.fit(maxiter=200)
    note = "; ".join(sorted({str(w.message)[:120] for w in wlist}))
    rho = res.model.cov_struct.dep_params
    note = f"exchangeable rho={float(rho):.3f}" + (f"; {note}" if note else "")
    return res, list(res.params.index), note


# ---------------------------------------------------------------------------------------------------------- summary
def write_summary(d, status, cells, gen, h2, models, tests, pairs, src, log_head, runtime, vb_judges, n_boot):
    J0, S0 = PRIMARY
    L = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    main_pending = [p for p in MAIN_PAIRS_PLANNED if not (p in pairs and pairs[p].modified_both)]
    L.append("# H1 and H2 on harmful held-out prompts (paper 33) -- " + (
        "all four main pairs judged" if not main_pending else "interim analysis"))
    L.append("")
    L.append(f"Generated by `scripts/analysis/h1h2.py` at {now} ({runtime:.0f} s) from "
             f"`{src.relative_to(ROOT)}` (md5 `{md5(src)[:12]}`; assembly run: {log_head}). Regenerated on every run; "
             "do not edit by hand.")
    L.append("")
    L.append(f"Software: Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, "
             f"scipy {scipy.__version__}, statsmodels {statsmodels.__version__}. **R is not installed on this machine**, "
             "so the planned lme4 model is fitted with statsmodels `BinomialBayesMixedGLM` (variational Bayes) and "
             "checked with GEE. The Methods todo \"name statistics software\" should say statsmodels, not lme4.")
    L.append("")

    # ---- status
    import freshness  # noqa: E402 (path added in load_inputs)
    L.append(freshness.summary_line(STALE))
    L.append("")
    main_judged = [p for p in MAIN_PAIRS_PLANNED if p in pairs and pairs[p].modified_both]
    main_pending = [p for p in MAIN_PAIRS_PLANNED if p not in main_judged]
    st_used = status[status["role"].isin(USED_ROLES)]
    L.append("## Status of the data")
    L.append("")
    L.append(f"- Main-study pairs judged: **{', '.join(main_judged) or 'none'}** ({len(main_judged)} of 4). "
             f"Pending: **{', '.join(main_pending) or 'none'}**.")
    for p in PAIR_ORDER:
        sp = st_used[st_used["pair"] == p]
        if len(sp) == 0:
            continue
        nj = int((sp["h1h2_use"] == "used").sum())
        pend = sp[sp["h1h2_use"] != "used"]
        txt = f"- {PAIR_DISPLAY[p]}: {nj}/{len(sp)} original+primary cells judged"
        if len(pend):
            txt += "; pending: " + ", ".join(f"`{r.label}` ({r.status})" for r in pend.itertuples())
        L.append(txt)
    other = status[~status["role"].isin(USED_ROLES)]
    L.append(f"- Not part of H1/H2 (by design): {len(other)} cells "
             f"(gamma-MPOA control {int((other.role == 'control').sum())}, community {int((other.role == 'external').sum())}, "
             f"seed-2 replicates {int((other.role == 'replicate').sum())}); judged so far: "
             f"{int((other.status == 'judged').sum())}.")
    if main_pending:
        L.append("")
        L.append(f"**The planned pooled model `refusal ~ generation x method_class + (1|prompt) + (1|family)` awaits "
                 f"{' and '.join({'olmo': 'OLMo'}.get(x, x.capitalize()) for x in main_pending)}.** With "
                 f"{len(main_judged)} judged families the family random intercept has {len(main_judged)} levels and "
                 "its variance is determined mainly by the prior, and a pooled estimate averages over families whose "
                 "effects may point in opposite directions (see the heterogeneity tests). Until the remaining pairs are "
                 "judged, the per-family estimates below are the evidence; pooled numbers are interim. Re-running "
                 "`assemble.py` and then this script picks the new judgments up automatically.")
    L.append("")

    # ---- definitions
    gm = gen[gen.method != "original"]
    bootdiff = 100 * float(np.nanmax(np.abs(np.r_[gm.delta_lo - gm.delta_boot_lo, gm.delta_hi - gm.delta_boot_hi])))
    L.append("## Definitions")
    L.append("")
    miss = d[d["y_wg"].isna()]
    used_kl = status[(status["h1h2_use"] == "used") & (status["role"] == "primary")]["dev_kl"].astype(float)
    L.append("- **Residual refusal**: share of the harmful held-out items (972; 928 without the 44 AdvBench-overlap "
             "items) that the response refuses. Primary judge: WildGuard refusal label (`ref_wg`; rows without a label "
             f"dropped: {len(miss)} harmful row(s){': ' if len(miss) else ''}"
             f"{', '.join(f'`{r.label}/{r.id}`' for r in miss.itertuples())}). Degenerate responses keep their "
             "WildGuard label in the primary outcome; the four-class outcome rule (degenerate first) is the `wg_class` "
             f"sensitivity. Every modified model is the selected trial at first-token KL <= 0.1 on the dev prompts. "
             f"\"Matched distribution shift\" therefore means a common KL cap, not equal KL: selected dev KL of the "
             f"analysed cells {used_kl.min():.3f}-{used_kl.max():.3f}, and it differs within pairs (column `Dev KL` "
             "below).")
    L.append("- **Generation effect** of a method in a pair = residual refusal of the newer model's modified version "
             "minus the older model's, on the same items (percentage points, pp). 95% CI: Newcombe hybrid score "
             "interval for paired proportions (the item-bootstrap percentile limits `delta_boot_*` differ from it by at "
             f"most {bootdiff:.1f} pp). Test: exact McNemar on the items where exactly one of the two models refuses. "
             "**Baseline-adjusted** = generation effect of the method minus that of the originals (bootstrap CI).")
    L.append("- **H2 contrast** = generation effect of ARA minus the mean generation effect of the direction-based "
             "methods (risk difference, bootstrap CI), and the generation x class interaction of the planned model "
             "(log-odds). H2 predicts a negative contrast/interaction.")
    L.append(f"- **Holm families** (alpha = {ALPHA}, two-sided, separately per judge and item subset): H1 = McNemar "
             "new vs old for every judged pair x method (main and small pair) + the two pooled main-study GEE "
             "generation effects (direction class, ARA); H2 = per-pair GEE interaction + pooled main-study GEE "
             "interaction; ablation (descriptive) = McNemar modified vs original per model. VB mixed-model results "
             "have no p-values and are not Holm-adjusted.")
    L.append(f"- Item bootstrap: {n_boot} multinomial resamples of items, seed {SEED}.")
    L.append("")

    # ---- H1 table, primary
    g0 = gen[(gen.judge == J0) & (gen.subset == S0)]
    L.append("## H1: generation effect per method (primary: WildGuard, 972 harmful items)")
    L.append("")
    stx = status.drop_duplicates("label").set_index("label")
    kl = stx["dev_kl"].astype(float)
    devref = stx["dev_refusals"].astype(float)
    cab = cells[(cells.judge == J0) & (cells.subset == S0) & (cells.method != "original")].set_index("label")
    failed = set(cab.index[cab.diff_vs_orig >= 0])  # held-out residual refusal at or above the original's
    mark = lambda lab, txt: txt + (" (edit failed)" if lab in failed else "")  # noqa: E731
    tbl = []
    for p in PAIR_ORDER:
        gp = g0[g0.pair == p]
        for m in METHOD_ORDER:
            r = gp[gp.method == m]
            if not len(r):
                continue
            r = r.iloc[0]
            tbl.append({"Pair": p, "Method": METHOD_DISPLAY[m], "Old %": mark(r.old_label, pp(r.p_old, sign=False)),
                        "New %": mark(r.new_label, pp(r.p_new, sign=False)),
                        "Dev KL old / new": ("" if m == "original" else
                                             f"{kl.get(r.old_label, np.nan):.3f} / {kl.get(r.new_label, np.nan):.3f}"),
                        "New - old, pp [95% CI]": f"{pp(r.delta)} [{pp(r.delta_lo)}, {pp(r.delta_hi)}]",
                        "b / c": f"{r.old_only_b} / {r.new_only_c}", "p": fp(r.p_mcnemar),
                        "p Holm": fp(r.get("p_holm_H1", np.nan)) if m != "original" else "(baseline)",
                        "Baseline-adj., pp [95% CI]": (f"{pp(r.did_baseline_adj)} [{pp(r.did_lo)}, {pp(r.did_hi)}]"
                                                       if m != "original" and "did_baseline_adj" in r and
                                                       not pd.isna(r.did_baseline_adj) else "")})
    L.append(md_table(pd.DataFrame(tbl)))
    L.append("")
    L.append("b = items refused only by the older model's version, c = only by the newer one. The `Original` rows give "
             "the generation difference of the unmodified models (not a test of H1). Dev KL = first-token KL of the "
             "selected trial on the dev prompts (all <= 0.1 by the selection rule). \"(edit failed)\" = held-out "
             "residual refusal at or above the model's original, i.e. the selected trial did not remove refusal.")
    L.append("")
    t0 = tests[(tests.judge == J0) & (tests.subset == S0)]
    h1 = t0[t0.family == "H1"]
    tx = tests[(tests.judge == J0) & (tests.subset == "excl_advbench") & (tests.family == "H1")].set_index("test_id")

    def robust(r):  # caveat for a Holm-significant H1 test whose decision or sign changes without the 44 items
        if r.test_id not in tx.index:
            return ""
        x = tx.loc[r.test_id]
        if bool(x.reject_holm) == bool(r.reject_holm) and np.sign(x.estimate) == np.sign(r.estimate):
            return ""
        return (f" [not robust to dropping the 44 AdvBench-overlap items: {pp(x.estimate)} pp, p_Holm "
                f"{fp(x.p_holm)}]")
    L.append("**Verdict per pair (Holm-adjusted, H1 family of "
             f"{int(h1.n_in_family.iloc[0]) if len(h1) else 0} tests):**")
    L.append("")
    for p in PAIR_ORDER:
        hp = h1[h1.pair == p]
        if not len(hp):
            continue
        up = hp[(hp.reject_holm) & (hp.estimate > 0)]
        dn = hp[(hp.reject_holm) & (hp.estimate < 0)]
        ns = hp[~hp.reject_holm]
        verdict = ("supported (every method significantly higher on the newer model)" if len(up) == len(hp)
                   else "not supported for every method")
        fmt = lambda s: ", ".join(f"{METHOD_DISPLAY[r.method]} {pp(r.estimate)} pp{robust(r)}"  # noqa: E731
                                  for r in s.itertuples()) or "none"
        gp = g0[(g0.pair == p) & (g0.method != "original")]
        adj = ""
        if "did_baseline_adj" in gp and gp.did_baseline_adj.notna().any():
            pos = gp[gp.did_lo > 0]; neg = gp[gp.did_hi < 0]
            adj = (" Baseline-adjusted (95% CI excludes 0): higher on the newer model for "
                   f"{', '.join(METHOD_DISPLAY[m] for m in pos.method) or 'none'}; lower for "
                   f"{', '.join(METHOD_DISPLAY[m] for m in neg.method) or 'none'}.")
        fl = []
        for r in gp.itertuples():
            for lab, which in ((r.old_label, "older"), (r.new_label, "newer")):
                if lab in failed:
                    fl.append(f" {METHOD_DISPLAY[r.method]}: the {which} model's edit failed (selected dev trial "
                              f"{devref.get(lab, np.nan):.0f}/100 keyword refusals at KL {kl.get(lab, np.nan):.3f}; "
                              f"held-out {pp(cab.loc[lab, 'rate'], sign=False)}% vs original "
                              f"{pp(cab.loc[lab, 'orig_rate'], sign=False)}%), so its generation effect reflects that "
                              "no trial at KL <= 0.1 removed refusal on that model, not residual refusal after a "
                              "successful ablation.")
        L.append(f"- {PAIR_DISPLAY[p]}: H1 {verdict}. Newer higher: {fmt(up)}; newer lower: {fmt(dn)}; "
                 f"not significant: {fmt(ns)}.{adj}{''.join(fl)}")
    gb = g0[g0.method == "original"]
    if len(gb):
        L.append("")
        L.append("Baseline differences of the originals: " + "; ".join(
            f"{r.pair} {pp(r.p_old, sign=False)}% -> {pp(r.p_new, sign=False)}% ({pp(r.delta)} pp, p {fp(r.p_mcnemar)})"
            for r in gb.itertuples()) + ". Where the newer original already refuses more, part of a positive "
                                        "generation effect after ablation is inherited from the original; the "
                                        "baseline-adjusted column removes it on the risk-difference scale.")
    L.append("")
    pm = models[(models.judge == J0) & (models.subset == S0) & (models.scope == "main")]
    if len(pm):
        L.append("**Pooled main-study estimates (" + ("interim, " if main_pending else "") + "+".join(main_judged)
                 + "), log-odds:**")
        L.append("")
        rows = []
        for mdl, lab in (("vb_planned", "VB mixed model (planned)"), ("gee_planned", "GEE (robustness)")):
            for term in ("gen effect | direction (= gen_new)", "gen effect | optimisation (derived)", "gen_new:opt"):
                r = pm[(pm.model == mdl) & (pm.term == term)]
                if not len(r):
                    continue
                r = r.iloc[0]
                tt = t0[(t0.source == "GEE Wald") & (t0.pair == "main (pooled)")]
                ph = ""
                if mdl == "gee_planned":
                    key = {"gen effect | direction (= gen_new)": "pooled main: gen effect | direction class",
                           "gen effect | optimisation (derived)": "pooled main: gen effect | optimisation class",
                           "gen_new:opt": "pooled main: generation x class"}[term]
                    q = tt[tt.test_id == key]
                    ph = fp(q.p_holm.iloc[0]) if len(q) else ""
                rows.append({"Model": lab, "Term": term.replace(" (= gen_new)", "").replace(" (derived)", "")
                            .replace("gen_new:opt", "generation x class (H2)"),
                             "Estimate [95% interval]": f"{fnum(r.estimate)} [{fnum(r.lo)}, {fnum(r.hi)}]",
                             "p": fp(r.p) if mdl == "gee_planned" else "(VB)", "p Holm": ph or "--"})
        L.append(md_table(pd.DataFrame(rows)))
        het = pm[pm.model == "gee_heterogeneity"]
        if len(het):
            L.append("")
            L.append("Heterogeneity across families (GEE Wald): " + "; ".join(
                f"{r.term}: chi2 = {r.estimate:.1f}, p {fp(r.p)}" for r in het.itertuples()) +
                     ". A significant heterogeneity test means the pooled coefficient averages effects of different "
                     "sign or size and should not be read as a common effect.")
        pmeth = pm[pm.model == "gee_per_method"]
        if len(pmeth):
            L.append("")
            L.append("Pooled per-method generation effects (GEE, log-odds): " + "; ".join(
                f"{METHOD_DISPLAY.get(r.term.split('| ')[1], r.term)} {fnum(r.estimate)} [{fnum(r.lo)}, {fnum(r.hi)}], "
                f"p {fp(r.p)}" for r in pmeth.itertuples()) + " (not Holm-adjusted; descriptive).")
        pwo = pm[pm.model == "gee_with_originals"]
        if len(pwo):
            L.append("")
            L.append("With the originals as reference class (GEE, log-odds): " + "; ".join(
                f"{r.term} {fnum(r.estimate)} [{fnum(r.lo)}, {fnum(r.hi)}], p {fp(r.p)}" for r in pwo.itertuples()) + ".")
        L.append("")

    # ---- H2
    L.append("## H2: generation x method class (primary: WildGuard, 972 harmful items)")
    L.append("")
    h20 = h2[(h2.judge == J0) & (h2.subset == S0)]
    rows = []
    for r in h20.itertuples():
        rows.append({"Pair": r.pair, "ARA gen. effect, pp": pp(r.delta_ara),
                     "Direction-based mean, pp": f"{pp(r.delta_direction_mean)} ({r.direction_methods})",
                     "RD contrast, pp [95% CI]": f"{pp(r.rd_contrast)} [{pp(r.rd_lo)}, {pp(r.rd_hi)}]",
                     "GEE interaction, log-odds [95% CI]": f"{fnum(r.gee_interaction)} [{fnum(r.gee_lo)}, {fnum(r.gee_hi)}]",
                     "p": fp(r.gee_p), "p Holm": fp(r.p_holm_H2),
                     "VB interaction [95% CrI]": (f"{fnum(r.vb_interaction)} [{fnum(r.vb_lo)}, {fnum(r.vb_hi)}]"
                                                  if hasattr(r, "vb_interaction") and not pd.isna(r.vb_interaction)
                                                  else "--"),
                     "GEE, method intercepts": f"{fnum(r.gee_mi_interaction)} [{fnum(r.gee_mi_lo)}, {fnum(r.gee_mi_hi)}]",
                     "Direction vs H2 (RD / log-odds)": ("H2" if r.rd_contrast < 0 else "opposite") + " / " +
                                                        ("H2" if r.gee_interaction < 0 else "opposite")})
    tp = t0[(t0.family == "H2") & (t0.pair == "main (pooled)")]
    vbm = pm[(pm.model == "vb_planned") & (pm.term == "gen_new:opt")] if len(pm) else pm
    mim = pm[(pm.model == "gee_method_intercepts") & (pm.term == "gen_new:opt")] if len(pm) else pm
    if len(tp):
        r = tp.iloc[0]
        rows.append({"Pair": "main pooled" + (" (interim)" if main_pending else ""), "ARA gen. effect, pp": "", "Direction-based mean, pp": "",
                     "RD contrast, pp [95% CI]": "",
                     "GEE interaction, log-odds [95% CI]": f"{fnum(r.estimate)} [{fnum(r.lo)}, {fnum(r.hi)}]",
                     "p": fp(r.p), "p Holm": fp(r.p_holm),
                     "VB interaction [95% CrI]": (f"{fnum(vbm.estimate.iloc[0])} [{fnum(vbm.lo.iloc[0])}, "
                                                  f"{fnum(vbm.hi.iloc[0])}]" if len(vbm) else "--"),
                     "GEE, method intercepts": (f"{fnum(mim.estimate.iloc[0])} [{fnum(mim.lo.iloc[0])}, "
                                                f"{fnum(mim.hi.iloc[0])}]" if len(mim) else "--"),
                     "Direction vs H2 (RD / log-odds)": "-- / " + ("H2" if r.estimate < 0 else "opposite")})
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("Negative values = ARA's residual refusal rises less (or falls more) from the older to the newer model "
             "than that of the direction-based methods, as H2 predicts. RD = risk-difference contrast; log-odds = "
             "planned-model interaction (GEE, population-averaged; the VB estimate is prompt-conditional and usually larger in "
             "magnitude). \"GEE, method "
             "intercepts\" adds one intercept per pair x method, so direction-based methods with different baseline "
             "rates are not pooled into one class rate. The scales can disagree in size and even in sign because "
             "ARA's rates sit near 0 in some cells, where small risk differences are large on the logit scale.")
    scale_flip = h20[np.sign(h20.rd_contrast) != np.sign(h20.gee_interaction)]
    if len(scale_flip):
        L.append("")
        L.append("**Scale-dependent H2 direction:** " + "; ".join(
            f"{r.pair}: RD contrast {pp(r.rd_contrast)} pp [{pp(r.rd_lo)}, {pp(r.rd_hi)}] but log-odds interaction "
            f"{fnum(r.gee_interaction)} (method intercepts {fnum(r.gee_mi_interaction)})" for r in scale_flip.itertuples())
                 + ". For these pairs H2 cannot be called supported or contradicted without fixing the scale in advance.")
    L.append("")

    # ---- judge sensitivity
    L.append("## How the conclusions change with the judge (keyword markers and Qwen3Guard vs WildGuard)")
    L.append("")
    rows = []
    gk = gen[(gen.subset == S0)]
    for p in PAIR_ORDER:
        for m in METHOD_ORDER:
            sel = gk[(gk.pair == p) & (gk.method == m)]
            if not len(sel):
                continue
            rr = {"Pair": p, "Method": METHOD_DISPLAY[m]}
            for j, lab in (("wg", "WildGuard"), ("q3g", "Qwen3Guard"), ("kw", "Keyword 512"), ("kw256", "Keyword 256")):
                r = sel[sel.judge == j]
                if not len(r):
                    rr[lab] = "--"
                    continue
                r = r.iloc[0]
                star = "" if m == "original" else ("*" if r.p_holm_H1 < ALPHA else "")
                rr[lab] = (f"{pp(r.p_old, sign=False)} -> {pp(r.p_new, sign=False)}: {pp(r.delta)} "
                           f"[{pp(r.delta_lo)}, {pp(r.delta_hi)}]{star}")
            rows.append(rr)
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("Residual refusal old -> new (%), generation effect in pp [95% CI]; * = Holm-significant in that judge's "
             "H1 family.")
    L.append("")
    changes = []
    for j in ("kw", "kw256", "q3g", "wg_class"):
        a = tests[(tests.judge == J0) & (tests.subset == S0) & tests.family.isin(["H1", "H2"])].set_index("test_id")
        b = tests[(tests.judge == j) & (tests.subset == S0) & tests.family.isin(["H1", "H2"])].set_index("test_id")
        for tid in a.index.intersection(b.index):
            ra, rb = a.loc[tid], b.loc[tid]
            ca = ("sig. " if ra.reject_holm else "n.s. ") + ("+" if ra.estimate > 0 else "-")
            cb = ("sig. " if rb.reject_holm else "n.s. ") + ("+" if rb.estimate > 0 else "-")
            if ca != cb and (ra.reject_holm or rb.reject_holm):
                unit = "pp" if "risk" in ra.scale else "log-odds"
                ea = pp(ra.estimate) if unit == "pp" else fnum(ra.estimate)
                eb = pp(rb.estimate) if unit == "pp" else fnum(rb.estimate)
                changes.append(f"- {JUDGES[j]}: `{tid}` ({ra.family}): WildGuard {ca} ({ea} {unit}, p_Holm "
                               f"{fp(ra.p_holm)}) -> {cb} ({eb} {unit}, p_Holm {fp(rb.p_holm)})")
    L.append("**Holm decisions that change relative to WildGuard (all 972 items):**")
    L.append("")
    L += changes or ["- none"]
    L.append("")
    hk = h2[(h2.subset == S0) & (h2.judge.isin(["kw", "kw256", "q3g"]))]
    if len(hk):
        L.append("H2 under keyword and Qwen3Guard scoring: " + "; ".join(
            f"{r.pair} ({r.judge}) RD contrast {pp(r.rd_contrast)} pp [{pp(r.rd_lo)}, {pp(r.rd_hi)}], GEE interaction "
            f"{fnum(r.gee_interaction)} (p_Holm {fp(r.p_holm_H2)})" for r in hk.itertuples()) + ".")
        L.append("")

    # ---- AdvBench + degenerate sensitivity
    L.append("## Sensitivity: AdvBench overlap and degenerate responses")
    L.append("")
    for j in ("wg", "kw"):
        a = gen[(gen.judge == j) & (gen.subset == "all") & (gen.method != "original")].set_index(["pair", "method"])
        b = gen[(gen.judge == j) & (gen.subset == "excl_advbench") & (gen.method != "original")].set_index(
            ["pair", "method"])
        diff = (b.delta - a.delta).abs()
        ta = tests[(tests.judge == j) & (tests.subset == "all") & tests.family.isin(["H1", "H2"])].set_index("test_id")
        tb = tests[(tests.judge == j) & (tests.subset == "excl_advbench") & tests.family.isin(["H1", "H2"])].set_index(
            "test_id")
        flips = [t for t in ta.index.intersection(tb.index)
                 if (ta.loc[t].reject_holm != tb.loc[t].reject_holm) or
                 (np.sign(ta.loc[t].estimate) != np.sign(tb.loc[t].estimate))]

        def show(t, ta=ta, tb=tb):
            f = (lambda x: f"{pp(x)} pp") if "risk" in ta.loc[t].scale else (lambda x: f"{fnum(x)} log-odds")
            return (f"`{t}` {f(ta.loc[t].estimate)}, p_Holm {fp(ta.loc[t].p_holm)} -> {f(tb.loc[t].estimate)}, "
                    f"p_Holm {fp(tb.loc[t].p_holm)}")
        L.append(f"- {JUDGES[j]}, without the 44 AdvBench-overlap items: generation effects change by at most "
                 f"{100 * diff.max():.1f} pp (median {100 * diff.median():.1f} pp); H1/H2 Holm decisions or signs that "
                 f"change: {'; '.join(show(t) for t in flips) or 'none'}.")
    a = tests[(tests.judge == "wg") & (tests.subset == "all") & tests.family.isin(["H1", "H2"])].set_index("test_id")
    b = tests[(tests.judge == "wg_class") & (tests.subset == "all") & tests.family.isin(["H1", "H2"])].set_index(
        "test_id")
    flips = [t for t in a.index.intersection(b.index)
             if (a.loc[t].reject_holm != b.loc[t].reject_holm) or (np.sign(a.loc[t].estimate) != np.sign(b.loc[t].estimate))]
    ga = gen[(gen.judge == "wg") & (gen.subset == "all")].set_index(["pair", "method"]).delta
    gc = gen[(gen.judge == "wg_class") & (gen.subset == "all")].set_index(["pair", "method"]).delta
    L.append(f"- Degenerate responses counted as not refused (outcome-class rule): generation effects change by at "
             f"most {100 * (gc - ga).abs().max():.1f} pp; decisions or signs that change: "
             f"{', '.join(f'`{t}`' for t in flips) or 'none'}. Degenerate rows per cell: "
             f"max {int(cells[(cells.judge == 'wg') & (cells.subset == 'all')].n_degenerate.max())} of 972.")
    L.append("")

    # ---- ablation effect
    L.append("## Ablation effect per model (modified vs original, WildGuard, 972 harmful items)")
    L.append("")
    c0 = cells[(cells.judge == J0) & (cells.subset == S0) & (cells.method != "original")]
    rows = []
    for p in PAIR_ORDER:
        for g in ("old", "new", "small-old", "small-new"):
            for m in METHOD_ORDER:
                r = c0[(c0.pair == p) & (c0.generation == g) & (c0.method == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append({"Model": r.label, "Residual refusal % [95% CI]":
                             f"{pp(r.rate, sign=False)} [{pp(r.lo, sign=False)}, {pp(r.hi, sign=False)}]",
                             "Original %": pp(r.orig_rate, sign=False),
                             "Mod - orig, pp [95% CI]": f"{pp(r.diff_vs_orig)} [{pp(r.diff_lo)}, {pp(r.diff_hi)}]",
                             "removed / introduced": f"{int(r.removed_b)} / {int(r.introduced_c)}",
                             "p Holm": fp(r.p_holm_ablation), "Capped at 512 %": pp(r.capped_share, sign=False)})
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    nsig = c0[c0.p_holm_ablation >= ALPHA]
    L.append("Not significantly different from the original after Holm: " +
             (", ".join(f"`{r.label}`" for r in nsig.itertuples()) or "none") + ".")
    up = c0[c0.diff_vs_orig >= 0]
    if len(up):
        L.append("")
        L.append("**Residual refusal at or above the original** (the selected trial at KL <= 0.1 did not remove refusal "
                 "on the held-out set): " + "; ".join(
                     f"`{r.label}` {pp(r.rate, sign=False)}% vs {pp(r.orig_rate, sign=False)}% ({pp(r.diff_vs_orig)} pp, "
                     f"p_Holm {fp(r.p_holm_ablation)})" for r in up.itertuples()) + ".")
    L.append("")

    # ---- caveats
    gk_ = g0[g0.method != "original"].copy()
    gk_["kl_old"] = gk_.old_label.map(kl)
    gk_["kl_new"] = gk_.new_label.map(kl)
    gk_["ratio"] = np.maximum(gk_.kl_old, gk_.kl_new) / np.minimum(gk_.kl_old, gk_.kl_new)
    kl_pairs = "; ".join(f"{r.pair} {METHOD_DISPLAY[r.method]} {r.kl_old:.3f} / {r.kl_new:.3f}"
                         for r in gk_.sort_values("ratio", ascending=False).head(4).itertuples())
    L.append("## Caveats")
    L.append("")
    cav = [
        (f"**{len(main_judged)} of four main families.** H1 says \"each pair\" and H2 is a pooled interaction; with "
         f"{len(main_judged)} judged main pairs, the pooled model and the family random intercept are interim. "
         if main_pending else
         "**Four main families.** The family random intercept of the planned model is estimated from 4 levels "
         "(VB posterior SD in h1h2_models.csv); the heterogeneity tests below show that the families differ in sign, "
         "so pooled coefficients are averages, not common effects. ")
        + "The small pair is analysed alongside but confounds generation with size (7B -> 4B) and was generated on a "
        "different GPU (RTX 4090).",
        "**One seed per cell.** Optimisation-seed variance is large (EXECUTION_PLAN_v2 finding 1; atlas_sweep D3: "
        "MPOA seed sensitivity). Prompt-level CIs and tests treat the selected model as fixed; they do not include "
        "seed-to-seed variation of the search, so a significant generation effect for one method can be smaller than "
        "the spread between seeds. Seed-2 replicates of the 10 small-pair cells are analysed in final.py "
        "(final_seed_*.csv); they are not part of H1/H2.",
        "**Baseline refusal differs between originals** (see H1 table). Raw generation effects mix the newer model's "
        "higher (or lower) starting refusal with the difficulty of removing it; report the baseline-adjusted values "
        "next to the raw ones.",
        "**Truncation.** Many modified responses hit the 512-token cap (up to 88% for Qwen3.8 MPOA), more often on the "
        "newer models; WildGuard and keyword labels on capped responses refer to truncated answers (atlas_sweep D7). "
        "The 256-token keyword column shows the sensitivity to length.",
        "**Judge validity.** WildGuard has not been audited against human labels on these edited-model outputs "
        "(atlas_sweep D1: judges can fail under distribution shift; LLM judges can over-count refusal when harmful "
        "content carries disclaimers). Keyword vs WildGuard disagreement is reported as divergence, not as one being "
        "right, until the human audit.",
        "**Scale dependence of H2.** ARA rates near 0 (e.g. Gemma-4 ARA) make log-odds contrasts large while the "
        "risk difference is moderate; the GEE (population-averaged) and VB (prompt-conditional) coefficients are on "
        "different scales and are not directly comparable.",
        "**VB posterior uncertainty** from the mean-field approximation is typically too narrow; the GEE robust SEs "
        "are the more conservative frequentist check, and per-pair McNemar tests are exact.",
        "**Matched distribution shift = a common KL cap, not equal KL.** All selected trials satisfy first-token "
        "KL <= 0.1 on the dev prompts, but their KL differs within pairs (largest old/new ratios: " + kl_pairs + ") "
        "and so do teacher-forced divergence and capability cost (`cell_rates.csv`: `dev_kl`, `tf_divergence`, "
        "`capability_mean_delta_pp`).",
        "**Dev selection uses keyword refusals**, which may push the search on newer models toward removing caveat "
        "vocabulary rather than refusal (atlas_sweep D6); the selected trial is not the WildGuard-optimal one.",
        "**Two-point pairs.** A difference between one older and one newer model is not a trend across generations; "
        "Gemma-2 refusal geometry is reported as unusual by several studies (atlas_sweep D2), so the Gemma pair "
        "should be reported separately rather than only pooled.",
        "Significance threshold alpha = 0.05 and the Holm families above are this script's choices; the Methods "
        "todos (\"define test families\", \"state significance threshold\", per-method H1 estimation) should be "
        "filled to match.",
    ]
    L += [f"- {c}" for c in cav]
    L.append("")
    L.append("## Files")
    L.append("")
    for f, desc in (("h1h2_status.csv", "every planned cell: judge status and whether it is used"),
                    ("h1h2_cells.csv", "residual refusal per cell x judge x subset; modified vs original (McNemar, Holm)"),
                    ("h1h2_generation_effects.csv", "generation effects per pair x method x judge x subset (H1)"),
                    ("h1h2_h2_contrasts.csv", "H2 contrasts per pair x judge x subset"),
                    ("h1h2_models.csv", "all VB and GEE model terms per scope x judge x subset"),
                    ("h1h2_tests.csv", "every test with its Holm family, raw and adjusted p")):
        L.append(f"- `results/analysis/{f}`: {desc}")
    L.append("")
    L.append("VB mixed models fitted for judges: " + (", ".join(sorted(vb_judges)) or "none (`--no-vb`)") +
             "; GEE models for every judge.")
    (OUT / "h1h2_summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boot", type=int, default=4000, help="item-bootstrap resamples (default 4000)")
    ap.add_argument("--no-vb", action="store_true", help="skip the BinomialBayesMixedGLM fits")
    ap.add_argument("--vb-judges", default="wg,kw",
                    help="judges for which the VB mixed models are fitted (default wg,kw; 'all' for every judge)")
    args = ap.parse_args()
    t0 = time.time()
    d, rates, src, log_head = load_inputs()
    status = build_status(rates, set(d["label"].unique()))
    print(f"read {src.relative_to(ROOT)}: {len(d)} harmful rows, {d['label'].nunique()} cells "
          f"({', '.join(sorted(d['pair'].unique()))})", flush=True)
    vb_judges = set() if args.no_vb else (set(JUDGES) if args.vb_judges == "all" else set(args.vb_judges.split(",")))
    cells, gen, h2, models, tests, pairs = analyse(d, args.boot, vb_judges)

    order = lambda df, cols: df.sort_values(cols, kind="stable")  # noqa: E731
    status.to_csv(OUT / "h1h2_status.csv", index=False)
    cells.to_csv(OUT / "h1h2_cells.csv", index=False, float_format="%.6g")
    gen.to_csv(OUT / "h1h2_generation_effects.csv", index=False, float_format="%.6g")
    h2.to_csv(OUT / "h1h2_h2_contrasts.csv", index=False, float_format="%.6g")
    models.to_csv(OUT / "h1h2_models.csv", index=False, float_format="%.6g")
    order(tests, ["judge", "subset", "family"]).to_csv(OUT / "h1h2_tests.csv", index=False, float_format="%.6g")
    write_summary(d, status, cells, gen, h2, models, tests, pairs, src, log_head, time.time() - t0, vb_judges,
                  args.boot)
    print(f"wrote h1h2_*.csv and h1h2_summary.md to {OUT.relative_to(ROOT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
