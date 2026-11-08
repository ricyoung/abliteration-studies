#!/usr/bin/env python3
"""Paper 33 -- secondary held-out analyses (a)-(d). Descriptive, no GPU, read-only on every input.

Run AFTER scripts/analysis/assemble.py (which must be re-run whenever a judge batch lands):
    scripts/figures/.venv/bin/python scripts/analysis/assemble.py
    scripts/figures/.venv/bin/python scripts/analysis/secondary.py

One-off helper mode (needs pandas + pyarrow; the figures venv has no pyarrow, the Heretic venv has):
    /data/heretic_2/heretic/.venv/bin/python -I scripts/analysis/secondary.py --build-advbench-map
  maps each of the 44 AdvBench-overlap items to the mlabonne/harmful_behaviors split/position it matches
  (train[:400] = direction / SOM / ARA fitting prompts; test[:100] = dev set used for search and selection)
  and writes results/analysis/secondary_advbench_overlap_splits.csv. The held-out set is frozen, so this is static.
  Main mode builds it in-process if pyarrow is importable, otherwise reads the CSV, otherwise skips the split subsets.

Inputs (all produced earlier; nothing here touches responses or judgments directly)
  results/analysis/responses_judged.{parquet,csv.gz}   per-response table from assemble.py (newer file wins)
  results/analysis/cell_rates.csv                     cell list incl. pending cells and their status
  paper/figures/data/trials.csv                       dev trials; selected trial = is_selected_kl0.1
  paper/figures/data/results.csv                      original-model dev baseline refusals; evaluated trial
  scripts/bench/refusal_sets.json                     dataset revision of mlabonne/harmful_behaviors (helper mode)

Analyses
  (a) over-refusal on benign items (XSTest-safe 250, JBB-benign 100) per cell under WildGuard, keyword (full 512-token
      response and first 256 tokens) and Qwen3Guard; change vs the cell's original model on the same items with an
      exact McNemar test, Holm-adjusted within each (judge, item group) family over all modified cells; per-item table;
      over-refusal vs harmful-prompt refusal across cells (Spearman).
  (b) dev vs held-out: Spearman between the selected trial's dev keyword refusals (100 dev prompts, 100 new tokens) and
      held-out harmful refusal under WildGuard (primary), keyword (512 and 256 tokens), Qwen3Guard and HarmBench
      non-compliance, across model x method cells; bootstrap (cells) 95% CIs; within-model method rankings.
  (c) harmful compliance (HarmBench classifier, WildGuard harmful response), compliance-conditional harmfulness,
      HarmBench/WildGuard agreement, degenerate and capped (512-token) rates, outcome4 shares, per cell and item group;
      capped rates also for cells whose responses exist but are not judged yet.
  (d) AdvBench-overlap sensitivity: every harmful measure with all 972 items, without the 44 overlap items, on the
      overlap items only and on the overlap items that match the fitting (train[:400]) or dev (test[:100]) prompts;
      change vs original on overlap vs non-overlap items (difference in differences, Wilcoxon over cells); old-vs-new
      contrasts per pair x method with and without the overlap items.

Outputs (results/analysis/, all overwritten): secondary_*.csv and secondary_summary.md (see the summary for columns).
Pending cells (responses not judged yet, or not generated) are never imputed: they are listed in secondary_cells.csv
and in the summary, and enter every table automatically once assemble.py finds their refusal/judgments.jsonl.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
RJ_PARQUET = OUT / "responses_judged.parquet"
RJ_CSV = OUT / "responses_judged.csv.gz"
CELL_RATES = OUT / "cell_rates.csv"
TRIALS_CSV = ROOT / "paper" / "figures" / "data" / "trials.csv"
RESULTS_CSV = ROOT / "paper" / "figures" / "data" / "results.csv"
REFUSAL_SETS = ROOT / "scripts" / "bench" / "refusal_sets.json"
ADV_MAP = OUT / "secondary_advbench_overlap_splits.csv"
HF_HUB = Path(os.environ.get("HF_HUB_CACHE", "/data/huggingface/hub"))
N_FIT, N_DEV = 400, 100  # Heretic: train[:400] for directions/SOM/ARA capture, test[:100] dev set (Methods)

N_BOOT = 10_000
SEED = 20261105
ALPHA = 0.05

PAIR_ORDER = ["qwen", "gemma", "olmo", "mistral", "qwen-small"]
GEN_RANK = {"old": 0, "new": 1, "small-old": 0, "small-new": 1}
METHOD_ORDER = ["original", "dim", "mpoa", "som", "jspace", "gamma_mpoa", "ara", "community"]
METHOD_SHORT = {"original": "Original", "dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space",
                "gamma_mpoa": "gamma-MPOA", "ara": "ARA", "community": "community"}
CELL_COLS = ["label", "model", "model_display", "family", "pair", "study", "generation", "gen_new", "method",
             "variant", "method_class", "role", "seed"]
BOOL_COLS = ["ref_wg", "ref_kw", "ref_kw256", "ref_q3g", "harm_hb", "nonharm_hb", "harm_wg", "harmreq_wg",
             "incomplete", "is_degenerate", "adv_overlap"]
HARMFUL_SETS = ["jbb_harmful", "xstest_unsafe", "strongreject", "harmbench_standard", "code_weapon"]
BENIGN_SETS = ["xstest_safe", "jbb_benign"]

# refusal-type measures on harmful items: name -> 0/1 column (1 = refused / not complied)
REFUSAL_MEASURES = {"wg_refusal": "ref_wg", "kw_refusal": "ref_kw", "kw256_refusal": "ref_kw256",
                    "q3g_refusal": "ref_q3g", "hb_noncompliance": "nonharm_hb"}
OVERREF_JUDGES = {"wg": "ref_wg", "kw": "ref_kw", "kw256": "ref_kw256", "q3g": "ref_q3g"}

LOG: list[str] = []
ADV_BENIGN: set = set()  # benign items that meet the AdvBench overlap rule (set in main from the AdvBench map)


def log(msg: str = "") -> None:
    print(msg)
    LOG.append(msg)


# ----------------------------------------------------------------------------------------------------------- helpers
def wilson(k: float, n: float) -> tuple[float, float]:
    from statsmodels.stats.proportion import proportion_confint
    if not n:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(int(k), int(n), alpha=ALPHA, method="wilson")
    return float(lo), float(hi)


def rate_block(x: pd.Series | np.ndarray, prefix: str, ci: bool = True) -> dict:
    """k, n (non-missing), rate, Wilson 95% CI of a 0/1/NaN vector."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n, k = len(a), float(a.sum()) if len(a) else 0.0
    d = {f"{prefix}_k": int(k), f"{prefix}_n": n, f"{prefix}_rate": (k / n) if n else np.nan}
    if ci:
        d[f"{prefix}_lo"], d[f"{prefix}_hi"] = wilson(k, n)
    return d


def to01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0, 1.0: 1.0, 0.0: 0.0}).astype(float)


def mcnemar_exact(orig: np.ndarray, mod: np.ndarray) -> tuple[int, int, int, float]:
    """Exact (binomial) McNemar test on paired 0/1 vectors; returns n_pairs, n10 (orig 1 -> mod 0), n01, p."""
    from scipy import stats
    o, m = np.asarray(orig, float), np.asarray(mod, float)
    ok = ~(np.isnan(o) | np.isnan(m))
    o, m = o[ok], m[ok]
    n10, n01 = int(((o == 1) & (m == 0)).sum()), int(((o == 0) & (m == 1)).sum())
    p = 1.0 if n10 + n01 == 0 else float(stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return int(ok.sum()), n10, n01, p


def holm(p: pd.Series) -> pd.Series:
    from statsmodels.stats.multitest import multipletests
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.sum():
        out[ok] = multipletests(p[ok].to_numpy(float), method="holm")[1]
    return out


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if n == 0:
        return 0, np.nan, np.nan
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return n, po, (np.nan if pe == 1 else (po - pe) / (1 - pe))


def spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho (scipy, two-sided t approximation), Kendall tau-b, and a percentile bootstrap CI over cells."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    d = dict(n_cells=n, rho=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, tau_b=np.nan, tau_p=np.nan)
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return d
    r = stats.spearmanr(x, y)
    t = stats.kendalltau(x, y)
    d.update(rho=float(r.statistic), p=float(r.pvalue), tau_b=float(t.statistic), tau_p=float(t.pvalue))
    rng = np.random.default_rng(SEED)
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


def cell_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_p"] = df["pair"].map({p: i for i, p in enumerate(PAIR_ORDER)}).fillna(99)
    df["_g"] = df["generation"].map(GEN_RANK).fillna(9)
    df["_m"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    df["_s"] = pd.to_numeric(df["seed"], errors="coerce").fillna(0)
    df = df.sort_values(["_p", "_g", "_m", "_s", "label"]).drop(columns=["_p", "_g", "_m", "_s"])
    return df


def short_cell(row) -> str:
    m = METHOD_SHORT.get(row["method"], row["method"])
    if row["method"] == "community" and isinstance(row.get("variant"), str):
        m = f"community {row['variant']}"
    s = pd.to_numeric(row.get("seed"), errors="coerce")
    if row.get("role") == "replicate" and pd.notna(s):
        m += f" (seed {int(s)})"
    return f"{row['model']} {m}"


def pct(v, d=1) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}"


def num(v, d=2) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def pval(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- AdvBench overlap splits
def build_advbench_map() -> pd.DataFrame:
    """Which mlabonne/harmful_behaviors prompt (split, index) each overlap item matches, with the same normalisation
    and fuzzy rule as scripts/build_refusal_sets.py (exact, or token Jaccard >= 0.6 and SequenceMatcher >= 0.8)."""
    meta = json.loads(REFUSAL_SETS.read_text())
    rev = meta["meta"]["datasets"]["mlabonne/harmful_behaviors"]
    snap = HF_HUB / "datasets--mlabonne--harmful_behaviors" / "snapshots" / rev / "data"
    splits = {sp: pd.read_parquet(snap / f"{sp}-00000-of-00001.parquet")["text"].tolist() for sp in ("train", "test")}
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()  # noqa: E731
    adv = [(sp, i, norm(t)) for sp in ("train", "test") for i, t in enumerate(splits[sp])]
    rows = []
    # Every item is re-matched, benign ones included: scripts/build_refusal_sets.py flags harmful items only
    # (`overlap(...) if kind == "harmful" else ""`), and benign jbb_benign_11 meets the fuzzy rule.
    for it in meta["items"]:
        q = norm(it["prompt"])
        qt = set(q.split())
        hits = []
        for sp, i, a in adv:
            if q == a:
                hits.append((sp, i, "exact"))
                continue
            at = set(a.split())
            if len(qt & at) / max(1, len(qt | at)) >= 0.6 and difflib.SequenceMatcher(None, q, a).ratio() >= 0.8:
                hits.append((sp, i, "fuzzy"))
        if not hits and not it.get("advbench_overlap"):
            continue
        recomputed = "exact" if any(t == "exact" for _, _, t in hits) else ("fuzzy" if hits else "")
        fit = any(sp == "train" and i < N_FIT for sp, i, _ in hits)
        dev = any(sp == "test" and i < N_DEV for sp, i, _ in hits)
        rows.append(dict(id=it["id"], set=it["set"], kind=it["kind"], overlap_type=it["advbench_overlap"],
                         overlap_recomputed=recomputed, n_matches=len(hits),
                         matches_fit_prompts=fit, matches_dev_prompts=dev, matches_unused_only=not (fit or dev),
                         matches=";".join(f"{sp}[{i}]:{t}" for sp, i, t in hits)))
    df = pd.DataFrame(rows)
    df.attrs["source"] = f"mlabonne/harmful_behaviors@{rev[:8]} (train {len(splits['train'])}, test {len(splits['test'])})"
    return df


def load_advbench_map() -> tuple[pd.DataFrame | None, str]:
    try:
        import pyarrow  # noqa: F401
        df = build_advbench_map()
        df.to_csv(ADV_MAP, index=False)
        return df, f"built from the HF cache ({df.attrs['source']})"
    except Exception as e:  # pyarrow missing or dataset not cached
        if ADV_MAP.exists():
            df = pd.read_csv(ADV_MAP, keep_default_na=False)
            if "kind" not in df:  # map written before the benign re-match was added: harmful items only
                df["kind"], df["overlap_recomputed"] = "harmful", df["overlap_type"]
            return df, f"read from {ADV_MAP.relative_to(ROOT)} (not rebuilt: {type(e).__name__})"
        return None, f"unavailable ({type(e).__name__}: {e}); run --build-advbench-map with a pyarrow Python"


# ------------------------------------------------------------------------------------------------------- data load
def load_inputs():
    cands = [p for p in (RJ_PARQUET, RJ_CSV) if p.exists()]
    if not cands:
        sys.exit("responses_judged.* not found: run scripts/analysis/assemble.py first")
    rj_path = max(cands, key=lambda p: p.stat().st_mtime)
    rj = None
    if rj_path.suffix == ".parquet":
        try:
            rj = pd.read_parquet(rj_path)
        except Exception:
            rj_path = RJ_CSV
    if rj is None:
        rj = pd.read_csv(rj_path, low_memory=False)
    for c in BOOL_COLS:
        rj[c] = to01(rj[c])
    for c in ("variant", "degenerate_reason", "outcome", "outcome4"):
        rj[c] = rj[c].astype(object).where(rj[c].notna(), "")
    cr = pd.read_csv(CELL_RATES, low_memory=False)
    cr["variant"] = cr["variant"].astype(object).where(cr["variant"].notna(), "")
    trials = pd.read_csv(TRIALS_CSV)
    results = pd.read_csv(RESULTS_CSV)
    return rj, rj_path, cr, trials, results


def stale_judgments_warning(rj_path: Path) -> list[str]:
    """Changes in the result trees since the last assemble.py run (content-based; see freshness.py). mtimes are not
    used: rsync -a keeps remote mtimes, so judgments synced after the assembly can look older than the table."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import freshness  # scripts/analysis/freshness.py
    return freshness.check()


# --------------------------------------------------------------------------------------------------- the analyses
def item_groups(df: pd.DataFrame, adv_ids: dict[str, set]) -> dict[str, pd.Series]:
    h = df["kind"] == "harmful"
    ov = df["adv_overlap"] == 1
    g = {"harmful_all": h, "harmful_excl_advbench": h & ~ov, "advbench_overlap": h & ov}
    if adv_ids.get("fit") is not None:
        g["advbench_fit_match"] = h & df["id"].isin(adv_ids["fit"])
        g["advbench_dev_match"] = h & df["id"].isin(adv_ids["dev"])
    for s in HARMFUL_SETS:
        g[s] = df["set"] == s
    g["benign_all"] = df["kind"] == "benign"
    if adv_ids.get("benign"):
        g["benign_excl_advbench"] = (df["kind"] == "benign") & ~df["id"].isin(adv_ids["benign"])
    for s in BENIGN_SETS:
        g[s] = df["set"] == s
    return g


def attach_original(rj: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    o = rj[rj["method"] == "original"][["model", "id"] + cols].rename(columns={c: f"o_{c}" for c in cols})
    return rj.merge(o, on=["model", "id"], how="left")


def analysis_a(rj, cells):
    """Over-refusal on benign items."""
    groups = ["benign_all", "benign_excl_advbench", "xstest_safe", "jbb_benign"]
    rows = []
    for _, c in cells.iterrows():
        d0 = rj[rj["label"] == c["label"]]
        g = item_groups(d0, {"benign": ADV_BENIGN} if ADV_BENIGN else {})
        if "benign_excl_advbench" not in g:
            g["benign_excl_advbench"] = g["benign_all"]
        for grp in groups:
            d = d0[g[grp]]
            row = {k: c[k] for k in CELL_COLS} | {"item_group": grp, "n_items": len(d)}
            for j, col in OVERREF_JUDGES.items():
                row |= rate_block(d[col], f"{j}_overrefusal")
            for cls in ("refusal", "compliance", "degenerate", "unscored"):
                row[f"out_{cls}_k"] = int((d["outcome"] == cls).sum())
            row["out_refusal_rate"] = row["out_refusal_k"] / len(d) if len(d) else np.nan
            row |= rate_block(d["incomplete"], "capped", ci=False)
            row |= rate_block(d["harmreq_wg"], "wg_harmful_request", ci=False)
            has_orig = d["o_ref_wg"].notna().any() or d["o_ref_kw"].notna().any()
            if c["method"] != "original" and has_orig:
                for j, col in OVERREF_JUDGES.items():
                    o = d[f"o_{col}"].to_numpy(float)
                    m = d[col].to_numpy(float)
                    npair, n10, n01, p = mcnemar_exact(o, m)
                    ok = ~(np.isnan(o) | np.isnan(m))
                    row[f"{j}_orig_rate"] = float(np.nanmean(o)) if (~np.isnan(o)).any() else np.nan
                    row[f"{j}_delta_pp"] = 100 * (m[ok].mean() - o[ok].mean()) if ok.any() else np.nan
                    row[f"{j}_mcnemar_npairs"], row[f"{j}_mcnemar_n10"], row[f"{j}_mcnemar_n01"] = npair, n10, n01
                    row[f"{j}_mcnemar_p"] = p
            rows.append(row)
    a = pd.DataFrame(rows)
    for j in OVERREF_JUDGES:
        col = f"{j}_mcnemar_p"
        if col not in a:
            a[col] = np.nan
        a[f"{j}_mcnemar_p_holm"] = np.nan
        for grp in groups:
            sel = (a["item_group"] == grp) & (a["method"] != "original")
            a.loc[sel, f"{j}_mcnemar_p_holm"] = holm(a.loc[sel, col])
    # per item: how often each benign prompt is refused, across cells
    b = rj[rj["kind"] == "benign"]
    is_orig = b["method"] == "original"
    items = b.groupby(["id", "set"]).agg(
        n_cells=("label", "nunique"),
        wg_refusal_share_all=("ref_wg", "mean"),
        kw_refusal_share_all=("ref_kw", "mean"),
        kw256_refusal_share_all=("ref_kw256", "mean"),
        q3g_refusal_share_all=("ref_q3g", "mean"),
        wg_harmful_request_share=("harmreq_wg", "mean"),
    ).reset_index()
    for nm, sel in (("originals", is_orig), ("modified", ~is_orig)):
        s = b[sel].groupby("id").agg(**{f"wg_refusal_share_{nm}": ("ref_wg", "mean"),
                                        f"n_cells_{nm}": ("label", "nunique")}).reset_index()
        items = items.merge(s, on="id", how="left")
    items = items.sort_values(["wg_refusal_share_all", "id"], ascending=[False, True])
    return a, items


def dev_table(cells: pd.DataFrame, trials: pd.DataFrame, results: pd.DataFrame, cr: pd.DataFrame) -> pd.DataFrame:
    sel = trials[trials["is_selected_kl0.1"].astype(str) == "True"].copy()
    sel["seed"] = pd.to_numeric(sel["seed"], errors="coerce")
    base = results[results["method"] == "base"][["model", "dev_refusals"]].dropna()
    ev = results.dropna(subset=["label"])[["label", "evaluated_trial"]]
    crd = cr.drop_duplicates("label")[["label", "dev_refusals"]].rename(columns={"dev_refusals": "cr_dev_refusals"})
    rows = []
    for _, c in cells.iterrows():
        r = {"label": c["label"], "dev_run": "", "dev_selected_trial": np.nan, "dev_refusals": np.nan,
             "dev_n_prompts": np.nan, "dev_kl": np.nan, "dev_source": ""}
        if c["method"] == "original":
            b = base[base["model"] == c["model"]]
            if len(b):
                r.update(dev_refusals=float(b["dev_refusals"].iloc[0]), dev_n_prompts=100, dev_kl=0.0,
                         dev_source="results.csv baseline (original model)")
        elif c["method"] != "community":
            s = sel[(sel["model"] == c["model"]) & (sel["method"] == c["method"]) &
                    (sel["seed"] == pd.to_numeric(c["seed"], errors="coerce"))]
            if len(s) == 1:
                s = s.iloc[0]
                r.update(dev_run=s["run"], dev_selected_trial=s["trial"], dev_refusals=float(s["refusals"]),
                         dev_n_prompts=float(s["n_prompts"]), dev_kl=float(s["kl"]),
                         dev_source="trials.csv is_selected_kl0.1")
            elif len(s) > 1:
                r["dev_source"] = f"AMBIGUOUS: {len(s)} selected rows"
        rows.append(r)
    d = pd.DataFrame(rows).merge(ev, on="label", how="left").merge(crd, on="label", how="left")
    d["dev_rate"] = d["dev_refusals"] / d["dev_n_prompts"]
    d["check_selected_eq_evaluated"] = np.where(
        d["evaluated_trial"].notna() & d["dev_selected_trial"].notna(),
        d["evaluated_trial"] == d["dev_selected_trial"], np.nan)
    d["check_dev_eq_cell_rates"] = np.where(d["dev_refusals"].notna() & d["cr_dev_refusals"].notna(),
                                            d["dev_refusals"] == d["cr_dev_refusals"], np.nan)
    return d.drop(columns=["cr_dev_refusals"])


def analysis_b(rj, cells, dev, adv_ids):
    subsets = ["harmful_all", "harmful_excl_advbench", "advbench_overlap"] + (["advbench_dev_match"] if adv_ids else [])
    rows = []
    for _, c in cells.iterrows():
        if c["method"] == "community":
            continue
        d0 = rj[rj["label"] == c["label"]]
        g = item_groups(d0, adv_ids)
        row = {k: c[k] for k in CELL_COLS}
        for sub in subsets:
            d = d0[g[sub]]
            for m, col in REFUSAL_MEASURES.items():
                a = d[col].to_numpy(float)
                a = a[~np.isnan(a)]
                row[f"{m}__{sub}"] = a.mean() if len(a) else np.nan
                if sub == "harmful_all":
                    row[f"{m}__{sub}_n"] = len(a)
        rows.append(row)
    t = pd.DataFrame(rows).merge(dev, on="label", how="left")
    t["wg_minus_dev_pp"] = 100 * (t["wg_refusal__harmful_all"] - t["dev_rate"])
    t["kw_minus_dev_pp"] = 100 * (t["kw_refusal__harmful_all"] - t["dev_rate"])
    t["kw256_minus_dev_pp"] = 100 * (t["kw256_refusal__harmful_all"] - t["dev_rate"])

    sets = {
        "primary_all": t["role"] == "primary",
        "primary_main": (t["role"] == "primary") & (t["study"] == "main"),
        "primary_small": (t["role"] == "primary") & (t["study"] == "small"),
        "primary_excl_jspace": (t["role"] == "primary") & (t["method"] != "jspace"),
        "modified_all_roles": t["role"].isin(["primary", "replicate", "control"]),
        "primary_plus_originals": t["role"].isin(["primary", "original"]),
    }
    corr = []
    for sname, smask in sets.items():
        tt = t[smask & t["dev_rate"].notna()]
        for m in REFUSAL_MEASURES:
            for sub in subsets:
                y = f"{m}__{sub}"
                corr.append(dict(analysis="b_dev_vs_heldout", cell_set=sname, x="dev_keyword_refusal_rate",
                                 y=f"heldout_{m}", item_subset=sub,
                                 cells=";".join(tt.loc[tt[y].notna(), "label"]))
                            | spearman_with_ci(tt["dev_rate"].to_numpy(), tt[y].to_numpy()))
    corr = pd.DataFrame(corr)

    within = []
    tp = t[t["role"] == "primary"]
    for model, d in tp.groupby("model", sort=False):
        d = d.dropna(subset=["dev_rate"])
        if len(d) < 3:
            continue

        def best(col, dd=d):
            v = dd[col]
            return "/".join(dd.loc[v == v.min(), "method"].map(METHOD_SHORT))

        def rank_order(col, dd=d):
            return " < ".join(dd.sort_values([col, "method"])["method"].map(METHOD_SHORT))

        r = {"model": model, "pair": d["pair"].iloc[0], "generation": d["generation"].iloc[0], "n_methods": len(d),
             "dev_order": rank_order("dev_rate"), "heldout_wg_order": rank_order("wg_refusal__harmful_all"),
             "heldout_kw_order": rank_order("kw_refusal__harmful_all"),
             "best_dev": best("dev_rate"), "best_heldout_wg": best("wg_refusal__harmful_all"),
             "best_heldout_kw": best("kw_refusal__harmful_all")}
        for m, y in (("wg", "wg_refusal__harmful_all"), ("kw", "kw_refusal__harmful_all"),
                     ("kw256", "kw256_refusal__harmful_all")):
            from scipy import stats
            rr = stats.spearmanr(d["dev_rate"], d[y]) if np.ptp(d[y]) > 0 and np.ptp(d["dev_rate"]) > 0 else None
            r[f"spearman_dev_vs_{m}"] = float(rr.statistic) if rr is not None else np.nan
        r["best_dev_is_best_wg"] = bool(set(r["best_dev"].split("/")) & set(r["best_heldout_wg"].split("/")))
        # dev ties: break them like the within-run selection rule (scripts/select_trial.py: min refusals, then min KL)
        r["best_dev_tied"] = "/" in r["best_dev"]
        r["best_dev_kl_tiebreak"] = METHOD_SHORT[d.sort_values(["dev_rate", "dev_kl"])["method"].iloc[0]]
        r["best_dev_kl_tiebreak_is_best_wg"] = r["best_dev_kl_tiebreak"] in r["best_heldout_wg"].split("/")
        r["best_dev_unique_is_best_wg"] = (not r["best_dev_tied"]) and r["best_dev_is_best_wg"]
        within.append(r)
    return t, corr, pd.DataFrame(within)


def analysis_c(rj, cells, cr, fp_items):
    groups = ["harmful_all", "harmful_excl_advbench"] + HARMFUL_SETS + ["benign_all"]
    rows = []
    for _, c in cells.iterrows():
        d0 = rj[rj["label"] == c["label"]]
        g = item_groups(d0, {})
        for grp in groups:
            d = d0[g[grp]]
            harmful = grp != "benign_all"
            row = {k: c[k] for k in CELL_COLS} | {"status": "judged", "item_group": grp, "n_items": len(d),
                                                 "n_tokens_mean": d["n_tokens"].mean()}
            if harmful:
                row |= rate_block(d["harm_hb"], "hb_harmful")
            row |= rate_block(d["harm_wg"], "wg_harmful_response")
            row |= rate_block(d["is_degenerate"], "degenerate")
            row |= rate_block(d.loc[~d["id"].isin(fp_items), "is_degenerate"], "degenerate_excl_fp")
            row |= rate_block(d["incomplete"], "capped")
            for cls in ("refusal", "compliance_complete", "compliance_incomplete", "degenerate", "unscored"):
                k = int((d["outcome4"] == cls).sum())
                row[f"out4_{cls}_k"], row[f"out4_{cls}_rate"] = k, (k / len(d) if len(d) else np.nan)
            comp = d["outcome"] == "compliance"
            if harmful:
                row |= rate_block(d.loc[comp, "harm_hb"], "hb_harmful_given_compliance")
                row |= rate_block(d.loc[comp & (d["incomplete"] == 0), "harm_hb"],
                                  "hb_harmful_given_compliance_complete", ci=False)
                row |= rate_block(d.loc[comp & (d["incomplete"] == 1), "harm_hb"],
                                  "hb_harmful_given_compliance_capped", ci=False)
                row["compliance_not_harmful_hb_rate"] = float(((comp) & (d["harm_hb"] == 0)).mean()) if len(d) else np.nan
                n, po, kap = cohen_kappa(d["harm_hb"], d["harm_wg"])
                row["hb_wg_n_both"], row["hb_wg_agreement"], row["hb_wg_kappa"] = n, po, kap
            row |= rate_block(d.loc[comp, "harm_wg"], "wg_harmful_given_compliance")
            row["capped_share_of_refusals"] = d.loc[d["outcome"] == "refusal", "incomplete"].mean()
            row["capped_share_of_compliance"] = d.loc[comp, "incomplete"].mean()
            if c["method"] != "original" and d["o_ref_wg"].notna().any():
                for nm, col in (("hb_harmful", "harm_hb"), ("wg_harmful_response", "harm_wg"),
                                ("capped", "incomplete")):
                    if nm == "hb_harmful" and not harmful:
                        continue
                    o, m = d[f"o_{col}"].to_numpy(float), d[col].to_numpy(float)
                    ok = ~(np.isnan(o) | np.isnan(m))
                    _, n10, n01, p = mcnemar_exact(o, m)
                    row[f"{nm}_orig_rate"] = float(np.nanmean(o)) if (~np.isnan(o)).any() else np.nan
                    row[f"{nm}_delta_pp"] = 100 * (m[ok].mean() - o[ok].mean()) if ok.any() else np.nan
                    row[f"{nm}_mcnemar_n10"], row[f"{nm}_mcnemar_n01"], row[f"{nm}_mcnemar_p"] = n10, n01, p
            rows.append(row)
    t = pd.DataFrame(rows)
    for nm in ("hb_harmful", "wg_harmful_response", "capped"):
        col = f"{nm}_mcnemar_p"
        if col not in t:
            t[col] = np.nan
        t[f"{nm}_mcnemar_p_holm"] = np.nan
        for grp in groups:
            sel = (t["item_group"] == grp) & (t["method"] != "original")
            t.loc[sel, f"{nm}_mcnemar_p_holm"] = holm(t.loc[sel, col])
    # cells with responses but no judgments yet: capped/length only (from cell_rates, generation-level fields)
    pend = cr[(cr["status"] == "pending_judging") & cr["set_group"].isin(groups)]
    if len(pend):
        p = pend[CELL_COLS + ["status", "set_group", "n_items", "n_tokens_mean", "capped_k", "capped_n",
                              "capped_rate", "capped_lo", "capped_hi"]].rename(columns={"set_group": "item_group"})
        t = pd.concat([t, p], ignore_index=True)
    return t


def analysis_d(rj, cells, adv_ids):
    measures = dict(REFUSAL_MEASURES) | {"hb_harmful": "harm_hb", "wg_harmful_response": "harm_wg",
                                         "degenerate": "is_degenerate", "capped": "incomplete"}
    subs = ["harmful_all", "harmful_excl_advbench", "advbench_overlap"] + (
        ["advbench_fit_match", "advbench_dev_match"] if adv_ids else [])
    rows = []
    for _, c in cells.iterrows():
        d0 = rj[rj["label"] == c["label"]]
        g = item_groups(d0, adv_ids)
        for m, col in measures.items():
            row = {k: c[k] for k in CELL_COLS} | {"measure": m}
            for sub in subs:
                row |= rate_block(d0.loc[g[sub], col], sub, ci=(sub != "harmful_all" or True))
            row["diff_all_minus_excl_pp"] = 100 * (row["harmful_all_rate"] - row["harmful_excl_advbench_rate"])
            row["diff_overlap_minus_excl_pp"] = 100 * (row["advbench_overlap_rate"] - row["harmful_excl_advbench_rate"])
            if c["method"] != "original":
                for sub, nm in (("advbench_overlap", "overlap"), ("harmful_excl_advbench", "nonoverlap")):
                    dd = d0[g[sub]]
                    o, mm = dd[f"o_{col}"].to_numpy(float), dd[col].to_numpy(float)
                    ok = ~(np.isnan(o) | np.isnan(mm))
                    row[f"orig_{nm}_rate"] = o[ok].mean() if ok.any() else np.nan
                    row[f"change_{nm}_pp"] = 100 * (mm[ok].mean() - o[ok].mean()) if ok.any() else np.nan
                row["did_overlap_minus_nonoverlap_pp"] = row["change_overlap_pp"] - row["change_nonoverlap_pp"]
                if m in REFUSAL_MEASURES:
                    # retention = P(modified still refuses | original refused): free of the ceiling that makes the DiD
                    # depend on the original's baseline (every original refuses ~100% of the overlap items)
                    rsubs = [("advbench_overlap", "overlap"), ("harmful_excl_advbench", "nonoverlap")] + (
                        [("advbench_fit_match", "fit_match"), ("advbench_dev_match", "dev_match")] if adv_ids else [])
                    for sub, nm in rsubs:
                        dd = d0[g[sub]]
                        keep = dd[f"o_{col}"] == 1
                        row |= rate_block(dd.loc[keep, col], f"retention_{nm}", ci=False)
                    row["retention_overlap_minus_nonoverlap_pp"] = 100 * (row["retention_overlap_rate"] -
                                                                         row["retention_nonoverlap_rate"])
            rows.append(row)
    s = pd.DataFrame(rows)

    # pooled: is the change vs original larger on the overlap items (fitting/dev prompts) than on the rest?
    from scipy import stats
    pooled = []
    for stat_col, stat_nm in (("retention_overlap_minus_nonoverlap_pp", "retention_diff"),
                              ("did_overlap_minus_nonoverlap_pp", "did")):
        for m in ("wg_refusal", "kw_refusal", "kw256_refusal", "hb_noncompliance", "hb_harmful"):
            if stat_nm == "retention_diff" and m not in REFUSAL_MEASURES:
                continue
            for cs, mask in (("primary_all", s["role"] == "primary"),
                             ("primary_main", (s["role"] == "primary") & (s["study"] == "main")),
                             ("modified_all_roles", s["role"].isin(["primary", "replicate", "control"]))):
                if stat_col not in s:
                    continue
                v = s.loc[mask & (s["measure"] == m), stat_col].dropna().to_numpy()
                r = dict(statistic=stat_nm, measure=m, cell_set=cs, n_cells=len(v),
                         median_pp=np.median(v) if len(v) else np.nan, mean_pp=v.mean() if len(v) else np.nan,
                         n_negative=int((v < 0).sum()), n_positive=int((v > 0).sum()),
                         wilcoxon_stat=np.nan, wilcoxon_p=np.nan)
                if len(v) >= 6 and np.any(v != 0):
                    w = stats.wilcoxon(v)
                    r.update(wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue))
                pooled.append(r)
    pooled = pd.DataFrame(pooled)

    # old-vs-new contrasts per pair x method, with and without the overlap items
    gc = []
    for m in ("wg_refusal", "kw_refusal", "kw256_refusal", "hb_harmful", "wg_harmful_response"):
        sm = s[(s["measure"] == m) & s["role"].isin(["original", "primary"])]
        for (pair, meth), d in sm.groupby(["pair", "method"], sort=False):
            old = d[d["gen_new"] == 0]
            new = d[d["gen_new"] == 1]
            if len(old) != 1 or len(new) != 1:
                continue
            o, n = old.iloc[0], new.iloc[0]
            r = dict(measure=m, pair=pair, method=meth, old_model=o["model"], new_model=n["model"])
            for sub in ("harmful_all", "harmful_excl_advbench"):
                r[f"old_{sub}"] = o[f"{sub}_rate"]
                r[f"new_{sub}"] = n[f"{sub}_rate"]
                r[f"new_minus_old_pp_{sub}"] = 100 * (n[f"{sub}_rate"] - o[f"{sub}_rate"])
            a, b = r["new_minus_old_pp_harmful_all"], r["new_minus_old_pp_harmful_excl_advbench"]
            r["shift_pp"] = b - a
            r["same_sign"] = bool(np.sign(np.round(a, 6)) == np.sign(np.round(b, 6)))
            gc.append(r)
    gc = pd.DataFrame(gc)
    if len(gc):
        gc["_p"] = gc["pair"].map({p: i for i, p in enumerate(PAIR_ORDER)})
        gc["_m"] = gc["method"].map({p: i for i, p in enumerate(METHOD_ORDER)})
        gc = gc.sort_values(["measure", "_p", "_m"]).drop(columns=["_p", "_m"])
    return s, pooled, gc


# ------------------------------------------------------------------------------------------------- consistency checks
def check_against_cell_rates(rj, cells, cr) -> list[str]:
    """Recompute a few rates exactly as cell_rates defines them; any mismatch means the inputs are out of sync."""
    msgs = []
    pairs = [("wg_refusal", "ref_wg"), ("kw_refusal", "ref_kw"), ("kw256_refusal", "ref_kw256"),
             ("hb_harmful", "harm_hb"), ("wg_harmful_response", "harm_wg"), ("degenerate", "is_degenerate"),
             ("capped", "incomplete")]
    n_bad = n_chk = 0
    for _, c in cells.iterrows():
        d0 = rj[rj["label"] == c["label"]]
        g = item_groups(d0, {})
        for grp in ("harmful_all", "harmful_excl_advbench", "benign_all", "xstest_safe", "jbb_benign"):
            ref = cr[(cr["label"] == c["label"]) & (cr["set_group"] == grp)]
            if len(ref) != 1:
                continue
            for m, col in pairs:
                a = d0.loc[g[grp], col].to_numpy(float)
                a = a[~np.isnan(a)]
                k = int(a.sum())
                rk = ref[f"{m}_k"].iloc[0]
                if pd.isna(rk):
                    continue
                n_chk += 1
                if k != int(rk) or len(a) != int(ref[f"{m}_n"].iloc[0]):
                    n_bad += 1
                    msgs.append(f"MISMATCH {c['label']} {grp} {m}: here {k}/{len(a)}, cell_rates {rk}/{ref[f'{m}_n'].iloc[0]}")
    msgs.insert(0, f"cross-check vs cell_rates.csv: {n_chk} counts compared, {n_bad} mismatches")
    return msgs


# ------------------------------------------------------------------------------------------------------- summary
def write_summary(ctx: dict) -> str:
    L = []
    w = L.append
    a, items, bt, bcorr, bwithin = ctx["a"], ctx["items"], ctx["bt"], ctx["bcorr"], ctx["bwithin"]
    c, s, pooled, gc = ctx["c"], ctx["d"], ctx["pooled"], ctx["gc"]
    cells, allcells = ctx["cells"], ctx["allcells"]
    w("# Paper 33 -- secondary held-out analyses (a)-(d)")
    w("")
    w(f"Generated {ctx['now']} by `scripts/analysis/secondary.py` from `{ctx['rj_path']}` "
      f"(assembled {ctx['rj_time']}). Regenerate after every judge batch:")
    w("")
    w("```\ncd /data/heretic_2/33_2611_does_refusal_removal_get_harder\n"
      "scripts/figures/.venv/bin/python scripts/analysis/assemble.py\n"
      "scripts/figures/.venv/bin/python scripts/analysis/secondary.py\n```")
    w("")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import freshness  # scripts/analysis/freshness.py
    w(freshness.summary_line(ctx["stale"]))
    w("")
    w("Software: Python with pandas, NumPy, SciPy and statsmodels (`scripts/figures/.venv`); R is not installed on this "
      "machine, so nothing here uses lme4. All analyses on this page are descriptive or simple paired/rank tests; the "
      "mixed-effects models for H1/H2 are a separate step.")
    w("")
    # ---- coverage
    st = allcells.drop_duplicates("label")
    w("## Coverage")
    w("")
    nj = len(cells)
    w(f"- **Included (judged or partially judged): {nj} cells**, {len(ctx['rj']):,} responses "
      f"({', '.join(f'{k} {v}' for k, v in cells.groupby('role').size().items())}).")
    for status, txt in (("partially_judged", "Partially judged (missing judges are empty)"),
                        ("pending_judging", "Pending judging (responses exist; judge batch 2), NOT included"),
                        ("pending_generation", "Pending generation (no responses yet), NOT included"),
                        ("judgments_mismatch", "Judgments do not match responses, NOT included")):
        sub = st[st["status"] == status]
        if len(sub):
            w(f"- **{txt}: {len(sub)}** -- " + ", ".join(f"`{x}`" for x in cell_sort_key(sub)["label"]))
    w("- Pending cells enter every table automatically once their `refusal/judgments.jsonl` exists and assemble.py "
      "has been re-run. Capped (512-token) rates of pending-judging cells are already in `secondary_harm_degenerate.csv` "
      "(`status = pending_judging`).")
    w(f"- AdvBench overlap split map: {ctx['adv_note']}.")
    for m in ctx["checks"]:
        w(f"- {m}")
    w("")
    w("Conventions: rates are k/n with Wilson 95% intervals; WildGuard rates exclude rows without a WildGuard label "
      "(see the table counts). Changes vs the original model are paired on identical items (exact McNemar test); Holm's "
      "correction is applied within each family = one judge x one item group over all modified cells in the table "
      "(the family covers every judged modified cell listed). Spearman correlations are over cells "
      f"(one selected model each) with a percentile bootstrap over cells ({N_BOOT:,} resamples, seed {SEED}).")
    w("")

    # ---- (a)
    w("## (a) Over-refusal on benign items")
    w("")
    w("Rate of refusals on the 350 benign held-out items (XSTest-safe 250, JBB-benign 100) under WildGuard (WG, primary), "
      "Heretic's keyword markers on the full response (KW) and on the first 256 tokens (KW256), and Qwen3Guard (Q3G). "
      "Delta = change vs the original model in percentage points (pp); p = Holm-adjusted exact McNemar.")
    w("")
    ab = cell_sort_key(a[a["item_group"] == "benign_all"])
    xs = a[a["item_group"] == "xstest_safe"].set_index("label")
    jb = a[a["item_group"] == "jbb_benign"].set_index("label")
    tab = []
    for _, r in ab.iterrows():
        tab.append({"cell": short_cell(r), "gen": r["generation"],
                    "WG % [95% CI]": f"{pct(r['wg_overrefusal_rate'])} [{pct(r['wg_overrefusal_lo'])}, {pct(r['wg_overrefusal_hi'])}]",
                    "KW %": pct(r["kw_overrefusal_rate"]), "KW256 %": pct(r["kw256_overrefusal_rate"]),
                    "Q3G %": pct(r["q3g_overrefusal_rate"]),
                    "XS-safe WG/KW %": f"{pct(xs.loc[r['label'], 'wg_overrefusal_rate'])} / {pct(xs.loc[r['label'], 'kw_overrefusal_rate'])}",
                    "JBB-benign WG/KW %": f"{pct(jb.loc[r['label'], 'wg_overrefusal_rate'])} / {pct(jb.loc[r['label'], 'kw_overrefusal_rate'])}",
                    "WG delta pp (p)": "" if r["method"] == "original" else f"{num(r.get('wg_delta_pp'), 1)} ({pval(r.get('wg_mcnemar_p_holm'))})",
                    "KW delta pp (p)": "" if r["method"] == "original" else f"{num(r.get('kw_delta_pp'), 1)} ({pval(r.get('kw_mcnemar_p_holm'))})"})
    w(md_table(pd.DataFrame(tab)))
    w("")
    mod = ab[ab["method"] != "original"]
    org = ab[ab["method"] == "original"]
    if len(org):
        w(f"- Originals: WG over-refusal {pct(org['wg_overrefusal_rate'].min())}-{pct(org['wg_overrefusal_rate'].max())}% "
          f"of benign items; KW {pct(org['kw_overrefusal_rate'].min())}-{pct(org['kw_overrefusal_rate'].max())}%.")
    if len(mod):
        sig_dn = int(((mod["wg_mcnemar_p_holm"] < ALPHA) & (mod["wg_delta_pp"] < 0)).sum())
        sig_up = int(((mod["wg_mcnemar_p_holm"] < ALPHA) & (mod["wg_delta_pp"] > 0)).sum())
        w(f"- Modified models: WG over-refusal {pct(mod['wg_overrefusal_rate'].min())}-{pct(mod['wg_overrefusal_rate'].max())}%; "
          f"WG change vs original: median {num(mod['wg_delta_pp'].median(), 1)} pp; Holm-significant decreases in "
          f"{sig_dn}/{len(mod)} cells, increases in {sig_up}/{len(mod)}.")
        w(f"- KW over-refusal of modified models {pct(mod['kw_overrefusal_rate'].min())}-{pct(mod['kw_overrefusal_rate'].max())}% "
          f"(median {pct(mod['kw_overrefusal_rate'].median())}%) versus WG median {pct(mod['wg_overrefusal_rate'].median())}%: "
          "the two measures diverge on benign items. The keyword list contains words such as 'illegal', 'harmful', "
          "'disclaimer' and 'sorry' that can also occur in answers that help while naming risks; which measure is "
          "right on these responses is open until the planned hand audit (atlas_sweep D1).")
    if ADV_BENIGN:
        ex = a[a["item_group"] == "benign_excl_advbench"].set_index("label")
        mx = ex.loc[ab["label"], "wg_overrefusal_rate"].to_numpy() - ab["wg_overrefusal_rate"].to_numpy()
        w(f"- AdvBench overlap on benign items: {', '.join(sorted(ADV_BENIGN))} meets the build script's fuzzy rule "
          "(the script checks harmful items only). Without it (n = 349; item group `benign_excl_advbench` in "
          f"`secondary_overrefusal.csv`) WG over-refusal changes by at most {pct(np.nanmax(np.abs(mx)), 2)} pp per cell.")
    wr = a[a["item_group"] == "jbb_benign"]["wg_harmful_request_rate"].mean()
    wx = a[a["item_group"] == "xstest_safe"]["wg_harmful_request_rate"].mean()
    w(f"- WildGuard's prompt-level label calls {pct(wr)}% of JBB-benign and {pct(wx)}% of XSTest-safe prompts harmful "
      "requests (mean over cells): JBB-benign items are topically matched to the harmful behaviours and partly "
      "borderline, so JBB-benign 'over-refusal' is not purely unwarranted refusal.")
    top = items.head(8)
    w(f"- Most-refused benign items (share of all {int(items['n_cells'].max())} judged cells refused by WG): " +
      "; ".join(f"`{r.id}` {pct(r.wg_refusal_share_all, 0)}% (originals {pct(r.wg_refusal_share_originals, 0)}%)"
                for r in top.itertuples()) + ".")
    oc = ctx["overref_corr"]
    for _, r in oc.iterrows():
        j = "WG" if r["x"].startswith("benign_wg") else "KW"
        w(f"- Over-refusal (benign, {j}) vs harmful-prompt refusal ({j}), {r['cell_set']}: Spearman rho = {num(r['rho'])} "
          f"[{num(r['ci_lo'])}, {num(r['ci_hi'])}], p = {pval(r['p'])}, n = {r['n_cells']} cells.")
    w("")

    # ---- (b)
    w("## (b) Dev keyword refusals vs held-out refusal (does the selection transfer?)")
    w("")
    w("x = keyword refusals of the selected trial on the 100 dev prompts (`trials.csv`, `is_selected_kl0.1`; 100 new "
      "tokens, Heretic's scorer); y = held-out refusal rate on the harmful items of the exported model (512 tokens). "
      "Primary set = seed-1 study methods (DiM, MPOA, SOM, ARA, J-space) of every judged model; originals excluded "
      "(adding them only adds two extreme clusters).")
    w("")
    key = bcorr[bcorr["item_subset"].isin(["harmful_all", "harmful_excl_advbench"])]
    tab = []
    for _, r in key.iterrows():
        if r["cell_set"] not in ("primary_all", "primary_main", "primary_small", "primary_plus_originals"):
            continue
        tab.append({"cells": r["cell_set"], "held-out measure": r["y"].replace("heldout_", ""),
                    "items": r["item_subset"], "n": r["n_cells"], "rho [95% CI]": f"{num(r['rho'])} [{num(r['ci_lo'])}, {num(r['ci_hi'])}]",
                    "p": pval(r["p"]), "tau-b": num(r["tau_b"])})
    w(md_table(pd.DataFrame(tab)))
    w("")
    w("Per cell (harmful_all; dev = selected trial's keyword refusals of 100):")
    w("")
    tb = cell_sort_key(bt[bt["role"] != "original"])
    tab = [{"cell": short_cell(r), "dev trial": "" if pd.isna(r["dev_selected_trial"]) else int(r["dev_selected_trial"]),
            "dev KL": num(r["dev_kl"], 3), "dev KW %": pct(r["dev_rate"], 0),
            "held-out WG %": pct(r["wg_refusal__harmful_all"]), "KW %": pct(r["kw_refusal__harmful_all"]),
            "KW256 %": pct(r["kw256_refusal__harmful_all"]), "Q3G %": pct(r["q3g_refusal__harmful_all"]),
            "HB non-compl. %": pct(r["hb_noncompliance__harmful_all"]),
            "WG - dev pp": num(r["wg_minus_dev_pp"], 1)} for _, r in tb.iterrows()]
    w(md_table(pd.DataFrame(tab)))
    w("")
    if len(bwithin):
        w("Within-model method rankings (lowest refusal first; descriptive, 4-5 methods per model):")
        w("")
        w(md_table(bwithin[["model", "dev_order", "heldout_wg_order", "best_dev", "best_heldout_wg",
                            "spearman_dev_vs_wg", "spearman_dev_vs_kw"]].assign(
            spearman_dev_vs_wg=lambda x: x["spearman_dev_vs_wg"].map(num),
            spearman_dev_vs_kw=lambda x: x["spearman_dev_vs_kw"].map(num))))
        w("")
        bties = bwithin[bwithin["best_dev_tied"]]
        w(f"- Best method on dev is also best on held-out WG in "
          f"{int(bwithin['best_dev_kl_tiebreak_is_best_wg'].sum())}/{len(bwithin)} models (dev ties broken by lower dev "
          f"KL, the selection rule's tie-break); {int(bwithin['best_dev_unique_is_best_wg'].sum())}/{len(bwithin)} "
          f"counting only a unique dev best"
          + ("; dev ties: " + "; ".join(f"{r.model} {r.best_dev} -> {r.best_dev_kl_tiebreak} (held-out best "
                                        f"{r.best_heldout_wg})" for r in bties.itertuples()) if len(bties) else "")
          + ".")
    tbm = tb[tb["role"] == "primary"]
    if len(tbm):
        w(f"- Level shift: held-out WG minus dev keyword rate, median {num(tbm['wg_minus_dev_pp'].median(), 1)} pp "
          f"(range {num(tbm['wg_minus_dev_pp'].min(), 1)} to {num(tbm['wg_minus_dev_pp'].max(), 1)}); held-out KW minus dev, "
          f"median {num(tbm['kw_minus_dev_pp'].median(), 1)} pp; held-out KW256 minus dev, median "
          f"{num(tbm['kw256_minus_dev_pp'].median(), 1)} pp.")
    chk = bt["check_selected_eq_evaluated"].dropna()
    w(f"- Check: selected trial = evaluated trial (`results.csv`) in {int((chk == True).sum())}/{len(chk)} judged cells; "  # noqa: E712
      "the merged exports were not re-scored on the dev set (Appendix todo), so dev values are those of the LoRA trial.")
    w("")

    # ---- (c)
    w("## (c) Harmful compliance, degenerate and capped responses")
    w("")
    w("Harmful items (972). HB = HarmBench classifier harmful compliance; WG harm = WildGuard harmful response; "
      "HB|compl = HB-harmful share among responses WildGuard did not call refusals (non-degenerate); "
      "evasive = WG compliance but HB not harmful (share of all harmful items); kappa = HB vs WG harmful label agreement; "
      "degenerate excludes the prompt-driven false positive(s) "
      f"{', '.join('`' + x + '`' for x in ctx['fp_items']) or '(none)'}; capped = stopped at 512 tokens.")
    w("")
    ch = cell_sort_key(c[(c["item_group"] == "harmful_all") & (c["status"] == "judged")])
    cb = c[(c["item_group"] == "benign_all") & (c["status"] == "judged")].set_index("label")
    tab = []
    for _, r in ch.iterrows():
        tab.append({"cell": short_cell(r),
                    "HB % [CI]": f"{pct(r['hb_harmful_rate'])} [{pct(r['hb_harmful_lo'])}, {pct(r['hb_harmful_hi'])}]",
                    "WG harm %": pct(r["wg_harmful_response_rate"]),
                    "HB|compl % (complete / capped)": f"{pct(r['hb_harmful_given_compliance_rate'])} ({pct(r['hb_harmful_given_compliance_complete_rate'], 0)} / {pct(r['hb_harmful_given_compliance_capped_rate'], 0)})",
                    "evasive %": pct(r["compliance_not_harmful_hb_rate"]),
                    "kappa": num(r["hb_wg_kappa"]),
                    "degen n (excl FP)": f"{int(r['degenerate_k'])} ({int(r['degenerate_excl_fp_k'])})",
                    "capped harmful %": pct(r["capped_rate"]),
                    "capped benign %": pct(cb.loc[r["label"], "capped_rate"]),
                    "refusal / compl. complete / compl. capped %": f"{pct(r['out4_refusal_rate'], 0)} / {pct(r['out4_compliance_complete_rate'], 0)} / {pct(r['out4_compliance_incomplete_rate'], 0)}"})
    w(md_table(pd.DataFrame(tab)))
    w("")
    pj = cell_sort_key(c[(c["status"] == "pending_judging") & (c["item_group"] == "harmful_all")])
    if len(pj):
        w("Capped share of harmful responses for cells not judged yet: " +
          "; ".join(f"{short_cell(r)} {pct(r['capped_rate'], 0)}%" for _, r in pj.iterrows()) + ".")
        w("")
    mh = ch[ch["method"] != "original"]
    if len(mh):
        w(f"- Modified models: HB harmful {pct(mh['hb_harmful_rate'].min())}-{pct(mh['hb_harmful_rate'].max())}%, "
          f"WG harmful {pct(mh['wg_harmful_response_rate'].min())}-{pct(mh['wg_harmful_response_rate'].max())}%; "
          f"HB|compliance median {pct(mh['hb_harmful_given_compliance_rate'].median())}% "
          f"(range {pct(mh['hb_harmful_given_compliance_rate'].min())}-{pct(mh['hb_harmful_given_compliance_rate'].max())}%); "
          f"HB-WG kappa median {num(mh['hb_wg_kappa'].median())}.")
        w(f"- HB|compliance, complete vs capped responses (median over modified cells): "
          f"{pct(mh['hb_harmful_given_compliance_complete_rate'].median())}% vs "
          f"{pct(mh['hb_harmful_given_compliance_capped_rate'].median())}%.")
        w(f"- Capped: harmful responses {pct(ch['capped_rate'].min())}-{pct(ch['capped_rate'].max())}% across judged cells; "
          f"capped share among WG compliance (median over modified cells) {pct(mh['capped_share_of_compliance'].median())}% "
          f"vs among refusals {pct(mh['capped_share_of_refusals'].median())}%.")
        w(f"- Degenerate rows: {int(ch['degenerate_k'].sum())} on harmful items across judged cells "
          f"({int(ch['degenerate_excl_fp_k'].sum())} after removing the false-positive item); max in one cell "
          f"{int(ch['degenerate_excl_fp_k'].max())} ({short_cell(ch.loc[ch['degenerate_excl_fp_k'].idxmax()])}).")
    w("")

    # ---- (d)
    w("## (d) AdvBench-overlap sensitivity")
    w("")
    if ctx["adv_counts"]:
        w(f"The 44 overlap items: {ctx['adv_counts']}. Items matching the fitting prompts (train[:400]) were used to "
          "compute the refusal directions / SOM / ARA targets; items matching the dev prompts (test[:100]) were the "
          "search-and-selection set.")
        w("")
    sw = cell_sort_key(s[s["measure"] == "wg_refusal"])
    tab = []
    for _, r in sw.iterrows():
        d = {"cell": short_cell(r), "WG all 972 %": pct(r["harmful_all_rate"]),
             "excl. 44 %": pct(r["harmful_excl_advbench_rate"]), "diff pp": num(r["diff_all_minus_excl_pp"], 2),
             "overlap 44 %": pct(r["advbench_overlap_rate"])}
        if "advbench_fit_match_rate" in r:
            d[f"fit-match {int(r['advbench_fit_match_n'])} %"] = pct(r["advbench_fit_match_rate"])
            d[f"dev-match {int(r['advbench_dev_match_n'])} %"] = pct(r["advbench_dev_match_rate"])
        d["retention overlap / rest %"] = ("" if r["method"] == "original" else
                                           f"{pct(r.get('retention_overlap_rate'), 0)} / {pct(r.get('retention_nonoverlap_rate'), 0)}")
        d["DiD pp"] = num(r.get("did_overlap_minus_nonoverlap_pp"), 1)
        tab.append(d)
    w(md_table(pd.DataFrame(tab)))
    w("")
    w("Retention = share of the items the original refused that the modified model still refuses (overlap items vs "
      "the other 928). DiD = (change vs original on the 44 overlap items) - (change on the other 928); it is shown for "
      "completeness but is biased toward negative values because every original refuses all 44 overlap items "
      "(ceiling), so a uniform removal produces a larger absolute drop there. Negative retention difference = refusal "
      "removed more on prompts that match the fitting/dev data.")
    w("")
    for m in ("wg_refusal", "kw_refusal", "hb_harmful"):
        sm = s[s["measure"] == m]
        w(f"- {m}: |all - excl| max {num(sm['diff_all_minus_excl_pp'].abs().max(), 2)} pp over {len(sm)} cells "
          f"(median {num(sm['diff_all_minus_excl_pp'].abs().median(), 2)} pp).")
    for _, r in pooled[(pooled["cell_set"] == "primary_all") & (pooled["measure"] != "hb_noncompliance")].iterrows():
        nm = "retention difference (overlap - rest)" if r["statistic"] == "retention_diff" else "DiD (ceiling-biased)"
        w(f"- Pooled {nm}, {r['measure']} (primary cells, n = {r['n_cells']}): median {num(r['median_pp'], 1)} pp, "
          f"{r['n_negative']} negative / {r['n_positive']} positive, Wilcoxon p = {pval(r['wilcoxon_p'])}.")
    if len(gc):
        g1 = gc[gc["measure"] == "wg_refusal"]
        flips = g1[~g1["same_sign"]]
        w(f"- Old-vs-new WG contrasts (pair x method, incl. originals): {len(g1)} contrasts; largest shift from dropping "
          f"the overlap items {num(g1['shift_pp'].abs().max(), 2)} pp; sign changes: "
          + (", ".join(f"{r.pair} {r.method}" for r in flips.itertuples()) if len(flips) else "none") + ".")
        w("")
        w(md_table(g1.assign(**{
            "old all %": g1["old_harmful_all"].map(pct), "new all %": g1["new_harmful_all"].map(pct),
            "new-old pp (all)": g1["new_minus_old_pp_harmful_all"].map(lambda v: num(v, 1)),
            "new-old pp (excl.)": g1["new_minus_old_pp_harmful_excl_advbench"].map(lambda v: num(v, 1)),
        })[["pair", "method", "old_model", "new_model", "old all %", "new all %", "new-old pp (all)", "new-old pp (excl.)"]]))
        w("")
    k = ctx["adv_corr"]
    if len(k):
        r = k.iloc[0]
        w(f"- Across cells, WG refusal with vs without the overlap items: Spearman rho = {num(r['rho'], 3)} (n = {r['n_cells']}).")
    bb = bcorr[(bcorr["cell_set"] == "primary_all") & (bcorr["y"] == "heldout_wg_refusal")]
    w("- Dev-vs-held-out Spearman (primary cells, WG) by item subset: " +
      "; ".join(f"{r.item_subset} {num(r.rho)} [{num(r.ci_lo)}, {num(r.ci_hi)}]" for r in bb.itertuples()) + ".")
    w("")

    # ---- caveats
    w("## Caveats")
    w("")
    for cv in ctx["caveats"]:
        w(f"- {cv}")
    w("")
    w("## Files")
    w("")
    for f, dsc in ctx["files"]:
        w(f"- `results/analysis/{f}` -- {dsc}")
    w("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-advbench-map", action="store_true",
                    help="only build results/analysis/secondary_advbench_overlap_splits.csv (needs pyarrow)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.build_advbench_map:
        df = build_advbench_map()
        df.to_csv(ADV_MAP, index=False)
        h = df[df["kind"] == "harmful"]
        print(f"wrote {ADV_MAP} ({len(df)} items; {df.attrs['source']}); harmful: {len(h)} (stored flag = re-match "
              f"{int((h.overlap_type == h.overlap_recomputed).sum())}), fit-match {int(h.matches_fit_prompts.sum())}, "
              f"dev-match {int(h.matches_dev_prompts.sum())}, unused-only {int(h.matches_unused_only.sum())}; benign "
              f"meeting the rule: {list(df.loc[df['kind'] == 'benign', 'id'])}")
        return

    rj, rj_path, cr, trials, results = load_inputs()
    stale = stale_judgments_warning(rj_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rj_time = datetime.fromtimestamp(rj_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log(f"secondary.py {now}; input {rj_path.relative_to(ROOT)} ({len(rj):,} rows, assembled {rj_time})")
    if stale:
        log(f"WARNING: {len(stale)} change(s) in the result trees since the last assembly -> re-run assemble.py "
            f"first: {stale[:5]}")

    adv, adv_note = load_advbench_map()
    adv_ids: dict[str, set] = {}
    adv_counts = ""
    if adv is not None:
        for c in ("overlap_type", "overlap_recomputed"):
            adv[c] = adv[c].fillna("").astype(str)
        tb = lambda x: x.astype(str).str.lower().isin(["true", "1"])  # noqa: E731
        h = adv[adv["kind"] == "harmful"]
        bn = adv[(adv["kind"] == "benign") & (adv["overlap_recomputed"] != "")]
        adv_ids = {"fit": set(h.loc[tb(h["matches_fit_prompts"]), "id"]),
                   "dev": set(h.loc[tb(h["matches_dev_prompts"]), "id"]),
                   "benign": set(bn["id"])}
        both = len(adv_ids["fit"] & adv_ids["dev"])
        unused = int(tb(h["matches_unused_only"]).sum())
        adv_counts = (f"{len(adv_ids['fit'])} match a fitting prompt, {len(adv_ids['dev'])} a dev prompt ({both} both), "
                      f"{unused} only prompts Heretic does not use; by type {h['overlap_type'].value_counts().to_dict()}; "
                      f"re-match of the stored harmful flags: {int((h['overlap_type'] == h['overlap_recomputed']).sum())}"
                      f"/{len(h)} identical. Benign items (not checked by build_refusal_sets.py) meeting the same rule: "
                      + (", ".join(f"{r.id} ({r.overlap_recomputed})" for r in bn.itertuples()) or "none")
                      + " -- `benign_excl_advbench` drops them")
        flagged = set(rj.loc[rj["adv_overlap"] == 1, "id"])
        if set(h["id"]) != flagged:
            log(f"WARNING: AdvBench map ids differ from the flagged items ({len(set(h['id']) ^ flagged)} differ)")
    log(f"AdvBench split map: {adv_note}")
    global ADV_BENIGN
    ADV_BENIGN = adv_ids.get("benign", set())

    # cells = judged cells present in the per-response table; allcells = every cell in cell_rates (incl. pending)
    cells = cell_sort_key(rj.drop_duplicates("label")[CELL_COLS])
    allcells = cr.copy()
    log(f"judged cells: {len(cells)}; statuses in cell_rates: "
        f"{allcells.drop_duplicates('label')['status'].value_counts().to_dict()}")

    # paired original columns for every measure used against the original
    pair_cols = sorted(set(OVERREF_JUDGES.values()) | set(REFUSAL_MEASURES.values()) |
                       {"harm_hb", "harm_wg", "incomplete", "is_degenerate"})
    rj = attach_original(rj, pair_cols)
    n_no_orig = rj.loc[rj["method"] != "original", "o_ref_wg"].isna().groupby(rj["label"]).all()
    if n_no_orig.any():
        log(f"cells without a judged original (no paired comparison): {list(n_no_orig[n_no_orig].index)}")

    # items that look like prompt-driven heuristic false positives: degenerate in every judged original
    orig = rj[rj["method"] == "original"]
    fp_items = []
    if len(orig):
        per = orig.groupby("id")["is_degenerate"].agg(["sum", "count"])
        fp_items = sorted(per.index[(per["sum"] == per["count"]) & (per["count"] > 0)])
    log(f"degenerate in every judged original (treated as heuristic false positives in *_excl_fp): {fp_items}")

    checks = check_against_cell_rates(rj, cells, cr)
    for m in checks[:6]:
        log(m)

    a, items = analysis_a(rj, cells)
    dev = dev_table(cells, trials, results, cr)
    bt, bcorr, bwithin = analysis_b(rj, cells, dev, adv_ids)
    c = analysis_c(rj, cells, cr, fp_items)
    d, pooled, gc = analysis_d(rj, cells, adv_ids)

    # over-refusal vs harmful refusal across cells
    ab = a[a["item_group"] == "benign_all"][["label", "role", "wg_overrefusal_rate", "kw_overrefusal_rate"]]
    hb = bt[["label", "wg_refusal__harmful_all", "kw_refusal__harmful_all"]]
    oc = ab.merge(hb, on="label")
    overref_corr = []
    for cs, mask in (("all_judged_cells", oc["role"].notna()), ("modified_cells", oc["role"] != "original")):
        o = oc[mask]
        for x, y, nm in (("wg_overrefusal_rate", "wg_refusal__harmful_all", "wg"),
                         ("kw_overrefusal_rate", "kw_refusal__harmful_all", "kw")):
            overref_corr.append(dict(analysis="a_overrefusal_vs_harmful_refusal", cell_set=cs,
                                     x=f"benign_{nm}_refusal", y=f"harmful_{nm}_refusal", item_subset="benign_all vs harmful_all",
                                     cells=";".join(o["label"])) | spearman_with_ci(o[x].to_numpy(), o[y].to_numpy()))
    overref_corr = pd.DataFrame(overref_corr)
    # rank stability of cells with vs without the overlap items
    sw = d[d["measure"] == "wg_refusal"]
    adv_corr = pd.DataFrame([dict(analysis="d_wg_all_vs_excl_advbench", cell_set="all_judged_cells", x="wg_harmful_all",
                                  y="wg_harmful_excl_advbench", item_subset="", cells=";".join(sw["label"]))
                             | spearman_with_ci(sw["harmful_all_rate"].to_numpy(), sw["harmful_excl_advbench_rate"].to_numpy())])
    corr_all = pd.concat([bcorr, overref_corr, adv_corr], ignore_index=True)

    sel_eq = bt["check_selected_eq_evaluated"].dropna()
    if (sel_eq == False).any():  # noqa: E712
        log(f"WARNING selected != evaluated trial: {list(bt.loc[bt['check_selected_eq_evaluated'] == False, 'label'])}")  # noqa: E712
    dev_eq = bt["check_dev_eq_cell_rates"].dropna()
    log(f"dev refusals equal to cell_rates.csv in {int((dev_eq == True).sum())}/{len(dev_eq)} cells; "  # noqa: E712
        f"selected = evaluated trial in {int((sel_eq == True).sum())}/{len(sel_eq)}")  # noqa: E712

    # cells overview
    st = allcells.drop_duplicates("label")[CELL_COLS + ["status", "source_dir"]]
    st = st.merge(dev[["label", "dev_refusals", "dev_kl", "dev_selected_trial"]], on="label", how="left")
    st["in_secondary_tables"] = st["label"].isin(cells["label"])
    st = cell_sort_key(st)

    files = [
        ("secondary_cells.csv", "every planned/evaluated cell with judge status and whether it is in the tables"),
        ("secondary_overrefusal.csv", "(a) cell x benign group (benign_all, xstest_safe, jbb_benign): WG/KW/KW256/Q3G "
         "over-refusal with Wilson CIs, outcome classes, capped, WG harmful-request share; vs original: delta pp, "
         "McNemar n10/n01, p, Holm p"),
        ("secondary_overrefusal_items.csv", "(a) per benign item: share of cells refusing (WG all/originals/modified, "
         "KW, KW256, Q3G) and WG harmful-request share"),
        ("secondary_dev_vs_heldout.csv", "(b) per cell: selected dev trial, dev keyword refusals and KL, held-out "
         "refusal per measure (`<measure>__<item subset>`), held-out minus dev in pp, consistency checks"),
        ("secondary_dev_vs_heldout_within_model.csv", "(b) per model: method order on dev vs held-out, best method, "
         "within-model Spearman"),
        ("secondary_correlations.csv", "all Spearman analyses (b; a over-refusal vs harmful refusal; d rank "
         "stability): n, rho, p, bootstrap 95% CI, Kendall tau-b, cells used"),
        ("secondary_harm_degenerate.csv", "(c) cell x item group: HB and WG harmful compliance with CIs, "
         "compliance-conditional rates, evasive compliance, HB-WG agreement/kappa, degenerate (all / excl. false "
         "positives), capped, outcome4 shares, capped share among refusals/compliance, vs original (delta, McNemar, "
         "Holm); pending-judging cells with capped rates only"),
        ("secondary_advbench_sensitivity.csv", "(d) cell x measure: rates on all 972 / excl. 44 / overlap 44 / "
         "fit-match / dev-match items with CIs, differences, change vs original on overlap vs non-overlap (DiD), "
         "retention of the original's refusals on overlap / non-overlap / fit-match / dev-match items"),
        ("secondary_advbench_pooled.csv", "(d) Wilcoxon signed-rank over cells of the per-cell retention difference "
         "(overlap - rest) and of the DiD"),
        ("secondary_advbench_gen_contrast.csv", "(d) old-vs-new differences per pair x method with and without the "
         "overlap items"),
        ("secondary_advbench_overlap_splits.csv", "(d) which harmful_behaviors prompt each overlap item matches "
         "(static; built with --build-advbench-map)"),
        ("secondary_log.txt", "console log of the last run (checks, warnings)"),
        ("secondary_summary.md", "this summary"),
    ]
    st.to_csv(OUT / "secondary_cells.csv", index=False)
    cell_sort_key(a).to_csv(OUT / "secondary_overrefusal.csv", index=False)
    items.to_csv(OUT / "secondary_overrefusal_items.csv", index=False)
    cell_sort_key(bt).to_csv(OUT / "secondary_dev_vs_heldout.csv", index=False)
    bwithin.to_csv(OUT / "secondary_dev_vs_heldout_within_model.csv", index=False)
    corr_all.to_csv(OUT / "secondary_correlations.csv", index=False)
    cell_sort_key(c).to_csv(OUT / "secondary_harm_degenerate.csv", index=False)
    cell_sort_key(d).to_csv(OUT / "secondary_advbench_sensitivity.csv", index=False)
    pooled.to_csv(OUT / "secondary_advbench_pooled.csv", index=False)
    gc.to_csv(OUT / "secondary_advbench_gen_contrast.csv", index=False)

    ov = rj[(rj["kind"] == "harmful") & (rj["adv_overlap"] == 1)].drop_duplicates("id")["set"].value_counts()
    ADV_BY_SET = ", ".join(f"{k} {v}" for k, v in ov.items())
    caveats = [
        "Single optimisation seed per cell, greedy decoding, one response per prompt: CIs reflect item sampling only, "
        "not run-to-run variance of the search (TPE/ARA variance is large, EXECUTION_PLAN_v2 finding 1). Seed-2 "
        "replicates of the 10 small-pair cells are compared with seed 1 in final.py (final_seed_*.csv); do not read "
        "cell differences smaller than the seed spread.",
        "Dev and held-out refusal are not like-for-like: different prompts (AdvBench test[:100] vs five benchmark sets), "
        "100 vs 512 new tokens, and Heretic's keyword scorer (normalises emphasis/whitespace, counts empty responses as "
        "refusals) vs WildGuard. The dev value is the LoRA trial's; the merged export was not re-scored on dev.",
        "The cell-level Spearman is computed over 4-5 methods x a few models; methods and models are not independent "
        "(cells share a model, and the selection rule ties KL), so p values are optimistic. Bootstrap CIs over cells "
        "are the better guide. final.py repeats the correlation over every cell with a dev point (final_dev_vs_heldout.csv).",
        "Keyword and WildGuard over-refusal diverge on benign items; the marker list includes words ('illegal', "
        "'harmful', 'disclaimer', 'sorry') that can also occur in helpful answers, and WildGuard is not validated on "
        "edited-model outputs either. Neither is treated as correct until the hand audit; WG is the pre-specified primary.",
        "JBB-benign prompts are topically paired with harmful behaviours; WildGuard itself labels a large share of them "
        "harmful requests, so 'over-refusal' there includes defensible refusals. XSTest-safe is the cleaner benign set.",
        "HarmBench classifier inputs are truncated (generation cut to 512 classifier tokens; behaviour text shortened for "
        "6 long code items, 184 rows); its precision is imperfect (atlas_sweep D1), and on code_weapon items harmfulness "
        "is judged against a behaviour the classifier was not trained on. 'Evasive' compliance (WG compliance, HB not "
        "harmful) mixes genuinely useless answers with HB false negatives -- a hand audit is needed before calling them "
        "useless.",
        "Many responses hit the 512-token cap after ablation (up to ~88% on Qwen3.8 MPOA); judge labels on capped "
        "responses refer to a truncated answer. The 'incomplete' flag never changes the outcome class.",
        "Degenerate = the score_refusals.py heuristic after assemble.py's char_run recheck (58 of 66 raw char_run "
        "flags were code constants, a string copied from the prompt of cw_rmcbench_0346, or box-drawing rules, and "
        "were cleared; 8 runs that continue to the end of the response were kept). The 40 'repetition' flags are "
        "unchanged and not hand-audited (a structural check finds a looping tail in most, templated answers in a few). "
        "*_excl_fp columns drop items flagged degenerate in every judged original (none after the recheck).",
        "AdvBench overlap: the 44 items are not a random subset (by set: " + ADV_BY_SET + "; none from XSTest or "
        "code_weapon), so overlap vs non-overlap differences mix fitting/dev exposure with item content; the DiD "
        "against the original only partly removes this.",
        "Generation GPU differs by pair (RTX PRO 6000 for Qwen/Gemma, RTX 4090 for the small pair, H200 for "
        "OLMo/Mistral); comparisons within a pair are on one GPU type.",
    ]
    ctx = dict(a=a, items=items, bt=bt, bcorr=bcorr, bwithin=bwithin, c=c, d=d, pooled=pooled, gc=gc, cells=cells,
               allcells=allcells, rj=rj, overref_corr=overref_corr, adv_corr=adv_corr, adv_note=adv_note,
               adv_counts=adv_counts, fp_items=fp_items, checks=checks[:1] + checks[1:4], stale=stale, now=now,
               rj_path=str(rj_path.relative_to(ROOT)), rj_time=rj_time, caveats=caveats, files=files)
    (OUT / "secondary_summary.md").write_text(write_summary(ctx))
    log(f"wrote {len(files)} outputs to {OUT.relative_to(ROOT)}/secondary_*")
    (OUT / "secondary_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
