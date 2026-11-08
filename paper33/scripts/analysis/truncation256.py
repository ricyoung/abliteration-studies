#!/usr/bin/env python3
"""Paper 33 -- 256-token judge sensitivity of the held-out refusal labels and of the H3 keyword-minus-judge gap.

Purpose
  Every held-out response (at most 512 new tokens, greedy) was judged in full by WildGuard (primary judge), the
  HarmBench classifier and Qwen3Guard-Gen-8B. Heretic's keyword check and the paper's kw256 measure look at a prefix
  only. On 2026-11-06 the same three judges (same pinned revisions, templates and parsers;
  scripts/bench/score_refusals_256.py) were re-run on the stored 256-token prefix (`response_256`) of every response.
  This script
    0. verifies coverage: every cell used by the paper's analyses (the judged cells of responses_judged, i.e. the
       `used` rows of dir_status.csv) has refusal/judgments_256.jsonl with the same id set as its judgments.jsonl,
       the responses.jsonl that was judged is the one assembled (md5 in dir_status.csv), and the 256 run used the
       main run's judge revisions; it also checks that keyword_refusal_256, n_tokens and finish_reason in the 256
       file equal the assembled values, and that the judges reproduce their full-text label on responses of at most
       256 tokens (identical text);
    1. compares each judge's label on the full response with its label on the 256-token prefix, per cell
       (harmful items): Wilson 95% CIs for both rates, paired difference (256 - full) with the Newcombe hybrid score
       CI, exact McNemar, Holm over the judged cells within each judge;
    2. recomputes the H3 keyword-minus-judge gap on the SAME 256-token text (keyword_refusal_256 vs WildGuard on the
       256-token prefix; also vs Qwen3Guard, and Qwen3Guard vs WildGuard) per cell, with exactly the definitions of
       h3.py gap_cells (Newcombe CI, exact McNemar, Holm per comparison x set group across all judged cells), next to
       the paper's full-text gap read from h3_gap_cells.csv (which is also recomputed here as a check);
    3. pools the gaps and label changes over the same groups as h3.py gap_pooled (prompt-cluster bootstrap, B = 2000,
       percentile 95% CI), next to the paper's values from h3_gap_pooled.csv;
    4. refits the two linear H3 models of h3.py fit_models (primary linear GEE on the paired difference, cell-level
       OLS) with the 256-token gap as outcome, next to the paper's estimates from h3_models.csv.
  Statistical definitions (wilson, newcombe_paired, mcnemar_exact, pooled_groups, model_data, ALPHA, B, SEED) are
  imported from h3.py so that they cannot drift.

Inputs (read-only)
  results/analysis/responses_judged.(parquet|csv.gz)   assembled full-text labels (assemble.py)
  results/analysis/dir_status.csv                      judged dir of each cell, responses md5 at assembly
  results/analysis/h3_gap_cells.csv, h3_gap_pooled.csv, h3_models.csv   the paper's full-text H3 results (h3.py)
  <source_dir>/refusal/judgments.jsonl                 ids of the full-text judging
  <source_dir>/refusal/judgments_256.jsonl             256-token labels: wg_refusal_256, q3g_refusal_256,
                                                       hb_harmful_256, keyword_refusal_256, truncated, n_tokens, ...
  <source_dir>/refusal/summary.json, summary_256.json  judge revisions of the two runs
  <source_dir>/refusal/responses.jsonl                 md5 only (no text is read)

Outputs (results/analysis/; regenerated on every run, nothing else is written)
  trunc256_cells.csv    one row per judged cell (harmful items): n, share of responses longer than 256 tokens, and
                        for WildGuard refusal (wg), Qwen3Guard refusal (q3g), HarmBench harmful (hb): full and 256
                        rates with Wilson CIs, 256 - full with Newcombe CI, discordant counts, exact McNemar, Holm;
                        coverage checks per cell
  trunc256_h3.csv       per (cell, set group, comparison): 256-token gap (rater1 - rater2, both on the 256-token text)
                        with Newcombe CI, exact McNemar, Holm, next to the paper's full-text gap (h3_gap_cells.csv)
  trunc256_pooled.csv   per (subset, pooled group, variant): gap / label change with prompt-cluster bootstrap CI,
                        next to the paper's full-text value (h3_gap_pooled.csv) where one exists
  trunc256_models.csv   primary linear GEE and cell-level OLS of h3.py on the 256-token gap, next to h3_models.csv
  trunc256_summary.md   coverage, checks, headline numbers, caveats

Label mappings (as assemble.py): wg / wg256 = WildGuard 'Response refusal: yes' (None = N/A or unparsable = missing);
q3g / q3g256 = Qwen3Guard 'Refusal: Yes'; hb / hb256 = HarmBench classifier 'Yes' (harmful items only);
kw = any of Heretic's 33 markers in the full response; kw256 = the same in the 256-token prefix (keyword_refusal_256).
A response is "longer than 256 tokens" when n_tokens > 256 (field `truncated` of judgments_256.jsonl); only those
responses have a 256-token text that differs from the full text.

How to run (no GPU; after assemble.py and h3.py; about a minute):
    scripts/figures/.venv/bin/python scripts/analysis/truncation256.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3  # noqa: E402  (scripts/analysis/h3.py: statistics conventions and group definitions)
from h3 import (ALPHA, B, SEED, md_table, mcnemar_exact, newcombe_paired, pct, pp, pval,  # noqa: E402
                wilson)

ROOT = h3.ROOT
OUT = h3.OUT
REPOS = ["allenai/wildguard", "cais/HarmBench-Llama-2-13b-cls", "Qwen/Qwen3Guard-Gen-8B"]

# new 256-token label columns (1 = yes, 0 = no, NaN = missing)
COL = {"kw": "ref_kw", "kw256": "ref_kw256", "wg": "ref_wg", "q3g": "ref_q3g", "hb": "harm_hb",
       "wg256": "ref_wg256", "q3g256": "ref_q3g256", "hb256": "harm_hb256"}
# per-cell label change: name -> (full-text column, 256-token column)
JUDGE_CHANGE = {"wg": ("ref_wg", "ref_wg256"), "q3g": ("ref_q3g", "ref_q3g256"), "hb": ("harm_hb", "harm_hb256")}
JUDGE_DESC = {"wg": "WildGuard refusal", "q3g": "Qwen3Guard refusal", "hb": "HarmBench harmful compliance"}
# 256-token gap comparisons -> full-text counterpart in h3_gap_cells.csv / h3_gap_pooled.csv
GAP256 = {"kw256_vs_wg256": ("kw256", "wg256", "kw_vs_wg"),
          "kw256_vs_q3g256": ("kw256", "q3g256", "kw_vs_q3g"),
          "q3g256_vs_wg256": ("q3g256", "wg256", "q3g_vs_wg")}
SET_GROUPS = ["harmful_all", "harmful_excl_advbench", "benign_all"]


def log(*a):
    print(*a, flush=True)


def yes01(s: pd.Series) -> pd.Series:
    """'yes' -> 1, 'no' -> 0, None / anything else -> NaN (assemble.py nullable_bool(v == 'yes'))."""
    return s.map(lambda v: 1.0 if v == "yes" else (0.0 if v == "no" else np.nan)).astype(float)


def bool01(s: pd.Series) -> pd.Series:
    """True -> 1, False -> 0, None / NaN -> NaN (Python or numpy booleans)."""
    return s.map(lambda v: float(bool(v)) if isinstance(v, (bool, np.bool_)) else np.nan).astype(float)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_counts(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    a = int(((x == 1) & (y == 1)).sum()); b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum()); d = int(((x == 0) & (y == 0)).sum())
    return a, b, c, d


# ------------------------------------------------------------------------------------------------ load + coverage
def load():
    df, jpath = h3.load_judged()
    log(f"read {jpath.relative_to(ROOT)}: {len(df):,} rows, {df.label.nunique()} cells")
    df = df[df.judge_status.isin(["judged", "partially_judged"])].reset_index(drop=True)
    for c in set(h3.RATERS.values()) | {"harm_hb", "harm_wg"}:
        df[c] = h3.to01(df[c])
    for c in ["adv_overlap", "incomplete", "is_degenerate"]:
        df[c] = df[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    return df, jpath


def load_256(df: pd.DataFrame):
    ds = pd.read_csv(OUT / "dir_status.csv", low_memory=False)
    used = ds[ds.used.astype(str) == "True"].set_index("source_dir")
    cov, parts = [], []
    for src, sub in df.groupby("source_dir", sort=False):
        ref = ROOT / src / "refusal"
        j, j256 = ref / "judgments.jsonl", ref / "judgments_256.jsonl"
        row = dict(label=sub.label.iloc[0], source_dir=src, in_dir_status_used=src in used.index,
                   n_assembled=len(sub), has_256=j256.exists())
        ids_full = [json.loads(line)["id"] for line in open(j) if line.strip()]
        row["n_judgments"] = len(ids_full)
        if not j256.exists():
            row.update(gap="judgments_256.jsonl missing")
            cov.append(row)
            continue
        recs = [json.loads(line) for line in open(j256) if line.strip()]
        ids256 = [r["id"] for r in recs]
        row["n_judgments_256"] = len(ids256)
        row["ids_equal_judgments"] = (len(ids256) == len(set(ids256)) == len(ids_full) and set(ids256) == set(ids_full))
        row["ids_equal_assembled"] = set(ids256) == set(sub.id) and len(ids256) == len(sub)
        row["responses_md5_ok"] = (src in used.index and md5(ref / "responses.jsonl") == used.loc[src, "responses_md5"])
        main_rev = json.load(open(ref / "summary.json")).get("judge_revisions", {})
        s256 = json.load(open(ref / "summary_256.json")) if (ref / "summary_256.json").exists() else {}
        rev256 = s256.get("judge_revisions", {})
        row["revisions_ok"] = all(rev256.get(k) is not None and rev256.get(k) == main_rev.get(k) for k in REPOS)
        row["scored_256"] = s256.get("scored", "")
        r = pd.DataFrame(recs)
        r["source_dir"] = src
        row["n_wg256_missing"] = int(r.wg_refusal_256.isna().sum()) if "wg_refusal_256" in r else len(r)
        row["n_q3g256_missing"] = int(r.q3g_refusal_256.isna().sum()) if "q3g_refusal_256" in r else len(r)
        hm = r.kind == "harmful"
        row["n_hb256_missing_harmful"] = int(r.loc[hm, "hb_harmful_256"].isna().sum()) if "hb_harmful_256" in r \
            else int(hm.sum())
        problems = [k for k in ["ids_equal_judgments", "ids_equal_assembled", "responses_md5_ok", "revisions_ok"]
                    if not row[k]]
        row["gap"] = "; ".join(problems)
        cov.append(row)
        parts.append(r[["source_dir", "id", "n_tokens", "finish_reason", "truncated", "keyword_refusal_256",
                        "wg_refusal_256", "q3g_refusal_256", "hb_harmful_256"]].rename(
            columns={"n_tokens": "n_tokens_256file", "finish_reason": "finish_reason_256file",
                     "keyword_refusal_256": "keyword_refusal_256_256file"}))
    cov = pd.DataFrame(cov)
    not_in_df = sorted(set(used.index) - set(df.source_dir))
    j256 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return cov, j256, not_in_df


def merge_256(df: pd.DataFrame, j256: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    df = df.merge(j256, on=["source_dir", "id"], how="left", validate="one_to_one")
    assert len(df) == n0
    harmful = df.kind == "harmful"
    df["ref_wg256"] = yes01(df.wg_refusal_256)
    df["ref_q3g256"] = yes01(df.q3g_refusal_256)
    df["harm_hb256"] = bool01(df.hb_harmful_256.where(harmful, None))
    df["trunc256"] = df.truncated.map({True: True, False: False}).astype("boolean")
    kw_file = bool01(df.keyword_refusal_256_256file)
    chk = dict(
        rows=len(df), rows_with_256=int(df.truncated.notna().sum()),
        kw256_mismatch=int(((kw_file != df.ref_kw256) & kw_file.notna() & df.ref_kw256.notna()).sum()),
        n_tokens_mismatch=int((df.n_tokens_256file != df.n_tokens).sum()),
        finish_reason_mismatch=int((df.finish_reason_256file.astype(str) != df.finish_reason.astype(str)).sum()),
        trunc_flag_mismatch=int((df.trunc256.astype(bool) != (df.n_tokens > 256)).sum()),
    )
    df["trunc256"] = df.trunc256.astype(bool)
    # determinism check: on responses of at most 256 tokens the 256-token text is the full text
    short = ~df.trunc256
    for name, (full, cut) in JUDGE_CHANGE.items():
        a, b, c, d = paired_counts(df.loc[short, cut].values, df.loc[short, full].values)
        chk[f"short_{name}_n"] = a + b + c + d
        chk[f"short_{name}_discordant"] = b + c
    return df, chk


# ------------------------------------------------------------------------------------------------ 1. per-cell labels
def cells_table(df, meta, cov):
    rows = []
    hm = (df.kind == "harmful").values
    for label in meta.label:
        m = hm & (df.label == label).values
        sub = df[m]
        r = dict(label=label, n_harmful=int(m.sum()), n_gt256=int(sub.trunc256.sum()),
                 share_gt256=float(sub.trunc256.mean()), n_capped512=int(sub.incomplete.sum()),
                 share_capped512=float(sub.incomplete.mean()), n_tokens_median=float(sub.n_tokens.median()))
        for name, (full, cut) in JUDGE_CHANGE.items():
            a, b, c, d = paired_counts(sub[cut].values, sub[full].values)  # x = 256, y = full
            n = a + b + c + d
            lf, hf = wilson(a + c, n)
            l2, h2 = wilson(a + b, n)
            delta, lo, hi = newcombe_paired(a, b, c, d)
            sh = ~sub.trunc256.values
            a_, b_, c_, d_ = paired_counts(sub[cut].values[sh], sub[full].values[sh])
            r.update({f"{name}_n": n, f"{name}_k_full": a + c, f"{name}_k_256": a + b,
                      f"{name}_full": (a + c) / n if n else np.nan, f"{name}_full_lo": lf, f"{name}_full_hi": hf,
                      f"{name}_256": (a + b) / n if n else np.nan, f"{name}_256_lo": l2, f"{name}_256_hi": h2,
                      f"{name}_diff": delta, f"{name}_diff_lo": lo, f"{name}_diff_hi": hi,
                      f"{name}_n_256_only": b, f"{name}_n_full_only": c,
                      f"{name}_p_mcnemar_exact": mcnemar_exact(b, c),
                      f"{name}_n_discordant_le256": b_ + c_})
        rows.append(r)
    t = pd.DataFrame(rows)
    for name in JUDGE_CHANGE:
        t[f"{name}_p_holm"] = multipletests(t[f"{name}_p_mcnemar_exact"].values, method="holm")[1]
        t[f"{name}_direction"] = np.where(t[f"{name}_p_holm"] < ALPHA,
                                          np.where(t[f"{name}_diff"] > 0, "256>full", "256<full"), "n.s.")
    t["holm_family"] = "per judge, across all judged cells (harmful_all)"
    t["diff_definition"] = "rate on the 256-token prefix minus rate on the full response (paired, same responses)"
    covc = cov.drop(columns=["source_dir"]).rename(columns={c: f"cov_{c}" for c in cov.columns
                                                         if c not in ("label", "source_dir")})
    t = meta.merge(t, on="label", how="right").merge(covc, on="label", how="left")
    return t


# ------------------------------------------------------------------------------------------------ 2. per-cell gaps
def gap_cells_256(df, meta):
    """Same computation as h3.gap_cells, for the 256-token comparisons and the three main set groups."""
    rows = []
    sg = {k: v for k, v in h3.set_group_masks(df).items() if k in SET_GROUPS}
    comps = dict({k: (r1, r2) for k, (r1, r2, _) in GAP256.items()},
                 kw_vs_wg=("kw", "wg"), kw256_vs_wg=("kw256", "wg"), kw_vs_q3g=("kw", "q3g"), q3g_vs_wg=("q3g", "wg"))
    for label in meta.label:
        lm = (df.label == label).values
        for gname, gm in sg.items():
            m = lm & gm
            if not m.any():
                continue
            sub = df[m]
            for comp, (r1, r2) in comps.items():
                a, b, c, d = paired_counts(sub[COL[r1]].values, sub[COL[r2]].values)
                n = a + b + c + d
                delta, lo, hi = newcombe_paired(a, b, c, d)
                rows.append(dict(label=label, set_group=gname, comparison=comp, rater1=r1, rater2=r2, n=n,
                                 rate1=(a + b) / n if n else np.nan, rate2=(a + c) / n if n else np.nan,
                                 a_both_yes=a, b_r1_only=b, c_r2_only=c, d_both_no=d, gap=delta, gap_lo=lo,
                                 gap_hi=hi, p_mcnemar_exact=mcnemar_exact(b, c),
                                 capped_rate=sub.incomplete.mean(), gt256_rate=sub.trunc256.mean(),
                                 n_tokens_median=sub.n_tokens.median()))
    g = pd.DataFrame(rows)
    g["p_holm"] = np.nan
    for (comp, sgn), idx in g.groupby(["comparison", "set_group"]).groups.items():
        g.loc[idx, "p_holm"] = multipletests(g.loc[idx, "p_mcnemar_exact"].values, method="holm")[1]
    g["holm_family"] = "comparison x set_group across all judged cells"
    g["direction"] = np.where(g.p_holm < ALPHA, np.where(g.gap > 0, "rater1>rater2", "rater1<rater2"), "n.s.")
    return g


def h3_table(g, meta, paper_cells):
    """256-token gaps with the paper's full-text counterpart (and the mixed kw256-vs-full-WildGuard gap) alongside."""
    new = g[g.comparison.isin(GAP256)].copy()
    new["full_comparison"] = new.comparison.map({k: v[2] for k, v in GAP256.items()})
    keep = ["n", "rate1", "rate2", "b_r1_only", "c_r2_only", "gap", "gap_lo", "gap_hi", "p_holm", "direction"]
    pc = paper_cells[paper_cells.set_group.isin(SET_GROUPS)][["label", "set_group", "comparison"] + keep]
    full = pc.rename(columns={"comparison": "full_comparison", **{k: f"full_{k}" for k in keep}})
    new = new.merge(full, on=["label", "set_group", "full_comparison"], how="left")
    mixed = pc[pc.comparison == "kw256_vs_wg"].rename(
        columns={**{k: f"mixed_kw256_vs_wg_{k}" for k in ["gap", "gap_lo", "gap_hi", "p_holm"]}})[
        ["label", "set_group", "mixed_kw256_vs_wg_gap", "mixed_kw256_vs_wg_gap_lo", "mixed_kw256_vs_wg_gap_hi",
         "mixed_kw256_vs_wg_p_holm"]].assign(comparison="kw256_vs_wg256")
    new = new.merge(mixed, on=["label", "set_group", "comparison"], how="left")
    new["gap_minus_full_gap"] = new.gap - new.full_gap
    new["note"] = ("gap = rater1 - rater2 on the same 256-token prefix; full_* = the paper's full-text counterpart "
                   "(h3_gap_cells.csv); mixed_* = keyword on 256 tokens vs WildGuard on the full response "
                   "(h3_gap_cells.csv kw256_vs_wg)")
    return meta.merge(new, on="label", how="right")


def check_recompute_cells(g, paper_cells):
    """Our recomputation of h3's full-text gaps must equal h3_gap_cells.csv (guards against stale inputs)."""
    comps = ["kw_vs_wg", "kw256_vs_wg", "kw_vs_q3g", "q3g_vs_wg"]
    mine = g[g.comparison.isin(comps)].set_index(["label", "set_group", "comparison"])
    theirs = paper_cells[paper_cells.comparison.isin(comps) & paper_cells.set_group.isin(SET_GROUPS)].set_index(
        ["label", "set_group", "comparison"])
    j = mine.join(theirs.add_suffix("_paper"), how="outer")
    return dict(rows=len(j), missing=int(j.n.isna().sum() + j.n_paper.isna().sum()),
                max_abs_gap=float((j.gap - j.gap_paper).abs().max()), n_diff=int((j.n != j.n_paper).sum()),
                max_abs_p_holm=float((j.p_holm - j.p_holm_paper).abs().max()))


# ------------------------------------------------------------------------------------------------ 3. pooled
def pooled(df, groups, rng, paper_pooled):
    """Prompt-cluster bootstrap of the mean paired difference x - y, exactly as h3.gap_pooled."""
    trunc = df.trunc256.values
    cap = df.incomplete.values
    d_gap = lambda c1, c2: (df[COL[c1]] - df[COL[c2]]).values  # noqa: E731
    variants = {  # name: (x, y, row filter, rater1, rater2, paper counterpart variant)
        "kw_vs_wg": (df.ref_kw.values, df.ref_wg.values, None, "kw", "wg", "kw_vs_wg"),
        "kw256_vs_wg": (df.ref_kw256.values, df.ref_wg.values, None, "kw256", "wg", "kw256_vs_wg"),
        "kw256_vs_wg256": (df.ref_kw256.values, df.ref_wg256.values, None, "kw256", "wg256", "kw_vs_wg"),
        "kw256_vs_wg256_complete_only": (df.ref_kw256.values, df.ref_wg256.values, ~cap, "kw256", "wg256",
                                         "kw_vs_wg_complete_only"),
        "kw256_vs_wg256_capped_only": (df.ref_kw256.values, df.ref_wg256.values, cap, "kw256", "wg256",
                                       "kw_vs_wg_capped_only"),
        "kw_vs_wg_le256_only": (df.ref_kw.values, df.ref_wg.values, ~trunc, "kw", "wg", None),
        "kw_vs_wg_gt256_only": (df.ref_kw.values, df.ref_wg.values, trunc, "kw", "wg", None),
        "kw256_vs_wg256_le256_only": (df.ref_kw256.values, df.ref_wg256.values, ~trunc, "kw256", "wg256", None),
        "kw256_vs_wg256_gt256_only": (df.ref_kw256.values, df.ref_wg256.values, trunc, "kw256", "wg256", None),
        "kw256_vs_q3g256": (df.ref_kw256.values, df.ref_q3g256.values, None, "kw256", "q3g256", "kw_vs_q3g"),
        "q3g256_vs_wg256": (df.ref_q3g256.values, df.ref_wg256.values, None, "q3g256", "wg256", "q3g_vs_wg"),
        "wg256_vs_wg": (df.ref_wg256.values, df.ref_wg.values, None, "wg256", "wg", None),
        "q3g256_vs_q3g": (df.ref_q3g256.values, df.ref_q3g.values, None, "q3g256", "q3g", None),
        "hb256_vs_hb": (df.harm_hb256.values, df.harm_hb.values, None, "hb256", "hb", None),
        "wg256_vs_wg_gt256_only": (df.ref_wg256.values, df.ref_wg.values, trunc, "wg256", "wg", None),
        "wg256_vs_wg_complete_only": (df.ref_wg256.values, df.ref_wg.values, ~cap, "wg256", "wg", None),
        "wg256_vs_wg_capped_only": (df.ref_wg256.values, df.ref_wg.values, cap, "wg256", "wg", None),
        "kw256_vs_q3g256_complete_only": (df.ref_kw256.values, df.ref_q3g256.values, ~cap, "kw256", "q3g256", None),
        "kw256_vs_q3g256_capped_only": (df.ref_kw256.values, df.ref_q3g256.values, cap, "kw256", "q3g256", None),
        # change of the H3 gap: (kw256 - wg256) - (kw - wg) on responses with all four labels
        "gap256_minus_gapfull": (d_gap("kw256", "wg256"), d_gap("kw", "wg"), None, "kw256-wg256", "kw-wg", None),
    }
    pv = paper_pooled.set_index(["subset", "group", "variant"])
    rows = []
    for sname, smask in {"harmful_all": (df.kind == "harmful").values,
                         "harmful_excl_advbench": ((df.kind == "harmful") & ~df.adv_overlap).values,
                         "benign_all": (df.kind == "benign").values}.items():
        ids = np.sort(df.id[smask].unique())
        n_p = len(ids)
        pmap = pd.Series(np.arange(n_p), index=ids)
        W = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=B).astype(np.float64)
        for gname, gm in groups.items():
            for vname, (xa, ya, filt, r1, r2, pvar) in variants.items():
                m = smask & gm & (filt if filt is not None else True)
                if not m.any():
                    continue
                x, y = xa[m].astype(float), ya[m].astype(float)
                ok = ~(np.isnan(x) | np.isnan(y))
                if not ok.any():
                    continue
                x, y = x[ok], y[ok]
                p = pmap.loc[df.id.values[m][ok]].values
                Cg = np.stack([np.bincount(p, weights=x - y, minlength=n_p),
                               np.bincount(p, minlength=n_p).astype(float),
                               np.bincount(p, weights=x, minlength=n_p),
                               np.bincount(p, weights=y, minlength=n_p)], 1)
                s, nn, sx, sy = Cg.sum(0)
                Cb = W @ Cg
                with np.errstate(invalid="ignore", divide="ignore"):
                    gb = Cb[:, 0] / Cb[:, 1]
                lo, hi = np.nanpercentile(gb, [2.5, 97.5])
                binary = vname != "gap256_minus_gapfull"
                r = dict(subset=sname, group=gname, variant=vname, rater1=r1, rater2=r2,
                         n_cells=len(np.unique(df.label.values[m][ok])), n=int(nn),
                         share_gt256=float(trunc[m][ok].mean()), rate1=sx / nn, rate2=sy / nn,
                         gap=s / nn, gap_lo=lo, gap_hi=hi,
                         b_r1_only=int(((x == 1) & (y == 0)).sum()) if binary else np.nan,
                         c_r2_only=int(((x == 0) & (y == 1)).sum()) if binary else np.nan,
                         paper_variant=pvar or "")
                if pvar and (sname, gname, pvar) in pv.index:
                    q = pv.loc[(sname, gname, pvar)]
                    r.update(paper_n=q.n, paper_rate1=q.rate1, paper_rate2=q.rate2, paper_gap=q.gap,
                             paper_gap_lo=q.gap_lo, paper_gap_hi=q.gap_hi)
                rows.append(r)
    out = pd.DataFrame(rows)
    out["note"] = ("gap = mean(x - y), x/y = rater1/rater2 labels (rates for binary variants; per-response gaps for "
                   f"gap256_minus_gapfull); 95% CI = prompt-cluster bootstrap percentile (B = {B}, seed {SEED}, h3.py "
                   "convention); paper_* = h3_gap_pooled.csv value of paper_variant (its own bootstrap draws)")
    return out


# ------------------------------------------------------------------------------------------------ 4. models
def fit_linear_models(df, which):
    """The two linear H3 models of h3.fit_models, with outcome ref_kw - ref_wg replaced by the chosen pair."""
    kwc, wgc = {"gap256": ("ref_kw256", "ref_wg256"), "full": ("ref_kw", "ref_wg")}[which]
    d0 = df.copy()
    d0["ref_kw"], d0["ref_wg"] = df[kwc], df[wgc]
    term_names = {"Intercept": "mean gap (kw - wg), averaged over generation x class",
                  "gen_c": "generation effect on gap (new - old), averaged over classes",
                  "ara_c": "class effect on gap (ARA - direction), averaged over generations",
                  "gen_c:ara_c": "generation x class interaction on gap (DiD)"}
    rows = []
    for study in ["main", "small"]:
        for subset in ["harmful_all", "harmful_excl_advbench", "benign_all"]:
            d, pairs = h3.model_data(d0, study, subset)
            if d.empty or d.label.nunique() < 4:
                continue
            pair_term = " + C(pair, Sum)" if len(pairs) > 1 else ""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = smf.gee("gap ~ gen_c * ara_c" + pair_term, groups="pid", data=d,
                            cov_struct=sm.cov_struct.Exchangeable(), family=sm.families.Gaussian()).fit()
            for t, lab in term_names.items():
                rows.append(dict(outcome=which, model="GEE_linear_paired_gap (primary)", study=study, subset=subset,
                                 term=lab, estimate=r.params[t], se=r.bse[t], lo=r.params[t] - 1.96 * r.bse[t],
                                 hi=r.params[t] + 1.96 * r.bse[t], p=r.pvalues[t], n_obs=len(d),
                                 n_clusters=d.pid.nunique(), pairs="+".join(pairs)))
            cg = d.groupby(["label", "pair", "gen_c", "ara_c"], as_index=False).gap.mean()
            if len(cg) >= 4 + (len(pairs) - 1) + 2:
                r = smf.ols("gap ~ gen_c * ara_c" + pair_term, data=cg).fit()
                for t, lab in term_names.items():
                    rows.append(dict(outcome=which, model="OLS_cell_level_gap", study=study, subset=subset, term=lab,
                                     estimate=r.params[t], se=r.bse[t], lo=r.params[t] - 1.96 * r.bse[t],
                                     hi=r.params[t] + 1.96 * r.bse[t], p=r.pvalues[t], n_obs=len(cg),
                                     n_clusters=len(cg), pairs="+".join(pairs)))
    m = pd.DataFrame(rows)
    m["p_holm"] = np.nan
    prim = m.model == "GEE_linear_paired_gap (primary)"
    for _, idx in m[prim].groupby(["study", "subset"]).groups.items():
        m.loc[idx, "p_holm"] = multipletests(m.loc[idx, "p"].values, method="holm")[1]
    return m


def models_table(df, paper_models):
    m256 = fit_linear_models(df, "gap256")
    mfull = fit_linear_models(df[df.ref_wg256.notna()], "full")  # full-text gap on the rows that have wg256
    key = ["model", "study", "subset", "term"]
    keep = ["estimate", "lo", "hi", "p", "p_holm", "n_obs"]
    out = m256.merge(mfull[key + keep].rename(columns={k: f"full_samerows_{k}" for k in keep}), on=key, how="left")
    pm = paper_models[paper_models.model.isin(["GEE_linear_paired_gap (primary)", "OLS_cell_level_gap"])]
    out = out.merge(pm[key + keep].rename(columns={k: f"paper_{k}" for k in keep}), on=key, how="left")
    out["holm_family"] = np.where(out.p_holm.notna(), "4 terms of the primary GEE within study x subset", "")
    out["note"] = ("outcome gap256 = keyword_refusal_256 - WildGuard refusal on the 256-token prefix; specification "
                   "identical to h3.fit_models (models 1 and 2); paper_* = h3_models.csv (full text); "
                   "full_samerows_* = full-text gap refitted on the rows with a WildGuard 256-token label")
    return out


# ------------------------------------------------------------------------------------------------ summary
def write_summary(ctx):
    cov, chk, cells, h3t, pool, models, rchk = (ctx[k] for k in ["cov", "chk", "cells", "h3t", "pool", "models",
                                                                 "rchk"])
    L = ["# Paper 33 -- 256-token judge sensitivity (H3 robustness)", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by scripts/analysis/truncation256.py "
         "(read-only on every input). Numbers are relative to the judges, not to human labels.", "",
         "## Coverage", ""]
    gaps = cov[cov.gap.fillna("") != ""]
    L.append(f"- Judged cells in responses_judged: {len(cov)}; with refusal/judgments_256.jsonl: {int(cov.has_256.sum())}; "
             f"id set equal to judgments.jsonl: {int(cov.ids_equal_judgments.fillna(False).sum())}; equal to the "
             f"assembled rows: {int(cov.ids_equal_assembled.fillna(False).sum())}; responses.jsonl md5 equal to the "
             f"assembly: {int(cov.responses_md5_ok.fillna(False).sum())}; judge revisions equal to the main run: "
             f"{int(cov.revisions_ok.fillna(False).sum())}.")
    L.append(f"- `used` dirs of dir_status.csv without assembled rows: {ctx['not_in_df'] or 'none'}.")
    L.append("- Coverage gaps: " + ("none." if gaps.empty else "; ".join(f"{r.label}: {r.gap}" for r in gaps.itertuples())))
    miss = cov[(cov.n_wg256_missing > 0) | (cov.n_q3g256_missing > 0) | (cov.n_hb256_missing_harmful > 0)]
    L.append(f"- Missing labels on the 256-token text (unparsable judge output, treated as missing as in the main "
             f"analysis): WildGuard {int(cov.n_wg256_missing.sum())} rows, Qwen3Guard {int(cov.n_q3g256_missing.sum())}, "
             f"HarmBench (harmful rows) {int(cov.n_hb256_missing_harmful.sum())}"
             + (" (" + ", ".join(f"{r.label} {int(r.n_wg256_missing)}" for r in miss.itertuples()) + ")" if len(miss) else "")
             + f". 256-token runs scored {cov.scored_256.min()} to {cov.scored_256.max()}.")
    L += ["", "## Checks", ""]
    L.append(f"- keyword_refusal_256 in judgments_256.jsonl vs the assembled ref_kw256: {chk['kw256_mismatch']} "
             f"mismatches; n_tokens: {chk['n_tokens_mismatch']}; finish_reason: {chk['finish_reason_mismatch']}; "
             f"`truncated` vs n_tokens > 256: {chk['trunc_flag_mismatch']}.")
    L.append("- Responses of at most 256 tokens (256-token text = full text), label on the prefix vs on the full "
             "response: " + "; ".join(f"{JUDGE_DESC[k]} {chk[f'short_{k}_discordant']} discordant of "
                                       f"{chk[f'short_{k}_n']:,}" for k in JUDGE_CHANGE)
             + " (the judges are deterministic up to batch numerics).")
    L.append(f"- Recomputed full-text gaps vs h3_gap_cells.csv ({rchk['rows']} cell x set group x comparison rows): "
             f"max |gap difference| {rchk['max_abs_gap']:.2e}, n differs in {rchk['n_diff']} rows, max |Holm p "
             f"difference| {rchk['max_abs_p_holm']:.2e}, unmatched rows {rchk['missing']}.")
    pchk = pool[(pool.variant.isin(["kw_vs_wg", "kw256_vs_wg"])) & pool.paper_gap.notna()]
    L.append(f"- Recomputed pooled full-text gaps vs h3_gap_pooled.csv ({len(pchk)} rows): max |gap difference| "
             f"{(pchk.gap - pchk.paper_gap).abs().max():.2e}, n differs in {int((pchk.n != pchk.paper_n).sum())}; "
             f"bootstrap limits differ by up to {max((pchk.gap_lo - pchk.paper_gap_lo).abs().max(), (pchk.gap_hi - pchk.paper_gap_hi).abs().max()) * 100:.2f} pp "
             "(different random draws).")

    def P(group, variant, subset="harmful_all"):
        s = pool[(pool.subset == subset) & (pool.group == group) & (pool.variant == variant)]
        return None if s.empty else s.iloc[0]

    def gci(r, paper=False):
        if paper:
            return f"{pp(r.paper_gap)} pp [{pp(r.paper_gap_lo)}, {pp(r.paper_gap_hi)}]"
        return f"{pp(r.gap)} pp [{pp(r.gap_lo)}, {pp(r.gap_hi)}]"

    L += ["", "## Headline numbers (harmful items)", ""]
    for g, gl in [("ablated_primary", "42 ablated seed-1 cells"), ("original", "10 originals")]:
        w, q, h, sh = P(g, "wg256_vs_wg"), P(g, "q3g256_vs_q3g"), P(g, "hb256_vs_hb"), P(g, "wg256_vs_wg_gt256_only")
        if w is not None:
            L.append(f"- **Judge labels, {gl}:** WildGuard refusal {pct(w.rate2)}% on the full response vs "
                     f"{pct(w.rate1)}% on the 256-token prefix ({gci(w)}; {int(w.c_r2_only):,} responses lose and "
                     f"{int(w.b_r1_only):,} gain the refusal label; among the {sh.n:,} responses longer than 256 tokens "
                     f"{gci(sh)}). Qwen3Guard refusal {pct(q.rate2)}% vs {pct(q.rate1)}% ({gci(q)}). HarmBench "
                     f"harmful {pct(h.rate2)}% vs {pct(h.rate1)}% ({gci(h)}).")
    for g, gl in [("ablated_primary", "42 ablated cells"), ("ablated_primary:main", "main study (32)"),
                  ("ablated_primary:small", "small pair (10)"), ("original", "originals (10)"),
                  ("all_judged", "all 74 judged cells")]:
        a, b, c, dd = P(g, "kw_vs_wg"), P(g, "kw256_vs_wg"), P(g, "kw256_vs_wg256"), P(g, "gap256_minus_gapfull")
        if a is not None and c is not None:
            L.append(f"- **H3 gap, {gl}:** full text kw - WG {gci(a, True)} (paper); keyword on 256 tokens vs WG on "
                     f"the full response {gci(b, True)} (paper, mixed); both on the same 256-token text "
                     f"{gci(c)} (keyword {pct(c.rate1)}%, WildGuard {pct(c.rate2)}%); change vs full text {gci(dd)}.")
    c1, c2 = P("ablated_primary", "kw256_vs_wg256_complete_only"), P("ablated_primary", "kw256_vs_wg256_capped_only")
    s1, s2 = P("ablated_primary", "kw256_vs_wg256_le256_only"), P("ablated_primary", "kw256_vs_wg256_gt256_only")
    f2 = P("ablated_primary", "kw_vs_wg_gt256_only")
    if c1 is not None:
        L.append(f"- **By length, 42 ablated cells, 256-token gap:** responses that ended before the 512 cap "
                 f"{gci(c1)} (paper full text {gci(c1, True)}), capped responses {gci(c2)} (paper {gci(c2, True)}); "
                 f"responses of at most 256 tokens {gci(s1)} (n = {s1.n:,}), longer responses {gci(s2)} "
                 f"(n = {s2.n:,}; full-text gap on the same responses {gci(f2)}).")
    q1, q2 = P("ablated_primary", "kw256_vs_q3g256"), P("ablated_primary", "q3g256_vs_wg256")
    if q1 is not None:
        L.append(f"- **Second judge, 42 ablated cells, 256-token text:** kw256 - Qwen3Guard {gci(q1)} (paper full text "
                 f"kw - Q3G {gci(q1, True)}); Qwen3Guard - WildGuard {gci(q2)} (paper {gci(q2, True)}).")
    s = h3t[(h3t.set_group == "harmful_all") & (h3t.comparison == "kw256_vs_wg256") & (h3t.role == "primary")]
    if len(s):
        def cnt(gcol, pcol):
            return (int((s[gcol] > 0).sum()), int(((s[gcol] > 0) & (s[pcol] < ALPHA)).sum()),
                    int((s[gcol] < 0).sum()), int(((s[gcol] < 0) & (s[pcol] < ALPHA)).sum()))
        n1, n2 = cnt("gap", "p_holm"), cnt("full_gap", "full_p_holm")
        neg = s[(s.gap < 0) & (s.p_holm < ALPHA)].label.tolist()
        L.append(f"- **Per ablated cell (Holm over 74 cells):** 256-token gap > 0 in {n1[0]}/{len(s)} "
                 f"({n1[1]} significant), < 0 in {n1[2]} ({n1[3]} significant: {', '.join(neg) or 'none'}); "
                 f"full text (paper) > 0 in {n2[0]} ({n2[1]}), < 0 in {n2[2]} ({n2[3]}). Gap smaller on the 256-token "
                 f"text than on the full text in {int((s.gap < s.full_gap).sum())}/{len(s)} cells.")
    mm = models[(models.model == "GEE_linear_paired_gap (primary)") & (models.subset == "harmful_all")]
    for study in ["main", "small"]:
        x = mm[mm.study == study]
        o = models[(models.model == "OLS_cell_level_gap") & (models.subset == "harmful_all") & (models.study == study)]
        if len(x) == 4:
            L.append(f"- **H3 models, {study}, harmful (256-token gap; paper full-text value in brackets):** " +
                     "; ".join(f"{r.term.split(' (')[0].split(',')[0]} {pp(r.estimate)} pp [{pp(r.lo)}, {pp(r.hi)}], "
                               f"Holm p {pval(r.p_holm)} ({pp(r.paper_estimate)} pp, Holm p {pval(r.paper_p_holm)})"
                               for r in x.itertuples()) +
                     (". Cell-level OLS p: " + ", ".join(f"{pval(p)} ({pval(q)})" for p, q in zip(o.p, o.paper_p))
                      if len(o) == 4 else "") + ".")

    L += ["", "## Pooled table (harmful_all; source of paper/sections/08e_truncation256.tex, tab:trunc256)", ""]
    groups = [("original", ""), ("ablated_primary", ""), ("ablated_primary:main", ""), ("ablated_primary:small", ""),
              ("ablated_primary", "_complete_only"), ("ablated_primary", "_capped_only"), ("role:control", ""),
              ("role:external", ""), ("role:replicate", ""), ("all_judged", "")]
    rows = []
    for g, suf in groups:
        a, c = P(g, "kw256_vs_wg256" + suf), P(g, "kw256_vs_q3g256" + suf)
        w = P(g, "wg256_vs_wg" + suf)
        if a is None or c is None or w is None:
            continue
        rows.append({"group": g + (f" ({suf.strip('_')})" if suf else ""), "cells": int(a.n_cells), "n": int(a.n),
                     ">256 tok %": pct(a.share_gt256), "WG full %": pct(w.rate2), "WG 256 %": pct(w.rate1),
                     "WG 256-full": gci(w), "kw-WG full (paper)": gci(a, True), "kw256-WG256": gci(a),
                     "kw256-Q3G256": gci(c)})
    L.append(md_table(pd.DataFrame(rows)))
    L += ["", "## Caveats", "",
          "- The 256-token rerun answers whether the judges' labels, and the keyword-judge gap, depend on how much of a "
          "response the judge sees. It does not say which label is correct; there is no human audit.",
          "- Only responses longer than 256 tokens can change; on the others the prefix is the full text, and the "
          "discordant counts in 'Checks' measure judge nondeterminism (batching), not truncation.",
          "- The 256-token WildGuard label is missing for a handful of rows (unparsable output); every comparison uses "
          "the rows on which both of its labels exist (pairwise deletion, as h3.py), so n can differ from the "
          "full-text n by those rows.",
          "- Pooled CIs come from new bootstrap draws (same B, seed and method as h3.py, but h3.py consumes random "
          "numbers in its agreement step first), so the paper_* limits and the recomputed limits differ slightly.",
          "- Holm families: per-cell label changes, per judge across the 74 cells; per-cell gaps, per comparison x set "
          "group across the 74 cells (h3.py convention); GEE terms, 4 per study x subset."]
    (OUT / "trunc256_summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df, jpath = load()
    cov, j256, not_in_df = load_256(df)
    log(f"coverage: {len(cov)} cells, {int(cov.has_256.sum())} with judgments_256.jsonl, gaps: "
        f"{(cov.gap.fillna('') != '').sum()}")
    for r in cov[cov.gap.fillna("") != ""].itertuples():
        log(f"  GAP {r.label}: {r.gap}")
    if not cov.has_256.all():
        raise SystemExit("judgments_256.jsonl missing for some judged cells: see the list above")
    df, chk = merge_256(df, j256)
    log("checks:", chk)
    for c in ["wg_refusal_256", "q3g_refusal_256"]:
        log(f"  {c} values: {df[c].value_counts(dropna=False).to_dict()}")

    meta = h3.sort_key_cells(df.groupby("label", as_index=False).first()[
        ["label", "model", "family", "pair", "study", "generation", "gen_new", "method", "variant", "method_class",
         "role", "seed", "source_dir"]])
    paper_cells = pd.read_csv(OUT / "h3_gap_cells.csv", low_memory=False)
    paper_pooled = pd.read_csv(OUT / "h3_gap_pooled.csv", low_memory=False)
    paper_models = pd.read_csv(OUT / "h3_models.csv", low_memory=False)

    log("per-cell label changes ...")
    cells = cells_table(df, meta, cov)
    cells.to_csv(OUT / "trunc256_cells.csv", index=False)

    log("per-cell gaps ...")
    g = gap_cells_256(df, meta)
    rchk = check_recompute_cells(g, paper_cells)
    log("  recompute vs h3_gap_cells.csv:", rchk)
    h3t = h3_table(g, meta, paper_cells)
    h3t.to_csv(OUT / "trunc256_h3.csv", index=False)

    log("pooled (prompt-cluster bootstrap) ...")
    groups = h3.pooled_groups(df)
    pool = pooled(df, groups, rng, paper_pooled)
    pool.to_csv(OUT / "trunc256_pooled.csv", index=False)

    log("models ...")
    models = models_table(df, paper_models)
    models.to_csv(OUT / "trunc256_models.csv", index=False)

    write_summary(dict(cov=cov, chk=chk, cells=cells, h3t=h3t, pool=pool, models=models, rchk=rchk,
                       not_in_df=not_in_df))
    log(f"done in {time.time() - t0:.0f}s -> results/analysis/trunc256_{{cells,h3,pooled,models}}.csv, "
        "trunc256_summary.md")


if __name__ == "__main__":
    main()
