#!/usr/bin/env python3
"""Paper 33 -- revision 2026-11-08 (ARS review, REV-19, REV-36, REV-39): three small summaries of existing data.

  (1) REV-19  A scale-free sensitivity estimand for the baseline adjustment: conditional refusal retention, the share
              of the prompts refused by a model's own original that the edited model still refuses (WildGuard,
              972 harmful held-out prompts). Generation effect = retention(newer) - retention(older), per pair x
              method and per class (direction mean; ARA) and the H2 contrast, with prompt-cluster bootstrap 95%
              intervals (B = 2000, seed 20261108; prompts resampled jointly for both generations). Prompt sampling only.
  (2) REV-39  Mean absolute keyword-minus-WildGuard gap over the 42 ablated seed-1 models at each keyword window,
              next to the mean signed gap (from review_out_c4_gap_cells.csv, harmful_all).
  (3) REV-36  OLMo per-method and direction-class generation effects under the second judge (Qwen3Guard) next to
              WildGuard (h1h2_generation_effects.csv, subset all).

Read-only on every input. Writes results/analysis/revision_retention.csv, revision_kw_gap_abs.csv,
revision_olmo_q3g.csv and revision_misc_summary.md.

Run (from the paper root):
    scripts/figures/.venv/bin/python scripts/analysis/revision_misc_estimands.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
PAIRS = ["qwen", "gemma", "olmo", "mistral", "qwen-small"]
DIR = ["dim", "mpoa", "som", "jspace"]
B = 2000


def main() -> None:
    L = ["# Revision analyses REV-19 / REV-36 / REV-39\n"]
    # ------------------------------------------------------------------ (1) conditional retention
    rj = pd.read_csv(OUT / "responses_judged.csv.gz", low_memory=False,
                     usecols=["label", "pair", "role", "method", "gen_new", "kind", "id", "ref_wg"])
    h = rj[(rj.kind == "harmful") & rj.role.isin(["original", "primary"])].copy()
    h["y"] = h.ref_wg.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    W = h.pivot(index="id", columns="label", values="y")
    lab = {(r.pair, r.method, int(r.gen_new)): r.label for r in h.drop_duplicates("label").itertuples()}
    rng = np.random.default_rng(20261108)
    n = len(W)
    boots = rng.integers(0, n, size=(B, n))
    rows = []

    def retention(pair, method, g, idx=None):
        o = W[lab[(pair, "original", g)]].values
        e = W[lab[(pair, method, g)]].values
        if idx is not None:
            o, e = o[idx], e[idx]
        m = (o == 1) & ~np.isnan(e)
        return float(np.nanmean(e[m]))

    for pair in PAIRS:
        dms = [m for m in DIR if (pair, m, 0) in lab]
        est = {}
        bt = {}
        for meth in dms + ["ara"]:
            est[meth] = retention(pair, meth, 1) - retention(pair, meth, 0)
            bt[meth] = np.array([retention(pair, meth, 1, b) - retention(pair, meth, 0, b) for b in boots])
            rows.append(dict(pair=pair, estimand=meth, ret_old=retention(pair, meth, 0), ret_new=retention(pair, meth, 1),
                             effect_pp=100 * est[meth], lo=100 * np.percentile(bt[meth], 2.5),
                             hi=100 * np.percentile(bt[meth], 97.5)))
        dmean, dbt = np.mean([est[m] for m in dms]), np.mean([bt[m] for m in dms], axis=0)
        for name, e, b in (("direction class", dmean, dbt), ("H2 contrast", est["ara"] - dmean, bt["ara"] - dbt)):
            rows.append(dict(pair=pair, estimand=name, effect_pp=100 * e, lo=100 * np.percentile(b, 2.5),
                             hi=100 * np.percentile(b, 97.5)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "revision_retention.csv", index=False)
    L.append("## (1) Conditional refusal retention, generation effect (newer - older), pp [bootstrap 95%]")
    for r in R[R.estimand.isin(["direction class", "ara", "H2 contrast"])].itertuples():
        L.append(f"- {r.pair} {r.estimand}: {r.effect_pp:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]")

    # ------------------------------------------------------------------ (2) mean absolute keyword gap
    g = pd.read_csv(OUT / "review_out_c4_gap_cells.csv")
    g = g[(g.set_group == "harmful_all") & (g.role == "primary")]
    arows = []
    for comp in sorted(g.comparison.unique()):
        s = g[g.comparison == comp]
        arows.append(dict(comparison=comp, n_models=len(s), mean_gap_pp=100 * s.gap.mean(),
                          mean_abs_gap_pp=100 * s.gap.abs().mean(), median_abs_gap_pp=100 * s.gap.abs().median(),
                          max_abs_gap_pp=100 * s.gap.abs().max()))
    A = pd.DataFrame(arows)
    A.to_csv(OUT / "revision_kw_gap_abs.csv", index=False)
    L.append("\n## (2) Keyword - WildGuard gap over the 42 primary ablated models")
    for r in A.itertuples():
        L.append(f"- {r.comparison}: mean {r.mean_gap_pp:+.1f} pp, mean |gap| {r.mean_abs_gap_pp:.1f} pp "
                 f"(median {r.median_abs_gap_pp:.1f}, max {r.max_abs_gap_pp:.1f}), n = {r.n_models}")

    # ------------------------------------------------------------------ (3) OLMo under Qwen3Guard
    ge = pd.read_csv(OUT / "h1h2_generation_effects.csv")
    ge = ge[(ge.pair == "olmo") & (ge.subset == "all") & ge.judge.isin(["wg", "q3g"])]
    q = ge[["judge", "method", "p_old", "p_new", "delta", "delta_lo", "delta_hi", "p_holm_H1"]].copy()
    for j in ("wg", "q3g"):
        s = ge[(ge.judge == j) & ge.method.isin(["dim", "mpoa", "som"])]
        a = ge[(ge.judge == j) & (ge.method == "ara")].delta.iloc[0]
        q = pd.concat([q, pd.DataFrame([dict(judge=j, method="direction class", delta=s.delta.mean()),
                                        dict(judge=j, method="H2 contrast", delta=a - s.delta.mean())])])
    q.to_csv(OUT / "revision_olmo_q3g.csv", index=False)
    L.append("\n## (3) OLMo generation effects, WildGuard vs Qwen3Guard (pp)")
    for r in q.itertuples():
        L.append(f"- {r.judge} {r.method}: {100*r.delta:+.1f}")
    (OUT / "revision_misc_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
