#!/usr/bin/env python3
"""Paper 33 -- revision 2026-11-08 (ARS review, REV-16, REV-24, REV-27): per-pair H2 sensitivity analyses on the
held-out set, from responses already judged.

  (a) REV-16  Cap-bound units only. H2 contrast per pair (ARA generation effect minus the mean generation effect of
              the direction-based methods) restricted to the direction-based pair x method units in which the KL cap
              bound in both cells (review_sel_kl_units.csv: Qwen DiM; Gemma DiM, MPOA, SOM; Mistral Small DiM; OLMo
              none). ARA bound in every cell, so the restricted contrast compares two classes that were both selected
              at the cap. pp scale, paired item differences, with prompt SE and the paper's run-to-run variance
              (2 sigma^2 + 2 sigma^2 / m, sigma = 9.15 pp) and, as a sensitivity, the logit-level sigma of
              revision_seed_variance.py. Pooled over the estimable main pairs with the paper's random-effects model.
  (b) REV-27  Gemma pair without the failed Gemma-4-31B DiM edit: direction class = mean of MPOA and SOM.
  (c) REV-24  Generation difference of the gamma-correction effect per post-norm pair:
              [gMPOA(new) - MPOA(new)] - [gMPOA(old) - MPOA(old)] = gMPOA generation effect - MPOA generation effect,
              paired item differences with prompt SE (no run-to-run variance: each of the four cells was run once).

Outcome: WildGuard refusal (ref_wg) on the 972 harmful held-out items (as h1h2.py). Read-only on every input.
Reads:  results/analysis/responses_judged.csv.gz, review_sel_kl_units.csv, seednoise_sigma.csv, revision_sigma_model.csv
Writes: results/analysis/revision_h2_sensitivity.csv, revision_h2_sensitivity_meta.csv,
        revision_h2_sensitivity_summary.md

Run (from the paper root):
    scripts/figures/.venv/bin/python scripts/analysis/revision_h2_sensitivity.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_selection import meta  # noqa: E402

Z = stats.norm.ppf(0.975)
MAIN = ["qwen", "gemma", "olmo", "mistral"]
DIR = ["dim", "mpoa", "som"]


def mse(d):
    d = d[~np.isnan(d)]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(len(d))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rj = pd.read_csv(OUT / "responses_judged.csv.gz",
                     usecols=["label", "pair", "role", "method", "gen_new", "kind", "id", "ref_wg"], low_memory=False)
    h = rj[(rj.kind == "harmful") & rj.role.isin(["original", "primary", "control"])].copy()
    h["y"] = h.ref_wg.astype(float)
    W = h.pivot(index="id", columns="label", values="y")
    lab = {(r.pair, r.method, int(r.gen_new)): r.label for r in h.drop_duplicates("label").itertuples()}
    units = pd.read_csv(OUT / "review_sel_kl_units.csv")
    sig_c = float(pd.read_csv(OUT / "seednoise_sigma.csv").set_index("quantity")
                  .loc["sigma (per-cell run-to-run SD), pp", "estimate"]) / 100
    sig_L = float(pd.read_csv(OUT / "revision_sigma_model.csv").set_index("quantity")
                  .loc["sigma_L (constant per cell, logit)", "estimate"])

    def gen(pair, method):
        return W[lab[(pair, method, 1)]].values - W[lab[(pair, method, 0)]].values

    def rate(pair, method, g):
        return float(np.nanmean(W[lab[(pair, method, g)]].values))

    def lvar(pair, method):
        return sum((sig_L * rate(pair, method, g) * (1 - rate(pair, method, g))) ** 2 for g in (0, 1))

    rows = []
    # ---------------------------------------------------------------- (a) cap-bound units
    for pair in MAIN:
        u = units[(units.pair == pair) & (units.method_class == "direction")]
        dms_cap = list(u[u.unit_category == "cap bound in both cells"].method)
        for variant, dms in (("all direction methods", DIR), ("cap-bound direction units only", dms_cap)):
            if not dms:
                rows.append(dict(analysis="H2 contrast", pair=pair, variant=variant, direction_methods="",
                                 n_direction_methods=0, estimate_pp=np.nan))
                continue
            m = len(dms)
            dd = np.mean([gen(pair, x) for x in dms], axis=0)
            da = gen(pair, "ara")
            for est, d, cls in (("direction class", dd, "dir"), ("ARA", da, "ara"), ("H2 contrast", da - dd, "h2")):
                e, se, n = mse(d)
                vc = {"dir": 2 / m, "ara": 2.0, "h2": 2 + 2 / m}[cls] * sig_c ** 2
                vdir = sum(lvar(pair, x) for x in dms) / m ** 2
                vl = {"dir": vdir, "ara": lvar(pair, "ara"), "h2": vdir + lvar(pair, "ara")}[cls]
                tc, tl = np.sqrt(se ** 2 + vc), np.sqrt(se ** 2 + vl)
                rows.append(dict(analysis=est, pair=pair, variant=variant, direction_methods="+".join(dms),
                                 n_direction_methods=m, n_items=n, estimate_pp=100 * e, se_prompt_pp=100 * se,
                                 se_total_const_pp=100 * tc, lo_const=100 * (e - Z * tc), hi_const=100 * (e + Z * tc),
                                 se_total_level_pp=100 * tl, lo_level=100 * (e - Z * tl), hi_level=100 * (e + Z * tl),
                                 var_total_const=tc ** 2, var_prompt=se ** 2))
    # ---------------------------------------------------------------- (b) Gemma without the failed DiM edit
    for variant, dms in (("Gemma, all direction methods", DIR), ("Gemma without the failed Gemma-4-31B DiM edit",
                                                                 ["mpoa", "som"])):
        m = len(dms)
        dd = np.mean([gen("gemma", x) for x in dms], axis=0)
        for est, d, mult in (("direction class", dd, 2 / m), ("H2 contrast", gen("gemma", "ara") - dd, 2 + 2 / m)):
            e, se, n = mse(d)
            tc = np.sqrt(se ** 2 + mult * sig_c ** 2)
            rows.append(dict(analysis=est, pair="gemma", variant=variant, direction_methods="+".join(dms),
                             n_direction_methods=m, n_items=n, estimate_pp=100 * e, se_prompt_pp=100 * se,
                             se_total_const_pp=100 * tc, lo_const=100 * (e - Z * tc), hi_const=100 * (e + Z * tc)))
    # ---------------------------------------------------------------- (c) gamma-correction by generation
    for pair in ("gemma", "olmo"):
        for est, d in (("gMPOA - MPOA, older model", W[lab[(pair, "gamma_mpoa", 0)]].values - W[lab[(pair, "mpoa", 0)]].values),
                       ("gMPOA - MPOA, newer model", W[lab[(pair, "gamma_mpoa", 1)]].values - W[lab[(pair, "mpoa", 1)]].values),
                       ("generation difference of the gamma-correction effect", gen(pair, "gamma_mpoa") - gen(pair, "mpoa"))):
            e, se, n = mse(d)
            rows.append(dict(analysis=est, pair=pair, variant="gamma-MPOA control", direction_methods="mpoa",
                             n_items=n, estimate_pp=100 * e, se_prompt_pp=100 * se,
                             lo_prompt=100 * (e - Z * se), hi_prompt=100 * (e + Z * se),
                             p_prompt=float(2 * stats.norm.sf(abs(e / se)))))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "revision_h2_sensitivity.csv", index=False)

    # ---------------------------------------------------------------- pooled (estimable pairs)
    mrows = []
    for variant in ("all direction methods", "cap-bound direction units only"):
        s = R[(R.analysis == "H2 contrast") & (R.variant == variant) & R.estimate_pp.notna()]
        for vname, col in (("prompt sampling only", "var_prompt"), ("prompt + seed (constant sigma)", "var_total_const")):
            mm = meta(s.estimate_pp.values / 100, s[col].values)
            mrows.append(dict(variant=variant, variance=vname, pairs="+".join(s.pair), k=mm["k"],
                              re_estimate_pp=100 * mm["re_estimate"], hksj_lo_pp=100 * mm["hksj_lo"],
                              hksj_hi_pp=100 * mm["hksj_hi"], Q=mm["Q"], Q_p=mm["Q_p"], I2=mm["I2"],
                              tau_pp=100 * mm["tau"]))
    M = pd.DataFrame(mrows)
    M.to_csv(OUT / "revision_h2_sensitivity_meta.csv", index=False)

    L = [f"# Revision analyses REV-16 / REV-24 / REV-27 (held-out H2 sensitivity)\n",
         f"Generated {t0:%Y-%m-%d %H:%M} UTC by scripts/analysis/revision_h2_sensitivity.py.\n"]
    for r in R.itertuples():
        if np.isnan(r.estimate_pp):
            L.append(f"- {r.pair} | {r.variant} | {r.analysis}: not estimable (no cap-bound direction unit)")
            continue
        if r.variant == "gamma-MPOA control":
            L.append(f"- {r.pair} | {r.analysis}: {r.estimate_pp:+.1f} pp [{r.lo_prompt:+.1f}, {r.hi_prompt:+.1f}] (prompt only)")
        else:
            extra = (f"; logit-level [{r.lo_level:+.1f}, {r.hi_level:+.1f}]" if not np.isnan(getattr(r, "lo_level", np.nan)) else "")
            L.append(f"- {r.pair} | {r.variant} ({r.direction_methods}) | {r.analysis}: {r.estimate_pp:+.1f} pp, "
                     f"prompt + seed [{r.lo_const:+.1f}, {r.hi_const:+.1f}]{extra}")
    L.append("")
    for r in M.itertuples():
        L.append(f"- pooled {r.variant} ({r.pairs}), {r.variance}: {r.re_estimate_pp:+.1f} pp HKSJ [{r.hksj_lo_pp:+.1f}, "
                 f"{r.hksj_hi_pp:+.1f}], Q {r.Q:.1f} (p {r.Q_p:.3g}), I2 {100*r.I2:.0f}%")
    (OUT / "revision_h2_sensitivity_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
