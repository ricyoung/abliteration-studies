#!/usr/bin/env python3
"""Paper 33 -- revision 2026-11-08 (ARS review, REV-14 and REV-15): between-pair heterogeneity with seed-inclusive
within-pair variances, and run-to-run variance as a function of the refusal level.

Purpose
  REV-14: report Cochran's Q, I^2 and the DerSimonian-Laird tau^2 of the four main-pair generation effects with
          within-pair variances that include run-to-run (seed) variance, on both scales, and tabulate the per-pair
          variances used as weights. The prompt-level Wald chi^2 (547) treats prompts, not pairs, as the unit.
  REV-15: the constant per-cell sigma = 9.15 pp of seednoise_flags.py ignores that a binomial-type rate varies less
          near 0% and 100%. Here the run effect is modelled on the logit scale (constant sigma_L per cell, estimated
          from the ten small-pair seed pairs) and mapped to percentage points by the delta method at each cell's
          observed rate: sigma_pp(cell) = sigma_L * p (1 - p). Per-pair verdicts are recomputed across the 95%
          interval of sigma_L and compared with the constant-sigma verdicts across the interval of sigma.

Analysis only: no GPU, no network. Reads (read-only):
  results/analysis/review_sel_pair_effects.csv   per-pair estimates and prompt SEs (pp and log-odds)
  results/analysis/seednoise_seed_pairs.csv      seed-1 / seed-2 held-out refusal of the ten small-pair cells
  results/analysis/seednoise_sigma.csv           constant pp sigma and its chi-square interval (reference)
  results/analysis/h1h2_generation_effects.csv   per pair x method rates and effects (wg, all)
Writes (results/analysis/, overwritten on every run):
  revision_sigma_model.csv          sigma_L and sigma_pp with chi-square intervals; fit of both variance models
  revision_sigma_cells.csv          per main-study modified cell: rate, sigma_pp under the logit model (lo/point/hi)
  revision_sigma_method_flags.csv   16 main pair x method effects (raw, adjusted): seed SE and "exceeds 2 SE"
                                    under constant sigma and logit-level sigma, each at lo/point/hi
  revision_sigma_pair_verdicts.csv  per pair x class estimand (pp): interval with prompt + seed variance under each
                                    sigma variant; whether it excludes zero
  revision_pair_variances.csv       per main pair x estimand x scale: estimate and the variance components
  revision_heterogeneity.csv        Q, df, p, I^2, tau^2, tau per estimand x scale x variance variant
  revision_seed_variance_summary.md key numbers

Run (from the paper root):
    scripts/figures/.venv/bin/python scripts/analysis/revision_seed_variance.py
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
from review_selection import meta  # noqa: E402  (FE, Q, I^2, DL tau^2, HKSJ, PI; identical to the paper's pooling)

Z = stats.norm.ppf(0.975)
MAIN = ["qwen", "gemma", "olmo", "mistral"]
PAIRS = MAIN + ["qwen-small"]
DIR = ["dim", "mpoa", "som", "jspace"]


def logit(p):
    p = np.asarray(p, float)
    return np.log(p / (1 - p))


def sigma_ci(d: np.ndarray) -> tuple[float, float, float]:
    """sigma = sqrt(sum d^2 / (2n)) with a chi-square(n) interval (as seednoise_flags.py)."""
    n = len(d)
    ss = float(np.sum(d ** 2)) / 2
    s = np.sqrt(ss / n)
    return s, np.sqrt(ss / stats.chi2.ppf(0.975, n)), np.sqrt(ss / stats.chi2.ppf(0.025, n))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    eff = pd.read_csv(OUT / "review_sel_pair_effects.csv")
    sp = pd.read_csv(OUT / "seednoise_seed_pairs.csv")
    sref = pd.read_csv(OUT / "seednoise_sigma.csv")
    ge = pd.read_csv(OUT / "h1h2_generation_effects.csv")
    ge = ge[(ge.judge == "wg") & (ge.subset == "all")]

    # ------------------------------------------------------------------ sigma on both scales and model fit
    p1, p2 = sp.s1_pp.values / 100, sp.s2_pp.values / 100
    d_pp = sp.d_pp.values / 100
    d_lo = logit(p2) - logit(p1)
    s_pp, s_pp_lo, s_pp_hi = sigma_ci(d_pp)
    s_L, s_L_lo, s_L_hi = sigma_ci(d_lo)
    ref = float(sref.loc[sref.quantity.str.startswith("sigma (per-cell"), "estimate"].iloc[0])
    assert abs(100 * s_pp - ref) < 1e-3, (100 * s_pp, ref)
    pbar = (p1 + p2) / 2
    v = pbar * (1 - pbar)
    # constant-pp model: d_i ~ N(0, 2 s^2); logit-level model: d_i ~ N(0, 2 s_L^2 v_i^2), each with its own MLE
    ll_const = float(np.sum(stats.norm.logpdf(d_pp, 0, np.sqrt(2) * s_pp)))
    sL_mle_pp = np.sqrt(np.mean(d_pp ** 2 / (2 * v ** 2)))
    ll_level = float(np.sum(stats.norm.logpdf(d_pp, 0, np.sqrt(2) * sL_mle_pp * v)))
    rho, rho_p = stats.spearmanr(np.abs(d_pp), v)
    rho_l, rho_lp = stats.spearmanr(np.abs(d_lo), v)
    model = pd.DataFrame([
        dict(quantity="sigma_pp (constant per cell, pp)", estimate=100 * s_pp, lo=100 * s_pp_lo, hi=100 * s_pp_hi,
             note="sqrt(sum d^2 / 20), chi-square(10) interval; = seednoise_sigma.csv"),
        dict(quantity="sigma_L (constant per cell, logit)", estimate=s_L, lo=s_L_lo, hi=s_L_hi,
             note="d = logit(s2) - logit(s1); chi-square(10) interval; = review_sel_seed_sigma.csv logit row"),
        dict(quantity="sigma_L fitted to the pp differences (delta method at the seed pair's mean rate)",
             estimate=sL_mle_pp, lo=np.nan, hi=np.nan, note="MLE of d_pp ~ N(0, 2 sigma_L^2 (pbar(1-pbar))^2)"),
        dict(quantity="log-likelihood, constant-pp model", estimate=ll_const, lo=np.nan, hi=np.nan, note="1 parameter"),
        dict(quantity="log-likelihood, logit-level model (pp differences)", estimate=ll_level, lo=np.nan, hi=np.nan,
             note="1 parameter; AIC difference = 2 x difference of the log-likelihoods"),
        dict(quantity="Spearman rho(|d_pp|, pbar(1-pbar))", estimate=rho, lo=np.nan, hi=np.nan, note=f"p = {rho_p:.3g}"),
        dict(quantity="Spearman rho(|d_logit|, pbar(1-pbar))", estimate=rho_l, lo=np.nan, hi=np.nan,
             note=f"p = {rho_lp:.3g}"),
    ])
    model.to_csv(OUT / "revision_sigma_model.csv", index=False)

    sig_const = {"lo": s_pp_lo, "point": s_pp, "hi": s_pp_hi}
    sig_L = {"lo": s_L_lo, "point": s_L, "hi": s_L_hi}

    # ------------------------------------------------------------------ per-cell sigma (main study)
    rows = []
    for r in ge[ge.study == "main"].itertuples():
        if r.method == "original":
            continue
        for gen, lab, p in (("old", r.old_label, r.p_old), ("new", r.new_label, r.p_new)):
            d = dict(pair=r.pair, method=r.method, generation=gen, label=lab, rate=p)
            for k, s in sig_L.items():
                d[f"sigma_level_pp_{k}"] = 100 * s * p * (1 - p)
            rows.append(d)
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "revision_sigma_cells.csv", index=False)

    # ------------------------------------------------------------------ per-method flags (16 main effects)
    frows = []
    for r in ge[(ge.study == "main") & (ge.method != "original")].itertuples():
        vo, vn = r.p_old * (1 - r.p_old), r.p_new * (1 - r.p_new)
        d = dict(pair=r.pair, method=r.method, p_old=r.p_old, p_new=r.p_new, effect_pp=100 * r.delta,
                 adjusted_pp=100 * r.did_baseline_adj)
        for k in ("lo", "point", "hi"):
            se_c = 100 * np.sqrt(2) * sig_const[k]
            se_l = 100 * sig_L[k] * np.sqrt(vo ** 2 + vn ** 2)
            d[f"seed_se_const_{k}"] = se_c
            d[f"seed_se_level_{k}"] = se_l
            d[f"exceeds_const_{k}"] = abs(d["effect_pp"]) > 2 * se_c
            d[f"exceeds_level_{k}"] = abs(d["effect_pp"]) > 2 * se_l
            d[f"adj_exceeds_const_{k}"] = abs(d["adjusted_pp"]) > 2 * se_c
            d[f"adj_exceeds_level_{k}"] = abs(d["adjusted_pp"]) > 2 * se_l
        frows.append(d)
    flags = pd.DataFrame(frows)
    flags.to_csv(OUT / "revision_sigma_method_flags.csv", index=False)

    # ------------------------------------------------------------------ per-pair class verdicts (pp)
    vrows = []
    pp = eff[eff.scale == "pp"]
    for pair in PAIRS:
        gp = ge[ge.pair == pair]
        dms = [m for m in DIR if m in set(gp.method)]
        m = len(dms)
        cv = {}
        for k in ("lo", "point", "hi"):
            var_dir = sum((sig_L[k] * gp.loc[gp.method == mm, "p_old"].iloc[0] * (1 - gp.loc[gp.method == mm, "p_old"].iloc[0])) ** 2
                          + (sig_L[k] * gp.loc[gp.method == mm, "p_new"].iloc[0] * (1 - gp.loc[gp.method == mm, "p_new"].iloc[0])) ** 2
                          for mm in dms) / m ** 2
            a = gp[gp.method == "ara"].iloc[0]
            var_ara = (sig_L[k] * a.p_old * (1 - a.p_old)) ** 2 + (sig_L[k] * a.p_new * (1 - a.p_new)) ** 2
            cv[("level", k)] = {"direction": var_dir, "ARA": var_ara, "H2": var_dir + var_ara}
            sc = sig_const[k] ** 2
            cv[("const", k)] = {"direction": 2 * sc / m, "ARA": 2 * sc, "H2": 2 * sc + 2 * sc / m}
        for est in ("direction raw", "direction adjusted", "ARA raw", "ARA adjusted", "H2 contrast (ARA - direction)"):
            row = pp[(pp.pair == pair) & (pp.estimand == est)].iloc[0]
            cls = "H2" if est.startswith("H2") else ("ARA" if est.startswith("ARA") else "direction")
            d = dict(pair=pair, estimand=est, n_direction_methods=m, estimate_pp=100 * row.estimate,
                     se_prompt_pp=100 * row.se_prompt)
            for (model_name, k), vv in cv.items():
                se = np.sqrt(row.se_prompt ** 2 + vv[cls])
                lo, hi = row.estimate - Z * se, row.estimate + Z * se
                tag = f"{model_name}_{k}"
                d[f"se_seed_{tag}"] = 100 * np.sqrt(vv[cls])
                d[f"lo_{tag}"], d[f"hi_{tag}"] = 100 * lo, 100 * hi
                d[f"excl0_{tag}"] = bool(lo > 0 or hi < 0)
            vrows.append(d)
    verd = pd.DataFrame(vrows)
    # reproduction check: constant-sigma point intervals equal review_sel_pair_effects.csv lo_total / hi_total
    chk = pp.set_index(["pair", "estimand"])
    mx = max(abs(r.lo_const_point / 100 - chk.loc[(r.pair, r.estimand), "lo_total"]) for r in verd.itertuples())
    assert mx < 1e-6, mx  # seednoise_seed_pairs.csv stores d to 4 decimals
    verd.to_csv(OUT / "revision_sigma_pair_verdicts.csv", index=False)

    # ------------------------------------------------------------------ heterogeneity (4 main pairs)
    var_rows, het_rows = [], []
    for (est, scale), g in eff[eff.pair.isin(MAIN)].groupby(["estimand", "scale"], sort=False):
        g = g.set_index("pair").loc[MAIN]
        y = g.estimate.values
        variants = {"prompt sampling only": g.se_prompt.values ** 2}
        if scale == "pp":
            cls = "H2" if est.startswith("H2") else ("ARA" if est.startswith("ARA") else "direction")
            for k in ("lo", "point", "hi"):
                vc = np.array([{"direction": 2 / n, "ARA": 2.0, "H2": 2 + 2 / n}[cls] for n in g.n_direction_methods])
                variants[f"prompt + seed, constant sigma_pp ({k})"] = g.se_prompt.values ** 2 + vc * sig_const[k] ** 2
                vl = np.array([cv_level(ge, p, sig_L[k], cls) for p in MAIN])
                variants[f"prompt + seed, logit-level sigma ({k})"] = g.se_prompt.values ** 2 + vl
        else:
            cls = "H2" if est.startswith("H2") else ("ARA" if est.startswith("ARA") else "direction")
            for k in ("lo", "point", "hi"):
                vc = np.array([{"direction": 2 / n, "ARA": 2.0, "H2": 2 + 2 / n}[cls] for n in g.n_direction_methods])
                variants[f"prompt + seed, constant sigma_L ({k})"] = g.se_prompt.values ** 2 + vc * sig_L[k] ** 2
        for vname, vv in variants.items():
            mres = meta(y, vv)
            het_rows.append(dict(estimand=est, scale=scale, variance=vname, k=4,
                                 Q=mres["Q"], df=mres["Q_df"], Q_p=mres["Q_p"], I2=mres["I2"], tau2=mres["tau2"],
                                 tau=mres["tau"], re_estimate=mres["re_estimate"], hksj_lo=mres["hksj_lo"],
                                 hksj_hi=mres["hksj_hi"], pi_lo=mres["pi_lo"], pi_hi=mres["pi_hi"]))
            for pair, yy, vvv in zip(MAIN, y, vv):
                var_rows.append(dict(pair=pair, estimand=est, scale=scale, variance=vname, estimate=yy, var_within=vvv,
                                     se_within=np.sqrt(vvv), weight_fe=1 / vvv))
    het = pd.DataFrame(het_rows)
    het.to_csv(OUT / "revision_heterogeneity.csv", index=False)
    pd.DataFrame(var_rows).to_csv(OUT / "revision_pair_variances.csv", index=False)
    # reproduction: constant-sigma point Q equals review_sel_pair_meta.csv
    rm = pd.read_csv(OUT / "review_sel_pair_meta.csv")
    for est in ("direction raw", "direction adjusted"):
        a = het[(het.estimand == est) & (het.scale == "pp") & (het.variance == "prompt + seed, constant sigma_pp (point)")].Q.iloc[0]
        b = rm[(rm.estimand == est) & (rm.scale == "pp") & (rm.variance == "prompt + seed (sigma, 10 seed pairs)")].Q.iloc[0]
        assert abs(a - b) < 1e-3, (est, a, b)

    # ------------------------------------------------------------------ summary
    L = [f"# Revision analyses REV-14 / REV-15 (seed variance)\n",
         f"Generated {t0:%Y-%m-%d %H:%M} UTC by scripts/analysis/revision_seed_variance.py. Read-only on its inputs.\n",
         "## Run-to-run sigma",
         f"- constant pp: {100*s_pp:.2f} pp [{100*s_pp_lo:.2f}, {100*s_pp_hi:.2f}]; logit: {s_L:.3f} [{s_L_lo:.3f}, {s_L_hi:.3f}]",
         f"- fit to the 10 seed-pair pp differences: log-likelihood constant {ll_const:.2f}, logit-level {ll_level:.2f} "
         f"(AIC difference {2*(ll_level-ll_const):+.2f}); Spearman |d_pp| vs pbar(1-pbar) {rho:.2f} (p {rho_p:.2g}).",
         "", "## Per-method flags (16 main effects), number exceeding 2 x seed SE",
         ]
    for k in ("lo", "point", "hi"):
        L.append(f"- sigma {k}: constant {int(flags[f'exceeds_const_{k}'].sum())} raw / {int(flags[f'adj_exceeds_const_{k}'].sum())} adjusted; "
                 f"logit-level {int(flags[f'exceeds_level_{k}'].sum())} raw / {int(flags[f'adj_exceeds_level_{k}'].sum())} adjusted")
    L.append("")
    L.append("Exceeding under logit-level sigma (point): " + "; ".join(
        f"{r.pair} {r.method} {r.effect_pp:+.1f}" for r in flags[flags.exceeds_level_point].itertuples()))
    L += ["", "## Per-pair class intervals excluding zero (pp)"]
    for tag in ("const_lo", "const_point", "const_hi", "level_lo", "level_point", "level_hi"):
        sub = verd[verd[f"excl0_{tag}"]]
        L.append(f"- {tag}: {len(sub)} of {len(verd)}: " + "; ".join(f"{r.pair} {r.estimand}" for r in sub.itertuples()))
    L += ["", "## Heterogeneity (4 main pairs)", "| estimand | scale | variance | Q | p | I2 | tau |", "|---|---|---|---|---|---|---|"]
    for r in het.itertuples():
        L.append(f"| {r.estimand} | {r.scale} | {r.variance} | {r.Q:.1f} | {r.Q_p:.3g} | {100*r.I2:.0f}% | {r.tau:.3f} |")
    (OUT / "revision_seed_variance_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cv_level(ge: pd.DataFrame, pair: str, sL: float, cls: str) -> float:
    gp = ge[ge.pair == pair]
    dms = [m for m in DIR if m in set(gp.method)]
    m = len(dms)
    vd = sum((sL * r.p_old * (1 - r.p_old)) ** 2 + (sL * r.p_new * (1 - r.p_new)) ** 2
             for r in gp[gp.method.isin(dms)].itertuples()) / m ** 2
    a = gp[gp.method == "ara"].iloc[0]
    va = (sL * a.p_old * (1 - a.p_old)) ** 2 + (sL * a.p_new * (1 - a.p_new)) ** 2
    return {"direction": vd, "ARA": va, "H2": vd + va}[cls]


if __name__ == "__main__":
    main()
