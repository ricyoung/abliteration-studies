#!/usr/bin/env python3
"""seednoise_flags.py -- run-to-run (optimisation-seed) noise band for the H1 generation effects (paper 33).

Reads (never writes) results/analysis/final_seed_cells.csv and h1h2_generation_effects.csv; writes ONLY
results/analysis/seednoise_*.csv / seednoise_summary.md. Run with the figures venv (pandas, numpy, scipy):
    scripts/figures/.venv/bin/python scripts/analysis/seednoise_flags.py

Estimand
  For each of the 10 small-pair cells (Qwen2.5-7B and Qwen3.5-4B x DiM, MPOA, SOM, J-space, ARA) the same
  configuration was optimised with seed 1 and seed 2 (100 TPE trials each) and both selected models were judged on
  the 972 harmful held-out prompts (WildGuard refusal). With d_i = rate(seed 2) - rate(seed 1) in pp and the
  assumption that the two seeds are exchangeable draws with run-to-run SD sigma (E d_i = 0),
      sigma   = sqrt(mean(d_i^2) / 2)                         (per-cell run-to-run SD of held-out refusal)
      SD_diff = sigma * sqrt(2)                               (SD of an older-minus-newer difference of two
                                                               independently optimised cells)
      threshold = 2 * SD_diff                                 (approximate 95% run-to-run band)
  95% intervals for sigma: (a) chi-square, sum(d_i^2) / (2 sigma^2) ~ chi2(10) under normality;
  (b) percentile bootstrap over the 10 cells (20,000 resamples, seed 20261106).
  Sensitivity: robust estimate from the median absolute difference, sigma_mad = median|d| / (0.6745 * sqrt 2).

Flags
  Each of the 16 main-study per-pair generation effects (h1h2_generation_effects.csv; judge wg, subset all;
  seed-1 models) is 'exceeds seed noise' if |effect| > threshold, else 'within seed noise'. The same rule is
  applied to the baseline-adjusted effect (effect minus the originals' generation difference; the originals are
  not optimised, so their difference carries no run-to-run noise) and, for reference only, to the small pair
  (whose own seed-2 replicate is the source of sigma, so its flags are not independent of the estimate).
  The band is also evaluated at the lower and upper chi-square limits of sigma.

Assumption (limitation): run-to-run variability measured on two 4-7B models transfers to the 22-32B main-study
models and is the same for every method and refusal level. Neither is tested: the main-study cells have one seed.
The band ignores prompt-sampling error, which the McNemar / Newcombe analysis already covers.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AN = os.path.join(ROOT, "results", "analysis")
PREFIX = "seednoise_"
N_BOOT = 20000
BOOT_SEED = 20261106
STAMP = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

MAIN_PAIRS = ["qwen", "gemma", "olmo", "mistral"]
PAIR_TEX = {"qwen": "Qwen", "gemma": "Gemma", "olmo": "OLMo", "mistral": "Mistral Small",
            "qwen-small": "Small pair"}
METHOD_TEX = {"dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space", "ara": "ARA"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def sigma_from(d):
    d = np.asarray(d, float)
    return float(np.sqrt(np.mean(d ** 2) / 2.0))


def main():
    f_seed = os.path.join(AN, "final_seed_cells.csv")
    f_eff = os.path.join(AN, "h1h2_generation_effects.csv")
    seed = pd.read_csv(f_seed)
    eff = pd.read_csv(f_eff)

    # ---------------- seed pairs ----------------
    seed = seed.copy()
    seed["s1_pp"] = 100 * seed["s1_wg_rate"]
    seed["s2_pp"] = 100 * seed["s2_wg_rate"]
    seed["d_pp"] = seed["s2_pp"] - seed["s1_pp"]
    # consistency check with the column written by final.py
    assert np.allclose(seed["d_pp"], 100 * seed["diff_s2_minus_s1"], atol=1e-3), "d mismatch vs diff_s2_minus_s1"
    n = len(seed)
    assert n == 10, f"expected 10 small-pair cells, got {n}"
    d = seed["d_pp"].to_numpy()
    ss = float(np.sum(d ** 2))
    sigma = sigma_from(d)
    sd_diff = sigma * np.sqrt(2)
    thr = 2 * sd_diff

    # chi-square interval: ss / (2 sigma^2) ~ chi2(n)
    q_hi, q_lo = stats.chi2.ppf(0.975, n), stats.chi2.ppf(0.025, n)
    sig_lo, sig_hi = np.sqrt(ss / (2 * q_hi)), np.sqrt(ss / (2 * q_lo))
    # bootstrap over cells
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.sqrt(np.mean(d[idx] ** 2, axis=1) / 2.0)
    bs_lo, bs_hi = np.percentile(boot, [2.5, 97.5])
    # robust sensitivity
    med_abs = float(np.median(np.abs(d)))
    sigma_mad = med_abs / (stats.norm.ppf(0.75) * np.sqrt(2))

    est = pd.DataFrame([
        dict(quantity="n_cells (small pair, seed 1 vs seed 2)", estimate=n, lo=np.nan, hi=np.nan,
             method="Qwen2.5-7B and Qwen3.5-4B x DiM, MPOA, SOM, J-space, ARA"),
        dict(quantity="mean d = seed2 - seed1, pp", estimate=float(np.mean(d)), lo=np.nan, hi=np.nan,
             method="arithmetic mean (assumed 0 by the estimator)"),
        dict(quantity="median |d|, pp", estimate=med_abs, lo=np.nan, hi=np.nan, method="median"),
        dict(quantity="max |d|, pp", estimate=float(np.max(np.abs(d))), lo=np.nan, hi=np.nan, method="max"),
        dict(quantity="sigma (per-cell run-to-run SD), pp", estimate=sigma, lo=sig_lo, hi=sig_hi,
             method="sqrt(mean(d^2)/2); 95% chi-square interval, df = 10"),
        dict(quantity="sigma bootstrap interval, pp", estimate=sigma, lo=bs_lo, hi=bs_hi,
             method=f"percentile bootstrap over cells, {N_BOOT} resamples, seed {BOOT_SEED}"),
        dict(quantity="sigma_mad (robust sensitivity), pp", estimate=sigma_mad, lo=np.nan, hi=np.nan,
             method="median|d| / (0.6745 * sqrt 2)"),
        dict(quantity="SD_diff (older-minus-newer difference), pp", estimate=sd_diff, lo=sig_lo * np.sqrt(2),
             hi=sig_hi * np.sqrt(2), method="sigma * sqrt(2); interval from the chi-square interval of sigma"),
        dict(quantity="threshold = 2 * SD_diff, pp", estimate=thr, lo=2 * np.sqrt(2) * sig_lo,
             hi=2 * np.sqrt(2) * sig_hi, method="flag |effect| > threshold as 'exceeds seed noise'"),
        dict(quantity="threshold from sigma_mad, pp", estimate=2 * np.sqrt(2) * sigma_mad, lo=np.nan, hi=np.nan,
             method="sensitivity"),
    ])
    est.to_csv(os.path.join(AN, PREFIX + "sigma.csv"), index=False, float_format="%.4f")

    seed_out = seed[["model", "generation", "method", "s1_pp", "s2_pp", "d_pp", "p_holm"]].copy()
    seed_out.to_csv(os.path.join(AN, PREFIX + "seed_pairs.csv"), index=False, float_format="%.4f")

    # ---------------- flags ----------------
    e = eff[(eff["judge"] == "wg") & (eff["subset"] == "all") & (eff["method"] != "original")].copy()
    e["effect_pp"] = 100 * e["delta"]
    e["effect_lo_pp"] = 100 * e["delta_lo"]
    e["effect_hi_pp"] = 100 * e["delta_hi"]
    e["baseline_adj_pp"] = 100 * e["did_baseline_adj"]
    e["holm_sig"] = e["p_holm_H1"] < 0.05
    e["holm_direction"] = np.where(~e["holm_sig"], "n.s.", np.where(e["delta"] > 0, "newer higher", "newer lower"))

    def flag(x, t):
        return np.where(np.abs(x) > t, "exceeds seed noise", "within seed noise")

    e["seed_noise_flag"] = flag(e["effect_pp"], thr)
    e["seed_noise_flag_at_sigma_lo"] = flag(e["effect_pp"], 2 * np.sqrt(2) * sig_lo)
    e["seed_noise_flag_at_sigma_hi"] = flag(e["effect_pp"], 2 * np.sqrt(2) * sig_hi)
    e["seed_noise_flag_mad"] = flag(e["effect_pp"], 2 * np.sqrt(2) * sigma_mad)
    e["baseline_adj_flag"] = flag(e["baseline_adj_pp"], thr)
    e["ratio_to_sd_diff"] = e["effect_pp"] / sd_diff
    e["scope"] = np.where(e["pair"].isin(MAIN_PAIRS), "main", "small (reference only; source of sigma)")
    cols = ["scope", "pair", "method", "old_label", "new_label", "effect_pp", "effect_lo_pp", "effect_hi_pp",
            "p_holm_H1", "holm_direction", "ratio_to_sd_diff", "seed_noise_flag", "seed_noise_flag_at_sigma_lo",
            "seed_noise_flag_at_sigma_hi", "seed_noise_flag_mad", "baseline_adj_pp", "baseline_adj_flag"]
    out = e[cols].reset_index(drop=True)
    out.to_csv(os.path.join(AN, PREFIX + "flags.csv"), index=False, float_format="%.4f")

    m = out[out["scope"] == "main"]
    assert len(m) == 16, f"expected 16 main-study effects, got {len(m)}"
    exc = m[m["seed_noise_flag"] == "exceeds seed noise"]

    def lst(df, col="effect_pp"):
        return "; ".join(f"{PAIR_TEX[r.pair]} {METHOD_TEX[r.method]} {getattr(r, col):+.1f} pp"
                         for r in df.itertuples())

    # cross-tab with the Holm verdicts
    xt = pd.crosstab(m["holm_direction"], m["seed_noise_flag"]).reindex(
        index=["newer higher", "newer lower", "n.s."], columns=["exceeds seed noise", "within seed noise"],
        fill_value=0)
    xt.to_csv(os.path.join(AN, PREFIX + "crosstab.csv"))

    n_lo = int((m["seed_noise_flag_at_sigma_lo"] == "exceeds seed noise").sum())
    n_hi = int((m["seed_noise_flag_at_sigma_hi"] == "exceeds seed noise").sum())
    n_mad = int((m["seed_noise_flag_mad"] == "exceeds seed noise").sum())
    exc_ba = m[m["baseline_adj_flag"] == "exceeds seed noise"]
    sm = out[out["scope"] != "main"]
    exc_sm = sm[sm["seed_noise_flag"] == "exceeds seed noise"]
    # closest calls
    m_sorted = m.assign(margin=np.abs(m["effect_pp"]) - thr).sort_values("margin", key=np.abs)

    L = []
    L.append("# Seed-noise band for the H1 generation effects (paper 33)\n")
    L.append(f"Generated by `scripts/analysis/seednoise_flags.py` at {STAMP} from "
             f"`results/analysis/final_seed_cells.csv` (md5 `{md5(f_seed)}`) and "
             f"`results/analysis/h1h2_generation_effects.csv` (md5 `{md5(f_eff)}`). Regenerated on every run; do "
             f"not edit by hand. Software: pandas {pd.__version__}, numpy {np.__version__}, scipy "
             f"{__import__('scipy').__version__}.\n")
    L.append("## Numbers to cite\n")
    L.append(f"- Seed pairs: n = {n} small-pair cells (Qwen2.5-7B, Qwen3.5-4B x DiM, MPOA, SOM, J-space, ARA); "
             f"d = seed 2 - seed 1 held-out WildGuard refusal, pp. Mean d = {np.mean(d):+.2f} pp, median |d| = "
             f"{med_abs:.2f} pp, max |d| = {np.max(np.abs(d)):.2f} pp.")
    L.append(f"- **sigma** (run-to-run SD of one cell's held-out refusal) = sqrt(mean(d^2)/2) = **{sigma:.1f} pp** "
             f"(95% chi-square interval [{sig_lo:.1f}, {sig_hi:.1f}], df = {n}; cell bootstrap [{bs_lo:.1f}, "
             f"{bs_hi:.1f}]). Robust sensitivity from median |d|: {sigma_mad:.1f} pp.")
    L.append(f"- **SD_diff** (older-minus-newer difference of two independently optimised cells) = sigma x sqrt 2 "
             f"= **{sd_diff:.1f} pp** [{sig_lo * np.sqrt(2):.1f}, {sig_hi * np.sqrt(2):.1f}].")
    L.append(f"- **Threshold** 2 x SD_diff = **{thr:.1f} pp** [{2 * np.sqrt(2) * sig_lo:.1f}, "
             f"{2 * np.sqrt(2) * sig_hi:.1f}] (interval from the chi-square interval of sigma).")
    L.append(f"- **Main study: {len(exc)} of 16** per-pair generation effects exceed the band: {lst(exc)}. "
             f"The other {16 - len(exc)} are within seed noise.")
    L.append(f"- Across the 95% interval of sigma the count is {n_hi} (upper limit, threshold "
             f"{2 * np.sqrt(2) * sig_hi:.1f} pp) to {n_lo} (lower limit, threshold {2 * np.sqrt(2) * sig_lo:.1f} pp); "
             f"with the robust sigma it is {n_mad}.")
    L.append(f"- Closest calls (|effect| - threshold, pp): " + "; ".join(
        f"{PAIR_TEX[r.pair]} {METHOD_TEX[r.method]} {r.effect_pp:+.1f} ({r.margin:+.1f})"
        for r in m_sorted.head(4).itertuples()) + ".")
    L.append(f"- Baseline-adjusted effects exceeding the same band: {len(exc_ba)} of 16 ({lst(exc_ba, 'baseline_adj_pp')}).")
    L.append(f"- Small pair (reference only; its seed-2 replicate is the source of sigma): {len(exc_sm)} of 5 exceed "
             f"({lst(exc_sm) if len(exc_sm) else 'none'}).\n")
    L.append("## Cross-tabulation with the Holm verdicts (main study, WildGuard, 972 items, seed 1)\n")
    L.append("| Holm verdict | exceeds seed noise | within seed noise |")
    L.append("|---|---|---|")
    for k, r in xt.iterrows():
        L.append(f"| {k} | {r['exceeds seed noise']} | {r['within seed noise']} |")
    L.append("")
    L.append("## Seed pairs (small pair)\n")
    L.append("| Cell | Seed 1, % | Seed 2, % | d, pp |")
    L.append("|---|---|---|---|")
    for r in seed_out.itertuples():
        L.append(f"| {r.model} {METHOD_TEX[r.method]} | {r.s1_pp:.1f} | {r.s2_pp:.1f} | {r.d_pp:+.1f} |")
    L.append("")
    L.append("## Per-pair flags (main study)\n")
    L.append("| Pair | Method | Effect, pp [95% CI] | Holm | Effect / SD_diff | Flag | Baseline-adj., pp | Flag (adj.) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in m.itertuples():
        L.append(f"| {PAIR_TEX[r.pair]} | {METHOD_TEX[r.method]} | {r.effect_pp:+.1f} [{r.effect_lo_pp:+.1f}, "
                 f"{r.effect_hi_pp:+.1f}] | {r.holm_direction} | {r.ratio_to_sd_diff:+.2f} | {r.seed_noise_flag} | "
                 f"{r.baseline_adj_pp:+.1f} | {r.baseline_adj_flag} |")
    L.append("")
    L.append("## Assumptions and limitations\n")
    L.append("- The run-to-run SD is estimated on the small pair (4B and 7B models) and assumed to transfer to the "
             "22-32B main-study models; the main-study cells have one seed each, so this is not tested.")
    L.append("- One SD is pooled over methods and refusal levels; on the percentage-point scale, variability is "
             "likely smaller near 0% (e.g. ARA 4.7% vs 2.2% on Qwen2.5-7B) and larger at intermediate rates, so "
             "the band is conservative for cells near the floor and may be liberal for mid-range cells.")
    L.append("- The two seeds are treated as exchangeable draws (E d = 0); with n = 10 the interval for sigma is "
             "wide (chi-square, assumes normal d).")
    L.append("- d also contains the paired prompt-sampling error between the two selected models (same 972 "
             "items), so sigma slightly overstates pure optimisation noise; the band is a heuristic screen, not "
             "a test, and is applied in addition to (not instead of) the McNemar / Holm analysis.")
    L.append("- The originals are not optimised, so the baseline-adjusted effect carries the same run-to-run "
             "SD_diff as the raw effect.")
    with open(os.path.join(AN, PREFIX + "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))


if __name__ == "__main__":
    main()
