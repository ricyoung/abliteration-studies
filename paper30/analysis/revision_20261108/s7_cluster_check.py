#!/usr/bin/env python3
"""Paper 30, re-review item S7 (2026-11-08). Existing data only; reads analysis/cells.csv, writes two CSVs here.

1. Recomputes the keyword-judge gap correlation (kw_minus_wg_pp vs wg_reduction_pp) with the same base-model
   cluster bootstrap used for the other cross-model correlations in revision_analyses.py (resample the six base
   models with replacement, keep all their cells, 4000 resamples, percentile interval), and its within-base value.
2. Diagnoses the dev-KL correlation (wg_rate vs dev_kl, general dev protocol, 15 cells, 6 base models): how many
   resamples are degenerate (one variable constant or fewer than 4 cells; such resamples are skipped), how many
   distinct cells and base models each resample holds, how often rho = -1, and the interval when resamples with
   fewer than 3 distinct base models are excluded.
3. Records how the within-base p-values are obtained: Spearman rho on within-model-centred values (models with at
   least 2 cells), p from scipy.stats.spearmanr (t approximation with n_cells_within - 2 df; it does not subtract the
   degrees of freedom used by centring, so it is anti-conservative).

Run:
    /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures/.venv/bin/python \
        /data/heretic_2/paper30/v3_2611/analysis/revision_20261108/s7_cluster_check.py
Outputs: s7_correlations.csv, s7_devkl_diagnostics.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

OUT = os.path.dirname(os.path.abspath(__file__))
P30 = "/data/heretic_2/paper30/v3_2611"
B = 4000

c = pd.read_csv(f"{P30}/analysis/cells.csv")
added = c.label.str.contains("kl0.1|obliteratus|jwest33") | c.model.str.startswith("qwen3.5")
md = c[(c.cond != "base") & ~added & (c.status == "judged")].copy()
assert len(md) == 23, len(md)
md["dev_kl_general"] = np.where(md.method == "1x_code", np.nan, md.dev_kl)


def analyse(a, b, seed):
    rng = np.random.default_rng(seed)
    x = md[["label", "model", a, b]].dropna().reset_index(drop=True)
    models = sorted(x.model.unique())
    rho = spearmanr(x[a], x[b])[0]
    res = []
    for _ in range(B):
        pick = rng.choice(models, len(models), replace=True)
        xi = pd.concat([x[x.model == m] for m in pick])
        ok = xi[a].nunique() > 1 and xi[b].nunique() > 1 and len(xi) >= 4
        r = spearmanr(xi[a], xi[b])[0] if ok else np.nan
        res.append(dict(rho=r, valid=ok, n_distinct_models=len(set(pick)), n_distinct_cells=xi.label.nunique()))
    bs = pd.DataFrame(res)
    v = bs[bs.valid]
    lo, hi = np.percentile(v.rho, [2.5, 97.5])
    v3 = v[v.n_distinct_models >= 3]
    lo3, hi3 = np.percentile(v3.rho, [2.5, 97.5])
    # within-base: centre on base-model means, models with >= 2 cells
    xc = x.groupby("model").filter(lambda g: len(g) >= 2).copy()
    for col in (a, b):
        xc[col] = xc[col] - xc.groupby("model")[col].transform("mean")
    rfe, pfe = spearmanr(xc[a], xc[b])
    n_w, k_w = len(xc), xc.model.nunique()
    # same rho, df reduced by the k_w centred means (conservative variant)
    df_adj = n_w - k_w - 1
    tstat = rfe * np.sqrt(df_adj / max(1e-12, 1 - rfe ** 2))
    p_adj = 2 * tdist.sf(abs(tstat), df_adj) if df_adj > 0 else np.nan
    sizes = x.groupby("model").size()
    row = dict(x=a, y=b, n_cells=len(x), n_models=len(models), n_single_cell_models=int((sizes == 1).sum()),
               spearman_rho=rho, cluster_boot_lo=lo, cluster_boot_hi=hi, n_boot=B, n_boot_valid=int(v.shape[0]),
               n_boot_degenerate=int(B - v.shape[0]), frac_rho_minus1=float((v.rho <= -0.999999).mean()),
               frac_rho_plus1=float((v.rho >= 0.999999).mean()),
               median_distinct_cells=float(v.n_distinct_cells.median()),
               cluster_boot_lo_ge3models=lo3, cluster_boot_hi_ge3models=hi3, n_boot_ge3models=int(v3.shape[0]),
               within_model_rho=rfe, within_model_p_scipy=pfe, n_cells_within=n_w, n_models_within=k_w,
               within_model_p_df_adjusted=p_adj, df_adjusted=df_adj,
               p_method="spearmanr on within-model-centred values; t approximation, df = n_cells_within - 2"
                        " (scipy); df_adjusted = n_cells_within - n_models_within - 1")
    return row, bs.assign(x=a, y=b)


rows, diag = [], []
for i, (a, b) in enumerate([("kw_minus_wg_pp", "wg_reduction_pp"), ("wg_rate", "dev_kl_general"),
                            ("wg_rate", "tf_divergence"), ("wg_rate", "capability_mean_delta_pp")]):
    r, bs = analyse(a, b, 20261108 + i)
    rows.append(r)
    if b == "dev_kl_general":
        diag.append(bs)
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/s7_correlations.csv", index=False)
pd.concat(diag).to_csv(f"{OUT}/s7_devkl_diagnostics.csv", index=False)
pd.set_option("display.width", 250)
print(out.T.to_string())
print(md.groupby("model").dev_kl_general.count())
