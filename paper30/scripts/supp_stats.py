#!/usr/bin/env python3
"""Paper 30 v3 supplementary statistics (reads only analysis/*.csv written by analysis/assemble_p30.py).

Writes analysis/supp_stats.csv (one row per statistic) and prints it. Added 2026-11-06 for the revision:
  * within-model Spearman of WildGuard residual refusal on teacher-forced divergence (efficiency check, Fig. 3a),
    with exact permutation p-values (all n! orderings);
  * keyword-on-the-first-256-tokens minus WildGuard gaps (H3 secondary measure), counts and range;
  * share of keyword-only refusals among responses that reached the 512-token limit;
  * H1 over-refusal contrasts by judge;
  * the cells first reported in the companion study (paper 33).
Revision 2026-11-06 (v3, after the added runs): all original-design statistics are restricted to the pre-stated design
cells (Qwen2.5 originals, Heretic 1.x, Heretic 2.0 / fork plug-ins, community releases), so that the cells added after
it (external tools, the re-exported 1.x general cell at KL <= 0.1, Qwen3.5-4B) do not enter them; the added cells get
their own statistics (prefix "added_"), read from the separate families vs_base_added / kw_vs_judge_added.

Run:
    /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures/.venv/bin/python \
        /data/heretic_2/paper30/v3_2611/scripts/supp_stats.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

A = Path("/data/heretic_2/paper30/v3_2611/analysis")
OUT = A / "supp_stats.csv"

# Cells whose models were selected and evaluated in the companion study (paper 33): the Qwen2.5-7B and
# Qwen2.5-32B originals, the nine Heretic 2.0 / fork-plug-in cells and the two community releases.
COMPANION = {"qwen2.5-7b__base", "qwen2.5-7b__dim", "qwen2.5-7b__mpoa", "qwen2.5-7b__som", "qwen2.5-7b__jspace",
             "qwen2.5-7b__ara", "qwen2.5-32b__base", "qwen2.5-32b__dim", "qwen2.5-32b__mpoa", "qwen2.5-32b__som",
             "qwen2.5-32b__ara", "qwen2.5-32b__community-huihui", "qwen2.5-32b__community-zetasepic-v2"}


def exact_spearman(x, y):
    """Spearman rho with an exact two-sided permutation p-value over all n! orderings (vectorised; n <= 9)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    n = len(x)
    perms = np.array(list(itertools.permutations(range(n))))
    a = rx - rx.mean()
    b = ry[perms] - ry.mean()
    r = (b @ a) / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum(axis=1)))
    return rho, float(np.mean(np.abs(r) >= abs(rho) - 1e-12)), n


def is_added(df):
    """Cells added after the pre-stated design."""
    return (df["tool"] == "external") | (df["method"] == "1x_general_kl01") | (df["model"] == "qwen3.5-4b")


def main():
    cells = pd.read_csv(A / "cells.csv")
    tests = pd.read_csv(A / "tests_refusal.csv")
    resp = pd.read_csv(A / "responses_p30.csv.gz", low_memory=False)
    rows = []

    def add(stat, value, note=""):
        rows.append({"stat": stat, "value": value, "note": note})

    judged_all = cells[cells["status"] == "judged"]
    added = judged_all[is_added(judged_all)]
    judged = judged_all[~is_added(judged_all)]          # pre-stated design cells
    edited = judged[judged["method"] != "original"]

    # --- source of each cell
    src = judged["label"].isin(COMPANION)
    add("cells_judged", int(len(judged)))
    add("cells_companion_judged", int(src.sum()), ";".join(sorted(judged.loc[src, "label"])))
    add("cells_new_judged", int((~src).sum()), ";".join(sorted(judged.loc[~src, "label"])))
    add("added_cells_judged", int(len(added)), ";".join(sorted(added["label"])))

    # --- within-model efficiency: residual refusal vs teacher-forced divergence
    for mod in ["qwen2.5-32b", "qwen2.5-7b"]:
        e = edited[edited["model"] == mod]
        rho, p, n = exact_spearman(e["tf_divergence"], e["wg_rate"])
        add(f"spearman_wg_tf_within_{mod}", round(rho, 4), f"n={n}; exact permutation p={p:.4f}")
    rho_all = stats.spearmanr(edited["tf_divergence"], edited["wg_rate"]).statistic
    add("spearman_wg_tf_pooled", round(rho_all, 4), f"n={len(edited)} (cross-check of correlations.csv)")
    band = edited[(edited["tf_divergence"] >= 0.016) & (edited["tf_divergence"] <= 0.021)]
    add("tf_band_0.016_0.021_reduction_min_pp", round(float(band["wg_reduction_pp"].abs().min()), 2),
        ";".join(f"{r.label}:{r.tf_divergence:.4f}:{abs(r.wg_reduction_pp):.1f}" for r in band.itertuples()))
    add("tf_band_0.016_0.021_reduction_max_pp", round(float(band["wg_reduction_pp"].abs().max()), 2))

    # --- keyword-256 minus WildGuard (H3 secondary measure)
    kv = tests[(tests["family"] == "kw_vs_judge") & (tests["group"] == "harmful_all")]
    for judge in ["kw_minus_wg", "kw256_minus_wg", "kw_minus_q3g"]:
        g = kv[kv["judge"] == judge].set_index("a")
        ge = g.loc[g.index.isin(edited["label"])]
        go = g.loc[g.index.isin(judged.loc[judged["method"] == "original", "label"])]
        add(f"{judge}_edited_sig_pos", int(((ge["p_holm"] < 0.05) & (ge["diff"] > 0)).sum()), f"of {len(ge)}")
        add(f"{judge}_edited_sig_neg", int(((ge["p_holm"] < 0.05) & (ge["diff"] < 0)).sum()),
            ";".join(f"{i}:{100 * r['diff']:+.1f}pp:pHolm={r['p_holm']:.3f}"
                     for i, r in ge[(ge["p_holm"] < 0.05) & (ge["diff"] < 0)].iterrows()))
        add(f"{judge}_edited_min_pp", round(100 * ge["diff"].min(), 2), ge["diff"].idxmin())
        add(f"{judge}_edited_max_pp", round(100 * ge["diff"].max(), 2), ge["diff"].idxmax())
        add(f"{judge}_originals_sig", int((go["p_holm"] < 0.05).sum()),
            f"of {len(go)}; range {100 * go['diff'].min():+.1f} to {100 * go['diff'].max():+.1f} pp")
        for name, labs in [("new1x", edited.loc[edited["tool"] == "heretic1x", "label"]),
                           ("companion", edited.loc[edited["label"].isin(COMPANION), "label"])]:
            gg = g.loc[g.index.isin(labs)]
            add(f"{judge}_{name}_sig_pos", int(((gg["p_holm"] < 0.05) & (gg["diff"] > 0)).sum()),
                f"of {len(gg)}; range {100 * gg['diff'].min():+.1f} to {100 * gg['diff'].max():+.1f} pp "
                f"(max {gg['diff'].idxmax()})")
    full = kv[kv["judge"] == "kw_minus_wg"].set_index("a")
    k256 = kv[kv["judge"] == "kw256_minus_wg"].set_index("a")
    lost = [i for i in full.index if i in edited["label"].values and full.loc[i, "p_holm"] < 0.05
            and full.loc[i, "diff"] > 0 and not (k256.loc[i, "p_holm"] < 0.05 and k256.loc[i, "diff"] > 0)]
    add("kw_gap_sig_full_not_256", len(lost),
        ";".join(f"{i}:full {100 * full.loc[i, 'diff']:+.1f}->256 {100 * k256.loc[i, 'diff']:+.1f}pp "
                 f"pHolm256={k256.loc[i, 'p_holm']:.3f}" for i in lost))
    for lab in ["qwen2.5-7b__dim", "qwen2.5-32b__mpoa", "qwen2.5-32b__ara", "qwen2.5-32b__dim"]:
        add(f"kw_gap_full_vs_256_{lab}", f"{100 * full.loc[lab, 'diff']:+.1f} / {100 * k256.loc[lab, 'diff']:+.1f}")
    shrink = 100 * (full.loc[edited["label"], "diff"] - k256.loc[edited["label"], "diff"])
    add("kw_gap_shrink_full_to_256_median_pp", round(float(shrink.median()), 2),
        f"edited cells n={len(shrink)}; range {shrink.min():.1f} to {shrink.max():.1f}")

    # --- keyword-only refusals and the 512-token limit (edited models, harmful prompts, WildGuard labelled)
    r = resp[(resp["kind"] == "harmful") & resp["label"].isin(edited["label"]) & resp["ref_wg"].notna()].copy()
    r["ref_wg_b"] = r["ref_wg"].astype(str).str.lower().isin(["true", "1", "yes"])
    kwonly = r["ref_kw"].astype(bool) & ~r["ref_wg_b"]
    add("edited_harmful_responses", int(len(r)))
    add("edited_harmful_capped_share", round(float(r["capped"].astype(bool).mean()), 4))
    add("edited_kw_only_n", int(kwonly.sum()))
    add("edited_kw_only_capped_share", round(float(r.loc[kwonly, "capped"].astype(bool).mean()), 4))
    wgc = ~r["ref_wg_b"]
    add("edited_wg_comply_capped_share", round(float(r.loc[wgc, "capped"].astype(bool).mean()), 4),
        "responses WildGuard did not label refusals")

    # --- H1 over-refusal (benign prompts) by judge
    h1b = tests[(tests["family"] == "h1_1x_vs_2.0") & (tests["group"] == "benign_all")]
    for judge in ["wg", "kw", "q3g"]:
        g = h1b[h1b["judge"] == judge]
        add(f"h1_overrefusal_sig_{judge}", int((g["p_holm"] < 0.05).sum()), f"of {len(g)}")
    pr = h1b[(h1b["a"] == "qwen2.5-32b__heretic1x-general") & (h1b["b"] == "qwen2.5-32b__dim")]
    for _, x in pr.iterrows():
        add(f"h1_primary_overrefusal_{x['judge']}",
            f"{100 * x['rate_a']:.1f} vs {100 * x['rate_b']:.1f}; pHolm={x['p_holm']:.3f}")

    # --- added cells (separate families; not part of the pre-stated design)
    ed_all = judged_all[judged_all["method"] != "original"]
    e32 = ed_all[ed_all["model"] == "qwen2.5-32b"]
    rho, p, n = exact_spearman(e32["tf_divergence"], e32["wg_rate"])
    add("added_spearman_wg_tf_within_qwen2.5-32b_incl_kl01", round(rho, 4),
        f"n={n} (design cells + 1.x general KL<=0.1); exact permutation p={p:.4f}")
    e4 = ed_all[ed_all["model"] == "qwen3.5-4b"]
    rho, p, n = exact_spearman(e4["tf_divergence"], e4["wg_rate"])
    add("added_spearman_wg_tf_within_qwen3.5-4b", round(rho, 4),
        f"n={n} (Heretic 2.0 + jwest33 as released + hybrid fix); exact permutation p={p:.4f}")
    vba = tests[(tests["family"] == "vs_base_added") & (tests["judge"] == "wg") & (tests["group"] == "harmful_all")]
    ns = vba[vba["p_holm"] >= 0.05]
    add("added_vs_base_wg_nonsig", int(len(ns)), f"of {len(vba)}; " + ";".join(
        f"{r.a}:{100 * r['diff']:+.1f}pp:pHolm={r.p_holm:.3f}" for _, r in ns.iterrows()))
    kva = tests[(tests["family"] == "kw_vs_judge_added") & (tests["judge"] == "kw_minus_wg")
                & (tests["group"] == "harmful_all")].set_index("a")
    kve = kva.loc[kva.index.isin(ed_all["label"])]
    add("added_kw_minus_wg_edited_sig_pos", int(((kve["p_holm"] < 0.05) & (kve["diff"] > 0)).sum()),
        f"of {len(kve)}; range {100 * kve['diff'].min():+.1f} to {100 * kve['diff'].max():+.1f} pp")
    add("added_kw_minus_wg_edited_sig_neg", int(((kve["p_holm"] < 0.05) & (kve["diff"] < 0)).sum()),
        ";".join(f"{i}:{100 * r['diff']:+.1f}pp:pHolm={r['p_holm']:.3f}"
                 for i, r in kve[(kve["p_holm"] < 0.05) & (kve["diff"] < 0)].iterrows()))
    rr = resp[(resp["kind"] == "harmful") & resp["label"].isin(added["label"])].copy()
    deg = rr.groupby("label")["is_degenerate"].apply(lambda x: int(x.astype(str).str.lower().isin(["true", "1"]).sum()))
    add("added_degenerate_harmful_counts", int(deg.sum()), ";".join(f"{k}:{v}" for k, v in deg.items() if v))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 200, "display.max_rows", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
