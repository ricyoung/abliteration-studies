#!/usr/bin/env python3
"""Paper 33 -- revision 2026-11-08 (ARS review, REV-16 and REV-32): do the per-pair H1/H2 verdicts depend on the
KL budget at which a model is selected?

Only the trial selected at first-token KL <= 0.1 was evaluated on the held-out set, so other budgets can be read
only on the development (dev) set: Heretic's keyword refusal count on its 100 dev prompts, for every completed TPE
trial in trials.csv. For every optimisation run this script computes the refusal-KL front
    f(t) = fewest dev keyword refusals among the run's completed trials with KL <= t
(the unedited original, KL = 0, is included, so f is defined for every t >= 0), reads it at the budgets
t = 0.05, 0.1, 0.2, 0.5 and 1.0, and summarises the whole front without a budget as the area under it on a
log-KL axis: AUC = mean of f(t) over 200 log-spaced t in [0.01, 1] (in refusals of 100; lower = refusal removed at
smaller divergence). Beyond the largest KL a run sampled, f is carried forward (the run's overall minimum); the
share of the grid where this happens is reported per run.

Generation effects (newer minus older model) are formed per pair x method; per pair the direction class is the
mean over DiM, MPOA, SOM (and J-space on the small pair) and the H2 contrast is ARA minus that mean. Run-to-run
variation is estimated on the same dev quantity from the ten small-pair seed-1 / seed-2 run pairs,
sigma_dev(t) = sqrt(mean(d^2) / 2), and propagated as in review_selection.py (2 sigma^2 for one method, 2 sigma^2/m
for a mean of m, their sum for the H2 contrast). These intervals cover run-to-run variation only, not sampling of
the 100 dev prompts. At t = 0.1 the dev effects are those of the selected models, so a change of sign between
t = 0.1 and another budget isolates the budget from the dev-versus-held-out difference.

Reads (read-only): paper/figures/data/trials.csv, paper/figures/data/runs.csv,
                   results/analysis/review_sel_pair_effects.csv (held-out per-pair effects, for comparison).
Writes (results/analysis/, overwritten):
  revision_kl_budget_runs.csv     per run: f(t) at each budget, AUC, max KL sampled, carried-forward share
  revision_kl_budget_sigma.csv    sigma_dev per budget and for the AUC (10 small-pair seed pairs)
  revision_kl_budget_effects.csv  per pair x (method | class | H2) x budget: generation effect, seed SE, sign,
                                  beyond-noise flag, held-out sign
  revision_kl_budget_summary.md   key numbers

Run (from the paper root):
    scripts/figures/.venv/bin/python scripts/analysis/revision_kl_budgets.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "figures" / "data"
OUT = ROOT / "results" / "analysis"
BUDGETS = [0.05, 0.1, 0.2, 0.5, 1.0]
GRID = np.logspace(-2, 0, 200)
PAIR_OF = {"qwen2.5-32b": "qwen", "qwen3.8-27b": "qwen", "gemma-2-27b": "gemma", "gemma-4-31b": "gemma",
           "olmo-2-32b": "olmo", "olmo-3.1-32b": "olmo", "mistral-small-2409": "mistral",
           "mistral-small-3.2": "mistral", "qwen2.5-7b": "qwen-small", "qwen3.5-4b": "qwen-small"}
PAIRS = ["qwen", "gemma", "olmo", "mistral", "qwen-small"]
DIR = ["dim", "mpoa", "som", "jspace"]
QUANT = [f"t{t:g}" for t in BUDGETS] + ["auc"]


def front(kl: np.ndarray, ref: np.ndarray, base: float, t: np.ndarray) -> np.ndarray:
    """f(t) = min(base, min{ref_i : kl_i <= t}) for every t."""
    o = np.argsort(kl)
    kl_s, cm = kl[o], np.minimum.accumulate(ref[o])
    idx = np.searchsorted(kl_s, t, side="right") - 1
    return np.where(idx >= 0, np.minimum(base, cm[np.clip(idx, 0, None)]), base)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    tr = pd.read_csv(DATA / "trials.csv")
    runs = pd.read_csv(DATA / "runs.csv")
    tr = tr[tr.method != "gamma_mpoa"]
    base = runs.drop_duplicates(["model"]).set_index("model").baseline_refusals
    rows = []
    for run, g in tr.groupby("run"):
        r0 = g.iloc[0]
        kl, ref = g.kl.values.astype(float), g.refusals.values.astype(float)
        b = float(base[r0.model])
        f_b = front(kl, ref, b, np.array(BUDGETS))
        f_g = front(kl, ref, b, GRID)
        d = dict(run=run, model=r0.model, pair=PAIR_OF[r0.model], study=r0.study,
                 gen_new=int("new" in str(r0.generation)), method=r0.method, seed=int(r0.seed), n_trials=len(g),
                 baseline_refusals=b, max_kl=float(kl.max()), min_kl=float(kl.min()),
                 carried_forward_share=float(np.mean(GRID > kl.max())), auc=float(f_g.mean()))
        for t, v in zip(BUDGETS, f_b):
            d[f"t{t:g}"] = float(v)
            d[f"n_trials_le_t{t:g}"] = int((kl <= t).sum())
        rows.append(d)
    R = pd.DataFrame(rows)
    # check: f(0.1) equals the selected trial's dev count (selection rule of select_trial.py)
    sel = runs.set_index("run").selected_refusals
    chk = R[R.run.isin(sel.index)]
    bad = chk[(chk["t0.1"] - chk.run.map(sel)).abs() > 0]
    assert bad.empty, bad[["run", "t0.1"]]
    R.to_csv(OUT / "revision_kl_budget_runs.csv", index=False)

    # ------------------------------------------------------------- run-to-run sigma on the dev quantities
    sm = R[R.study == "small"].pivot_table(index=["model", "method"], columns="seed", values=QUANT)
    srows, sig = [], {}
    for q in QUANT:
        d = (sm[(q, 2)] - sm[(q, 1)]).values
        s = float(np.sqrt(np.mean(d ** 2) / 2))
        sig[q] = s
        srows.append(dict(quantity=q, n_pairs=len(d), sigma_dev=s, median_abs_d=float(np.median(np.abs(d))),
                          max_abs_d=float(np.max(np.abs(d)))))
    pd.DataFrame(srows).to_csv(OUT / "revision_kl_budget_sigma.csv", index=False)

    # ------------------------------------------------------------- generation effects per pair
    s1 = R[R.seed == 1]
    ho = pd.read_csv(OUT / "review_sel_pair_effects.csv")
    ho = ho[ho.scale == "pp"].set_index(["pair", "estimand"]).estimate
    erows = []
    for pair in PAIRS:
        g = s1[s1.pair == pair]
        dms = [m for m in DIR if m in set(g.method)]
        m = len(dms)
        for q in QUANT:
            eff = {}
            for meth in dms + ["ara"]:
                o = g[(g.method == meth) & (g.gen_new == 0)][q].iloc[0]
                n = g[(g.method == meth) & (g.gen_new == 1)][q].iloc[0]
                eff[meth] = n - o
            dmean = float(np.mean([eff[x] for x in dms]))
            items = [(x, eff[x], 2.0, None) for x in dms] + [
                ("ara", eff["ara"], 2.0, ho[(pair, "ARA raw")]),
                ("direction class", dmean, 2.0 / m, ho[(pair, "direction raw")]),
                ("H2 contrast", eff["ara"] - dmean, 2.0 + 2.0 / m, ho[(pair, "H2 contrast (ARA - direction)")])]
            for name, e, mult, hos in items:
                se = np.sqrt(mult) * sig[q]
                erows.append(dict(pair=pair, quantity=q, estimand=name, n_direction_methods=m, effect=e,
                                  seed_se=se, lo=e - 1.96 * se, hi=e + 1.96 * se,
                                  beyond_noise=bool(abs(e) > 1.96 * se), sign=int(np.sign(e)),
                                  heldout_effect_pp=np.nan if hos is None else 100 * hos,
                                  heldout_sign=np.nan if hos is None else int(np.sign(hos))))
    E = pd.DataFrame(erows)
    ref01 = E[E.quantity == "t0.1"].set_index(["pair", "estimand"]).sign
    E["sign_as_at_0.1"] = [int(s == ref01[(p, e)]) if s != 0 and ref01[(p, e)] != 0 else np.nan
                           for p, e, s in zip(E.pair, E.estimand, E.sign)]
    E.to_csv(OUT / "revision_kl_budget_effects.csv", index=False)

    # ------------------------------------------------------------- summary
    L = [f"# Revision analysis REV-16 / REV-32 (KL budgets, dev set)\n",
         f"Generated {t0:%Y-%m-%d %H:%M} UTC by scripts/analysis/revision_kl_budgets.py.\n",
         "Dev keyword refusals (of 100) of the best trial at KL <= t; AUC = mean front value over log KL in [0.01, 1].\n",
         "## Run-to-run sigma (small pair, 10 seed pairs)"]
    for r in srows:
        L.append(f"- {r['quantity']}: sigma_dev {r['sigma_dev']:.1f} (median |d| {r['median_abs_d']:.1f}, max {r['max_abs_d']:.1f})")
    L.append(f"\nRuns with the front carried forward over > 50% of the AUC grid: "
             f"{int((s1.carried_forward_share > 0.5).sum())} of {len(s1)} seed-1 runs\n")
    L += ["## Class-level effects (newer - older), refusals of 100; * = beyond 1.96 x run-to-run SE", "",
          "| pair | estimand | " + " | ".join(QUANT) + " | held-out pp |", "|---" * (len(QUANT) + 3) + "|"]
    for pair in PAIRS:
        for est in ("direction class", "ara", "H2 contrast"):
            sub = E[(E.pair == pair) & (E.estimand == est)].set_index("quantity")
            cells = [f"{sub.loc[q, 'effect']:+.1f}{'*' if sub.loc[q, 'beyond_noise'] else ''}" for q in QUANT]
            L.append(f"| {pair} | {est} | " + " | ".join(cells) + f" | {sub.iloc[0].heldout_effect_pp:+.1f} |")
    L.append("")
    for est in ("direction class", "ara", "H2 contrast"):
        for q in QUANT:
            sub = E[(E.estimand == est) & (E.quantity == q) & E.pair.isin(PAIRS[:4])]
            L.append(f"- {est} {q}: sign as at 0.1 in {int(np.nansum(sub['sign_as_at_0.1']))} of 4 main pairs; "
                     f"sign as held-out in {int((sub.sign == sub.heldout_sign).sum())} of 4; "
                     f"beyond noise {int(sub.beyond_noise.sum())} of 4")
    (OUT / "revision_kl_budget_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
