# Paper 33 -- Results notes (held-out refusal; interim, judge batch 1)

**Superseded on 2026-11-06 by `FINAL_NOTES.md` (all 74 cells judged). Kept for the record; do not cite numbers from here.**

Hand-written on 2026-11-06 from the run of `scripts/analysis/run_all.sh` at 00:37Z (assembly 2026-11-06T00:37Z).
These are the numbers the Results section will report, each with its source file in `results/analysis/`. The
regenerated summaries (`h1h2_summary.md`, `h3_summary.md`, `secondary_summary.md`) are authoritative; **re-run
`scripts/analysis/run_all.sh` after judge batch 2 and update this file** (every judged-cell number below can change once
OLMo/Mistral enter the Holm families and the pooled models).

Scope of every number: 32 judged cells = Qwen and Gemma main pairs x {original, DiM, MPOA, SOM, ARA} (20) and the small
pair Qwen2.5-7B -> Qwen3.5-4B x {original, DiM, MPOA, SOM, J-space, ARA} (12); 1,322 held-out items per cell
(972 harmful, 350 benign); greedy decoding, one response per item, 512-token cap, thinking off, one optimisation seed.

---------------------------------------------------------------------------------------------------------------------

## 0. What changed in this pass (fixes of the verified issues; all scripts re-run)

| # | Verified issue | Change | Effect on numbers |
|---|---|---|---|
| 1 | Degeneracy heuristic: 66 `char_run` flags, nearly all in coherent code answers | `assemble.py`: char_run recheck (module docstring "CHAR_RUN RECHECK"). A char_run flag is kept only if the run continues to the end of the response (model stuck on one character until it stopped), or a non-separator, non-box-drawing run lies outside fenced/inline code, or a separator run is > 50% of the response. Raw flag kept as `degenerate_reason_raw` / `is_degenerate_raw` (`cell_rates.csv`: `degenerate_raw_*`). | 58 of 66 char_run flags cleared; 8 kept (all terminal runs, e.g. `0000...`/`1111...` until the cap, 285 x `'` in `gemma-4-31b__som/cw_astra_1144`). Degenerate rows 106 -> 48 (40 repetition, 8 char_run); 58 rows change outcome class (56 -> compliance, 2 -> refusal). `cw_rmcbench_0346` no longer flagged anywhere. H1/H2 primary unaffected (uses `ref_wg`). |
| 2 | Primary `ref_wg` vs outcome-class refusal differ on 4 rows | Docstrings, README and `h1h2_summary.md` now state: **primary outcome = WildGuard refusal label on every WildGuard-labelled row, degenerate rows included**; the four-class outcome is the secondary view (`wg_class`). | After fix 1 they differ on 2 harmful rows (`gemma-4-31b__som/cw_astra_1144`, `qwen3.8-27b__ara/strongreject_86`); generation effects change by <= 0.1 pp. |
| 3 | AdvBench overlap checked for harmful items only | `secondary.py --build-advbench-map` re-matches all 1,322 items (same normalisation and exact / Jaccard>=0.6 + SequenceMatcher>=0.8 rule, `mlabonne/harmful_behaviors@01cead01` from the local HF cache). Over-refusal reported also for `benign_excl_advbench` (349 items). | Harmful flags reproduce 44/44 (36 exact, 8 fuzzy). Benign `jbb_benign_11` meets the fuzzy rule (it matches train[41], train[106] and dev prompt test[11]). WildGuard over-refusal changes by <= 0.06 pp per cell without it (WildGuard never refused it). |
| 4 | Two significant negative H1 effects not robust to dropping the AdvBench items | `h1h2.py` verdict now annotates every Holm-significant H1 test whose decision or sign changes without the 44 items. | Qwen SOM and Gemma MPOA carry the caveat (section 2). |
| 5 | Gemma-4 DiM is a failed edit; "matched" = KL cap, not equal KL | `h1h2.py`: H1 table gains `Dev KL old / new` and an "(edit failed)" marker (held-out refusal >= the original's); the verdict explains the failed cell; caveat lists the largest within-pair KL ratios. | Text only. |
| 6 | Pending lists stale; mtime-based staleness checks | New `scripts/analysis/freshness.py` (content-based: new/removed dirs, responses/judgments that appeared, changed row counts, changed judge `scored` stamp). Used by `h1h2.py` (had none), `h3.py` and `secondary.py` (replacing their mtime checks) and as step 6 of `run_all.sh` (exit 3 if the trees changed during the run). `assemble.py` lists the new root `runs/mc-rtx-1/results/import_h200` (OLMo/Mistral copies for judge batch 2, appeared 00:2xZ) explicitly. | Pending counts updated (section 1). |
| -- | Wording | `secondary.py` no longer states that keyword over-refusal "is not a refusal measure"; it says the measures diverge pending the audit (atlas_sweep D1). Its AdvBench caveat had "none from HarmBench"; it now prints the counts by set (jbb_harmful 17, strongreject 26, harmbench_standard 1). | Text only. |
| -- | Re-run | `scripts/analysis/run_all.sh` (new) re-runs assemble -> AdvBench map -> h1h2 -> h3 -> secondary -> freshness. Log: `run_all_log.txt`. | |

Not changed: the 40 `repetition` flags (not hand-audited; a structural check -- duplicate 4-grams in the last 80 words --
finds a looping tail in about 32, and templated, list-like answers with little tail repetition in about 4, e.g.
`qwen2.5-32b__ara/strongreject_27`, `qwen3.5-4b__som/harmbench_breathalyzer_test_fooling`). Not re-verified here: VB
mixed-model estimates, bootstrap CIs, the H3 cell-level OLS.

---------------------------------------------------------------------------------------------------------------------

## 1. Coverage and what is pending

Source: `h1h2_status.csv`, `secondary_cells.csv`, `cell_rates.csv` (`status`), `dir_status.csv`.

- **Judged and analysed: 32 cells** (above). One harmful row has no WildGuard refusal label (`qwen2.5-32b__base/cw_astra_1459`,
  denominator 971) and one benign row (`qwen2.5-7b__som/jbb_benign_65`, denominator 349); both are dropped pairwise.
- **Pending judging (judge batch 2), not analysed -- 35 cells at 00:37Z:** OLMo pair (10 original + primary cells) and
  Mistral pair (10) -- the remaining two of the four main pairs; gamma-MPOA controls `olmo-2-32b__gmpoa`,
  `olmo-3.1-32b__gmpoa` (2); community models (7: `qwen2.5-32b__community-huihui`, `qwen2.5-32b__community-zetasepic-v2`,
  `qwen3.8-27b__community-hereticorg-ara`, `qwen3.8-27b__community-huihui`, `mistral-small-3.2__community-huihui`,
  `mistral-small-3.2__community-llmfan46`, `olmo-3.1-32b__community-muxodious`); seed-2 replicates (6:
  `qwen2.5-7b__{dim,mpoa}-s2`, `qwen3.5-4b__{dim,mpoa,som,ara}-s2`). OLMo/Mistral copies for judging are arriving in
  `runs/mc-rtx-1/results/import_h200` (25 dirs, not yet judged).
- **Pending generation -- 7 cells:** gamma-MPOA for both Gemma models; community `gemma-4-31b__community-coder3101`;
  seed-2 `qwen2.5-7b__{som,jspace,ara}-s2`, `qwen3.5-4b__jspace-s2`.
- Counts move while results sync (25 / 17 at the 23:33Z assembly, 33 / 9 at 00:23Z, 35 / 7 at 00:37Z). Pending cells
  are never imputed; they enter every table automatically when `refusal/judgments.jsonl` exists and `run_all.sh` is
  re-run; `freshness.py` flags any change since the last assembly.
- By design not in H1/H2: gamma-MPOA (control), community (external), seed-2 (replicate). They feed the seed-spread
  caveat (atlas_sweep D3), the Gemma post-norm control (D2) and the community audit once judged.

---------------------------------------------------------------------------------------------------------------------

## 2. Conventions to state in Methods (fill the `sec:setup-stats` todos)

- Primary outcome: WildGuard `Response refusal` label on the 972 harmful items (`ref_wg`), degenerate rows included;
  rows with N/A dropped. Secondary: four-class outcome (degenerate > unscored > WildGuard refusal > compliance;
  "incomplete" = `finish_reason == length` is a flag that never changes the class), keyword (full and 256-token),
  Qwen3Guard, HarmBench classifier, VINAY, HCAI.
- Intervals: 95%. Proportions: Wilson. Paired differences: Newcombe hybrid score interval (item bootstrap, 4,000
  resamples, agrees within 0.2 pp). Paired tests: exact McNemar. Baseline-adjusted effects and H2 risk-difference
  contrasts: item bootstrap.
- alpha = 0.05, two-sided; Holm within families, separately per judge x item subset: **H1** = 13 pair x method McNemar
  tests + 2 pooled main-study GEE generation effects (15); **H2** = 3 per-pair GEE interactions + pooled main-study
  interaction (4); **ablation** = 26 modified-vs-original McNemar tests (descriptive); secondary: one family per judge x
  item group over the 26 modified cells; H3: per-cell McNemar family per comparison x item group, 4 primary GEE terms.
- Models: **R is not installed**, so the planned `refusal ~ generation x method_class + (1|prompt) + (1|family)` is fitted
  with statsmodels 0.15.0 `BinomialBayesMixedGLM` (variational Bayes; no p-values); robustness = GEE logistic,
  exchangeable working correlation clustered by prompt, family as a fixed effect. Per-method H1 = paired McNemar per pair
  plus GEE per-method generation effects. H3 class test = GEE linear model of the paired gap (kw - wg) clustered by
  prompt, with a cell-level OLS (cell as unit) as the check. Software: Python 3.12.15, pandas 3.1.1, numpy 2.5.4,
  scipy 1.18.1, statsmodels 0.15.0.
- "Matched distribution shift" = every modified model is the selected trial at first-token KL <= 0.1 on the dev
  prompts -- a common cap, not equal KL. Selected KL 0.018-0.099; within-pair differences up to 3x (small ARA
  0.021 / 0.064, Qwen DiM 0.018 / 0.053; Gemma-2 MPOA and ARA at 0.099).
- AdvBench overlap: 44 harmful items flagged (36 exact, 8 fuzzy; jbb_harmful 17, strongreject 26, harmbench_standard 1);
  the build script checks harmful items only; every H1/H2/H3 result is also reported on the 928 remaining items.

---------------------------------------------------------------------------------------------------------------------

## 3. H1 -- residual refusal higher on the newer model of each pair, for every method

Sources: `h1h2_generation_effects.csv` (judge `wg`, subset `all` / `excl_advbench`), `h1h2_cells.csv` (Wilson CIs),
`h1h2_tests.csv` (family `H1`), `h1h2_summary.md`.

**WildGuard residual refusal, 972 harmful items** (rate % [Wilson 95% CI]; new - old in pp [Newcombe 95% CI]; b/c =
items refused only by the older / only by the newer model's version; exact McNemar p; Holm over the 15-test H1 family;
baseline-adjusted = generation effect minus the originals' [bootstrap CI])

| Pair | Method | Older % | Newer % | New - old, pp | b / c | p | p Holm | Baseline-adj., pp | Dev KL old / new |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 2.5-32B -> 3.8-27B | Original | 76.8 [74.1, 79.4] | 89.9 [87.9, 91.7] | +13.1 [+10.9, +15.4] | 6 / 133 | 2.7e-32 | (baseline) | | |
| | DiM | 47.4 [44.3, 50.6] | 74.3 [71.4, 76.9] | +26.9 [+23.8, +29.8] | 17 / 278 | 5.7e-62 | 6.8e-61 | +13.7 [+10.1, +17.1] | 0.018 / 0.053 |
| | MPOA | 3.9 [2.9, 5.3] | 14.0 [12.0, 16.3] | +10.1 [+7.8, +12.5] | 23 / 121 | 2.9e-17 | 2.0e-16 | -3.1 [-6.3, +0.2] | 0.060 / 0.085 |
| | SOM | 42.7 [39.6, 45.8] | 38.0 [35.0, 41.1] | **-4.7 [-8.0, -1.4]** | 157 / 111 | 0.0059 | 0.018 (a) | -17.8 [-21.5, -14.2] | 0.075 / 0.084 |
| | ARA | 6.1 [4.7, 7.8] | 40.4 [37.4, 43.5] | +34.4 [+31.2, +37.5] | 11 / 345 | 3.5e-87 | 5.3e-86 | +21.3 [+17.1, +25.4] | 0.062 / 0.061 |
| Gemma 2-27B -> 4-31B | Original | 88.3 [86.1, 90.1] | 89.3 [87.2, 91.1] | +1.0 [-1.4, +3.5] | 69 / 79 | 0.46 | (baseline) | | |
| | DiM | 73.8 [70.9, 76.4] | 90.1 [88.1, 91.8] (b) | +16.4 [+13.5, +19.3] | 35 / 194 | 6.9e-28 | 6.9e-27 | +15.3 [+13.0, +17.8] | 0.069 / 0.093 |
| | MPOA | 45.2 [42.1, 48.3] | 40.4 [37.4, 43.5] | **-4.7 [-8.6, -0.9]** | 206 / 160 | 0.019 | 0.037 (a) | -5.8 [-9.7, -1.7] | 0.099 / 0.069 |
| | SOM | 65.2 [62.2, 68.2] | 67.8 [64.8, 70.7] | +2.6 [-1.1, +6.2] | 151 / 176 | 0.18 | 0.18 | +1.5 [-1.9, +4.9] | 0.086 / 0.055 |
| | ARA | 26.6 [24.0, 29.5] | 0.6 [0.3, 1.3] | **-26.0 [-28.9, -23.3]** | 253 / 0 | 1.4e-76 | 1.9e-75 | -27.1 [-30.3, -23.6] | 0.099 / 0.084 |
| Small Qwen2.5-7B -> Qwen3.5-4B | Original | 72.1 [69.2, 74.8] | 88.5 [86.3, 90.3] | +16.4 [+13.9, +18.9] | 11 / 170 | 8.8e-38 | (baseline) | | |
| | DiM | 20.6 [18.2, 23.2] | 54.6 [51.5, 57.7] | +34.1 [+30.5, +37.5] | 43 / 374 | 5.4e-67 | 7.1e-66 | +17.7 [+13.5, +21.8] | 0.078 / 0.054 |
| | MPOA | 13.8 [11.8, 16.1] | 18.0 [15.7, 20.5] | +4.2 [+1.6, +6.9] | 66 / 107 | 0.0023 | 0.009 | -12.1 [-15.7, -8.3] | 0.074 / 0.054 |
| | SOM | 12.2 [10.3, 14.5] | 21.9 [19.4, 24.6] | +9.7 [+6.9, +12.4] | 50 / 144 | 9.5e-12 | 5.7e-11 | -6.7 [-10.4, -3.0] | 0.076 / 0.055 |
| | J-space | 52.6 [49.4, 55.7] | 73.8 [70.9, 76.4] | +21.2 [+18.1, +24.3] | 37 / 243 | 2.6e-38 | 2.8e-37 | +4.8 [+1.3, +8.3] | 0.022 / 0.019 |
| | ARA | 4.7 [3.6, 6.3] | 18.9 [16.6, 21.5] | +14.2 [+11.6, +16.9] | 24 / 162 | 2.4e-26 | 2.2e-25 | -2.2 [-6.0, +1.6] | 0.021 / 0.064 |

(a) Not robust to dropping the 44 AdvBench-overlap items (928 items): Qwen SOM -3.9 pp [-7.2, -0.6], b/c 141/105,
p = 0.025, p_Holm = 0.052; Gemma MPOA -4.8 pp [-8.7, -0.9], b/c 194/149, p = 0.017, p_Holm = 0.052. Report both as
"lower on the newer model, but not significant after Holm without the overlap items".
(b) **Failed edit.** Gemma-4 DiM: the selected trial had 98/100 dev keyword refusals at KL 0.093 (no trial at
KL <= 0.1 -- or <= 0.5 -- went below 98/100; the original had 99/100), and held-out refusal is 90.1% vs the original's
89.3% (+0.8 pp [+0.1, +1.6], 2/10, p_Holm = 0.039). The +16.4 pp Gemma DiM generation effect therefore means "no trial at
KL <= 0.1 removed refusal from Gemma-4", not "residual refusal after a successful ablation".

**Without the 44 AdvBench-overlap items** (`subset = excl_advbench`): generation effects change by at most 1.9 pp
(median 0.4); the only Holm decisions that change are (a). Small-pair ARA becomes +12.3 pp [+9.8, +14.9] with
baseline-adjusted -4.8 [-8.7, -1.0] (CI now excludes 0).

**Originals already differ** (not a test of H1): Qwen +13.1 pp and small pair +16.4 pp (both p < 1e-30), Gemma +1.0 pp
(p = 0.46). Baseline-adjusted, the positive generation effect survives for Qwen DiM and ARA, Gemma DiM (failed edit),
small DiM and J-space; it is negative for Qwen SOM, Gemma MPOA and ARA, small MPOA and SOM; null for Qwen MPOA,
Gemma SOM and small ARA.

**Pooled main study (Qwen + Gemma only; interim), log-odds** (`h1h2_models.csv`, scope `main`):
- GEE generation effect, direction class +0.32 [+0.25, +0.40], p = 1.3e-17, p_Holm = 1.0e-16; optimisation class (ARA)
  +0.29 [+0.15, +0.44], p = 6.3e-5, p_Holm = 3.1e-4. VB mixed model (planned): +0.44 [+0.38, +0.49] and
  +0.35 [+0.21, +0.48] (no p-values; SD(family) 0.74 [0.29, 1.88] from 2 levels, prior-dominated).
- Heterogeneity (GEE Wald): generation x family chi2 = 14.3, p = 1.6e-4; generation x class x family chi2 = 180.0,
  p = 4.8e-41 -- the pooled coefficient averages effects of opposite sign (Qwen ARA +34 pp, Gemma ARA -26 pp).
- With the originals as reference (baseline-adjusted): direction -0.28 [-0.41, -0.14], p = 4.1e-5; optimisation
  -0.31 [-0.50, -0.12], p = 0.0015; originals' own generation difference +0.60 [+0.45, +0.74].

**Judge sensitivity** (`h1h2_summary.md`, "How the conclusions change with the judge"): with keyword scoring, Gemma MPOA
flips to +18.9 pp (p_Holm < 0.001), Gemma SOM becomes significant (+11.8), Gemma ARA becomes null (24.9 -> 22.3%,
-2.6 pp, p_Holm 0.65), and small-pair MPOA, J-space and ARA become null. The H1 pattern depends on the judge.

**H1 status: NOT SUPPORTED (interim; 2 of 4 main pairs judged).** In neither judged main pair is residual refusal higher
on the newer model for every method: Qwen 3 of 4 methods higher, SOM lower (not robust to the AdvBench exclusion);
Gemma 1 of 4 higher and that one is a failed edit, ARA much lower (26.6% -> 0.6%), MPOA lower (not robust), SOM n.s.
The small pair satisfies H1 as stated (5/5 methods higher), but its newer original already refuses 16.4 pp more, and
baseline-adjusted only DiM and J-space remain higher. Final status pending OLMo and Mistral (H1 says "each pair").

---------------------------------------------------------------------------------------------------------------------

## 4. H2 -- generation x method class: ARA loses less effectiveness on newer models than direction-based methods

Sources: `h1h2_h2_contrasts.csv`, `h1h2_tests.csv` (family `H2`, 4 tests), `h1h2_models.csv`. H2 predicts a negative
contrast / interaction.

| Pair | ARA gen. effect | Direction-based mean | RD contrast, pp [95% CI] | GEE interaction, log-odds [95% CI] | p | p Holm | VB interaction [95% CrI] | Direction vs H2 |
|---|---|---|---|---|---|---|---|---|
| Qwen | +34.4 | +10.7 (DiM, MPOA, SOM) | +23.6 [+19.9, +27.1] | +1.89 [+1.61, +2.16] | 7.5e-42 | 3.0e-41 | +2.29 [+2.14, +2.44] | opposite |
| Gemma | -26.0 | +4.7 (DiM, MPOA, SOM) | -30.8 [-34.0, -27.5] | -4.27 [-5.07, -3.47] | 1.3e-25 | 4.0e-25 | -4.90 [-5.64, -4.17] | as predicted |
| Small pair | +14.2 | +17.3 (DiM, MPOA, SOM, J-space) | -3.1 [-6.0, -0.1] | +0.76 [+0.46, +1.06] | 7.6e-7 | 1.5e-6 | +0.80 [+0.62, +0.99] | scale-dependent |
| Main pooled (Qwen + Gemma) | | | | -0.03 [-0.18, +0.12] | 0.69 | 0.69 | -0.09 [-0.21, +0.04] | null |

- Gemma's H2-consistent contrast does not depend on the failed DiM edit: with MPOA and SOM only (mean -1.1 pp) the RD
  contrast is about -25 pp (computed from the H1 table).
- Method-intercept GEE (one intercept per pair x method): Qwen +1.77 [+1.50, +2.04], Gemma -4.35 [-5.20, -3.50],
  small +0.62 [+0.30, +0.93], pooled main -0.11 [-0.26, +0.04] (p = 0.15). Main + small pooled GEE: +0.09
  [-0.04, +0.22], p = 0.19.
- Without the AdvBench items: pooled main -0.07 [-0.22, +0.08], p = 0.39; small pair RD -5.2 [-8.2, -2.2] but GEE +0.60
  [+0.30, +0.90]; per-pair signs unchanged.
- Keyword scoring reverses the pooled conclusion: +0.46 log-odds, p_Holm < 0.001 (kw256 +0.51), i.e. opposite to H2.

**H2 status: NOT SUPPORTED (interim).** The pooled interaction is null (GEE -0.03, p = 0.69; VB -0.09, CrI covers 0) and
the per-pair interactions are large and of opposite sign (Qwen opposite to H2, Gemma as predicted, small pair depends on
the scale: risk difference as predicted, log-odds opposite). The family x class heterogeneity (chi2 = 180) means a
pooled H2 estimate is not interpretable as a common effect with these two pairs. Pending OLMo/Mistral; a two-family
random intercept cannot be estimated from the data.

---------------------------------------------------------------------------------------------------------------------

## 5. H3 -- keyword counts underestimate refusal relative to an independent judge; the gap differs by method class

Sources: `h3_gap_pooled.csv`, `h3_gap_cells.csv`, `h3_agreement_pairs.csv`, `h3_agreement_fleiss.csv`, `h3_models.csv`,
`h3_summary.md`. Gap = keyword rate - WildGuard rate on the same responses (H3 predicts negative).

**5.1 Direction of the gap (harmful items)**

| Cells | n responses | Keyword % | WildGuard % | Gap kw - wg, pp [95% CI] | kw-only / wg-only |
|---|---|---|---|---|---|
| Originals (6) | 5,831 | 85.5 | 84.2 | +1.4 [+0.6, +2.2] | 215 / 136 |
| Ablated, main study (16) | 15,552 | 49.3 | 42.3 | +7.0 [+5.4, +8.6] | 2,714 / 1,627 |
| Ablated, small pair (10) | 9,720 | 33.3 | 29.1 | +4.2 [+2.5, +5.9] | 1,606 / 1,200 |
| All ablated (26) | 25,272 | 43.1 | 37.2 | +5.9 [+4.3, +7.4] | 4,320 / 2,827 |

- Per ablated cell: keyword > WildGuard in 20/26 (12 Holm-significant), keyword < WildGuard in 6/26 (3 Holm-significant:
  Gemma-2 DiM, Gemma-2 SOM, Qwen3.5-4B J-space). Largest: Gemma-4 MPOA +28.1 pp, Gemma-4 ARA +21.7 pp (22.3% vs 0.6%),
  Qwen3.8 ARA +20.1 pp; most negative: Qwen3.5-4B J-space -21.1 pp [-24.4, -17.7] (keyword 52.7% vs WildGuard 73.8%; 55 / 260).
- Against Qwen3Guard the direction is the same (ablated main kw - q3g +4.7 pp [+3.1, +6.4]; ablated small +1.8
  [-0.1, +3.7]); Qwen3Guard counts slightly more refusals than WildGuard (+2.3 pp ablated).
- **The sign depends on length.** All ablated cells: responses that ended before the cap -4.7 pp [-6.8, -2.8]
  (n = 12,759; the H3-predicted direction), capped responses +16.8 pp [+15.2, +18.5] (n = 12,513); 256-token keyword
  window +2.9 pp [+1.3, +4.4].
- Wording of the 4,320 keyword-only refusals (string heuristics, not validated): 88.6% contain only safety/disclaimer
  vocabulary, 5.7% a first-person refusal phrase; Qwen3Guard calls 16.1% of them refusals, HCAI labels 52.3%
  "disclaimer", HarmBench calls 62.5% harmful compliance, 62.6% are capped. Of the 2,827 WildGuard-only refusals
  Qwen3Guard agrees on 85.2%; 52.2% carry only redirect cues.
- Without the AdvBench items: main-study mean gap (GEE) +8.7 pp [+7.1, +10.3] (vs +9.1 with them).

**5.2 Agreement** (harmful items; Cohen's kappa [bootstrap 95% CI over prompts], raw agreement p_o)

| Pair | Originals | Ablated main | Ablated small |
|---|---|---|---|
| Keyword vs WildGuard | 0.77 [0.74, 0.79], p_o 0.940 | 0.44 [0.42, 0.46], p_o 0.721 | 0.33 [0.31, 0.35], p_o 0.711 |
| Keyword-256 vs WildGuard | 0.78 | 0.47 [0.45, 0.49] | 0.35 |
| WildGuard vs Qwen3Guard | 0.88 [0.86, 0.91], p_o 0.968 | 0.83 [0.81, 0.84], p_o 0.915 | 0.81 [0.78, 0.83] |
| Keyword vs Qwen3Guard | 0.77 | 0.46 [0.44, 0.48] | 0.33 |
| WildGuard vs HCAI | 0.42 | 0.35 [0.34, 0.37] | 0.37 |
| WildGuard vs VINAY | 0.25 | 0.09 [0.07, 0.11] | 0.07 |

- Per ablated cell, keyword vs WildGuard kappa ranges from -0.01 (Qwen3.8 MPOA) and 0.04 (Gemma-4 ARA) to 0.68
  (Gemma-4 DiM, the failed edit); originals 0.72-0.78.
- Fleiss' kappa, harmful: all six measures 0.49 [0.46, 0.51] originals, 0.30 [0.29, 0.32] ablated; WildGuard +
  Qwen3Guard + HCAI 0.51 [0.49, 0.52] ablated; the same plus keyword 0.42 [0.41, 0.44].
- VINAY (Heretic mapping full + partial refusal) calls 80.0% of ablated harmful and 51.4% of ablated benign responses
  refusals -- near-uninformative on edited outputs; report it, but not as a refusal estimate.
- Benign items, ablated main: keyword vs WildGuard kappa 0.12 [0.09, 0.16] (rates 14.8% vs 2.2%); keyword > WildGuard
  in 32/32 cells (median +10.3 pp).
- Kappa is depressed by prevalence near 0 or 1 (PABAK and marginals in `h3_agreement_pairs.csv`).

**5.3 Does the gap differ between method classes?** (main study, harmful; GEE linear model of the paired gap,
clustered by prompt, n = 15,552 / 972 prompts; Holm over 4 terms; cell-level OLS with the cell as unit, n = 16)

| Term | GEE estimate, pp [95% CI] | p Holm | Cell-level OLS p |
|---|---|---|---|
| Mean gap | +9.1 [+7.5, +10.7] | 7.1e-29 | 0.007 |
| Generation effect on gap (new - old) | +9.6 [+7.7, +11.5] | 3.7e-22 | 0.11 |
| Class effect on gap (ARA - direction) | +8.4 [+6.6, +10.2] | 7.0e-20 | 0.16 |
| Generation x class | +11.2 [+7.9, +14.5] | 3.0e-11 | 0.33 |

Small pair: class effect +0.8 pp [-1.5, +3.0] (p_Holm 0.77), interaction -1.8 pp (p_Holm 0.77), generation effect on
the gap -13.2 pp [-15.8, -10.6] (the newer model's gap is smaller); cell-level OLS: all p >= 0.16. The prompt-level GEE
treats the 16 selected checkpoints as fixed; with 2 ARA cells per generation the cell-level test is the honest one.

**H3 status: NOT SUPPORTED IN THE PREDICTED DIRECTION (part 1); class difference NOT ROBUSTLY SUPPORTED (part 2).**
After ablation, keyword matching counts more refusals than WildGuard (and than Qwen3Guard), not fewer: +7.0 pp main,
+4.2 pp small; only on responses that finished before the 512-token cap is the gap negative (-4.7 pp). Write that the
**measures diverge** -- agreement falls from kappa 0.77 to 0.44 / 0.33 after ablation while the two LLM judges keep
0.81-0.88 -- not that keyword matching over-counts: no human audit exists yet (atlas_sweep D1; an LLM judge has been shown
to over-count refusal on moralising outputs). The ARA-vs-direction difference in the gap is large at prompt level in the
main study but not significant with the cell as unit and absent in the small pair. The hypothesis was written before the
judge runs; report it as stated.

---------------------------------------------------------------------------------------------------------------------

## 6. Secondary outcomes

### 6.1 Ablation effect per model (modified vs its original, WildGuard, 972 harmful items)
Source: `h1h2_cells.csv` (judge `wg`, subset `all`), `h1h2_summary.md` "Ablation effect per model".
- All 26 modified models differ from their original after Holm (26-test family); 25 are lower, by 14.5 pp (Gemma-2
  DiM) to 88.7 pp (Gemma-4 ARA, 89.3% -> 0.6%, 862 removed / 0 introduced).
- Lowest residual refusal per model: Qwen2.5-32B MPOA 3.9% [2.9, 5.3]; Qwen3.8-27B MPOA 14.0% [12.0, 16.3];
  Gemma-2 ARA 26.6% [24.0, 29.5]; Gemma-4 ARA 0.6% [0.3, 1.3]; Qwen2.5-7B ARA 4.7% [3.6, 6.3]; Qwen3.5-4B MPOA
  18.0% [15.7, 20.5].
- Absolute levels stay high for several cells (atlas_sweep D2: "not harder" is not "easy"): Gemma-2 DiM 73.8%, SOM 65.2%;
  Gemma-4 SOM 67.8%; Qwen3.8 DiM 74.3%; Qwen3.5-4B J-space 73.8%.
- Gemma-4 ARA, the lowest WildGuard figure, needs context next to it: Qwen3Guard 2.6%, keyword 22.3%, HarmBench harmful
  compliance 80.9%, WildGuard harmful response 65.2%, 79.9% capped, teacher-forced divergence 0.279 (largest of the 26),
  capability -2.2 pp (`cell_rates.csv`). Human audit pending.

### 6.2 Over-refusal on benign items (350; WildGuard)
Source: `secondary_overrefusal.csv` (`item_group` benign_all / benign_excl_advbench / xstest_safe / jbb_benign),
`secondary_correlations.csv`.
- Originals: Qwen2.5-32B 5.1% [3.3, 8.0] (18/350), Qwen3.8-27B 11.7% [8.8, 15.5] (41), Gemma-2 22.3% [18.2, 26.9] (78),
  Gemma-4 4.6% [2.8, 7.3] (16), Qwen2.5-7B 4.3% [2.6, 6.9] (15), Qwen3.5-4B 9.7% [7.0, 13.3] (34). Keyword 14.6-39.1%.
- Modified: 0.0-7.4% (median 0.6%); Holm-significant decrease vs the original in 25/26 (exact McNemar, Holm over 26),
  no significant increase. Exception: Gemma-4 DiM 5.7% [3.7, 8.7] vs 4.6% (+1.1 pp, b/c 0/4, p = 0.125; the
  failed edit). XSTest-safe <= 6.4% in every modified model; most residual over-refusal is on JBB-benign, which WildGuard's
  prompt label itself calls a harmful request in 37.9% of cases (XSTest-safe 1.1%).
- Keyword over-refusal of modified models 4.6-31.1% (median 9.3%; 25-31% for Gemma-4 DiM/MPOA/SOM): the measures diverge
  on benign items (kappa 0.12); do not call either one correct before the audit.
- Without `jbb_benign_11` (AdvBench fuzzy match): WildGuard rates change by <= 0.06 pp.
- Across cells over-refusal tracks harmful refusal: Spearman WildGuard 0.91 [0.81, 0.95] (32 cells), 0.87 [0.73, 0.93]
  (26 modified); keyword 0.86 [0.71, 0.93] / 0.79 [0.55, 0.92].

### 6.3 Dev selection vs held-out refusal (26 method cells)
Source: `secondary_dev_vs_heldout.csv`, `secondary_correlations.csv` (`b_dev_vs_heldout`).
- Spearman between the selected trial's dev keyword refusals and held-out WildGuard refusal: 0.75 [0.55, 0.87] (26 cells),
  0.82 [0.49, 0.97] (main, 16), 0.56 [-0.09, 0.84], p = 0.093 (small pair, 10). Held-out keyword: 0.83 [0.66, 0.92] /
  0.79 [0.41, 0.94] / 0.77 [0.32, 0.93]. Unchanged without the AdvBench items (0.75).
- The best method on dev is also best on held-out WildGuard in 5 of 6 models. Held-out WildGuard minus dev keyword rate:
  median -10.0 pp (range -41.6 to +27.3). Selected trial = evaluated trial in 26/26 cells.
- Not like-for-like: different prompts, 100 vs 512 tokens, keyword vs WildGuard; cells share models, so p values are
  optimistic.

### 6.4 Harmful compliance, outcome classes, degenerate and capped responses
Source: `secondary_harm_degenerate.csv`, `cell_rates.csv` (`out_*`, `capped_*`, `degenerate*`).
- Harmful compliance in modified models: HarmBench classifier 6.5-85.3% of harmful items, WildGuard harmful response
  1.5-72.9%; HarmBench-harmful among WildGuard compliance median 74.2% (complete 57.5% vs capped 86.9%); HarmBench vs
  WildGuard kappa median 0.59.
- Outcome classes, harmful items, 26 modified cells (25,272 responses): refusal 9,404 (37.2%), compliance 15,828
  (62.6%; 10,654 of them capped), degenerate 40 (0.16%), unscored 0. Originals (5,832): 4,907 / 920 / 4 / 1.
- Degenerate after the char_run recheck: 44 harmful + 4 benign rows over 32 cells (raw heuristic: 100 + 6); max 6 in one
  cell (Qwen3.5-4B ARA). The 40 repetition flags are unaudited.
- Capped at 512 tokens: 11.2% (Gemma-2 original) to 87.8% (Qwen3.8 MPOA) of harmful responses; Qwen3.8 ARA 83.7%,
  Gemma-4 ARA 79.9%. Among WildGuard compliance 70.8% are capped (median over modified cells) vs 12.3% among refusals.

### 6.5 AdvBench-overlap sensitivity
Source: `secondary_advbench_sensitivity.csv`, `secondary_advbench_pooled.csv`, `secondary_advbench_overlap_splits.csv`.
- 31 of the 44 items match a direction/SOM/ARA fitting prompt (train[:400]), 14 a dev prompt (test[:100]), 1 only unused
  prompts.
- WildGuard refusal with vs without the 44 items: |difference| max 1.95 pp, median 0.49 pp over 32 cells; Spearman across
  cells 0.9995.
- Retention of the original's refusals, overlap minus non-overlap items (26 cells): WildGuard median -0.2 pp, Wilcoxon
  p = 0.53; keyword median +13.3 pp, p = 8e-5 (keyword refusals are retained more on items resembling the dev/fit
  prompts). WildGuard difference-in-differences median -10.7 pp, p = 0.031 (ceiling-biased; the overlap items are not a
  random subset).

---------------------------------------------------------------------------------------------------------------------

## 7. Cautions for the Results and Discussion text (notes/atlas_sweep.md D, E)

- **Keyword vs judge (D1).** Write "keyword matching and WildGuard diverge", never "keyword over-counts", until a hand
  audit of keyword-yes / WildGuard-no responses exists. The HarmBench classifier is not a tiebreaker (over-flags
  compliance); agreement among measures with correlated errors is not independent confirmation; WildGuard itself is
  unvalidated on edited-model outputs. Give the 0.6% Gemma-4 ARA figure with its Wilson CI [0.3, 1.3], the Qwen3Guard and
  keyword rates and the teacher-forced divergence.
- **Gemma (D2).** Report the Gemma pair separately; write "the newer model of the pair", not "newer generations";
  report absolute residual refusal next to old/new differences; the Gemma gamma-MPOA control is not generated yet and the
  OLMo gamma-MPOA cells are not judged.
- **Gemma-4 DiM** is a failed edit at the KL cap, not residual refusal after ablation.
- **Seeds (D3).** One optimisation seed per cell; CIs cover item sampling only. Seed-2 replicates are pending; do not
  interpret old/new differences smaller than the seed spread (e.g. the -4.7 pp effects).
- **Search budget (D5) and keyword-driven selection (D6).** Not analysed here; the dev-vs-held-out correlation (0.75) is
  only a partial check.
- **Length and thinking (D7).** 512-token cap, 256-token keyword results and the incomplete share must be reported
  together; conclusions are for non-thinking mode.
- **Scale dependence of H2.** Fix the scale before reading the small-pair contrast (risk difference vs log-odds).
- **Two-point pairs, confounds.** Each pair changes recipe, architecture and (small pair, Gemma) size together; the
  small pair was generated on an RTX 4090, Qwen/Gemma on an RTX PRO 6000.
- **Not preregistered**; hypotheses were written after pilot data but before the judge runs -- report them as stated.

## 8. Still open after this pass

- Judge batch 2 (OLMo, Mistral, OLMo gamma-MPOA, community, seed-2) and generation of the remaining cells (section 1).
- Human-labelled audit sample (keyword-yes / WildGuard-no, WildGuard-yes / keyword-no, Gemma-4 ARA, repetition flags).
- Not re-verified in this pass: VB mixed-model estimates, bootstrap CIs, the H3 cell-level OLS.
