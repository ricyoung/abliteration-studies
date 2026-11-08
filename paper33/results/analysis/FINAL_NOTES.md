# Paper 33 -- Final results notes (held-out refusal; judge batches 1 + 2 complete)

Hand-written on 2026-11-06 from the run of `scripts/analysis/run_all.sh` that finished at 05:43:07Z (assembly
2026-11-06T05:35:28Z); corrected and re-checked against the re-run that finished at 06:10:27Z (assembly
2026-11-06T06:02:44Z, same `responses_judged.csv.gz` md5 `5a5514a72949`; `run_all_log.txt`; freshness check: "outputs
match the result trees"; changes in section 0, row 7). Supersedes
`RESULTS_NOTES.md` (interim, 32 cells). The regenerated summaries (`h1h2_summary.md`, `h3_summary.md`,
`secondary_summary.md`, `final_log.txt`) are authoritative; every number below names its source file in
`results/analysis/`. All rates are WildGuard (primary judge) on the 972 harmful held-out items unless stated otherwise;
95% intervals throughout (Wilson for rates, Newcombe for paired differences, item bootstrap for contrasts, cell bootstrap
for Spearman correlations). pp = percentage points. Hypotheses are quoted from `paper/sections/02_introduction.tex` and
reported as stated (written before the judge runs; not preregistered).

---------------------------------------------------------------------------------------------------------------------

## 0. What was done in this pass

| # | Change | Files |
|---|---|---|
| 1 | Re-ran the whole chain after judge batch 2. Assembly now holds **74 judged cells, 97,828 responses** (role: original 10, primary 42, control/gamma-MPOA 4, external/community 8, replicate/seed-2 10). | `assemble_log.txt`, `dir_status.csv` |
| 2 | Refreshed `paper/figures/data/{results,trials,runs}.csv` (`bash scripts/figures/make_figures.sh --data`; figures not re-rendered) so that the capability / teacher-forcing columns of `cell_rates.csv` include gamma-MPOA and community cells, and `trials.csv` includes the gamma-MPOA and seed-2 runs. 64 cells were then in `results.csv`; the 10 seed-2 cells are read from their `summary.json` by `final.py` (`final_capability.csv`, column `check_results_csv`: 64 agree with `results.csv`, 10 read only there). | `paper/figures/data/*.csv` |
| 3 | `h1h2.py`: Qwen3Guard-Gen-8B (`ref_q3g`) added as a fifth outcome ("judge" `q3g`) next to WildGuard, WildGuard outcome class, keyword 512 and keyword 256; summary wording no longer says "interim" once all four main pairs are judged. | `scripts/analysis/h1h2.py` |
| 4 | `freshness.py`: a new result dir whose model prefix is not a paper-33 model no longer marks the tables stale (`runs/mc-rtx-1/results/bench/qwen2.5-coder-32b__base` appeared during the run; `assemble.py` skips it as "unknown model"). | `scripts/analysis/freshness.py` |
| 5 | New `scripts/analysis/final.py` (step 6 of `run_all.sh`; freshness is now step 7): analyses (a)-(g) below, each with its own CSV (`final_*.csv`) and console report `final_log.txt`. | `scripts/analysis/final.py`, `run_all.sh` |
| 6 | Stale caveat wording in `h3.py` / `secondary.py` (Qwen + Gemma only, batch 2 pending) made data-driven. Numbers unaffected. | `scripts/analysis/h3.py`, `secondary.py` |
| 7 | Verification pass (06:10Z re-run). Corrected here: section 3 table, ablated primary keyword 40.0 -> 39.9% (16,308 / 40,824 = 0.39947); 4(c) the "method share of variance 0.95 / 0.91" was MS_method / (MS_method + MS_within), a ratio of mean squares, now reported as eta-squared 0.94 / 0.88 (variance-component share 0.90 / 0.81); 4(d) the dev tie behind "7 of 12" is now stated (6 of 12 with a unique dev best required); 4(f) originals' median keyword-yes share 0.233 -> 0.232, and the discordant at-cap pairs are 2 older-only vs 4 newer-only (order was reversed). `final.py` writes `eta_sq_method`, `var_component_share_method`, `ms_ratio_method` (was `share_variance_method`) and the tie columns `best_dev_tied`, `best_dev_tie_kl`, `best_dev_unique_is_best_wg`, `best_dev_kl_tiebreak(_is_best_wg)`; `secondary.py` reports the same tie (6/10 with the KL tie-break, 5/10 with a unique dev best). All other numbers in this file are unchanged by the re-run (VB fits differ only in the 6th significant digit). | `scripts/analysis/final.py`, `secondary.py`, `README.md` |

Coverage check (task item 1): `dir_status.csv` lists 74 used dirs, all `judged`: 4 main families x {original, DiM,
MPOA, SOM, ARA} = 40; gamma-MPOA 4 (Gemma-2/4, OLMo-2/3.1); community 8; small pair 12; seed-2 10. No judged dir was
missed. The only other `judgments.jsonl` in the project (`runs/local-4090/judge_smoke/qwen3.5-4b__mpoa`) is a judge smoke
test outside the result roots and is correctly not assembled. The H200 originals of the OLMo/Mistral cells and the 4090 originals of the small-pair cells are recorded as
duplicates with identical `responses.jsonl` (md5) of the judged import copies; 18 dirs are skipped (3 unknown models:
`qwen2.5-14b`, `qwen2.5-coder-14b`, `qwen2.5-coder-32b`; 13 superseded attempts with a suffix such as `.publish-403`;
2 non-protocol copies `qwen2.5-14b__heretic1x-general`, `qwen2.5-coder-14b__heretic1x-general` of non-paper-33
models, new in the 06:02Z assembly). Judge revisions: one set for all 74 cells
(`assemble_log.txt`). WildGuard gave no refusal label on 5 rows (2 harmful: `mistral-small-2409__base/cw_astra_1912`,
`qwen2.5-32b__base/cw_astra_1459`), dropped pairwise.

---------------------------------------------------------------------------------------------------------------------

## 1. H1 -- residual refusal higher for the newer model of each pair, for every method

> H1: "at matched distribution shift (first-token KL divergence <= 0.1), residual refusal after ablation would be
> higher for the newer model of each pair than for its predecessor, for every method."

**Status: NOT SUPPORTED.** In none of the four main-study pairs is residual refusal significantly higher on the newer
model for every method; across the 16 main-study pair x method comparisons 8 are higher on the newer model, 4 lower and
4 not significant after Holm. The small pair meets H1 with seed 1 (5/5 methods) but not with seed 2 (ARA +0.0 pp).

Source: `h1h2_generation_effects.csv` (judge `wg`, subset `all`), `h1h2_tests.csv` (family `H1`, 23 tests: 21 pair x
method McNemar tests + 2 pooled GEE terms), `h1h2_summary.md`.

| Pair | Method | Older % | Newer % | Dev KL old / new | New - old, pp [95% CI] | b / c | p | p Holm | Baseline-adj., pp [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 2.5-32B -> 3.8-27B | Original | 76.8 | 89.9 | | +13.1 [+10.9, +15.4] | 6 / 133 | <0.001 | (baseline) | |
| | DiM | 47.4 | 74.3 | 0.018 / 0.053 | +26.9 [+23.8, +29.8] | 17 / 278 | <0.001 | <0.001 | +13.7 [+10.1, +17.1] |
| | MPOA | 3.9 | 14.0 | 0.060 / 0.085 | +10.1 [+7.8, +12.5] | 23 / 121 | <0.001 | <0.001 | -3.1 [-6.3, +0.2] |
| | SOM | 42.7 | 38.0 | 0.075 / 0.084 | -4.7 [-8.0, -1.4] | 157 / 111 | 0.006 | 0.041 (a) | -17.8 [-21.5, -14.2] |
| | ARA | 6.1 | 40.4 | 0.062 / 0.061 | +34.4 [+31.2, +37.5] | 11 / 345 | <0.001 | <0.001 | +21.3 [+17.1, +25.4] |
| Gemma 2-27B -> 4-31B | Original | 88.3 | 89.3 | | +1.0 [-1.4, +3.5] | 69 / 79 | 0.460 | (baseline) | |
| | DiM | 73.8 | 90.1 (b) | 0.069 / 0.093 | +16.4 [+13.5, +19.3] | 35 / 194 | <0.001 | <0.001 | +15.3 [+13.0, +17.8] |
| | MPOA | 45.2 | 40.4 | 0.099 / 0.069 | -4.7 [-8.6, -0.9] | 206 / 160 | 0.019 | 0.093 | -5.8 [-9.7, -1.7] |
| | SOM | 65.2 | 67.8 | 0.086 / 0.055 | +2.6 [-1.1, +6.2] | 151 / 176 | 0.184 | 0.369 | +1.5 [-1.9, +4.9] |
| | ARA | 26.6 | 0.6 | 0.099 / 0.084 | -26.0 [-28.9, -23.3] | 253 / 0 | <0.001 | <0.001 | -27.1 [-30.3, -23.6] |
| OLMo 2-32B -> 3.1-32B | Original | 83.2 | 84.1 | | +0.8 [-0.6, +2.2] | 19 / 27 | 0.302 | (baseline) | |
| | DiM | 33.5 | 68.3 | 0.089 / 0.089 | +34.8 [+31.5, +37.9] | 14 / 352 | <0.001 | <0.001 | +34.0 [+30.5, +37.6] |
| | MPOA | 3.6 | 65.0 | 0.037 / 0.092 | +61.4 [+58.2, +64.4] | 2 / 599 | <0.001 | <0.001 | +60.6 [+57.1, +64.1] |
| | SOM | 27.3 | 41.7 | 0.047 / 0.064 | +14.4 [+11.5, +17.3] | 45 / 185 | <0.001 | <0.001 | +13.6 [+10.4, +16.7] |
| | ARA | 13.3 | 10.8 | 0.049 / 0.053 | -2.5 [-4.6, -0.3] | 67 / 43 | 0.028 | 0.111 | -3.3 [-5.8, -0.9] |
| Mistral Small-2409 -> Small-3.2 | Original | 55.7 | 73.6 | | +17.9 [+15.2, +20.6] | 21 / 195 | <0.001 | (baseline) | |
| | DiM | 32.8 | 28.5 | 0.099 / 0.093 | -4.3 [-7.4, -1.2] | 141 / 99 | 0.008 | 0.048 (a) | -22.2 [-26.1, -18.4] |
| | MPOA | 6.9 | 4.0 | 0.049 / 0.094 | -2.9 [-4.7, -1.2] | 51 / 23 | 0.002 | 0.014 | -20.8 [-24.0, -17.7] |
| | SOM | 3.7 | 7.2 | 0.062 / 0.100 | +3.5 [+1.7, +5.3] | 22 / 56 | <0.001 | 0.001 | -14.4 [-17.7, -11.1] |
| | ARA | 4.9 | 4.2 | 0.066 / 0.083 | -0.7 [-2.1, +0.6] | 23 / 16 | 0.337 | 0.369 | -18.6 [-21.6, -15.7] |
| Small Qwen2.5-7B -> Qwen3.5-4B | Original | 72.1 | 88.5 | | +16.4 [+13.9, +18.9] | 11 / 170 | <0.001 | (baseline) | |
| | DiM | 20.6 | 54.6 | 0.078 / 0.054 | +34.1 [+30.5, +37.5] | 43 / 374 | <0.001 | <0.001 | +17.7 [+13.5, +21.8] |
| | MPOA | 13.8 | 18.0 | 0.074 / 0.054 | +4.2 [+1.6, +6.9] | 66 / 107 | 0.002 | 0.018 | -12.1 [-15.7, -8.3] |
| | SOM | 12.2 | 21.9 | 0.076 / 0.055 | +9.7 [+6.9, +12.4] | 50 / 144 | <0.001 | <0.001 | -6.7 [-10.4, -3.0] |
| | J-space | 52.6 | 73.8 | 0.022 / 0.019 | +21.2 [+18.1, +24.3] | 37 / 243 | <0.001 | <0.001 | +4.8 [+1.3, +8.3] |
| | ARA | 4.7 | 18.9 | 0.021 / 0.064 | +14.2 [+11.6, +16.9] | 24 / 162 | <0.001 | <0.001 | -2.2 [-6.0, +1.6] |

b / c = items refused only by the older / only by the newer model's version (exact McNemar). Baseline-adjusted =
generation effect of the method minus that of the originals.
(a) Not robust to dropping the 44 AdvBench-overlap items: Qwen SOM -3.9 pp, p_Holm 0.127; Mistral DiM -2.9 pp,
p_Holm 0.320 (`h1h2_summary.md`, "Sensitivity"). Without the 44 items the generation effects change by at most 1.9 pp
(median 0.4 pp).
(b) **Failed edit.** `gemma-4-31b__dim` is the only modified cell whose held-out refusal is at or above its original
(90.1% vs 89.3%, +0.8 pp [+0.1, +1.6], 2 removed / 10 introduced, p_Holm 0.039; `h1h2_cells.csv`, ablation family
of 42 tests). Its selected dev trial had 98/100 keyword refusals at KL 0.093 (`final_search_runs.csv`). The Gemma DiM
"generation effect" therefore means that no trial at KL <= 0.1 removed refusal from Gemma-4. Weak but successful
edits (held-out reduction < 20 pp, all p_Holm < 0.001): Gemma-2 DiM -14.5, Qwen3.5-4B J-space -14.7, Qwen3.8 DiM -15.6,
OLMo-3.1 DiM -15.7, OLMo-3.1 MPOA -19.0, Qwen2.5-7B J-space -19.5 pp (`h1h2_cells.csv`).

**Per pair.** Qwen: 3 of 4 methods higher on the newer model, SOM lower (not robust, (a)). Gemma: only the failed DiM
edit is higher; ARA much lower (26.6% -> 0.6%), MPOA and SOM not significant. OLMo: DiM, MPOA and SOM higher (MPOA
3.6% -> 65.0%, the largest effect in the study), ARA not significant (-2.5 pp, p_Holm 0.111). Mistral: SOM higher
(+3.5 pp), DiM and MPOA lower, ARA not significant. Small pair: all five higher (seed 1).

**Baseline.** The newer original already refuses more in Qwen (+13.1 pp), Mistral (+17.9 pp) and the small pair
(+16.4 pp); Gemma (+1.0, p 0.46) and OLMo (+0.8, p 0.30) do not differ. Baseline-adjusted, the generation effect
stays positive (CI excludes 0) for Qwen DiM and ARA, Gemma DiM (failed edit), OLMo DiM, MPOA and SOM, small DiM and
J-space; it is negative for Qwen SOM, Gemma MPOA and ARA, OLMo ARA, all four Mistral methods, small MPOA and SOM.

**Pooled main study, four families** (`h1h2_models.csv`, scope `main`; `h1h2_tests.csv`; log-odds):
- GEE (exchangeable, clustered by prompt, family fixed effect): generation effect, direction class +0.62
  [+0.57, +0.68], p_Holm < 0.001; optimisation class (ARA) +0.12 [+0.00, +0.24], p = 0.045, p_Holm = 0.135.
- Planned model, VB mixed (`BinomialBayesMixedGLM`, random intercepts prompt + family): direction +0.86 [+0.82, +0.90];
  optimisation +0.15 [+0.04, +0.26]; SD(family) 1.19 [0.63, 2.26], SD(prompt) 1.58 [1.51, 1.65].
- Per method (GEE, descriptive): DiM +0.90 [+0.82, +0.98], MPOA +1.06 [+0.96, +1.16], SOM +0.20 [+0.12, +0.27], ARA
  +0.12 [+0.00, +0.24], p = 0.043.
- With the originals as reference (baseline-adjusted): originals' own generation difference +0.58 [+0.49, +0.66];
  direction class **+0.03 [-0.05, +0.11], p = 0.529**; optimisation class -0.46 [-0.58, -0.34], p < 0.001.
- Heterogeneity (GEE Wald): generation x family (direction class) chi2 = 547.4, p < 0.001.
- Robustness (`final_pooled_robustness.csv`): without the failed Gemma DiM pair, direction +0.61 [+0.55, +0.66];
  leaving one family out, direction +0.24 (without OLMo) to +0.81 (without Gemma), optimisation -1.37 (without Qwen) to
  +1.02 (without Gemma).

**Judge sensitivity** (`h1h2_summary.md`, "How the conclusions change with the judge"; `h1h2_tests.csv`). No judge
makes H1 hold in a main pair under the full-response measures. Qwen3Guard: Qwen SOM becomes n.s. (-0.4 pp), Gemma SOM
significant (+7.4 pp), Mistral MPOA n.s. (-2.1 pp, p_Holm 0.090); pooled optimisation class +0.21 [+0.10, +0.32],
p_Holm < 0.001. Keyword 512: Gemma MPOA +18.9 pp and SOM +11.8 pp (both p_Holm < 0.001), Gemma ARA -2.6 pp (n.s.),
Mistral DiM n.s.; small-pair MPOA, J-space and ARA n.s. Keyword 256: OLMo ARA +5.2 pp (p_Holm 0.009), so all four OLMo
methods are higher on the newer model under that measure only. Degenerate rows counted as not refused (`wg_class`):
effects change by <= 0.1 pp, no decision changes.

---------------------------------------------------------------------------------------------------------------------

## 2. H2 -- generation x method class: ARA loses less effectiveness on newer models

> H2: "the generation effect would interact with method class, such that optimisation-based ablation (ARA) would lose
> less effectiveness on newer models than direction-based methods (DiM, MPOA, SOM and, on a small pair, J-space)."

**Status: SUPPORTED in the pre-specified pooled main-study test (WildGuard); NOT SUPPORTED as a consistent pattern
across pairs** (as predicted in Gemma and OLMo, opposite in Qwen, null in Mistral, scale-dependent in the small pair;
the pooled estimate reverses without Gemma and vanishes without OLMo, and is null under keyword scoring).

Source: `h1h2_h2_contrasts.csv`, `h1h2_tests.csv` (family `H2`, 6 tests), `h1h2_models.csv`, `final_pooled_robustness.csv`.

| Pair | ARA gen. effect, pp | Direction mean, pp | RD contrast, pp [95% CI] | GEE interaction, log-odds [95% CI] | p Holm | VB interaction [95% CrI] | Direction |
|---|---|---|---|---|---|---|---|
| Qwen | +34.4 | +10.7 | +23.6 [+19.9, +27.1] | +1.89 [+1.61, +2.16] | <0.001 | +2.29 [+2.14, +2.44] | opposite |
| Gemma | -26.0 | +4.7 | -30.8 [-34.0, -27.5] | -4.27 [-5.07, -3.47] | <0.001 | -4.90 [-5.64, -4.17] | as predicted |
| OLMo | -2.5 | +36.9 | -39.3 [-42.2, -36.4] | -1.87 [-2.08, -1.65] | <0.001 | -3.25 [-3.49, -3.01] | as predicted |
| Mistral | -0.7 | -1.2 | +0.5 [-1.4, +2.4] | -0.06 [-0.37, +0.25] | 0.696 | -0.08 [-0.42, +0.26] | null |
| Small pair | +14.2 | +17.3 (incl. J-space) | -3.1 [-6.0, -0.1] | +0.76 [+0.46, +1.06] | <0.001 | +0.80 [+0.62, +0.99] | RD as predicted, log-odds opposite |
| **Main pooled (4 families)** | | | | **-0.50 [-0.62, -0.38]** | **<0.001** | **-0.71 [-0.81, -0.60]** | as predicted |

- Method-intercept GEE (one intercept per pair x method): pooled main -0.59 [-0.71, -0.48]; main + small pair GEE
  -0.29 [-0.41, -0.17], VB -0.46 [-0.55, -0.37] (`h1h2_models.csv`, scopes `main`, `main+small`).
- Heterogeneity: generation x class x family chi2 = 430.4, p < 0.001 (`h1h2_models.csv`, `gee_heterogeneity`).
- Without the 44 AdvBench items: pooled -0.54 [-0.66, -0.42], p_Holm < 0.001; per-pair signs unchanged; small-pair RD
  -5.2 [-8.2, -2.2], GEE +0.60 [+0.30, +0.90] (`h1h2_h2_contrasts.csv`, `h1h2_tests.csv`, subset `excl_advbench`).
- Qwen3Guard: pooled -0.46 [-0.57, -0.36], p_Holm < 0.001 (same per-pair signs as WildGuard; Mistral null).
  Keyword 512: pooled +0.02 [-0.08, +0.12], p_Holm 1.000; keyword 256: +0.03 [-0.08, +0.13], p_Holm 1.000 (`h1h2_tests.csv`).
- Pooled-estimate robustness (`final_pooled_robustness.csv`, GEE as above): without the failed Gemma DiM pair -0.49
  [-0.60, -0.38]; gamma-MPOA in place of MPOA for Gemma and OLMo -0.42 [-0.53, -0.30]; leave one family out: without
  Qwen -2.06 [-2.28, -1.85], **without Gemma +0.20 [+0.05, +0.36], p = 0.008 (opposite to H2)**, **without OLMo -0.00
  [-0.15, +0.15], p = 0.98**, without Mistral -0.59 [-0.71, -0.47].
- Seed dependence (small pair, `final_seed_generation.csv`): RD contrast -3.1 [-6.0, -0.1] with seed 1, -28.7
  [-31.0, -26.5] with seed 2, -15.9 [-18.0, -13.8] seed-averaged (section 3c).

---------------------------------------------------------------------------------------------------------------------

## 3. H3 -- keyword counts vs an independent judge (all 74 cells)

> H3: "keyword-based refusal counts would underestimate residual refusal relative to an independent judge, and the size
> of this gap would differ between method classes."

**Status, part 1 (direction): NOT SUPPORTED IN THE PREDICTED DIRECTION.** After ablation keyword matching counts more
refusals than WildGuard (and than Qwen3Guard), not fewer. **Part 2 (class difference): NOT ROBUSTLY SUPPORTED**
(significant at prompt level in the main study, not with the cell as unit, absent in the small pair).

Source: `h3_gap_pooled.csv` (subset `harmful_all`), `h3_gap_cells.csv`, `h3_agreement_pairs.csv`, `h3_models.csv`,
`h3_summary.md`. Gap = keyword rate - WildGuard rate on the same responses (H3 predicts negative).

| Cells (harmful items) | n cells | n responses | Keyword % | WildGuard % | Gap kw - wg, pp [95% CI] | kw-only / wg-only |
|---|---|---|---|---|---|---|
| Originals | 10 | 9,718 | 82.8 | 80.2 | +2.6 [+1.8, +3.4] | 531 / 277 |
| Ablated, primary (seed 1), all | 42 | 40,824 | 39.9 | 31.5 | +8.4 [+6.9, +10.0] | 7,503 / 4,059 |
| -- main study | 32 | 31,104 | 42.0 | 32.3 | +9.8 [+8.2, +11.3] | 5,897 / 2,859 |
| -- small pair | 10 | 9,720 | 33.3 | 29.1 | +4.2 [+2.5, +5.9] | 1,606 / 1,200 |
| gamma-MPOA (control) | 4 | 3,888 | 47.0 | 29.9 | +17.1 [+14.9, +19.4] | 1,038 / 372 |
| Community (external) | 8 | 7,776 | 22.9 | 6.4 | +16.5 [+15.2, +17.8] | 1,475 / 192 |
| Seed 2 (replicate) | 10 | 9,720 | 32.8 | 29.2 | +3.5 [+2.0, +5.0] | 1,351 / 1,010 |
| All 74 cells | 74 | 71,926 | 43.3 | 35.0 | +8.3 [+7.0, +9.6] | 11,898 / 5,910 |

- Per ablated primary cell: keyword > WildGuard in 34/42 (24 Holm-significant), keyword < WildGuard in 7/42 (3
  Holm-significant: `gemma-2-27b__dim`, `gemma-2-27b__som`, `qwen3.5-4b__jspace`) (`h3_summary.md`). Per role (median
  cell gap): gamma-MPOA 4/4 positive (+15.3 pp), community 7/8 positive (+13.5 pp), seed 2 7/10 positive (+4.6 pp)
  (`h3_gap_cells.csv`, comparison `kw_vs_wg`).
- Against Qwen3Guard: ablated primary kw - q3g +6.2 pp [+4.7, +7.7]; Qwen3Guard - WildGuard +2.3 pp [+1.5, +3.1].
- **Length dependence.** Ablated primary cells: responses that ended before the 512-token cap -1.2 pp [-3.2, +0.9]
  (n = 20,011; small pair alone -7.6 pp [-10.3, -5.0], the predicted direction), capped responses +17.7 pp [+16.2, +19.3]
  (n = 20,813); 256-token keyword window +4.5 pp [+3.0, +5.9].
- Wording of the 7,503 keyword-only refusals (string heuristics, not validated): 90.9% contain only safety/disclaimer
  vocabulary, 5.2% a first-person refusal phrase; Qwen3Guard calls 14.2% refusals; HarmBench calls 64.5% harmful
  compliance; 60.0% are capped. Qwen3Guard agrees with 83.9% of the 4,059 WildGuard-only refusals (`h3_summary.md`).
- **Agreement** (harmful, Cohen's kappa [prompt-bootstrap CI]): keyword vs WildGuard 0.73 [0.70, 0.75] in originals,
  0.40 [0.39, 0.42] ablated main, 0.33 [0.31, 0.35] ablated small; WildGuard vs Qwen3Guard 0.90 [0.88, 0.92] originals,
  0.82 [0.80, 0.83] ablated. Fleiss, all ablated cells: panel of six 0.27; WildGuard + Qwen3Guard + HCAI 0.50; the same
  plus keyword 0.40 (`h3_agreement_pairs.csv`, `h3_agreement_fleiss.csv`).
- **Class difference** (main study, 4 families, primary GEE on the paired gap, n = 31,104 / 972 prompts; Holm over 4
  terms; cell-level OLS n = 32): mean gap +11.3 pp [+9.8, +12.7]; generation effect on the gap +4.4 [+3.1, +5.7]; class
  effect (ARA - direction) **+6.0 [+4.6, +7.4], p_Holm < 0.001, cell-level OLS p = 0.151**; generation x class +9.4
  [+7.2, +11.6], p_Holm < 0.001, OLS p = 0.257. Without the AdvBench items the class effect is +5.9 [+4.5, +7.3].
  Small pair: class effect +0.8 [-1.5, +3.0], p_Holm 0.774; generation effect on the gap -13.2 [-15.8, -10.6]
  (`h3_models.csv`).
- Benign items: keyword > WildGuard in 74/74 cells (median +10.0 pp) (`h3_summary.md`).

---------------------------------------------------------------------------------------------------------------------

## 4. New analyses (final.py)

### (a) gamma-MPOA control (post-norm leakage; Gemma-2/4, OLMo-2/3.1)

Sources: `final_gmpoa_control.csv`, `final_gmpoa_benchmarks.csv`, `final_gmpoa_generation.csv`,
`final_kl_vs_capability.csv`, `final_kl_vs_capability_cells.csv`, `final_pooled_robustness.csv`.

| Model | gamma-MPOA % [95% CI] | MPOA % | gamma - MPOA, pp [95% CI] | b / c | p Holm (4) | Capability mean, gamma / MPOA / orig. | HumanEval, gamma / MPOA / orig. | Dev KL gamma / MPOA | TF divergence gamma / MPOA |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-27B | 40.9 [37.9, 44.1] | 45.2 | -4.2 [-6.8, -1.6] | 103 / 62 | 0.0018 | 67.2 / 66.7 / 67.1 | 76.8 / 76.8 / 76.2 | 0.097 / 0.099 | 0.096 / 0.063 |
| Gemma-4-31B | 20.6 [18.2, 23.2] | 40.4 | -19.9 [-22.7, -16.9] | 218 / 25 | <0.001 | **66.2** / 87.0 / 87.6 | **39.6** / 90.9 / 92.7 | 0.061 / 0.069 | 0.128 / 0.007 |
| OLMo-2-32B | 0.5 [0.2, 1.2] | 3.6 | -3.1 [-4.4, -2.1] | 30 / 0 | <0.001 | **58.0** / 64.7 / 66.1 | **47.6** / 57.9 / 61.0 | 0.085 / 0.037 | 0.196 / 0.088 |
| OLMo-3.1-32B | 57.5 [54.4, 60.6] | 65.0 | -7.5 [-9.4, -5.6] | 84 / 11 | <0.001 | 80.4 / 80.3 / 79.9 | 90.2 / 90.2 / 87.2 | 0.099 / 0.092 | 0.015 / 0.008 |

b / c = items refused only by MPOA / only by gamma-MPOA.

- gamma-MPOA has lower held-out refusal than MPOA in all four post-norm models (Holm over 4, all p_Holm <= 0.0018);
  under Qwen3Guard also in all four (p <= 0.047), under keyword in 3 of 4 (Gemma-2 +2.2 pp, p = 0.17).
- **Generation effects** (`final_gmpoa_generation.csv`): Gemma MPOA 45.2 -> 40.4% (-4.7 pp [-8.6, -0.9]) vs gamma-MPOA
  40.9 -> 20.6% (-20.4 pp [-23.6, -17.1], p < 0.001); OLMo MPOA 3.6 -> 65.0% (+61.4 pp [+58.2, +64.4]) vs gamma-MPOA
  0.5 -> 57.5% (+57.0 pp [+53.8, +60.1], 0 / 554, p < 0.001). The large OLMo generation effect therefore survives the
  gamma correction; the post-norm leak does not explain it. With gamma-MPOA in place of MPOA the pooled H2 interaction
  is -0.42 [-0.53, -0.30] and the direction-class generation effect +0.54 [+0.49, +0.59] (`final_pooled_robustness.csv`).
- **Capability damage at KL <= 0.1.** Gemma-4 gamma-MPOA (selected dev KL 0.061, well inside the cap) loses 21.4 pp of
  mean capability; paired per-item McNemar vs the original (`final_gmpoa_benchmarks.csv`, Holm over 20 per reference):
  HumanEval 39.6 vs 92.7 (-53.0 pp [-60.5, -44.4], 89 / 2 items), MBPP 37.8 vs 71.4 (-33.6 [-37.8, -29.2]), IFEval
  81.1 vs 91.5 (-10.4 [-13.5, -7.3]), MMLU-Pro 75.9 vs 85.6 (-9.7 [-11.9, -7.6]), all p_Holm < 0.001; GSM8K 96.4 vs
  96.6 (n.s.). It is also the only cell with many degenerate responses (28 of 1,322 rows, char_run after the recheck;
  `final_gmpoa_control.csv` `degenerate_all_k`; `assemble_log.txt`) and has 73.7% of harmful responses capped. OLMo-2
  gamma-MPOA (KL 0.085) loses 8.1 pp: MBPP 24.8 vs 44.0 (-19.2 [-22.8, -15.6]), HumanEval 47.6 vs 61.0 (-13.4
  [-20.7, -5.8], p_Holm 0.013), IFEval 77.8 vs 84.3 (-6.5 [-9.4, -3.6]). Gemma-2 and OLMo-3.1 gamma-MPOA show no
  significant benchmark change (all p_Holm = 1.0).
  (Traceability: the raw `judgments.jsonl` of `gemma-4-31b__gmpoa` flags 29 rows as char_run; the 28 above is the
  count after `assemble.py`'s char_run recheck, `assemble_log.txt` "char_run recheck".)
- These are the two largest capability losses among all 64 modified cells (next: OLMo-2 ARA -4.2 pp, Gemma-4 SOM
  -2.8 pp; `final_kl_vs_capability_cells.csv`, `final_capability.csv`). **The first-token KL cap did not prevent
  capability damage:** across the 56 modified cells with a dev point, selected dev KL is unrelated to the capability
  change (Spearman rho -0.03 [-0.29, +0.24], p = 0.81), whereas teacher-forced divergence on the benign held-out items
  tracks it (rho -0.48 [-0.66, -0.24], p < 0.001; with community cells, n = 64: -0.46 [-0.63, -0.24])
  (`final_kl_vs_capability.csv`). Teacher-forced divergence is not a sufficient guard either: Gemma-4 gamma-MPOA has
  0.128 (8th largest of 64), below Gemma-4 ARA (0.279, capability -2.2 pp).

### (b) Community checkpoints vs the selected models of this study

Source: `final_community_vs_selected.csv` (each community model vs each of the 4 seed-1 cells of the same model; Holm
over 4 per community model). "Best ours" = lowest held-out WildGuard refusal among DiM, MPOA, SOM, ARA.

| Community model | Community % [95% CI] | Best ours (method, %) | Community - best ours, pp [95% CI] | b / c | p Holm | TF divergence comm. / ours | Capability delta comm. / ours, pp | Keyword % comm. / ours |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-32b huihui | 1.3 [0.8, 2.3] | MPOA 3.9 | -2.6 [-3.9, -1.5] | 29 / 4 | <0.001 | 0.151 / 0.042 | -0.94 / -0.03 | 8.2 / 17.1 |
| qwen2.5-32b zetasepic-v2 | 1.1 [0.6, 2.0] | MPOA 3.9 | -2.8 [-4.1, -1.7] | 30 / 3 | <0.001 | 0.085 / 0.042 | -0.45 / -0.03 | 12.9 / 17.1 |
| qwen3.8-27b hereticorg-ara | 26.4 [23.8, 29.3] | MPOA 14.0 | +12.4 [+9.6, +15.3] | 49 / 170 | <0.001 | 0.026 / 0.060 | -0.47 / +0.33 | 65.7 / 24.4 |
| qwen3.8-27b huihui | 1.4 [0.9, 2.4] | MPOA 14.0 | -12.6 [-14.8, -10.5] | 125 / 3 | <0.001 | 0.111 / 0.060 | -0.73 / +0.33 | 26.2 / 24.4 |
| gemma-4-31b coder3101 | 0.9 [0.5, 1.8] | ARA 0.6 | +0.3 [-0.4, +1.1] | 3 / 6 | 0.51 | 0.188 / 0.279 | +0.02 / -2.18 | 16.3 / 22.3 |
| mistral-small-3.2 huihui | 5.0 [3.8, 6.6] | MPOA 4.0 | +1.0 [-0.5, +2.6] | 22 / 32 | 0.44 | 0.131 / 0.066 | +0.05 / +0.17 | 15.0 / 13.3 |
| mistral-small-3.2 llmfan46 | 13.8 [11.8, 16.1] | MPOA 4.0 | +9.8 [+7.5, +12.2] | 23 / 118 | <0.001 | 0.191 / 0.066 | -0.63 / +0.17 | 8.8 / 13.3 |
| olmo-3.1-32b muxodious | 0.7 [0.3, 1.5] | ARA 10.8 | -10.1 [-12.1, -8.3] | 98 / 0 | <0.001 | 0.106 / 0.049 | -0.15 / +0.41 | 29.6 / 34.3 |

b / c = items refused only by our best cell / only by the community model.

- 4 of 8 community checkpoints have lower held-out refusal than the best model selected here (two Qwen2.5-32B, Qwen3.8
  huihui, OLMo-3.1 muxodious), 2 do not differ (Gemma-4 coder3101, Mistral huihui), 2 are higher (Qwen3.8
  hereticorg-ara, Mistral llmfan46). All four lower ones have a larger teacher-forced divergence than our best cell
  (0.085-0.151 vs 0.042-0.060), i.e. they sit at a larger distribution shift than the KL <= 0.1 cap allowed here; their
  capability change is within -0.94 to -0.15 pp.
- Two of the four "harder" newer models are removable to about 1% by a public checkpoint: Qwen3.8-27B (huihui 1.4% vs
  our best 14.0%) and OLMo-3.1-32B (muxodious 0.7% vs 10.8%). Residual refusal under the KL cap is therefore an upper
  bound on removability, not a floor.
- The community ARA checkpoint for Qwen3.8 (hereticorg-ara) has lower refusal than our ARA cell (26.4 vs 40.4%,
  -14.0 pp, 187 / 51, p_Holm < 0.001) at lower divergence (0.026 vs 0.042), but a keyword rate of 65.7%.
- Over-refusal (benign, WildGuard): 0.0-0.3% for 7 of 8, 5.1% for Mistral llmfan46 (`final_community_vs_selected.csv`).
  HarmBench harmful compliance 47.1-91.0% (`community_harm_hb_rate`).

### (c) Seed 2 vs seed 1 (small pair, 10 cells)

Sources: `final_seed_cells.csv`, `final_seed_generation.csv`, `final_seed_method_vs_seed.csv`.

| Cell | Seed 1 % | Seed 2 % | s2 - s1, pp [95% CI] | s1-only / s2-only | p Holm (10) | item kappa | Dev refusals s1 / s2 | Dev KL s1 / s2 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B DiM | 20.6 | 4.4 | -16.2 [-18.7, -13.7] | 169 / 12 | <0.001 | 0.20 | 55 / 20 | 0.078 / 0.054 |
| Qwen2.5-7B MPOA | 13.8 | 3.6 | -10.2 [-12.4, -8.2] | 107 / 8 | <0.001 | 0.28 | 19 / 18 | 0.074 / 0.042 |
| Qwen2.5-7B SOM | 12.2 | 17.5 | +5.2 [+3.2, +7.3] | 25 / 76 | <0.001 | 0.59 | 27 / 45 | 0.076 / 0.091 |
| Qwen2.5-7B J-space | 52.6 | 61.2 | +8.6 [+6.5, +10.7] | 15 / 99 | <0.001 | 0.76 | 84 / 87 | 0.022 / 0.018 |
| Qwen2.5-7B ARA | 4.7 | 2.2 | -2.6 [-4.0, -1.3] | 34 / 9 | <0.001 | 0.34 | 8 / 4 | 0.021 / 0.053 |
| Qwen3.5-4B DiM | 54.6 | 48.8 | -5.9 [-8.9, -2.8] | 142 / 85 | <0.001 | 0.53 | 30 / 28 | 0.054 / 0.047 |
| Qwen3.5-4B MPOA | 18.0 | 33.7 | +15.7 [+13.2, +18.3] | 14 / 167 | <0.001 | 0.53 | 6 / 18 | 0.054 / 0.023 |
| Qwen3.5-4B SOM | 21.9 | 47.1 | +25.2 [+22.1, +28.2] | 23 / 268 | <0.001 | 0.38 | 10 / 20 | 0.055 / 0.018 |
| Qwen3.5-4B J-space | 73.8 | 71.8 | -2.0 [-3.8, -0.1] | 50 / 31 | 0.045 | 0.79 | 48 / 54 | 0.019 / 0.017 |
| Qwen3.5-4B ARA | 18.9 | 2.2 | -16.8 [-19.4, -14.3] | 172 / 9 | <0.001 | 0.08 | 6 / 28 | 0.064 / 0.069 |

- All 10 cells differ between seeds after Holm; |s2 - s1| median 9.4 pp, mean 10.8 pp, max 25.2 pp (Qwen3.5-4B SOM);
  item-level kappa between seeds 0.08-0.79 (median 0.46). The direction of the seed difference is not consistent
  (6 lower, 4 higher with seed 2).
- **Generation effects by seed** (`final_seed_generation.csv`; new - old, pp): DiM +34.1 (s1) / +44.3 (s2); MPOA +4.2 /
  +30.1; SOM +9.7 / +29.6; J-space +21.2 / +10.6; **ARA +14.2 / +0.0 [-1.2, +1.2], p = 1.0**. Over the four old/new seed
  combinations the effect ranges over 10.6 (J-space) to 30.5 pp (SOM); ARA ranges from -2.6 [-4.2, -1.0] to +16.8. With
  seed 2, H1 fails in the small pair (ARA). Seed-averaged: DiM +39.2, MPOA +17.2, SOM +19.7, J-space +15.9, ARA +7.1
  [+5.7, +8.5].
- **Are method differences larger than seed differences?** Mostly yes, at the level of the method ranking
  (`final_seed_method_vs_seed.csv`): one-way ANOVA of the 10 cell rates per model on method (seeds as replicates)
  Qwen2.5-7B F(4,5) = 19.46, p = 0.003, eta-squared (method) 0.94; Qwen3.5-4B F(4,5) = 9.58, p = 0.015, eta-squared
  0.88 (method variance-component share 0.90 / 0.81; columns `eta_sq_method`, `var_component_share_method`; the
  0.95 / 0.91 in `ms_ratio_method` is MS_method / (MS_method + MS_within), a ratio of mean squares, not a share of
  variance). Range of seed-averaged method means 53.4 pp and 62.2 pp vs median seed difference 8.6 pp and 15.7 pp. Pairwise
  method orderings with the same sign under both seeds: 8/10 and 9/10; robust to all four seed combinations 5/10 and
  8/10. But the **generation effects of single methods are not robust to the seed**: the old/new differences that H1
  and H2 rest on (4-34 pp) are of the same size as the seed differences (2-25 pp).

### (d) Dev selection vs held-out refusal, all cells with a dev point

Sources: `final_dev_vs_heldout.csv`, `final_dev_vs_heldout_within_model.csv` (also `secondary_correlations.csv`).
x = dev keyword refusals of the selected trial (of 100); y = held-out refusal on the 972 harmful items.

| Cell set | n | Spearman vs WildGuard [95% CI] | vs keyword 512 | vs Qwen3Guard |
|---|---|---|---|---|
| All with a dev point (originals + primary + gamma-MPOA + seed 2) | 66 | 0.83 [0.73, 0.89] | 0.92 [0.87, 0.95] | 0.83 [0.73, 0.89] |
| All modified (primary + gamma-MPOA + seed 2) | 56 | 0.74 [0.61, 0.83] | 0.88 [0.80, 0.92] | 0.73 [0.60, 0.82] |
| Primary, seed 1 | 42 | 0.77 [0.61, 0.87] | 0.90 [0.81, 0.95] | 0.75 [0.59, 0.85] |
| Primary, main study | 32 | 0.84 [0.67, 0.93] | 0.92 [0.79, 0.97] | 0.81 [0.66, 0.92] |
| Primary, small pair | 10 | 0.56 [-0.08, 0.81], p = 0.093 | 0.77 [0.35, 0.94] | 0.35 [-0.36, 0.79] |
| Primary, older models | 21 | 0.85 [0.55, 0.97] | 0.98 [0.92, 0.99] | 0.83 [0.56, 0.95] |
| Primary, newer models | 21 | 0.73 [0.47, 0.88] | 0.89 [0.72, 0.96] | 0.73 [0.45, 0.87] |
| Seed 2 | 10 | 0.65 [0.02, 0.97], p = 0.043 | 0.69 [0.15, 0.93] | 0.71 [0.10, 0.95] |
| Small pair, both seeds | 20 | 0.59 [0.19, 0.80] | 0.75 [0.48, 0.88] | 0.55 [0.14, 0.80] |

- Per family (primary, n = 8 each): Qwen 0.83, Gemma 0.93, OLMo 0.69 (p = 0.058), Mistral 0.76. gamma-MPOA (n = 4):
  0.80, p = 0.20.
- Within model (4-5 methods per model x seed): the best dev method is also the best held-out (WildGuard) method in 7 of
  12 model x seed sets, counting one dev tie broken as the selection rule breaks ties (lower dev KL;
  `scripts/select_trial.py`): Qwen3.5-4B seed 1, MPOA and ARA both 6/100 dev refusals, MPOA at KL 0.054 before ARA at
  0.064, and MPOA is also the held-out best (18.0 vs 18.9%). Requiring a unique dev best gives 6 of 12
  (`final_dev_vs_heldout_within_model.csv`, columns `best_dev_kl_tiebreak_is_best_wg`, `best_dev_unique_is_best_wg`).
  Median within-model Spearman 0.85. Misses: Mistral-Small-2409 (dev ARA, held-out SOM),
  Mistral-Small-3.2 (ARA vs MPOA), OLMo-2 (DiM vs MPOA; within-model rho -0.4), Qwen2.5-32B (ARA vs MPOA), Qwen3.5-4B
  seed 2 (MPOA vs ARA).
- Transfer is weaker on newer than older models (0.73 vs 0.85) and weakest in the small pair; dev keyword transfers
  better to held-out keyword (0.90) than to either LLM judge (0.75-0.77). Example of a large dev / held-out mismatch:
  Qwen3.5-4B ARA seed 2 had 28/100 dev refusals but 2.2% held-out WildGuard refusal (seed 1: 6/100 and 18.9%;
  `final_seed_cells.csv`). Cells share models and the KL rule, so p values are optimistic.

### (e) Search convergence (D5)

Sources: `final_search_runs.csv`, `final_search_summary.csv`, `final_search_tests.csv` (dev set, `trials.csv`;
"first reach" = first trial at which the best feasible (KL <= 0.1) keyword count equals the selected count; "late" =
first reach after trial 75). Definitions match `fig7_search_convergence`.

- 42 core seed-1 runs: 8 late (19%), median first reach trial 48. Late runs: Qwen2.5-32B MPOA (79), Gemma-2 MPOA (99),
  Gemma-2 SOM (96), Gemma-4 MPOA (78), OLMo-3.1 SOM (89), Mistral-2409 ARA (91), Mistral-3.2 ARA (86), Mistral-3.2 SOM
  (95); gains after trial 75 of 1-14 refusals (mean 1.3 over all 42).
- **No clustering on newer models:** 4/21 older vs 4/21 newer runs late (Fisher p = 1.0); first-reach trial new vs old
  over 21 matched pair x method runs: newer later in 10, earlier in 11 (Wilcoxon p = 0.945); gain after trial 50
  larger on the newer model in 10, smaller in 6, tie 5 (p = 0.551).
- By method: DiM 0/10, MPOA 3/10, SOM 3/10, ARA 2/10, J-space 0/2 (chi2 p = 0.356, small counts). By family: Gemma 3/8,
  Mistral 3/8, Qwen 1/8, OLMo 1/8, small pair 0/10.
- The failed Gemma-4 DiM run reached its selected 98/100 at trial 40 and did not improve over the remaining 60 trials;
  within KL <= 0.1 its failure is not a search-budget effect. Seed 2 (small pair): 1/10 late, median first reach 18.5
  (seed 1, same cells: 54.5). gamma-MPOA: 1/4 late (OLMo-2, trial 87, gain 1).

### (f) Keyword-driven selection bias (Atlas D6)

Question: does minimising dev keyword refusals pick stronger edits (higher KL, divergence or capability cost) on newer
models? Sources: `final_selection_bias_pairs.csv` (21 matched pair x method, seed 1), `final_selection_bias_cells.csv`,
`final_selection_bias_tests.csv`. Excess KL = selected KL minus the lowest KL among feasible trials within 5 dev
refusals of the selected count (KL spent on the last few keyword refusals).

| Measure (new vs old, paired over 21 pair x method; main study n = 16 in brackets) | Median old | Median new | new > old | Wilcoxon p |
|---|---|---|---|---|
| Selected dev KL | 0.066 [0.064] | 0.069 [0.084] | 12/21 [11/16] | 0.243 [0.093] |
| Excess KL (within 5 refusals) | 0.0034 [0.0068] | 0.0031 [0.0077] | 11/21 [8/16] | 0.494 [0.331] |
| Teacher-forced divergence | 0.031 [0.046] | 0.035 [0.032] | 11/21 [7/16] | 0.838 [0.632] |
| Capability change vs original, pp | -0.23 [-0.44] | -0.24 [-0.23] | 8/21 [8/16] | 0.946 [0.562] |
| Held-out keyword-yes share among WildGuard-compliant harmful answers | 0.274 [0.280] | 0.3125 [0.322] | **16/21 [13/16]** | **0.014 [0.0027]** |
| Held-out keyword - WildGuard gap | 0.097 [0.085] | 0.056 [0.102] | 11/21 [11/16] | 0.946 [0.159] |

- Selected trial at the KL cap (KL > 0.09): 3 older vs 5 newer of 21 (discordant pairs: 2 older-only vs 4 newer-only,
  sign test p = 0.688; all eight are main-study cells).
- Originals: keyword-yes share among WildGuard-compliant answers does not differ by generation (median 0.232 vs 0.241,
  5 pairs, p = 0.81), so the difference arises after ablation.
- Across the 42 primary cells the keyword-yes share is unrelated to selected KL (rho -0.12 [-0.38, +0.16]), excess KL
  (-0.08) and capability change (+0.11), and negatively related to teacher-forced divergence (-0.53 [-0.73, -0.27],
  p < 0.001). Selected KL is unrelated to capability change (+0.08) and to divergence (+0.08).
- **Verdict:** the first link of D6 holds -- after ablation, the newer models' compliant answers more often contain
  Heretic's keyword markers, so the keyword objective penalises them more -- but the second link does not: the
  selection did not pick measurably stronger edits on newer models (KL, excess KL, divergence, capability: all
  p >= 0.09). The main-study selected KL leans higher on newer models (0.064 vs 0.084, 11/16, p = 0.093). The sharper
  check proposed in atlas_sweep D6 (re-selecting on the Pareto front with a WildGuard objective) needs judge labels on
  non-selected trials, which do not exist.

### (g) Pooled H1/H2 robustness

Source: `final_pooled_robustness.csv` (GEE of `h1h2.py` re-fitted; numbers quoted in sections 1, 2 and 4a). The full
model reproduces `h1h2_models.csv` (direction +0.62, optimisation +0.12, interaction -0.50).

---------------------------------------------------------------------------------------------------------------------

## 5. Cautions for Results and Discussion (atlas_sweep D1-D7; still open)

- **Keyword vs judge (D1).** Write "keyword matching and WildGuard diverge", not "keyword over-counts": no human audit
  exists; HarmBench is not an independent tiebreaker; agreement among LLM judges with correlated errors is not
  confirmation. The gap is driven by capped (truncated) answers; on complete answers it is null (main) or negative (small).
- **Gemma (D2)** and the **failed Gemma-4 DiM edit**: report per pair; "the newer model of the pair", not "newer
  generations". The pooled H2 result depends on Gemma and OLMo (section 2).
- **Seeds (D3).** Seed differences (2-25 pp, all 10 significant) are as large as many generation effects; small-pair H1
  and H2 change with the seed. Main-study cells have one seed only.
- **Budget (D5).** 19% of runs still improved after trial 75; not related to generation. Community checkpoints show
  that lower refusal is reachable on Qwen3.8 and OLMo-3.1 at larger divergence.
- **KL cap is not a capability guard (D4).** Gemma-4 gamma-MPOA lost 21.4 pp of capability at KL 0.061.
- **Length / thinking (D7).** 512-token cap, 256-token keyword and incomplete share go together; non-thinking mode only.
- Not done in this pass: no paper text was edited, so no Atlas BibTeX entries were merged into `paper/references.bib`;
  the human-labelled audit sample is still missing; figures were not re-rendered (only their data CSVs refreshed).
