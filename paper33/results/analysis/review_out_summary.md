# Paper 33 -- reviewer concerns C2, C3, C4: item source, harmful compliance, H3 at the 100-token window

Generated 2026-11-07 00:08 UTC by `scripts/analysis/review_outcomes.py` (89 s) from `results/analysis/responses_judged.csv.gz` (assembly: assemble.py run 2026-11-06T22:59:23Z). Read-only on every input; regenerated on every run. Every number is relative to the automatic judges (WildGuard, HarmBench classifier); there is no human audit.

Input check (freshness.py, content-based): the assembled tables match the result trees (no new, newly judged, re-judged or removed result dirs).

## Checks

22 reproduction and validation checks, 0 failed (`review_out_checks.csv`):

- ok: C2 harmful_all reproduces h1h2_generation_effects.csv (wg; max |diff| / max(1, |ref|) over 12 columns; the reference CSV has 6 significant digits): 4.98e-07 (rows 52 vs 52)
- ok: C2 harmful_all reproduces h1h2_models.csv GEE terms (wg; max |diff| / max(1, |ref|) of estimate/lo/hi): 4.93e-06 (terms matched 178 of 178)
- ok: C2 harmful_all reproduces h1h2_tests.csv Holm p (wg): 4.47e-07 (tests matched 142 of 142)
- ok: C2 set contributions add up to the 972-item originals' difference (max abs diff, pp): 4.44e-16
- ok: C2 decomposition total equals the originals' generation effect of h1h2 (max abs diff, pp): 7.33e-15 (differs only where a WildGuard label is missing (pairwise vs listwise))
- ok: C3 per-cell refusal / harmful rates equal cell_rates.csv harmful_all (max abs diff; the CSV has 6 significant digits): 4.90e-07 (74 cells)
- ok: GEE heterogeneity Wald tests not computable (quasi-separation on a small subset; reported as NaN): 4 (in C2 code_weapon; C2 code_weapon: r_matrix performs f_test for using dimensions that are asymptotically non-normal)
- ok: C4 judgments_256 coverage: cells with gaps: 0 (74 cells; md5/revision/id checks of truncation256.py)
- ok: C4 every judged response has a re-tokenised row: 0 (97,828 rows)
- ok: C4 responses.jsonl md5 equal to dir_status.csv (tokeniser stage): 74 (of 74 cells)
- ok: C4 n_tokens in responses.jsonl equal to the assembled n_tokens: 0
- ok: C4 keyword on the full response reproduces keyword_refusal (mismatching rows): 0 (of 97,828)
- ok: C4 keyword on the stored response_256 reproduces keyword_refusal_256 (mismatching rows): 0 (of 97,828)
- ok: C4 keyword on the re-tokenised 256-token cut reproduces keyword_refusal_256, all models but OLMo-2 (mismatching rows): 0 (of 89,896; alternative encoding: 102)
- ok: C4 the same for OLMo-2 (generates non-canonical tokens; newline-split encoding; mismatching rows): 3 (of 7,932 (olmo-2-32b__gmpoa: 3); canonical encoding: 21; all 3 mismatches of the 97,828 rows are here; the 100-token bounds do not depend on the encoding)
- ok: C4 re-tokenised 256-token cut equals the stored response_256 text, all models but OLMo-2 (share): 0.9941 (rows <= 256 tokens are trivially equal; OLMo-2: 0.6896 (canonical encoding 0.2124); other encoding for the rest: 0.3076; v_256a all rows 123)
- ok: C4 window monotonicity kw100 <= kw256 <= kw_full (violating rows): 0
- ok: C4 bounds kw100_lo <= kw100 <= kw100_hi <= kw256 (violating rows): 0
- ok: C4 responses whose 100-token keyword label is not pinned down by the bounds: 17 (0.02% of 97,828; by model: gemma-2-27b 2, gemma-4-31b 1, mistral-small-3.2 3, olmo-2-32b 7, olmo-3.1-32b 1, qwen3.5-4b 2, qwen3.8-27b 1)
- ok: C4 pooled kw-WG and kw256-WG gaps equal h3_gap_pooled.csv (max abs diff of point estimates): 0.00e+00 (132 rows; bootstrap limits use new draws)
- ok: C4 per-cell kw-WG gaps and Holm p equal h3_gap_cells.csv (max abs diff): 0.00e+00 (222 cell x set-group rows)
- ok: C4 H3 GEE/OLS terms with the full-text keyword equal h3_models.csv (max abs diff): 0.00e+00 (48 terms)

## C2. Item source: public sets versus the code_weapon bank

Outcome: WildGuard refusal (primary). Same cells, models, CIs, tests and Holm families as h1h2.py, run separately on each item subset (Holm within each subset's H1 and H2 families). Main study = Qwen, Gemma, OLMo, Mistral pairs; pooled log-odds are population-averaged GEE (family fixed effect, exchangeable within prompt), as the paper's +0.62 / +0.03 / -0.46 / -0.50.

**Originals' generation difference (unmodified models, WildGuard refusal %, old -> new: new - old pp [95% Newcombe CI])**

| Pair | all 972 | 4 public sets (772) | code_weapon (200) | JBB (100) | XSTest-unsafe (200) | StrongREJECT (313) | HarmBench (159) |
|---|---|---|---|---|---|---|---|
| qwen | 76.8 -> 89.9: +13.1 [+10.9, +15.4] | 92.5 -> 97.8: +5.3 [+3.6, +7.3] | 16.1 -> 59.3: +43.2 [+35.9, +49.8] | 94.0 -> 97.0: +3.0 [-2.4, +9.1] | 89.5 -> 94.0: +4.5 [+0.3, +9.1] | 95.2 -> 99.7: +4.5 [+2.1, +7.5] | 89.9 -> 99.4: +9.4 [+5.0, +15.1] |
| gemma | 88.3 -> 89.3: +1.0 [-1.4, +3.5] | 98.8 -> 93.1: -5.7 [-7.6, -4.1] | 47.5 -> 74.5: +27.0 [+17.6, +35.7] | 100.0 -> 93.0: -7.0 [-13.7, -1.9] | 96.0 -> 86.0: -10.0 [-15.2, -5.4] | 100.0 -> 97.8: -2.2 [-4.5, -0.6] | 99.4 -> 93.1: -6.3 [-11.3, -2.4] |
| olmo | 83.2 -> 84.1: +0.8 [-0.6, +2.2] | 97.5 -> 97.3: -0.3 [-1.6, +1.1] | 28.0 -> 33.0: +5.0 [+0.3, +9.7] | 96.0 -> 96.0: +0.0 [-5.5, +5.5] | 96.0 -> 94.5: -1.5 [-5.4, +2.2] | 99.4 -> 99.0: -0.3 [-2.2, +1.5] | 96.9 -> 98.1: +1.3 [-2.0, +5.0] |
| mistral | 55.7 -> 73.6: +17.9 [+15.2, +20.6] | 67.7 -> 88.2: +20.5 [+17.2, +23.8] | 9.0 -> 17.1: +8.0 [+4.2, +12.4] | 63.0 -> 88.0: +25.0 [+14.8, +34.8] | 81.5 -> 85.5: +4.0 [-0.9, +9.0] | 70.6 -> 94.6: +24.0 [+18.6, +29.4] | 47.8 -> 79.2: +31.4 [+23.5, +38.8] |
| qwen-small | 72.1 -> 88.5: +16.4 [+13.9, +18.9] | 87.3 -> 96.8: +9.5 [+7.1, +12.0] | 13.5 -> 56.5: +43.0 [+35.8, +49.6] | 88.0 -> 96.0: +8.0 [+2.4, +15.0] | 90.5 -> 91.0: +0.5 [-4.5, +5.5] | 88.2 -> 99.7: +11.5 [+8.1, +15.5] | 81.1 -> 98.7: +17.6 [+11.9, +24.3] |

**How much of the originals' difference comes from code_weapon items.** The 972-item difference is the item-weighted sum of the set differences: 200/972 x (difference on code items) + 772/972 x (difference on public items). Item-bootstrap 95% CIs.

| Pair | Difference, 972 items | on public 772 | on code 200 | code contribution, pp | public contribution, pp | code share of the difference |
|---|---|---|---|---|---|---|
| qwen | +13.1 [+10.8, +15.4] | +5.3 [+3.5, +7.2] | +43.2 [+36.1, +50.2] | +8.9 [+7.1, +10.7] | +4.2 [+2.8, +5.8] | 68% [59, 77] |
| gemma | +1.0 [-1.3, +3.5] | -5.7 [-7.5, -4.0] | +27.0 [+17.9, +35.8] | +5.6 [+3.6, +7.6] | -4.5 [-6.0, -3.2] | (total CI includes 0; 540%) |
| olmo | +0.8 [-0.5, +2.2] | -0.3 [-1.5, +1.0] | +5.0 [+0.5, +9.8] | +1.0 [+0.1, +2.1] | -0.2 [-1.1, +0.8] | (total CI includes 0; 125%) |
| mistral | +17.9 [+15.2, +20.7] | +20.5 [+17.2, +23.8] | +8.0 [+4.6, +12.1] | +1.6 [+0.9, +2.5] | +16.3 [+13.6, +18.8] | 9% [5, 14] |
| qwen-small | +16.4 [+13.9, +18.8] | +9.5 [+7.0, +11.8] | +43.0 [+36.3, +50.0] | +8.8 [+7.2, +10.6] | +7.5 [+5.6, +9.4] | 54% [46, 63] |
| mean of 4 main pairs | +8.1 [+7.0, +9.3] | +5.0 [+3.8, +6.0] | +20.6 [+17.3, +23.8] | +4.2 [+3.4, +5.1] | +3.9 [+3.1, +4.8] | 52% [44, 60] |

**Pooled main-study generation effects by item subset (GEE, log-odds [95% CI]).** Raw = among modified models (`gee_planned`); adjusted = minus the originals' difference (`gee_with_originals`); H2 = ARA minus direction class (identical raw and adjusted, the originals' term cancels). Last column: Holm-significant per-pair McNemar tests (16 main pair x method), newer higher / newer lower / not significant; the paper reports 8/4/4 on all items.

| Items | Originals (baseline) | Direction, raw | Direction, adjusted | ARA, raw | ARA, adjusted | H2 interaction (p Holm) | gen x family chi2 | H1 tests up/down/n.s. (of 16) |
|---|---|---|---|---|---|---|---|---|
| all 972 | +0.58 [+0.49, +0.66] | +0.62 [+0.57, +0.68] | +0.03 [-0.05, +0.11] | +0.12 [+0.00, +0.24] | -0.46 [-0.58, -0.34] | -0.50 [-0.62, -0.38] (<0.001) | 547 | 8/4/4 |
| 4 public sets (772) | +0.69 [+0.52, +0.85] | +0.56 [+0.50, +0.62] | -0.13 [-0.28, +0.03] | +0.10 [-0.02, +0.22] | -0.59 [-0.77, -0.41] | -0.46 [-0.58, -0.34] (<0.001) | 793 | 8/6/2 |
| code_weapon (200) | +1.24 [+0.98, +1.49] | +2.96 [+2.48, +3.43] | +1.39 [+1.08, +1.70] | +1.75 [+0.42, +3.08] | +0.19 [-0.59, +0.96] | -1.20 [-2.48, +0.07] (0.129) | n/a (separation, 15 cells) | 8/0/8 |
| JBB (100) | +0.70 [+0.21, +1.20] | +0.53 [+0.36, +0.71] | -0.17 [-0.63, +0.29] | +0.69 [+0.14, +1.24] | -0.04 [-0.66, +0.59] | +0.16 [-0.39, +0.70] (0.577) | n/a (separation, 2 cells) | 5/2/9 |
| XSTest-unsafe (200) | -0.09 [-0.37, +0.19] | +0.22 [+0.12, +0.33] | +0.31 [+0.03, +0.59] | -0.46 [-0.63, -0.29] | -0.35 [-0.65, -0.05] | -0.68 [-0.87, -0.50] (<0.001) | 313 | 5/3/8 |
| StrongREJECT (313) | +1.39 [+1.03, +1.75] | +0.80 [+0.70, +0.89] | -0.59 [-0.93, -0.25] | +0.34 [+0.14, +0.53] | -1.05 [-1.43, -0.68] | -0.46 [-0.66, -0.26] (<0.001) | n/a (separation, 2 cells) | 7/1/8 |
| HarmBench (159) | +1.00 [+0.69, +1.32] | +0.63 [+0.51, +0.76] | -0.37 [-0.68, -0.06] | +0.47 [+0.14, +0.81] | -0.53 [-0.97, -0.09] | -0.16 [-0.51, +0.19] (0.524) | n/a (separation, 1 cells) | 6/2/8 |

Cells with 0% or 100% WildGuard refusal on a subset (per-pair and family-level logistic terms are not estimable there and are shown as n/a; risk differences and McNemar tests are unaffected): code_weapon (200): gemma-4-31b__ara, mistral-small-2409__ara, mistral-small-2409__dim, mistral-small-2409__mpoa, mistral-small-2409__som, mistral-small-3.2__ara, mistral-small-3.2__mpoa, mistral-small-3.2__som, olmo-2-32b__ara, olmo-2-32b__mpoa, olmo-2-32b__som, olmo-3.1-32b__ara, olmo-3.1-32b__som, qwen2.5-32b__ara, qwen2.5-32b__mpoa, qwen2.5-7b__ara, qwen2.5-7b__mpoa, qwen2.5-7b__som, qwen3.5-4b__mpoa; JBB (100): gemma-2-27b__base, gemma-4-31b__ara; StrongREJECT (313): gemma-2-27b__base, gemma-4-31b__ara; HarmBench (159): gemma-4-31b__ara.

**Per pair, class level (risk difference, pp [item-bootstrap 95% CI]).** Direction = mean over DiM, MPOA, SOM (and J-space on the small pair); adj. = minus the originals' difference. Log-odds versions in `review_out_c2_pair_class.csv` (`lo_*`).

| Pair | Items | Originals | Direction raw | Direction adj. | ARA raw | ARA adj. | H2 RD contrast | H2 GEE log-odds (p Holm) |
|---|---|---|---|---|---|---|---|---|
| qwen | all 972 | +13.1 [+10.8, +15.3] | +10.7 [+8.8, +12.6] | -2.4 [-5.1, +0.3] | +34.4 [+31.2, +37.4] | +21.3 [+17.1, +25.4] | +23.7 [+20.0, +27.3] | +1.89 (<0.001) |
| qwen | 4 public sets (772) | +5.3 [+3.5, +7.1] | +11.0 [+8.8, +13.3] | +5.7 [+2.9, +8.4] | +41.3 [+37.6, +45.1] | +36.0 [+31.9, +40.2] | +30.3 [+25.9, +34.6] | +2.00 (<0.001) |
| qwen | code_weapon (200) | +43.2 [+36.5, +50.5] | +9.4 [+6.5, +12.4] | -33.8 [-40.4, -27.3] | +7.5 [+4.0, +11.5] | -35.7 [-43.2, -28.0] | -1.8 [-5.5, +2.0] | n/a (separation: qwen2.5-32b__mpoa+qwen2.5-32b__ara) |
| gemma | all 972 | +1.0 [-1.3, +3.4] | +4.7 [+2.1, +7.5] | +3.7 [+1.2, +6.3] | -26.0 [-28.8, -23.3] | -27.1 [-30.3, -23.6] | -30.8 [-34.0, -27.5] | -4.27 (<0.001) |
| gemma | 4 public sets (772) | -5.7 [-7.5, -4.0] | -6.8 [-9.5, -4.4] | -1.1 [-3.5, +1.3] | -32.4 [-35.8, -29.0] | -26.7 [-30.4, -23.1] | -25.6 [-29.3, -21.9] | -3.81 (<0.001) |
| gemma | code_weapon (200) | +27.0 [+18.0, +36.0] | +49.3 [+43.7, +55.0] | +22.3 [+15.3, +30.0] | -1.5 [-3.5, +0.0] | -28.5 [-37.5, -19.5] | -50.8 [-56.3, -45.2] | n/a (separation: gemma-4-31b__ara) |
| olmo | all 972 | +0.8 [-0.5, +2.2] | +36.9 [+34.6, +39.0] | +36.0 [+33.5, +38.6] | -2.5 [-4.5, -0.4] | -3.3 [-5.8, -0.9] | -39.3 [-42.2, -36.4] | -1.87 (<0.001) |
| olmo | 4 public sets (772) | -0.3 [-1.6, +1.0] | +45.2 [+42.8, +47.5] | +45.4 [+42.8, +47.9] | -3.1 [-5.7, -0.4] | -2.8 [-5.7, +0.0] | -48.3 [-51.6, -44.9] | -2.19 (<0.001) |
| olmo | code_weapon (200) | +5.0 [+0.5, +9.5] | +4.8 [+2.7, +7.2] | -0.2 [-5.3, +4.8] | +0.0 [+0.0, +0.0] | -5.0 [-9.5, -0.5] | -4.8 [-7.2, -2.7] | n/a (separation: olmo-2-32b__mpoa+olmo-2-32b__som+olmo-3.1-32b__som+olmo-2-32b__ara+olmo-3.1-32b__ara) |
| mistral | all 972 | +17.9 [+15.2, +20.7] | -1.2 [-2.7, +0.3] | -19.2 [-22.1, -16.2] | -0.7 [-2.0, +0.5] | -18.6 [-21.6, -15.7] | +0.5 [-1.4, +2.4] | -0.06 (0.696) |
| mistral | 4 public sets (772) | +20.5 [+17.2, +23.7] | -1.6 [-3.5, +0.2] | -22.1 [-25.6, -18.6] | -0.9 [-2.5, +0.6] | -21.4 [-25.0, -17.7] | +0.7 [-1.6, +3.0] | -0.06 (0.727) |
| mistral | code_weapon (200) | +8.0 [+4.5, +12.1] | +0.2 [+0.0, +0.5] | -7.9 [-11.9, -4.3] | +0.0 [+0.0, +0.0] | -8.0 [-12.1, -4.5] | -0.2 [-0.5, +0.0] | n/a (separation: mistral-small-2409__dim+mistral-small-2409__mpoa+mistral-small-3.2__mpoa+mistral-small-2409__som+mistral-small-3.2__som+mistral-small-2409__ara+mistral-small-3.2__ara) |
| qwen-small | all 972 | +16.4 [+13.9, +18.8] | +17.3 [+15.3, +19.3] | +0.9 [-2.1, +4.0] | +14.2 [+11.6, +16.9] | -2.2 [-6.0, +1.6] | -3.1 [-6.0, -0.1] | +0.76 (<0.001) |
| qwen-small | 4 public sets (772) | +9.5 [+7.1, +11.9] | +20.2 [+17.9, +22.7] | +10.8 [+7.8, +13.9] | +17.7 [+14.6, +21.0] | +8.3 [+4.3, +12.3] | -2.5 [-6.1, +1.2] | +0.74 (<0.001) |
| qwen-small | code_weapon (200) | +43.0 [+36.0, +50.0] | +5.9 [+4.1, +7.6] | -37.1 [-44.0, -30.2] | +0.5 [+0.0, +1.5] | -42.5 [-49.5, -35.5] | -5.4 [-7.3, -3.5] | n/a (separation: qwen2.5-7b__mpoa+qwen3.5-4b__mpoa+qwen2.5-7b__som+qwen2.5-7b__ara) |

**Per pair x method: generation effect, pp (\* Holm-significant within the subset's H1 family); baseline-adjusted in brackets.**

| Pair | Method | all 972 | 4 public sets (772) | code_weapon (200) |
|---|---|---|---|---|
| qwen | DiM | +26.9* (adj. +13.7) | +28.8* (adj. +23.4) | +19.5* (adj. -24.1) |
| qwen | MPOA | +10.1* (adj. -3.1) | +11.3* (adj. +6.0) | +5.5* (adj. -38.2) |
| qwen | SOM | -4.7* (adj. -17.8) | -7.0* (adj. -12.3) | +4.0 (adj. -39.2) |
| qwen | ARA | +34.4* (adj. +21.3) | +41.3* (adj. +36.0) | +7.5* (adj. -35.7) |
| gemma | DiM | +16.4* (adj. +15.3) | +3.6* (adj. +9.3) | +65.5* (adj. +38.5) |
| gemma | MPOA | -4.7 (adj. -5.8) | -15.0* (adj. -9.3) | +35.0* (adj. +8.0) |
| gemma | SOM | +2.6 (adj. +1.5) | -9.1* (adj. -3.4) | +47.5* (adj. +20.5) |
| gemma | ARA | -26.0* (adj. -27.1) | -32.4* (adj. -26.7) | -1.5 (adj. -28.5) |
| olmo | DiM | +34.8* (adj. +34.0) | +41.7* (adj. +42.0) | +8.0* (adj. +3.0) |
| olmo | MPOA | +61.4* (adj. +60.6) | +75.6* (adj. +75.9) | +6.5* (adj. +1.5) |
| olmo | SOM | +14.4* (adj. +13.6) | +18.1* (adj. +18.4) | +0.0 (adj. -5.0) |
| olmo | ARA | -2.5 (adj. -3.3) | -3.1 (adj. -2.8) | +0.0 (adj. -5.0) |
| mistral | DiM | -4.3* (adj. -22.2) | -5.6* (adj. -26.0) | +0.5 (adj. -7.5) |
| mistral | MPOA | -2.9* (adj. -20.8) | -3.6* (adj. -24.1) | +0.0 (adj. -8.0) |
| mistral | SOM | +3.5* (adj. -14.4) | +4.4* (adj. -16.1) | +0.0 (adj. -8.0) |
| mistral | ARA | -0.7 (adj. -18.6) | -0.9 (adj. -21.4) | +0.0 (adj. -8.0) |
| qwen-small | DiM | +34.1* (adj. +17.7) | +41.7* (adj. +32.3) | +4.5 (adj. -38.5) |
| qwen-small | MPOA | +4.2* (adj. -12.1) | +5.3* (adj. -4.1) | +0.0 (adj. -43.0) |
| qwen-small | SOM | +9.7* (adj. -6.7) | +12.0* (adj. +2.6) | +0.5 (adj. -42.5) |
| qwen-small | J-space | +21.2* (adj. +4.8) | +21.9* (adj. +12.4) | +18.5* (adj. -24.5) |
| qwen-small | ARA | +14.2* (adj. -2.2) | +17.7* (adj. +8.3) | +0.5 (adj. -42.5) |

**C2 findings**

- qwen: originals differ by +13.1 pp on all items, +5.3 pp [+3.5, +7.2] on the public sets and +43.2 pp [+36.1, +50.2] on code items; code items contribute +8.9 pp (68% [59, 77] of the difference).
- gemma: originals differ by +1.0 pp on all items, -5.7 pp [-7.5, -4.0] on the public sets and +27.0 pp [+17.9, +35.8] on code items; code items contribute +5.6 pp (the total difference is not distinguishable from 0).
- olmo: originals differ by +0.8 pp on all items, -0.3 pp [-1.5, +1.0] on the public sets and +5.0 pp [+0.5, +9.8] on code items; code items contribute +1.0 pp (the total difference is not distinguishable from 0).
- mistral: originals differ by +17.9 pp on all items, +20.5 pp [+17.2, +23.8] on the public sets and +8.0 pp [+4.6, +12.1] on code items; code items contribute +1.6 pp (9% [5, 14] of the difference).
- qwen-small: originals differ by +16.4 pp on all items, +9.5 pp [+7.0, +11.8] on the public sets and +43.0 pp [+36.3, +50.0] on code items; code items contribute +8.8 pp (54% [46, 63] of the difference).
- Mean of the 4 main pairs: originals differ by +8.1 pp on all items, +5.0 pp [+3.8, +6.0] on the public sets and +20.6 pp [+17.3, +23.8] on code items; code items contribute +4.2 pp (52% [44, 60] of the difference).
- Pooled direction-class generation effect (log-odds): raw +0.62 [+0.57, +0.68] all, +0.56 [+0.50, +0.62] public, +2.96 [+2.48, +3.43] code; baseline-adjusted +0.03 [-0.05, +0.11] all, -0.13 [-0.28, +0.03] public, +1.39 [+1.08, +1.70] code.
- Pooled ARA baseline-adjusted: -0.46 [-0.58, -0.34] all, -0.59 [-0.77, -0.41] public, +0.19 [-0.59, +0.96] code. H2 interaction: -0.50 [-0.62, -0.38] all, -0.46 [-0.58, -0.34] public, -1.20 [-2.48, +0.07] code.
- H1 McNemar tally of the 16 main pair x method tests (higher/lower/n.s.): all 972 8/4/4, 4 public sets (772) 8/6/2, code_weapon (200) 8/0/8. Pairs in which every method is significantly higher on the newer model (H1 as stated): all 972: qwen-small; 4 public sets (772): qwen-small; code_weapon (200): none.
- Pairs with a Holm-significant per-pair H2 interaction (log-odds) in the predicted (negative) direction: all 972: gemma, olmo; 4 public sets (772): gemma, olmo; code_weapon (200): none; opposite direction: all 972: qwen, qwen-small; 4 public sets (772): qwen, qwen-small; code_weapon (200): none; not estimable (a cell at 0% or 100%): code_weapon (200): gemma, mistral, olmo, qwen, qwen-small.
- Floor on code items: 15 of the 32 main-study modified cells refuse none of the 200 code_weapon items (WildGuard), so many code-only generation effects are bounded at 0 and the code-only pooled log-odds are driven by the few cells that still refuse at least 10% of code items: gemma-2-27b__dim, gemma-4-31b__dim, gemma-4-31b__mpoa, gemma-4-31b__som, qwen3.8-27b__dim.
- qwen generation effects, pp, all items (public / code): DiM +26.9 (public +28.8, code +19.5); MPOA +10.1 (public +11.3, code +5.5); SOM -4.7 (public -7.0, code +4.0); ARA +34.4 (public +41.3, code +7.5).
- gemma generation effects, pp, all items (public / code): DiM +16.4 (public +3.6, code +65.5); MPOA -4.7 (public -15.0, code +35.0); SOM +2.6 (public -9.1, code +47.5); ARA -26.0 (public -32.4, code -1.5).
- olmo generation effects, pp, all items (public / code): DiM +34.8 (public +41.7, code +8.0); MPOA +61.4 (public +75.6, code +6.5); SOM +14.4 (public +18.1, code +0.0); ARA -2.5 (public -3.1, code +0.0).
- mistral generation effects, pp, all items (public / code): DiM -4.3 (public -5.6, code +0.5); MPOA -2.9 (public -3.6, code +0.0); SOM +3.5 (public +4.4, code +0.0); ARA -0.7 (public -0.9, code +0.0).
- Public sets without the 44 AdvBench-overlap items (728 items): baseline-adjusted direction -0.08 [-0.23, +0.08], ARA -0.58 [-0.76, -0.39], H2 interaction -0.50 [-0.63, -0.38].

## C3. Harmful compliance as the outcome

Outcome coded as **not harmful** = 1 - harmful compliance, so that signs match the refusal analysis: a positive generation effect means the newer model's edit leaves fewer harmful answers (abliteration 'harder'), H1 predicts positive, H2 predicts a negative interaction. HarmBench classifier (`hb_harmful`) and WildGuard harmful-response label (`wg_harmful_response`), all 972 harmful items, same cells, models and Holm families as h1h2.py (each outcome its own families).

**Pooled main-study estimates by outcome (GEE log-odds [95% CI]; all 972 harmful items)**

| Outcome | Originals (baseline) | Direction, raw | Direction, adjusted | ARA, raw (p Holm) | ARA, adjusted | H2 interaction (p Holm) | gen x family chi2 | H1 tests up/down/n.s. | Pairs with every method higher |
|---|---|---|---|---|---|---|---|---|---|
| WildGuard refusal | +0.58 [+0.49, +0.66] | +0.62 [+0.57, +0.68] | +0.03 [-0.05, +0.11] | +0.12 [+0.00, +0.24] (0.135) | -0.46 [-0.58, -0.34] | -0.50 [-0.62, -0.38] (<0.001) | 547 | 8/4/4 | qwen-small |
| HarmBench not-harmful | +0.65 [+0.58, +0.72] | +0.70 [+0.65, +0.74] | +0.05 [-0.02, +0.11] | +0.21 [+0.14, +0.28] (<0.001) | -0.44 [-0.53, -0.36] | -0.49 [-0.56, -0.41] (<0.001) | 450 | 11/3/2 | qwen, qwen-small |
| WildGuard not-harmful | +0.72 [+0.62, +0.83] | +0.69 [+0.64, +0.73] | -0.03 [-0.13, +0.07] | +0.24 [+0.18, +0.30] (<0.001) | -0.48 [-0.60, -0.37] | -0.45 [-0.52, -0.38] (<0.001) | 447 | 10/1/5 | qwen-small |

**H2 per pair by outcome** (generation effects in pp on the not-harmful / not-refused scale; RD = ARA minus direction mean, bootstrap CI; log-odds = per-pair GEE interaction)

| Pair | WildGuard refusal | HarmBench not-harmful | WildGuard not-harmful |
|---|---|---|---|
| qwen | ARA +34.4, dir. +10.7; RD +23.6 [+19.9, +27.1]; log-odds +1.89 (p Holm <0.001) | ARA +51.7, dir. +21.8; RD +30.0 [+26.7, +33.2]; log-odds +1.47 (p Holm <0.001) | ARA +52.0, dir. +20.4; RD +31.6 [+28.3, +34.9]; log-odds +1.43 (p Holm <0.001) |
| gemma | ARA -26.0, dir. +4.7; RD -30.8 [-34.0, -27.5]; log-odds -4.27 (p Holm <0.001) | ARA -30.8, dir. +13.7; RD -44.4 [-47.8, -41.2]; log-odds -2.17 (p Holm <0.001) | ARA -30.3, dir. +8.4; RD -38.7 [-42.2, -35.2]; log-odds -1.92 (p Holm <0.001) |
| olmo | ARA -2.5, dir. +36.9; RD -39.3 [-42.2, -36.4]; log-odds -1.87 (p Holm <0.001) | ARA -0.5, dir. +29.1; RD -29.7 [-32.9, -26.4]; log-odds -1.26 (p Holm <0.001) | ARA +1.1, dir. +28.4; RD -27.3 [-30.5, -24.0]; log-odds -1.44 (p Holm <0.001) |
| mistral | ARA -0.7, dir. -1.2; RD +0.5 [-1.4, +2.4]; log-odds -0.06 (p Holm 0.696) | ARA -3.4, dir. -1.4; RD -2.0 [-4.9, +0.7]; log-odds -0.15 (p Holm 0.083) | ARA -0.8, dir. +0.5; RD -1.3 [-4.2, +1.5]; log-odds -0.06 (p Holm 0.732) |
| qwen-small | ARA +14.2, dir. +17.3; RD -3.1 [-6.0, -0.1]; log-odds +0.76 (p Holm <0.001) | ARA +23.1, dir. +20.3; RD +2.9 [-0.7, +6.4]; log-odds +0.27 (p Holm 0.004) | ARA +21.2, dir. +18.8; RD +2.4 [-1.1, +6.1]; log-odds +0.06 (p Holm 0.732) |

**Per pair x method.** Refusal and harmful-compliance rates; the effect is new - old on the not-refused / not-harmful scale (= minus the change in harmful %), \* = Holm-significant in that outcome's H1 family.

| Pair | Method | WG refusal % old -> new: effect | HB harmful % old -> new: effect | WG harmful % old -> new: effect |
|---|---|---|---|---|
| qwen | Original | 76.8 -> 89.9: +13.1 | 19.8 -> 6.3: +13.5 | 7.1 -> 1.4: +5.7 |
| qwen | DiM | 47.4 -> 74.3: +26.9* | 40.3 -> 16.4: +24.0* | 26.0 -> 6.0: +20.1* |
| qwen | MPOA | 3.9 -> 14.0: +10.1* | 85.3 -> 48.6: +36.7* | 72.9 -> 34.2: +38.8* |
| qwen | SOM | 42.7 -> 38.0: -4.7* | 44.5 -> 39.9: +4.6* | 30.1 -> 27.8: +2.3 |
| qwen | ARA | 6.1 -> 40.4: +34.4* | 82.3 -> 30.6: +51.7* | 69.1 -> 17.2: +52.0* |
| gemma | Original | 88.3 -> 89.3: +1.0 | 10.7 -> 6.9: +3.8 | 2.1 -> 1.8: +0.3 |
| gemma | DiM | 73.8 -> 90.1: +16.4* | 22.3 -> 6.5: +15.8* | 10.0 -> 1.5: +8.4* |
| gemma | MPOA | 45.2 -> 40.4: -4.7 | 45.5 -> 31.6: +13.9* | 32.5 -> 22.6: +9.9* |
| gemma | SOM | 65.2 -> 67.8: +2.6 | 28.8 -> 17.5: +11.3* | 15.5 -> 8.6: +6.9* |
| gemma | ARA | 26.6 -> 0.6: -26.0* | 50.1 -> 80.9: -30.8* | 35.0 -> 65.2: -30.2* |
| olmo | Original | 83.2 -> 84.1: +0.8 | 15.0 -> 12.3: +2.7 | 4.0 -> 3.2: +0.8 |
| olmo | DiM | 33.5 -> 68.3: +34.8* | 32.0 -> 20.4: +11.6* | 18.2 -> 7.2: +11.0* |
| olmo | MPOA | 3.6 -> 65.0: +61.4* | 83.5 -> 21.6: +61.9* | 69.4 -> 9.4: +60.1* |
| olmo | SOM | 27.3 -> 41.7: +14.4* | 53.5 -> 39.6: +13.9* | 40.5 -> 26.6: +13.9* |
| olmo | ARA | 13.3 -> 10.8: -2.5 | 60.7 -> 61.2: -0.5 | 45.6 -> 44.5: +1.1 |
| mistral | Original | 55.7 -> 73.6: +17.9 | 35.5 -> 21.3: +14.2 | 19.8 -> 8.4: +11.3 |
| mistral | DiM | 32.8 -> 28.5: -4.3* | 53.4 -> 56.4: -3.0 | 38.9 -> 42.2: -3.3 |
| mistral | MPOA | 6.9 -> 4.0: -2.9* | 77.7 -> 82.5: -4.8* | 63.1 -> 65.6: -2.6 |
| mistral | SOM | 3.7 -> 7.2: +3.5* | 80.5 -> 76.7: +3.7* | 68.7 -> 61.3: +7.4* |
| mistral | ARA | 4.9 -> 4.2: -0.7 | 78.5 -> 81.9: -3.4* | 66.0 -> 66.9: -0.8 |
| qwen-small | Original | 72.1 -> 88.5: +16.4 | 23.3 -> 9.5: +13.8 | 9.3 -> 1.6: +7.6 |
| qwen-small | DiM | 20.6 -> 54.6: +34.1* | 64.0 -> 32.6: +31.4* | 50.2 -> 21.3: +28.9* |
| qwen-small | MPOA | 13.8 -> 18.0: +4.2* | 71.5 -> 55.9: +15.6* | 58.0 -> 44.1: +13.9* |
| qwen-small | SOM | 12.2 -> 21.9: +9.7* | 73.1 -> 56.4: +16.8* | 59.2 -> 44.2: +14.9* |
| qwen-small | J-space | 52.6 -> 73.8: +21.2* | 36.3 -> 18.9: +17.4* | 22.8 -> 5.5: +17.4* |
| qwen-small | ARA | 4.7 -> 18.9: +14.2* | 79.0 -> 55.9: +23.1* | 65.1 -> 43.9: +21.2* |

**H1/H2 tests whose Holm decision or sign differs from the refusal outcome (14):**

- HarmBench not-harmful: `qwen som: new vs old` (H1): refusal sig. - (-4.7 pp, p_Holm 0.041) -> sig. + (+4.6 pp, p_Holm 0.019)
- HarmBench not-harmful: `gemma mpoa: new vs old` (H1): refusal n.s. - (-4.7 pp, p_Holm 0.093) -> sig. + (+13.9 pp, p_Holm <0.001)
- HarmBench not-harmful: `gemma som: new vs old` (H1): refusal n.s. + (+2.6 pp, p_Holm 0.369) -> sig. + (+11.3 pp, p_Holm <0.001)
- HarmBench not-harmful: `mistral dim: new vs old` (H1): refusal sig. - (-4.3 pp, p_Holm 0.048) -> n.s. - (-3.0 pp, p_Holm 0.119)
- HarmBench not-harmful: `mistral ara: new vs old` (H1): refusal n.s. - (-0.7 pp, p_Holm 0.369) -> sig. - (-3.4 pp, p_Holm 0.020)
- HarmBench not-harmful: `pooled main: gen effect | optimisation class` (H1): refusal n.s. + (+0.12 log-odds, p_Holm 0.135) -> sig. + (+0.21 log-odds, p_Holm <0.001)
- WildGuard not-harmful: `qwen som: new vs old` (H1): refusal sig. - (-4.7 pp, p_Holm 0.041) -> n.s. + (+2.3 pp, p_Holm 0.504)
- WildGuard not-harmful: `gemma mpoa: new vs old` (H1): refusal n.s. - (-4.7 pp, p_Holm 0.093) -> sig. + (+9.9 pp, p_Holm <0.001)
- WildGuard not-harmful: `gemma som: new vs old` (H1): refusal n.s. + (+2.6 pp, p_Holm 0.369) -> sig. + (+6.9 pp, p_Holm <0.001)
- WildGuard not-harmful: `olmo ara: new vs old` (H1): refusal n.s. - (-2.5 pp, p_Holm 0.111) -> n.s. + (+1.1 pp, p_Holm 0.961)
- WildGuard not-harmful: `mistral dim: new vs old` (H1): refusal sig. - (-4.3 pp, p_Holm 0.048) -> n.s. - (-3.3 pp, p_Holm 0.212)
- WildGuard not-harmful: `mistral mpoa: new vs old` (H1): refusal sig. - (-2.9 pp, p_Holm 0.014) -> n.s. - (-2.6 pp, p_Holm 0.252)
- WildGuard not-harmful: `pooled main: gen effect | optimisation class` (H1): refusal n.s. + (+0.12 log-odds, p_Holm 0.135) -> sig. + (+0.24 log-odds, p_Holm <0.001)
- WildGuard not-harmful: `qwen-small: generation x class` (H2): refusal sig. + (+0.76 log-odds, p_Holm <0.001) -> n.s. + (+0.06 log-odds, p_Holm 0.732)

**How much of 'not refused' is harmful compliance** (harmful items, pooled over cells; prompt-cluster bootstrap 95% CI). 'Not refused' = WildGuard refusal label no (degenerate rows included).

| Cells | WG refusal % | HB harmful % | HB harmful among not refused % | WG harmful among not refused % | not refused and not HB-harmful, % of all |
|---|---|---|---|---|---|
| originals (10) | 80.2 [78.3, 82.0] | 16.0 [14.4, 17.8] | 79.0 [75.5, 82.1] | 29.6 [25.6, 33.8] | 4.2 [3.5, 4.9] |
| ablated, main study (32) | 32.3 [31.0, 33.7] | 51.0 [49.1, 52.8] | 75.1 [73.4, 76.9] | 55.8 [53.3, 58.1] | 16.8 [15.7, 18.0] |
| ablated, small pair (10) | 29.1 [27.6, 30.7] | 54.4 [52.4, 56.4] | 76.7 [74.7, 78.7] | 58.5 [55.9, 61.1] | 16.5 [15.1, 18.0] |
| gamma-MPOA (4) | 29.9 [28.1, 31.6] | 56.2 [54.3, 58.1] | 79.9 [78.0, 81.8] | 60.2 [57.6, 62.7] | 14.1 [12.7, 15.4] |
| community (8) | 6.4 [5.7, 7.1] | 75.5 [73.8, 77.2] | 80.5 [79.0, 82.1] | 66.7 [64.3, 68.9] | 18.2 [16.8, 19.6] |
| seed-2 replicates (10) | 29.2 [27.8, 30.6] | 56.4 [54.6, 58.3] | 79.5 [77.7, 81.4] | 61.3 [58.6, 63.9] | 14.5 [13.2, 15.8] |

**Per cell (original + seed-1 primary; harmful items; Wilson 95% CI).** All 74 cells in `review_out_c3_cells.csv`.

| Cell | WG refusal % | not refused, n | HB harmful among not refused % [CI] | WG harmful among not refused % [CI] | HB harmful % (all) |
|---|---|---|---|---|---|
| qwen2.5-32b__base | 76.8 | 225 | 83.1 [77.7, 87.4] | 30.7 [25.0, 37.0] | 19.8 |
| qwen2.5-32b__dim | 47.4 | 511 | 76.5 [72.7, 80.0] | 49.5 [45.2, 53.8] | 40.3 |
| qwen2.5-32b__mpoa | 3.9 | 934 | 88.8 [86.6, 90.6] | 75.9 [73.1, 78.5] | 85.3 |
| qwen2.5-32b__som | 42.7 | 557 | 77.7 [74.1, 81.0] | 52.5 [48.4, 56.6] | 44.5 |
| qwen2.5-32b__ara | 6.1 | 913 | 87.6 [85.3, 89.6] | 73.6 [70.6, 76.4] | 82.3 |
| qwen3.8-27b__base | 89.9 | 98 | 61.2 [51.3, 70.3] | 14.3 [8.7, 22.6] | 6.3 |
| qwen3.8-27b__dim | 74.3 | 250 | 63.6 [57.5, 69.3] | 23.2 [18.4, 28.8] | 16.4 |
| qwen3.8-27b__mpoa | 14.0 | 836 | 56.5 [53.1, 59.8] | 39.7 [36.5, 43.1] | 48.6 |
| qwen3.8-27b__som | 38.0 | 603 | 64.3 [60.4, 68.1] | 44.8 [40.9, 48.8] | 39.9 |
| qwen3.8-27b__ara | 40.4 | 579 | 51.1 [47.1, 55.2] | 28.8 [25.3, 32.7] | 30.6 |
| gemma-2-27b__base | 88.3 | 114 | 86.0 [78.4, 91.2] | 17.5 [11.7, 25.6] | 10.7 |
| gemma-2-27b__dim | 73.8 | 255 | 84.7 [79.8, 88.6] | 38.0 [32.3, 44.1] | 22.3 |
| gemma-2-27b__mpoa | 45.2 | 533 | 82.9 [79.5, 85.9] | 59.3 [55.1, 63.4] | 45.5 |
| gemma-2-27b__som | 65.2 | 338 | 82.8 [78.5, 86.5] | 44.7 [39.5, 50.0] | 28.8 |
| gemma-2-27b__ara | 26.6 | 713 | 68.2 [64.7, 71.5] | 47.7 [44.0, 51.4] | 50.1 |
| gemma-4-31b__base | 89.3 | 104 | 60.6 [51.0, 69.4] | 16.5 [10.6, 24.9] | 6.9 |
| gemma-4-31b__dim | 90.1 | 96 | 62.5 [52.5, 71.5] | 15.8 [9.8, 24.4] | 6.5 |
| gemma-4-31b__mpoa | 40.4 | 579 | 52.3 [48.3, 56.4] | 38.0 [34.1, 42.0] | 31.6 |
| gemma-4-31b__som | 67.8 | 313 | 52.4 [46.9, 57.9] | 26.8 [22.2, 32.0] | 17.5 |
| gemma-4-31b__ara | 0.6 | 966 | 81.4 [78.8, 83.7] | 65.6 [62.6, 68.6] | 80.9 |
| olmo-2-32b__base | 83.2 | 163 | 88.3 [82.5, 92.4] | 23.9 [18.0, 31.0] | 15.0 |
| olmo-2-32b__dim | 33.5 | 646 | 48.1 [44.3, 52.0] | 27.4 [24.1, 31.0] | 32.0 |
| olmo-2-32b__mpoa | 3.6 | 937 | 86.4 [84.1, 88.5] | 72.0 [69.1, 74.8] | 83.5 |
| olmo-2-32b__som | 27.3 | 707 | 73.6 [70.2, 76.7] | 55.6 [51.9, 59.2] | 53.5 |
| olmo-2-32b__ara | 13.3 | 843 | 70.0 [66.8, 73.0] | 52.6 [49.2, 55.9] | 60.7 |
| olmo-3.1-32b__base | 84.1 | 155 | 76.8 [69.5, 82.7] | 20.0 [14.5, 27.0] | 12.3 |
| olmo-3.1-32b__dim | 68.3 | 308 | 63.6 [58.1, 68.8] | 23.1 [18.7, 28.1] | 20.4 |
| olmo-3.1-32b__mpoa | 65.0 | 340 | 61.5 [56.2, 66.5] | 26.8 [22.3, 31.7] | 21.6 |
| olmo-3.1-32b__som | 41.7 | 567 | 67.7 [63.8, 71.4] | 45.6 [41.5, 49.7] | 39.6 |
| olmo-3.1-32b__ara | 10.8 | 867 | 68.6 [65.5, 71.6] | 49.9 [46.6, 53.2] | 61.2 |
| mistral-small-2409__base | 55.7 | 430 | 79.1 [75.0, 82.6] | 44.7 [40.0, 49.4] | 35.5 |
| mistral-small-2409__dim | 32.8 | 653 | 79.5 [76.2, 82.4] | 57.9 [54.1, 61.6] | 53.4 |
| mistral-small-2409__mpoa | 6.9 | 905 | 83.4 [80.9, 85.7] | 67.7 [64.6, 70.7] | 77.7 |
| mistral-small-2409__som | 3.7 | 936 | 83.5 [81.0, 85.8] | 71.4 [68.4, 74.2] | 80.5 |
| mistral-small-2409__ara | 4.9 | 924 | 82.6 [80.0, 84.9] | 69.5 [66.4, 72.4] | 78.5 |
| mistral-small-3.2__base | 73.6 | 257 | 79.8 [74.4, 84.2] | 32.3 [26.9, 38.2] | 21.3 |
| mistral-small-3.2__dim | 28.5 | 695 | 78.8 [75.7, 81.7] | 59.0 [55.3, 62.6] | 56.4 |
| mistral-small-3.2__mpoa | 4.0 | 933 | 85.9 [83.5, 87.9] | 68.4 [65.3, 71.3] | 82.5 |
| mistral-small-3.2__som | 7.2 | 902 | 82.7 [80.1, 85.0] | 66.1 [62.9, 69.1] | 76.7 |
| mistral-small-3.2__ara | 4.2 | 931 | 85.5 [83.1, 87.6] | 69.8 [66.8, 72.7] | 81.9 |
| qwen2.5-7b__base | 72.1 | 271 | 80.8 [75.7, 85.1] | 33.2 [27.9, 39.0] | 23.3 |
| qwen2.5-7b__dim | 20.6 | 772 | 80.6 [77.6, 83.2] | 63.2 [59.8, 66.5] | 64.0 |
| qwen2.5-7b__mpoa | 13.8 | 838 | 82.8 [80.1, 85.2] | 67.3 [64.1, 70.4] | 71.5 |
| qwen2.5-7b__som | 12.2 | 853 | 83.4 [80.7, 85.7] | 67.4 [64.2, 70.5] | 73.1 |
| qwen2.5-7b__jspace | 52.6 | 461 | 76.4 [72.3, 80.0] | 48.2 [43.6, 52.7] | 36.3 |
| qwen2.5-7b__ara | 4.7 | 926 | 82.9 [80.4, 85.2] | 68.4 [65.3, 71.3] | 79.0 |
| qwen3.5-4b__base | 88.5 | 112 | 78.6 [70.1, 85.2] | 14.3 [9.0, 22.0] | 9.5 |
| qwen3.5-4b__dim | 54.6 | 441 | 71.9 [67.5, 75.9] | 46.9 [42.3, 51.6] | 32.6 |
| qwen3.5-4b__mpoa | 18.0 | 797 | 68.1 [64.8, 71.3] | 53.8 [50.4, 57.3] | 55.9 |
| qwen3.5-4b__som | 21.9 | 759 | 72.2 [68.9, 75.3] | 56.7 [53.1, 60.1] | 56.4 |
| qwen3.5-4b__jspace | 73.8 | 255 | 72.2 [66.4, 77.3] | 20.8 [16.3, 26.2] | 18.9 |
| qwen3.5-4b__ara | 18.9 | 788 | 68.9 [65.6, 72.0] | 54.2 [50.7, 57.6] | 55.9 |

**C3 findings**

- HarmBench not-harmful: direction raw +0.70 [+0.65, +0.74], baseline-adjusted +0.05 [-0.02, +0.11]; ARA raw +0.21 [+0.14, +0.28], adjusted -0.44 [-0.53, -0.36]; H2 -0.49 [-0.56, -0.41]; originals +0.65 [+0.58, +0.72]; H1 tally 11/3/2; pairs with every method higher: qwen, qwen-small (refusal: direction raw +0.62 [+0.57, +0.68], adjusted +0.03 [-0.05, +0.11], H2 -0.50 [-0.62, -0.38]).
- WildGuard not-harmful: direction raw +0.69 [+0.64, +0.73], baseline-adjusted -0.03 [-0.13, +0.07]; ARA raw +0.24 [+0.18, +0.30], adjusted -0.48 [-0.60, -0.37]; H2 -0.45 [-0.52, -0.38]; originals +0.72 [+0.62, +0.83]; H1 tally 10/1/5; pairs with every method higher: qwen-small (refusal: direction raw +0.62 [+0.57, +0.68], adjusted +0.03 [-0.05, +0.11], H2 -0.50 [-0.62, -0.38]).
- H2 per pair, WildGuard refusal: qwen +1.89 (p_Holm <0.001); gemma -4.27 (p_Holm <0.001); olmo -1.87 (p_Holm <0.001); mistral -0.06 (p_Holm 0.696); qwen-small +0.76 (p_Holm <0.001).
- H2 per pair, HarmBench not-harmful: qwen +1.47 (p_Holm <0.001); gemma -2.17 (p_Holm <0.001); olmo -1.26 (p_Holm <0.001); mistral -0.15 (p_Holm 0.083); qwen-small +0.27 (p_Holm 0.004).
- H2 per pair, WildGuard not-harmful: qwen +1.43 (p_Holm <0.001); gemma -1.92 (p_Holm <0.001); olmo -1.44 (p_Holm <0.001); mistral -0.06 (p_Holm 0.732); qwen-small +0.06 (p_Holm 0.732).
- Of the ablated main-study responses that WildGuard does not call a refusal, 75.1% [73.4, 76.9] are harmful per HarmBench and 55.8% per WildGuard (originals: 79.0% and 29.6%). Across the 42 ablated seed-1 cells the HarmBench share ranges 48.1-88.8% (lowest: olmo-2-32b__dim), the WildGuard share 15.8-75.9%.

## C4. H3 at Heretic's own keyword window (100 tokens)

Heretic 2.0 counts a dev response as a refusal if any of its 33 markers occurs in the first **100** generated tokens (`config.default.toml` @ 8fd4d76: `max_response_length = 100`; `model.py` `get_responses`: `max_new_tokens = max_response_length`; Methods sec:setup-opt, 'at most 100 new tokens'). The paper's H3 uses the same markers on the full response (<= 512 tokens). Here each stored response is re-tokenised with the evaluated model's tokenizer, cut to 100 tokens, and matched with the paper's keyword function. Because responses are greedy, the first 100 tokens of a 512-token greedy answer are what a 100-token greedy run would produce (up to numerics).

**Validation.** Encoding per base model (the one that reproduces the stored `response_256` text more often): gemma-2-27b canonical; gemma-4-31b canonical; mistral-small-2409 canonical; mistral-small-3.2 canonical; olmo-2-32b nl_split; olmo-3.1-32b canonical; qwen2.5-32b canonical; qwen2.5-7b canonical; qwen3.5-4b canonical; qwen3.8-27b canonical. Community cells: the repo's own tokenizer.json is used only if it reproduces the stored 256-token text at least as often as the base tokenizer; chosen own tokenizer: none (scores per candidate in `review_out_c4_tokenizer_cells.csv`: gemma-4-31b__community-coder3101 base: 1314/1322; community repo: 1314/1322; qwen2.5-32b__community-huihui base: 1321/1322; community repo: 1/1322; qwen2.5-32b__community-zetasepic-v2 base: 1320/1322; community repo: 1320/1322; qwen3.8-27b__community-hereticorg-ara base: 1317/1322; community repo: 0/1322; qwen3.8-27b__community-huihui base: 1317/1322; community repo: 1317/1322; mistral-small-3.2__community-huihui base (community repo ships no tokenizer.json): 1303/1322; mistral-small-3.2__community-llmfan46 base: 1311/1322; community repo: 1/1322; olmo-3.1-32b__community-muxodious base: 1312/1322; community repo: 1312/1322).

- Full response: re-matched keyword equals `keyword_refusal` in 97,828 of 97,828 responses.
- 256 tokens: the re-tokenised 256-token cut gives the stored `keyword_refusal_256` label in 97,825 of 97,828 responses (other encoding: 97,705); its text equals the stored `response_256` in 96.94% of all responses and 95.92% of those longer than 256 tokens.
- Cells below 98% text agreement or with any 256-token label mismatch: `gemma-4-31b__gmpoa` text 94.3% (other encoding 10.4%), label mismatches 0; `mistral-small-3.2__som` text 98.0% (other encoding 11.4%), label mismatches 0; `olmo-2-32b__base` text 80.3% (other encoding 53.2%), label mismatches 0; `olmo-2-32b__dim` text 63.9% (other encoding 14.1%), label mismatches 0; `olmo-2-32b__mpoa` text 67.2% (other encoding 14.8%), label mismatches 0; `olmo-2-32b__som` text 69.8% (other encoding 15.7%), label mismatches 0; `olmo-2-32b__gmpoa` text 68.0% (other encoding 17.0%), label mismatches 3; `olmo-2-32b__ara` text 64.4% (other encoding 12.6%), label mismatches 0.
- The two encodings give a different 100-token keyword label in 64 responses (0.07%), 10 of them in OLMo-2 cells.
- Bounds that do not rely on the re-encoding being exact (`response_256` is exactly the first 256 generated tokens, the keyword label only grows with the window, responses of <= 100 tokens lie wholly inside it): the 100-token label is pinned down in 99.98% of the responses; 17 are unresolved (gemma-2-27b 2, gemma-4-31b 1, mistral-small-3.2 3, olmo-2-32b 7, olmo-3.1-32b 1, qwen3.5-4b 2, qwen3.8-27b 1). The gaps are reported at both bounds below.
- Median length of the 100-token text: 478 characters (full responses: 1984); 92.1% of the responses are longer than 100 tokens.

**Pooled keyword-minus-WildGuard gap, harmful items** (pp, prompt-cluster bootstrap 95% CI; h3.py groups). H3 predicted a negative gap (keyword undercounts refusal).

| Cells | kw 512 % | kw 256 % | kw 100 % | WG full % | WG 256 % | kw512 - WG (paper) | kw256 - WG256 | **kw100 - WG** | **kw100 - WG256** | kw100 - WG at the bounds | kw100 (Heretic matcher) - WG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| originals (10) | 82.8 | 82.1 | 80.9 | 80.2 | 80.4 | +2.6 [+1.8, +3.4] | +1.6 [+0.8, +2.4] | +0.7 [-0.1, +1.5] | +0.4 [-0.4, +1.2] | +0.7 to +0.7 | +0.7 [-0.1, +1.5] |
| ablated, all (42) | 39.9 | 36.0 | 32.0 | 31.5 | 32.1 | +8.4 [+6.9, +10.0] | +3.8 [+2.4, +5.2] | +0.5 [-1.0, +1.9] | -0.1 [-1.6, +1.3] | +0.5 to +0.5 | +0.5 [-1.0, +1.9] |
| ablated, main (32) | 42.0 | 37.8 | 33.8 | 32.3 | 32.9 | +9.8 [+8.2, +11.3] | +4.9 [+3.5, +6.4] | +1.5 [+0.0, +2.9] | +0.9 [-0.6, +2.2] | +1.5 to +1.5 | +1.5 [+0.0, +2.9] |
| ablated, small (10) | 33.3 | 30.2 | 26.5 | 29.1 | 29.8 | +4.2 [+2.4, +5.9] | +0.4 [-1.4, +2.0] | -2.6 [-4.3, -0.9] | -3.3 [-5.0, -1.6] | -2.6 to -2.6 | -2.6 [-4.3, -0.9] |
| gamma-MPOA (4) | 47.0 | 42.5 | 38.7 | 29.9 | 30.6 | +17.1 [+15.0, +19.3] | +12.0 [+9.9, +14.0] | +8.8 [+6.8, +10.9] | +8.2 [+6.1, +10.2] | +8.8 to +8.8 | +8.8 [+6.8, +10.9] |
| community (8) | 22.9 | 20.5 | 17.3 | 6.4 | 6.8 | +16.5 [+15.2, +17.8] | +13.6 [+12.4, +14.8] | +11.0 [+9.8, +12.1] | +10.5 [+9.3, +11.6] | +11.0 to +11.0 | +11.0 [+9.8, +12.1] |
| seed-2 (10) | 32.8 | 30.0 | 26.4 | 29.2 | 30.0 | +3.5 [+2.0, +4.9] | -0.0 [-1.5, +1.4] | -2.9 [-4.3, -1.5] | -3.7 [-5.1, -2.3] | -2.9 to -2.9 | -2.9 [-4.3, -1.5] |
| all 74 cells | 43.3 | 40.1 | 36.6 | 35.0 | 35.6 | +8.3 [+7.1, +9.6] | +4.5 [+3.3, +5.7] | +1.7 [+0.5, +2.8] | +1.1 [-0.1, +2.2] | +1.7 to +1.7 | +1.7 [+0.5, +2.8] |

**Per base model** (harmful items; pooled over the model's cells)

| Model | Cells | kw512 - WG | kw100 - WG | kw100 - WG256 | kw 100 % / WG % |
|---|---|---|---|---|---|
| qwen2.5-32b | original | -0.3 [-2.1, +1.4] | -2.1 [-3.8, -0.3] | -2.2 [-3.8, -0.4] | 74.8 / 76.8 |
| qwen2.5-32b | ablated (seed-1 methods) | +9.6 [+7.4, +11.7] | +2.1 [+0.2, +4.0] | +2.0 [+0.1, +4.0] | 27.2 / 25.0 |
| qwen3.8-27b | original | -1.3 [-2.7, +0.0] | -2.4 [-3.8, -0.9] | -2.5 [-3.9, -1.0] | 87.6 / 89.9 |
| qwen3.8-27b | ablated (seed-1 methods) | +7.3 [+4.9, +9.6] | -1.4 [-3.9, +1.0] | -4.3 [-6.9, -1.9] | 40.3 / 41.7 |
| gemma-2-27b | original | +2.9 [+1.4, +4.3] | +2.2 [+0.7, +3.6] | +1.7 [+0.3, +3.2] | 90.4 / 88.3 |
| gemma-2-27b | ablated (seed-1 methods) | -2.4 [-4.6, -0.2] | -7.8 [-9.9, -5.7] | -7.8 [-10.0, -5.7] | 44.9 / 52.7 |
| gemma-4-31b | original | +2.5 [+1.2, +3.8] | +1.3 [+0.2, +2.6] | +0.9 [-0.2, +2.1] | 90.6 / 89.3 |
| gemma-4-31b | ablated (seed-1 methods) | +13.5 [+11.5, +15.3] | +7.4 [+5.5, +9.2] | +6.7 [+4.8, +8.5] | 57.2 / 49.7 |
| olmo-2-32b | original | +0.0 [-1.5, +1.4] | -2.1 [-3.5, -0.7] | -2.3 [-3.7, -0.8] | 81.2 / 83.2 |
| olmo-2-32b | ablated (seed-1 methods) | +18.5 [+16.0, +21.1] | +4.3 [+2.1, +6.5] | +4.5 [+2.3, +6.7] | 23.8 / 19.4 |
| olmo-3.1-32b | original | +1.7 [+0.1, +3.5] | -1.2 [-2.8, +0.4] | -2.0 [-3.5, -0.4] | 82.8 / 84.1 |
| olmo-3.1-32b | ablated (seed-1 methods) | +11.5 [+8.9, +14.0] | +2.5 [-0.1, +4.9] | +1.7 [-0.8, +4.0] | 48.9 / 46.5 |
| mistral-small-2409 | original | +15.9 [+13.0, +18.8] | +12.4 [+9.5, +15.2] | +12.1 [+9.3, +14.9] | 68.1 / 55.7 |
| mistral-small-2409 | ablated (seed-1 methods) | +9.3 [+7.4, +11.1] | +2.1 [+0.5, +3.7] | +1.8 [+0.2, +3.4] | 14.2 / 12.1 |
| mistral-small-3.2 | original | +0.4 [-1.5, +2.3] | -1.9 [-3.7, +0.0] | -1.7 [-3.6, +0.1] | 71.7 / 73.6 |
| mistral-small-3.2 | ablated (seed-1 methods) | +10.9 [+9.0, +12.8] | +2.8 [+1.2, +4.3] | +2.6 [+1.0, +4.1] | 13.8 / 11.0 |
| qwen2.5-7b | original | +3.7 [+1.7, +5.7] | +1.9 [+0.1, +3.6] | +1.4 [-0.3, +3.1] | 74.0 / 72.1 |
| qwen2.5-7b | ablated (seed-1 methods) | +10.5 [+8.3, +12.7] | +4.0 [+1.9, +6.1] | +3.6 [+1.5, +5.6] | 24.8 / 20.8 |
| qwen3.5-4b | original | +0.7 [-0.7, +2.1] | -1.0 [-2.3, +0.2] | -1.5 [-2.8, -0.3] | 87.4 / 88.5 |
| qwen3.5-4b | ablated (seed-1 methods) | -2.1 [-4.3, +0.0] | -9.2 [-11.3, -7.2] | -10.2 [-12.2, -8.2] | 28.2 / 37.4 |

**Per evaluated model (cell)**, harmful items, Holm over all 74 cells per comparison (`review_out_c4_gap_cells.csv`):

- `kw_vs_wg`: keyword > WildGuard in 34/42 ablated seed-1 cells (24 Holm-significant), keyword < WildGuard in 7 (3 Holm-significant: gemma-2-27b__dim, gemma-2-27b__som, qwen3.5-4b__jspace).
- `kw100_vs_wg`: keyword > WildGuard in 24/42 ablated seed-1 cells (14 Holm-significant), keyword < WildGuard in 18 (11 Holm-significant: qwen3.8-27b__dim, gemma-2-27b__dim, gemma-2-27b__som, gemma-2-27b__ara, gemma-4-31b__som, olmo-2-32b__dim, olmo-3.1-32b__dim, olmo-3.1-32b__mpoa, qwen3.5-4b__dim, qwen3.5-4b__jspace, qwen3.5-4b__ara).
- `kw100_vs_wg256`: keyword > WildGuard in 24/42 ablated seed-1 cells (12 Holm-significant), keyword < WildGuard in 18 (12 Holm-significant: qwen3.8-27b__dim, qwen3.8-27b__som, gemma-2-27b__dim, gemma-2-27b__som, gemma-2-27b__ara, gemma-4-31b__som, olmo-2-32b__dim, olmo-3.1-32b__dim, olmo-3.1-32b__mpoa, qwen3.5-4b__dim, qwen3.5-4b__jspace, qwen3.5-4b__ara).
- `kw256_vs_wg256`: keyword > WildGuard in 28/42 ablated seed-1 cells (21 Holm-significant), keyword < WildGuard in 14 (6 Holm-significant: qwen3.8-27b__dim, gemma-2-27b__dim, gemma-2-27b__som, olmo-2-32b__dim, qwen3.5-4b__jspace, qwen3.5-4b__ara).

**H3 models (h3.py primary linear GEE on the paired difference, harmful items, pp).** Cell-level OLS in `review_out_c4_models.csv`.

| Outcome | Study | Mean gap | Generation effect on gap | Class effect (ARA - dir.) | Gen x class |
|---|---|---|---|---|---|
| kw_vs_wg | main | +11.3 [+9.8, +12.7] (p Holm <0.001) | +4.4 [+3.1, +5.7] (p Holm <0.001) | +6.0 [+4.6, +7.4] (p Holm <0.001) | +9.4 [+7.2, +11.6] (p Holm <0.001) |
| kw_vs_wg | small | +4.4 [+2.8, +6.0] (p Holm <0.001) | -13.2 [-15.8, -10.6] (p Holm <0.001) | +0.8 [-1.5, +3.0] (p Holm 0.774) | -1.8 [-6.0, +2.3] (p Holm 0.774) |
| kw100_vs_wg | main | +2.8 [+1.4, +4.2] (p Holm <0.001) | +4.7 [+3.4, +5.9] (p Holm <0.001) | +5.2 [+3.9, +6.4] (p Holm <0.001) | +8.1 [+6.0, +10.1] (p Holm <0.001) |
| kw100_vs_wg | small | -2.3 [-3.8, -0.8] (p Holm 0.008) | -12.2 [-14.5, -9.9] (p Holm <0.001) | +1.1 [-0.9, +3.2] (p Holm 0.280) | +3.6 [-0.2, +7.3] (p Holm 0.124) |
| kw100_vs_wg256 | main | +2.2 [+0.8, +3.5] (p Holm 0.002) | +3.5 [+2.2, +4.7] (p Holm <0.001) | +5.1 [+3.8, +6.3] (p Holm <0.001) | +7.6 [+5.6, +9.6] (p Holm <0.001) |
| kw100_vs_wg256 | small | -2.9 [-4.4, -1.4] (p Holm <0.001) | -12.9 [-15.3, -10.6] (p Holm <0.001) | +1.4 [-0.7, +3.5] (p Holm 0.287) | +2.8 [-1.0, +6.6] (p Holm 0.287) |
| kw256_vs_wg256 | main | +6.3 [+4.9, +7.7] (p Holm <0.001) | +4.4 [+3.2, +5.7] (p Holm <0.001) | +5.5 [+4.2, +6.8] (p Holm <0.001) | +7.5 [+5.4, +9.5] (p Holm <0.001) |
| kw256_vs_wg256 | small | +0.6 [-1.0, +2.1] (p Holm 1.000) | -11.9 [-14.4, -9.4] (p Holm <0.001) | +0.6 [-1.5, +2.8] (p Holm 1.000) | +1.8 [-2.2, +5.8] (p Holm 1.000) |

**C4 findings**

- all 42 ablated seed-1 cells: keyword refusal 39.9% on <= 512 tokens, 32.0% on 100 tokens (-7.9 pp); WildGuard 31.5% (full) / 32.1% (256 tokens). Gap keyword - WildGuard: +8.4 pp [+6.9, +10.0] (paper) -> +0.5 pp [-1.0, +1.9] at 100 tokens vs full-response WildGuard, -0.1 pp [-1.6, +1.3] vs 256-token WildGuard; Heretic's own matcher +0.5 pp [-1.0, +1.9].
- main study (32): keyword refusal 42.0% on <= 512 tokens, 33.8% on 100 tokens (-8.3 pp); WildGuard 32.3% (full) / 32.9% (256 tokens). Gap keyword - WildGuard: +9.8 pp [+8.2, +11.3] (paper) -> +1.5 pp [+0.0, +2.9] at 100 tokens vs full-response WildGuard, +0.9 pp [-0.6, +2.2] vs 256-token WildGuard; Heretic's own matcher +1.5 pp [+0.0, +2.9].
- small pair (10): keyword refusal 33.3% on <= 512 tokens, 26.5% on 100 tokens (-6.8 pp); WildGuard 29.1% (full) / 29.8% (256 tokens). Gap keyword - WildGuard: +4.2 pp [+2.4, +5.9] (paper) -> -2.6 pp [-4.3, -0.9] at 100 tokens vs full-response WildGuard, -3.3 pp [-5.0, -1.6] vs 256-token WildGuard; Heretic's own matcher -2.6 pp [-4.3, -0.9].
- originals (10): keyword refusal 82.8% on <= 512 tokens, 80.9% on 100 tokens (-1.9 pp); WildGuard 80.2% (full) / 80.4% (256 tokens). Gap keyword - WildGuard: +2.6 pp [+1.8, +3.4] (paper) -> +0.7 pp [-0.1, +1.5] at 100 tokens vs full-response WildGuard, +0.4 pp [-0.4, +1.2] vs 256-token WildGuard; Heretic's own matcher +0.7 pp [-0.1, +1.5].
- By length (ablated cells, 100-token keyword vs full WildGuard): responses that hit the 512 cap +11.0 pp [+9.7, +12.4], responses that ended earlier -10.4 pp [-12.2, -8.6].
- H3 GEE, main, 100-token keyword vs WildGuard: mean gap +2.8 pp [+1.4, +4.2] (p_Holm <0.001; full text +11.3); generation effect on gap +4.7 pp [+3.4, +5.9] (p_Holm <0.001; full text +4.4); class effect on gap +5.2 pp [+3.9, +6.4] (p_Holm <0.001; full text +6.0); generation x class interaction on gap +8.1 pp [+6.0, +10.1] (p_Holm <0.001; full text +9.4).
- H3 GEE, small, 100-token keyword vs WildGuard: mean gap -2.3 pp [-3.8, -0.8] (p_Holm 0.008; full text +4.4); generation effect on gap -12.2 pp [-14.5, -9.9] (p_Holm <0.001; full text -13.2); class effect on gap +1.1 pp [-0.9, +3.2] (p_Holm 0.280; full text +0.8); generation x class interaction on gap +3.6 pp [-0.2, +7.3] (p_Holm 0.124; full text -1.8).
- Per ablated seed-1 cell (100-token keyword vs full WildGuard): keyword lower in 18/42 cells (11 Holm-significant), higher in 24 (14).
- Bounds on the true 100-token label (all ablated cells): gap vs full WildGuard between +0.5 pp [-1.0, +1.9] (lower bound) and +0.5 pp [-1.0, +1.9] (upper bound).

## Verdict per reviewer concern

- **C2 (item source drives the result): partly supported.** Code items carry a large but uneven part of the originals' generation difference (qwen 68% [59, 77] of +13.1 pp; gemma: total +1.0 pp = -5.7 pp on public items and +27.0 pp on code items, weighted; olmo: total +0.8 pp = -0.3 pp on public items and +5.0 pp on code items, weighted; mistral 9% [5, 14] of +17.9 pp; mean of the four main pairs 52% [44, 60]), and Gemma's direction-method effects sit in the code items (DiM / MPOA / SOM +65.5 / +35.0 / +47.5 pp on code vs +3.6 / -15.0 / -9.1 pp on the public sets); the largest effect of each other main pair, by contrast: qwen ARA +41.3 public vs +7.5 code; olmo MPOA +75.6 public vs +6.5 code; mistral DiM -5.6 public vs +0.5 code. But on the 772 public items alone the paper's conclusions are unchanged: H1 is still not supported (baseline-adjusted direction effect -0.13 [-0.28, +0.03] vs +0.03 [-0.05, +0.11] on all items; every method higher only in qwen-small; Holm tally 8/6/2 vs 8/4/4), and the pooled H2 interaction keeps its size and sign (-0.46 [-0.58, -0.34] vs -0.50 [-0.62, -0.38]) with the same pairs carrying it (gemma, olmo) or opposing it (qwen, qwen-small). On code items alone the pooled baseline-adjusted direction effect would be positive (+1.39 [+1.08, +1.70]; per pair, risk difference: qwen -33.8, gemma +22.3, olmo -0.2, mistral -7.9 pp), so the item mix changes pooled log-odds, but the conclusions rest on the public sets. The paper should show the per-source table and the public-only results.
- **C3 (refusal is not harmful compliance): partly supported.** The construct point is right: of the ablated main-study responses that WildGuard does not call a refusal, 75.1% [73.4, 76.9] are harmful per HarmBench and 55.8% per WildGuard, so 16.8% [15.7, 18.0] of all ablated main-study responses are neither refused nor harmful (HarmBench); residual refusal understates how often an edit fails to yield harmful content. The headline conclusions do not change with either harm label: baseline-adjusted direction effect +0.05 [-0.02, +0.11] (HarmBench) and -0.03 [-0.13, +0.07] (WildGuard harmful) vs +0.03 [-0.05, +0.11] for refusal; H2 interaction -0.49 [-0.56, -0.41] and -0.45 [-0.52, -0.38] vs -0.50 [-0.62, -0.38]; per pair the interaction is significant in the predicted direction for gemma, olmo and opposite for qwen under every outcome. Some per-pair and raw results do change: under HarmBench not-harmful every method is significantly higher on the newer model in qwen, qwen-small (refusal: qwen-small), so H1 as stated would hold for qwen; the raw pooled ARA generation effect becomes Holm-significant (HarmBench not-harmful +0.21 [+0.14, +0.28], WildGuard not-harmful +0.24 [+0.18, +0.30]; refusal +0.12 [+0.00, +0.24], not significant after Holm); of the 21 per-pair H1 McNemar tests, 5 (HarmBench; 4 of them toward 'newer harder') and 6 (WildGuard harmful; 6 toward 'newer harder') change Holm decision or sign (HarmBench: significantly higher on the newer model now qwen som, gemma mpoa, gemma som; no longer significant mistral dim; significantly lower now mistral ara); Holm tally of the 16 main tests 11/3/2 (HarmBench) and 10/1/5 (WildGuard harmful) vs 8/4/4. Adding harmful compliance as a secondary H1/H2 outcome answers the concern.
- **C4 (H3 uses the wrong keyword window): supported.** At Heretic's own 100-token window the reversal disappears: keyword minus WildGuard over all 42 ablated seed-1 cells is +0.5 pp [-1.0, +1.9] against the full-response WildGuard label and -0.1 pp [-1.6, +1.3] against the 256-token label, versus +8.4 pp [+6.9, +10.0] at 512 tokens in the paper (main study +1.5 pp [+0.0, +2.9], small pair -2.6 pp [-4.3, -0.9]). The keyword count falls from 39.9% to 32.0% while WildGuard stays at 31.5%: most of the excess keyword refusals are markers that appear after the first 100 tokens. Neither the paper's 'keyword overcounts' nor H3's predicted 'keyword undercounts' holds pooled at the tool's window; per cell the sign varies (14 cells significantly higher, 11 lower, Holm). The bounds on the true 100-token label give the same result (+0.5 pp [-1.0, +1.9] to +0.5 pp [-1.0, +1.9]). The second part of H3 (the gap differs between method classes) is unchanged in the main study: at 100 tokens the gap is still larger for ARA (+5.2 pp, p_Holm <0.001), grows with generation (+4.7 pp) and shows the same generation x class interaction (+8.1 pp [+6.0, +10.1], p_Holm <0.001). H3 should be reported per window, with the 100-token result as the one that bears on the tool.

## Caveats

- Intervals reflect prompt sampling only (selected checkpoints fixed, one seed), as in the paper; per-set analyses have 100-313 items, so their per-pair tests have less power than the 972-item tests.
- Pooled GEE coefficients average across families that differ in sign (see the generation x family chi2); per-pair numbers are the primary evidence, as the reviewer's A1 asks.
- HarmBench and WildGuard harmfulness labels are themselves automatic judges; 'not harmful' includes degenerate, off-topic and unhelpful answers, so it is a weaker construct than refusal in the other direction. The HarmBench classifier sees at most 512 classifier tokens of each response.
- C4 re-tokenises stored text (token ids were not stored). For every model but OLMo-2 the canonical re-encoding reproduces the stored 256-token text in nearly every response; OLMo-2 needs the newline-split encoding and still differs by a few tokens in some responses (alternative-encoding sensitivity reported). No judge was run on 100-token text (no GPU here); the 100-token keyword is compared with WildGuard on the full response and on the 256-token prefix.
- Heretic's dev scoring generates with HF transformers on the 100 dev prompts (and, for Qwen3.x, after the detected '</think>' response prefix); this analysis applies its window to the held-out greedy vLLM responses, which start after the chat template's empty think block.

## Files

- `results/analysis/review_out_c2_generation_effects.csv`
- `results/analysis/review_out_c2_h2_contrasts.csv`
- `results/analysis/review_out_c2_models.csv`
- `results/analysis/review_out_c2_originals_decomposition.csv`
- `results/analysis/review_out_c2_pair_class.csv`
- `results/analysis/review_out_c2_tests.csv`
- `results/analysis/review_out_c3_cells.csv`
- `results/analysis/review_out_c3_generation_effects.csv`
- `results/analysis/review_out_c3_h2_contrasts.csv`
- `results/analysis/review_out_c3_models.csv`
- `results/analysis/review_out_c3_pair_class.csv`
- `results/analysis/review_out_c3_pooled.csv`
- `results/analysis/review_out_c3_tests.csv`
- `results/analysis/review_out_c4_gap_cells.csv`
- `results/analysis/review_out_c4_gap_pooled.csv`
- `results/analysis/review_out_c4_kw_rows.csv`
- `results/analysis/review_out_c4_models.csv`
- `results/analysis/review_out_c4_tokenizer_cells.csv`
- `results/analysis/review_out_c4_validation.csv`
- `results/analysis/review_out_checks.csv`
- `results/analysis/review_out_summary.md`

