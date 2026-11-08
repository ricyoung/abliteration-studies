# Paper 33 -- 256-token judge sensitivity (H3 robustness)

Generated 2026-11-06 23:06 UTC by scripts/analysis/truncation256.py (read-only on every input). Numbers are relative to the judges, not to human labels.

## Coverage

- Judged cells in responses_judged: 74; with refusal/judgments_256.jsonl: 74; id set equal to judgments.jsonl: 74; equal to the assembled rows: 74; responses.jsonl md5 equal to the assembly: 74; judge revisions equal to the main run: 74.
- `used` dirs of dir_status.csv without assembled rows: none.
- Coverage gaps: none.
- Missing labels on the 256-token text (unparsable judge output, treated as missing as in the main analysis): WildGuard 7 rows, Qwen3Guard 0, HarmBench (harmful rows) 0 (olmo-2-32b__dim 1, olmo-2-32b__ara 2, olmo-3.1-32b__som 1, qwen2.5-7b__mpoa 1, qwen3.5-4b__dim-s2 1, qwen3.5-4b__ara 1). 256-token runs scored 2026-11-06T16:01:10Z to 2026-11-06T16:01:10Z.

## Checks

- keyword_refusal_256 in judgments_256.jsonl vs the assembled ref_kw256: 0 mismatches; n_tokens: 0; finish_reason: 0; `truncated` vs n_tokens > 256: 0.
- Responses of at most 256 tokens (256-token text = full text), label on the prefix vs on the full response: WildGuard refusal 6 discordant of 24,507; Qwen3Guard refusal 14 discordant of 24,507; HarmBench harmful compliance 4 discordant of 17,967 (the judges are deterministic up to batch numerics).
- Recomputed full-text gaps vs h3_gap_cells.csv (888 cell x set group x comparison rows): max |gap difference| 0.00e+00, n differs in 0 rows, max |Holm p difference| 0.00e+00, unmatched rows 0.
- Recomputed pooled full-text gaps vs h3_gap_pooled.csv (132 rows): max |gap difference| 0.00e+00, n differs in 0; bootstrap limits differ by up to 0.29 pp (different random draws).

## Headline numbers (harmful items)

- **Judge labels, 42 ablated seed-1 cells:** WildGuard refusal 31.5% on the full response vs 32.1% on the 256-token prefix (+0.6 pp [+0.5, +0.7]; 131 responses lose and 385 gain the refusal label; among the 32,196 responses longer than 256 tokens +0.8 pp [+0.6, +0.9]). Qwen3Guard refusal 33.8% vs 35.1% (+1.3 pp [+1.1, +1.4]). HarmBench harmful 51.8% vs 50.2% (-1.6 pp [-1.8, -1.3]).
- **Judge labels, 10 originals:** WildGuard refusal 80.2% on the full response vs 80.4% on the 256-token prefix (+0.3 pp [+0.1, +0.5]; 11 responses lose and 40 gain the refusal label; among the 3,950 responses longer than 256 tokens +0.7 pp [+0.4, +1.2]). Qwen3Guard refusal 78.9% vs 79.3% (+0.4 pp [+0.3, +0.6]). HarmBench harmful 16.0% vs 15.0% (-1.1 pp [-1.3, -0.8]).
- **H3 gap, 42 ablated cells:** full text kw - WG +8.4 pp [+6.9, +10.0] (paper); keyword on 256 tokens vs WG on the full response +4.5 pp [+3.0, +5.9] (paper, mixed); both on the same 256-token text +3.8 pp [+2.4, +5.2] (keyword 36.0%, WildGuard 32.1%); change vs full text -4.6 pp [-4.9, -4.3].
- **H3 gap, main study (32):** full text kw - WG +9.8 pp [+8.2, +11.3] (paper); keyword on 256 tokens vs WG on the full response +5.5 pp [+4.1, +7.0] (paper, mixed); both on the same 256-token text +4.9 pp [+3.5, +6.4] (keyword 37.8%, WildGuard 32.9%); change vs full text -4.8 pp [-5.2, -4.5].
- **H3 gap, small pair (10):** full text kw - WG +4.2 pp [+2.5, +5.9] (paper); keyword on 256 tokens vs WG on the full response +1.0 pp [-0.6, +2.7] (paper, mixed); both on the same 256-token text +0.4 pp [-1.4, +2.0] (keyword 30.2%, WildGuard 29.8%); change vs full text -3.8 pp [-4.3, -3.4].
- **H3 gap, originals (10):** full text kw - WG +2.6 pp [+1.8, +3.4] (paper); keyword on 256 tokens vs WG on the full response +1.9 pp [+1.1, +2.7] (paper, mixed); both on the same 256-token text +1.6 pp [+0.8, +2.4] (keyword 82.1%, WildGuard 80.4%); change vs full text -1.0 pp [-1.3, -0.7].
- **H3 gap, all 74 judged cells:** full text kw - WG +8.3 pp [+7.0, +9.6] (paper); keyword on 256 tokens vs WG on the full response +5.1 pp [+3.9, +6.3] (paper, mixed); both on the same 256-token text +4.5 pp [+3.3, +5.7] (keyword 40.1%, WildGuard 35.6%); change vs full text -3.8 pp [-4.1, -3.6].
- **By length, 42 ablated cells, 256-token gap:** responses that ended before the 512 cap -5.6 pp [-7.6, -3.8] (paper full text -1.2 pp [-3.2, +0.9]), capped responses +13.0 pp [+11.5, +14.4] (paper +17.7 pp [+16.2, +19.3]); responses of at most 256 tokens -16.3 pp [-18.4, -14.2] (n = 8,624), longer responses +9.2 pp [+7.7, +10.6] (n = 32,196; full-text gap on the same responses +15.1 pp [+13.5, +16.5]).
- **Second judge, 42 ablated cells, 256-token text:** kw256 - Qwen3Guard +0.9 pp [-0.6, +2.4] (paper full text kw - Q3G +6.2 pp [+4.6, +7.7]); Qwen3Guard - WildGuard +2.9 pp [+2.1, +3.7] (paper +2.3 pp [+1.5, +3.1]).
- **Per ablated cell (Holm over 74 cells):** 256-token gap > 0 in 28/42 (21 significant), < 0 in 14 (6 significant: qwen3.8-27b__dim, gemma-2-27b__dim, gemma-2-27b__som, olmo-2-32b__dim, qwen3.5-4b__jspace, qwen3.5-4b__ara); full text (paper) > 0 in 34 (24), < 0 in 7 (3). Gap smaller on the 256-token text than on the full text in 42/42 cells.
- **H3 models, main, harmful (256-token gap; paper full-text value in brackets):** mean gap +6.3 pp [+4.9, +7.7], Holm p <0.001 (+11.3 pp, Holm p <0.001); generation effect on gap +4.4 pp [+3.2, +5.7], Holm p <0.001 (+4.4 pp, Holm p <0.001); class effect on gap +5.5 pp [+4.2, +6.8], Holm p <0.001 (+6.0 pp, Holm p <0.001); generation x class interaction on gap +7.5 pp [+5.4, +9.5], Holm p <0.001 (+9.4 pp, Holm p <0.001). Cell-level OLS p: 0.002 (<0.001), 0.226 (0.288), 0.139 (0.151), 0.308 (0.257).
- **H3 models, small, harmful (256-token gap; paper full-text value in brackets):** mean gap +0.6 pp [-1.0, +2.1], Holm p 1.000 (+4.4 pp, Holm p <0.001); generation effect on gap -11.9 pp [-14.4, -9.4], Holm p <0.001 (-13.2 pp, Holm p <0.001); class effect on gap +0.6 pp [-1.5, +2.8], Holm p 1.000 (+0.8 pp, Holm p 0.774); generation x class interaction on gap +1.8 pp [-2.2, +5.8], Holm p 1.000 (-1.8 pp, Holm p 0.774). Cell-level OLS p: 0.887 (0.328), 0.161 (0.162), 0.934 (0.930), 0.908 (0.916).

## Pooled table (harmful_all; source of paper/sections/08e_truncation256.tex, tab:trunc256)

| group | cells | n | >256 tok % | WG full % | WG 256 % | WG 256-full | kw-WG full (paper) | kw256-WG256 | kw256-Q3G256 |
|---|---|---|---|---|---|---|---|---|---|
| original | 10 | 9720 | 40.7 | 80.2 | 80.4 | +0.3 pp [+0.1, +0.5] | +2.6 pp [+1.8, +3.4] | +1.6 pp [+0.8, +2.4] | +2.7 pp [+1.9, +3.6] |
| ablated_primary | 42 | 40820 | 78.9 | 31.5 | 32.1 | +0.6 pp [+0.5, +0.7] | +8.4 pp [+6.9, +10.0] | +3.8 pp [+2.4, +5.2] | +0.9 pp [-0.6, +2.4] |
| ablated_primary:main | 32 | 31100 | 79.1 | 32.3 | 32.9 | +0.6 pp [+0.5, +0.7] | +9.8 pp [+8.2, +11.3] | +4.9 pp [+3.5, +6.4] | +2.1 pp [+0.6, +3.5] |
| ablated_primary:small | 10 | 9720 | 78.1 | 29.1 | 29.8 | +0.7 pp [+0.5, +0.9] | +4.2 pp [+2.5, +5.9] | +0.4 pp [-1.4, +2.0] | -2.7 pp [-4.5, -1.0] |
| ablated_primary (complete_only) | 42 | 20011 | 56.9 | 52.4 | 52.4 | -0.0 pp [-0.1, +0.1] | -1.2 pp [-3.2, +0.9] | -5.6 pp [-7.6, -3.8] | -7.6 pp [-9.8, -5.5] |
| ablated_primary (capped_only) | 42 | 20809 | 100.0 | 11.4 | 12.6 | +1.2 pp [+1.0, +1.4] | +17.7 pp [+16.2, +19.3] | +13.0 pp [+11.5, +14.4] | +9.1 pp [+7.9, +10.3] |
| role:control | 4 | 3888 | 82.0 | 29.9 | 30.6 | +0.7 pp [+0.4, +1.0] | +17.1 pp [+14.9, +19.4] | +12.0 pp [+9.9, +14.0] | +9.1 pp [+7.1, +11.1] |
| role:external | 8 | 7776 | 90.3 | 6.4 | 6.8 | +0.5 pp [+0.3, +0.7] | +16.5 pp [+15.2, +17.8] | +13.6 pp [+12.4, +14.8] | +8.1 pp [+6.8, +9.3] |
| role:replicate | 10 | 9719 | 78.2 | 29.3 | 30.0 | +0.8 pp [+0.5, +1.0] | +3.5 pp [+2.0, +5.0] | -0.0 pp [-1.5, +1.4] | -1.9 pp [-3.4, -0.4] |
| all_judged | 74 | 71923 | 75.0 | 35.0 | 35.6 | +0.6 pp [+0.5, +0.7] | +8.3 pp [+7.0, +9.6] | +4.5 pp [+3.3, +5.7] | +2.0 pp [+0.8, +3.2] |

## Caveats

- The 256-token rerun answers whether the judges' labels, and the keyword-judge gap, depend on how much of a response the judge sees. It does not say which label is correct; there is no human audit.
- Only responses longer than 256 tokens can change; on the others the prefix is the full text, and the discordant counts in 'Checks' measure judge nondeterminism (batching), not truncation.
- The 256-token WildGuard label is missing for a handful of rows (unparsable output); every comparison uses the rows on which both of its labels exist (pairwise deletion, as h3.py), so n can differ from the full-text n by those rows.
- Pooled CIs come from new bootstrap draws (same B, seed and method as h3.py, but h3.py consumes random numbers in its agreement step first), so the paper_* limits and the recomputed limits differ slightly.
- Holm families: per-cell label changes, per judge across the 74 cells; per-cell gaps, per comparison x set group across the 74 cells (h3.py convention); GEE terms, 4 per study x subset.
