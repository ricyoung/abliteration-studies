# Dev-set re-score of the released exports (paper 33)

Generated 2026-11-06T23:06:47Z by `scripts/analysis/dev_rescore_summary.py` from `results/rescore_dev/rescore_dev_all.csv` (46 rows), the per-host TSVs, `paper/figures/data/trials.csv` and the Gemma-2 config/vLLM artefacts listed in the script docstring. Per-export rows: `results/analysis/dev_rescore_cells.csv`.

Selected = the trial chosen by the KL <= 0.1 rule, values logged during the search (adapter applied to the base). Re-scored = the released merged export scored by Heretic's evaluator (`heretic --model <pinned base> --evaluate-model <export>`) with the run's own config, code version and batch size, on the same 100 dev prompts (refusals) and 100 harmless prompts (first-token KL). Delta = re-scored - selected. Every selected trial/refusal/KL in the re-score table matches `trials.csv` (is_selected_kl0.1).

## Headline

- 46 released exports (42 primary + 4 gamma-MPOA controls; the 10 seed-2 replicates are not released). All 46 re-scores rc = 0; the original model's dev refusals in the re-score equal the value Heretic recorded in the search (`paper/tables/t1_dev_refusals.csv`, method base) in 46/46 runs.
- Delta refusals: median +1, mean +1.00; 27 up, 9 unchanged, 10 down; 37/46 within +-2; range -7 to +9.
- Delta KL: median +0.0003, mean +0.0019, max |delta| 0.0356.
- Selected-trial refusal proportion inside the Wilson 95% interval of the re-scored proportion: 45/46.
- Re-scored KL > 0.1: 4 exports: Gemma-2-27B MPOA 0.0991 -> 0.1036; Gemma-2-27B gamma-MPOA 0.0969 -> 0.1059; Gemma-4-31B ARA 0.0845 -> 0.1070; OLMo-3.1-32B gamma-MPOA 0.0993 -> 0.1032.
- |Delta refusals| > 2: 9 exports: Qwen2.5-32B MPOA 13 -> 17 (+4); Qwen2.5-32B ARA 10 -> 16 (+6); Qwen3.8-27B DiM 47 -> 44 (-3); Qwen3.8-27B ARA 63 -> 72 (+9); Gemma-2-27B SOM 83 -> 86 (+3); Gemma-2-27B ARA 16 -> 21 (+5); Gemma-4-31B ARA 28 -> 21 (-7); Mistral-Small-2409 DiM 69 -> 72 (+3); Qwen3.5-4B DiM 30 -> 37 (+7).

## Tests (all 46 exports)

| test | n_pairs | median_diff | mean_diff | n_pos | n_neg | n_zero | wilcoxon_W | wilcoxon_p | sign_test_p | wilcoxon_p_holm | n | rho | p | ci_lo | ci_hi | tau_b | tau_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wilcoxon signed-rank, re-scored vs selected refusals (count of 100), all 46 exports | 46 | 1 | 1 | 27 | 10 | 9 | 153.5 | 0.002232 | 0.007632 | 0.004463 |  |  |  |  |  |  |  |
| Wilcoxon signed-rank, re-scored vs selected first-token KL, all 46 exports | 46 | 0.0003367 | 0.001898 | 25 | 21 | 0 | 427 | 0.2192 | 0.6587 | 0.2192 |  |  |  |  |  |  |  |
| Spearman, selected vs re-scored refusals, all 46 exports (bootstrap 95% CI, 4000 resamples) |  |  |  |  |  |  |  |  |  |  | 46 | 0.995 | 9.975e-46 | 0.9845 | 0.9975 | 0.9593 | 1.788e-20 |

Re-scored refusal counts are higher than the selected trial's more often than lower (27 vs 10, 9 ties; Wilcoxon p = 0.0022, Holm 0.0045; sign test p = 0.0076); KL shows no significant shift (Wilcoxon p = 0.219). Selected and re-scored counts rank the exports almost identically (Spearman rho = 0.995 [0.984, 0.997]).

## By group

| group | n | median d_ref | mean d_ref | n_up | n_same | n_down | within +-2 | min_d_ref | max_d_ref | max |d_ref| | median d_KL | mean d_KL | max |d_KL| | KL>0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all 46 released exports | 46 | 1.0 | 1.0 | 27 | 9 | 10 | 37 | -7 | 9 | 9 | 0.0003 | 0.0019 | 0.0356 | 4 |
| primary (42) | 42 | 1.0 | 1.12 | 26 | 8 | 8 | 33 | -7 | 9 | 9 | 0.0003 | 0.0017 | 0.0356 | 2 |
| gamma-MPOA controls (4) | 4 | -0.5 | -0.25 | 1 | 1 | 2 | 4 | -1 | 1 | 1 | 0.0037 | 0.0038 | 0.009 | 2 |
| main study: searched H200, re-scored RTX PRO 6000 (36) | 36 | 1.0 | 0.97 | 20 | 9 | 7 | 28 | -7 | 9 | 9 | 0.0003 | 0.0024 | 0.0356 | 4 |
| small pair: searched and re-scored RTX 4090 (10) | 10 | 1.0 | 1.1 | 7 | 0 | 3 | 9 | -1 | 7 | 7 | 0.0003 | 0.0002 | 0.0015 | 0 |
| method DiM | 10 | 1.0 | 1.1 | 6 | 1 | 3 | 7 | -3 | 7 | 7 | -0.0001 | -0.0003 | 0.0042 | 0 |
| method MPOA | 10 | 1.0 | 1.2 | 6 | 4 | 0 | 9 | 0 | 4 | 4 | 0.0004 | 0.0007 | 0.0045 | 1 |
| method SOM | 10 | 1.0 | 0.8 | 7 | 1 | 2 | 9 | -1 | 3 | 3 | -0.0012 | -0.0012 | 0.0053 | 0 |
| method J-space | 2 | 0.0 | 0.0 | 1 | 0 | 1 | 2 | -1 | 1 | 1 | -0.0001 | -0.0001 | 0.0007 | 0 |
| method ARA | 10 | 1.0 | 1.6 | 6 | 2 | 2 | 6 | -7 | 9 | 9 | 0.0011 | 0.008 | 0.0356 | 1 |
| method gamma-MPOA | 4 | -0.5 | -0.25 | 1 | 1 | 2 | 4 | -1 | 1 | 1 | 0.0037 | 0.0038 | 0.009 | 2 |
| all methods except ARA (36) | 36 | 1.0 | 0.83 | 21 | 7 | 8 | 31 | -3 | 7 | 7 | 0.0001 | 0.0002 | 0.009 | 3 |
| ARA, main study (8) | 8 | 1.5 | 2.0 | 5 | 2 | 1 | 4 | -7 | 9 | 9 | 0.0019 | 0.01 | 0.0356 | 1 |
| non-ARA, main study (28) | 28 | 1.0 | 0.68 | 15 | 7 | 6 | 24 | -3 | 4 | 4 | -0.0001 | 0.0002 | 0.009 | 3 |

ARA drifts most: 4/10 ARA exports move by more than 2 refusals (vs 5/36 for the other methods), and ARA has the largest KL changes (mean +0.0080, max +0.0356; other methods max |d_KL| 0.0090). The large ARA drifts are all main-study exports (searched on the H200, re-scored on the RTX PRO 6000); the two small-pair ARA exports (same 4090 for search and re-score) move by -1, +1.

## Exports whose re-scored KL exceeds 0.1, and the selection rule

Runner-up = the rule applied to the logged trials of the run without the selected trial. Drift rule = the rule with every trial's logged KL shifted by this export's d_KL (assumes equal drift for every trial; untested).

| model_display | method_display | role | sel_trial | sel_refusals | sel_kl | rescore_refusals | rescore_kl | d_kl | runnerup_trial | runnerup_refusals | runnerup_kl | drift_rule_trial | drift_rule_refusals | drift_rule_kl_logged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-27B | MPOA | primary | 99 | 74 | 0.0991 | 74 | 0.1036 | +0.0045 | 16 | 87 | 0.0835 | 16 | 87 | 0.0835 |
| Gemma-2-27B | gamma-MPOA | control | 54 | 73 | 0.0969 | 72 | 0.1059 | +0.0090 | 26 | 75 | 0.0898 | 26 | 75 | 0.0898 |
| Gemma-4-31B | ARA | primary | 37 | 28 | 0.0845 | 21 | 0.1070 | +0.0225 | 76 | 28 | 0.0921 | 66 | 45 | 0.0763 |
| OLMo-3.1-32B | gamma-MPOA | control | 34 | 82 | 0.0993 | 81 | 0.1032 | +0.0039 | 52 | 84 | 0.0948 | 52 | 84 | 0.0948 |

All 46 selected trials met KL <= 0.1 as logged. Held-out refusal, capability and teacher-forced divergence were measured on the released weights themselves (main study: downloaded back from the published repository; small pair: SHA-256-identical shards), so they describe the released models whatever the dev re-score. Only the 4 exports above lose eligibility if the re-scored KL replaced the logged KL; for the other 42 the selected trial stays within the budget on re-score (other trials were not re-scored, so their eligibility under re-measurement is unknown).

## Gemma-2-27B: SDPA vs eager re-score

| method_display | sel_refusals | sel_kl | sdpa_refusals | sdpa_kl | sdpa_d_refusals | rescore_refusals | rescore_kl | d_refusals | d_kl | merged_repo | evaluated_rev | patched_rev12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DiM | 89 | 0.0686 | 89 | 0.4036 | +0 | 90 | 0.0672 | +1 | -0.0014 | richardyoung/gemma-2-27b-it-DiM-heretic | 9ee29ab3ab16 | 3f2f3e1d2c36 |
| MPOA | 74 | 0.0991 | 52 | 0.4609 | -22 | 74 | 0.1036 | +0 | +0.0045 | richardyoung/gemma-2-27b-it-MPOA-heretic | a2c021b2e06a | 8aa25e304fdd |
| SOM | 83 | 0.0860 | 77 | 0.4912 | -6 | 86 | 0.0807 | +3 | -0.0053 | richardyoung/gemma-2-27b-it-SOM-heretic | 2a355f20c0d4 | c63892ab9b07 |
| ARA | 16 | 0.0990 | 23 | 0.4817 | +7 | 21 | 0.0996 | +5 | +0.0006 | richardyoung/gemma-2-27b-it-ARA-heretic | 29250dd15c03 | 958fda591386 |
| gamma-MPOA | 73 | 0.0969 | 63 | 0.4167 | -10 | 72 | 0.1059 | -1 | +0.0090 | richardyoung/gemma-2-27b-it-GMPOA-heretic | 78da037a4a28 | 9630b864b0fe |

- SDPA (config without `_attn_implementation`): KL 0.4036-0.4912, delta refusals -22 to +7. Eager (local config copy with `_attn_implementation: eager`, same pinned weights): KL 0.0672-0.1059, delta refusals -1 to +5. `rescore_dev_all.csv` carries the eager rows (merged path `.../gemma2_eager/models/...`; values identical to `rtx_gemma2_eager/rescore_dev.tsv`); the SDPA rows exist only in `rtx/rescore_dev.tsv`.
- Base config `google__gemma-2-27b-it@aaf20e6b9f4c/config.json` (sha256 8e560335c5f3): `_attn_implementation` = eager, transformers_version 4.42.0.dev0. Local copies of the pre-patch export configs: richardyoung__gemma-2-27b-it-ARA-heretic@29250dd15c03 sha256 6e6114858c6d `_attn_implementation` absent (transformers 5.18.0); richardyoung__gemma-2-27b-it-DiM-heretic@9ee29ab3ab16 sha256 6e6114858c6d `_attn_implementation` absent (transformers 5.18.0); richardyoung__gemma-2-27b-it-MPOA-heretic@a2c021b2e06a sha256 6e6114858c6d `_attn_implementation` absent (transformers 5.18.0); richardyoung__gemma-2-27b-it-SOM-heretic@2a355f20c0d4 sha256 6e6114858c6d `_attn_implementation` absent (transformers 5.18.0). No other base config in `models/paper33/base` sets `_attn_implementation` or attention-logit soft-capping (Gemma-4 sets only final-logit soft-capping).
- Mechanism: in transformers-5.18.0.dist-info (Heretic env; the RTX capture `pip-freeze_heretic_.venv.txt` lists the same version) `PretrainedConfig._remove_keys_not_serialized` drops `_attn_implementation_internal` when saving (string found in that function's source: True). `integrations/sdpa_attention.py` contains 'softcap' 0 times: the SDPA path ignores the soft-cap argument that `modeling_gemma2.py` passes to the attention function; the eager function applies it (the flash-attention and flex paths, not used here, also accept it: 'softcap' in flash/flex sources = 2).
- Public configs patched 2026-11-06 (`notes/gemma2_config_patch_20261106.log`; one commit per repo adding only `_attn_implementation: eager`; old -> new, 12 hex): richardyoung/gemma-2-27b-it-DiM-heretic 9ee29ab3ab16 -> 3f2f3e1d2c36; richardyoung/gemma-2-27b-it-MPOA-heretic a2c021b2e06a -> 8aa25e304fdd; richardyoung/gemma-2-27b-it-ARA-heretic 29250dd15c03 -> 958fda591386; richardyoung/gemma-2-27b-it-SOM-heretic 2a355f20c0d4 -> c63892ab9b07; richardyoung/gemma-2-27b-it-GMPOA-heretic 78da037a4a28 -> 9630b864b0fe.

## vLLM (held-out generations, benchmarks, teacher-forced divergence) is unaffected

| cell | bench_backend | vllm | backend | flash_attn | architecture | config_sha256 | tf_divergence |
|---|---|---|---|---|---|---|---|
| gemma-2-27b__base | vllm | 0.33.0 | FLASH_ATTN | 2 | Gemma2ForCausalLM | 8e560335c5f3 |  |
| gemma-2-27b__dim | vllm | 0.33.0 | FLASH_ATTN | 2 | Gemma2ForCausalLM | 6e6114858c6d | 0.0678 |
| gemma-2-27b__mpoa | vllm | 0.33.0 | FLASH_ATTN | 2 | Gemma2ForCausalLM | 6e6114858c6d | 0.0626 |
| gemma-2-27b__ara | vllm | 0.33.0 | FLASH_ATTN | 2 | Gemma2ForCausalLM | 6e6114858c6d | 0.1612 |
| gemma-2-27b__som | vllm | 0.33.0 | FLASH_ATTN | 2 | Gemma2ForCausalLM | 6e6114858c6d | 0.0313 |
| gemma-2-27b__gmpoa | vllm | 0.33.0 | FLASH_ATTN | 2 | Gemma2ForCausalLM | 6e6114858c6d | 0.0964 |

- All six Gemma-2 vLLM runs (original and five exports) used vLLM 0.33.0 with the same attention backend (FLASH_ATTN, FlashAttention 2) and architecture Gemma2ForCausalLM. The five exports were benchmarked from configs with sha256 6e6114858c6d = the pre-patch export config (no `_attn_implementation`).
- vLLM source in the bench environment (vllm-0.33.0.dist-info; `runs/mc-rtx-1/scripts/bench/.venv/.../vllm/model_executor/models/gemma2.py`): 106: attn_logits_soft_cap: float | None = None,; 165: logits_soft_cap=attn_logits_soft_cap,; 202: attn_logits_soft_cap=config.attn_logit_softcapping,; 343: config.vocab_size, soft_cap=config.final_logit_softcapping. `_attn_implementation` occurs 0 times in that file: vLLM takes the soft-cap values from `attn_logit_softcapping` / `final_logit_softcapping` in config.json (preserved in the exports) and passes them to its own attention backend; the transformers attention-implementation key is not read. The FLASH_ATTN backend (`vllm/v1/attention/backends/flash_attn.py`) passes `softcap=self.logits_soft_cap` to the kernel in 5 call sites.
- Empirical check: teacher-forced divergence (vLLM, 350 benign held-out prompts) of the four primary Gemma-2 exports 0.0313-0.1612 nats/token vs 0.0039-0.2790 for the other 38 primary cells (inside their range); the Gemma-2 capability changes are -0.50 to +0.21 pp (mean of five benchmarks; other 38 primary cells -4.21 to +0.86 pp). This is consistent with, but on its own does not prove, intact soft-capping; the basis is the source and backend lines above.

## Notes

- The per-prompt dev outcomes of the search-time trial are not stored (the Optuna journal keeps the count and KL only), so no per-prompt paired test (McNemar, Newcombe) is possible; the counts are compared across exports with the Wilcoxon signed-rank and Spearman helpers of final.py.
- Differences between the trial and the export combine: merged bf16 weights W+BA vs the adapter applied at run time; GPU and kernel differences for the 36 main-study exports (H200 search, RTX PRO 6000 re-score); and greedy decoding on borderline prompts. They are not separated here.
