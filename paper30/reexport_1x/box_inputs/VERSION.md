# Which Heretic 1.x produced the paper-32 studies?

**Answer: heretic-llm 1.4.0 = git tag `v1.4.0` = commit `6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d`
(p-e-w/heretic, "build: bump version to 1.4.0", 2026-06-14). The PyPI wheel `heretic_llm-1.4.0-py3-none-any.whl`
(sha256 `81d54d12f63620099a2f18063185967c5365a9af85c0e49e169d060a9ceb8fd7`, uploaded 2026-06-14 11:16 UTC) is
byte-identical to `src/heretic/*` at that tag (all 10 modules compared with `cmp`).**

Checkout used here: `heretic-1x/` = `git -C /data/heretic_2/heretic archive v1.4.0` (no branch of that repo was
changed). It is installed (`--no-deps`) into a venv whose other packages are pinned to v1.4.0's own `uv.lock`
(`requirements-1x-lock.txt`: transformers 5.6.2, peft 0.19.1, accelerate 1.13.0, optuna 4.8.0, datasets 4.8.4, ...);
only torch is taken separately (see run_on_rtx.sh).

## Evidence (all from the archived studies in /data/heretic_2/results/heretic/, June 19-21 2026)

1. **Settings field set and order.** Every 1.x journal stores `settings` = `Settings.model_dump_json()` (1.4.0
   `main.py` l.670) with exactly these 27 keys, in this order: `model, model_commit, dtypes, quantization, device_map,
   max_memory, offload_outputs_to_cpu, batch_size, max_response_length, response_prefix, kl_divergence_scale,
   kl_divergence_target, orthogonalize_direction, row_normalization, full_normalization_lora_rank,
   winsorization_quantile, n_trials, n_startup_trials, seed, export_strategy, max_shard_size, refusal_markers,
   system_prompt, good_prompts, bad_prompts, good_evaluation_prompts, bad_evaluation_prompts`, and dataset specs with
   `dataset, commit, split, column, prefix, suffix, system_prompt`. Parsing `class Settings` /
   `DatasetSpecification` (non-`exclude=True` fields) at every commit that touched `src/heretic/config.py` on all
   branches (`git log --all`), the exact key list *and order* occurs only for the config.py blob `7bc8a4d6`, introduced by
   `6757ada` (2026-06-13) and unchanged through `v1.4.0`, `b186d6c`, `554a58a`, `00185db` until `0146b27` (2026-06-27,
   headless operation). Older revisions (`b9d5b45`, `2fd163f`, 06-07/06-11) have the same set in a different order;
   everything before 06-07 lacks `export_strategy`. `export_1x_trial.py` re-checks this at run time: the 1.4.0
   `Settings.model_validate_json(journal).model_dump_json()` round-trips to the journal JSON exactly.
2. **Search space excludes later master.** All 1.x journals record `mlp.down_proj.max_weight` with
   `FloatDistribution(low=0.8, high=1.5)`. Commit `00185db` (2026-06-18 08:14 UTC, "let the optimizer disable MLP
   ablation via a 0 max_weight floor") changed that lower bound to -0.25. The studies ran 2026-06-19 20:49 to
   2026-06-21 08:23 (journal timestamps), i.e. after `00185db` was on master, so they were **not** run from a master
   checkout but from the 1.4.0 release. The only master commits between `v1.4.0` and `00185db` are `b186d6c` (aiohttp lock
   bump) and `554a58a` (trial count in the interactive "run additional trials" path) - functionally identical for these
   runs.
3. **Trial user attrs.** Each trial stores `index, direction_index, parameters, kl_divergence, refusals,
   base_refusals, n_bad_prompts` = exactly what 1.4.0 `objective()` writes (l.547-644; `n_bad_prompts` exists since
   `2fd163f`, 06-11). Objective values = `(kld_score, refusals/base_refusals)` as in 1.4.0 `Evaluator.get_score`.
4. **Journal format / naming.** Optuna `JournalFileBackend` JSONL, study name `heretic`, file name
   `models--Qwen__Qwen2--5-32B-Instruct.jsonl` = 1.4.0's checkpoint naming (non-alnum chars of `settings.model`
   `models/Qwen__Qwen2.5-32B-Instruct` -> `--`, l.290-296). `result.json`'s `config: configs/heretic_general.toml` and
   `model: models/...` come from the paper-32 driver `paper32/scripts/heretic_batch.py`, which monkeypatches
   `heretic.main.prompt_select/prompt_path/prompt_text` (names imported into 1.4.0 `main.py` from `.utils`) and passes
   `--export-strategy ADAPTER --residual-plot-path --study-checkpoint-dir --seed 0`.
5. **Library versions.** The adapters record `peft_version 0.19.1` (= v1.4.0 `uv.lock`; requirement `peft~=0.19`),
   LoRA rank 1, `lora_alpha 1`, target `o_proj, down_proj`, fp32 LoRA weights - as 1.4.0 `_apply_lora`.
6. **Contemporary note.** `from_2070/4090-HERETIC-GUIDE.md`: "Use the standard PyPI release, Heretic 1.4.0 (Ric's
   choice; his June uploads used it ...)".

## Settings that are not in the journal but matter
`Settings` fields marked `exclude=True` are not stored. From `paper32/configs/heretic_general.toml` /
`heretic_code_specific.toml`: `batch_size = 0` (auto -> 64, stored), `print_residual_geometry = true`,
`plot_residuals = true`. The last two make 1.4.0 take the `needs_full_residuals` branch (residuals of all prompts,
fp32 `.mean(dim=0)`) instead of `get_residuals_mean` (fp64 running sum) - `export_1x_trial.py --residual-mode batched`
(default). `response_prefix` was auto-detected as `""` and stored, so no prefix check is repeated.

## Inputs not in /data
The *code* studies use local text files `configs/code_abliteration_data/{benign_code_prompts,malicious_code_prompts,
benign_code_eval,malicious_code_eval}.txt` (relative to the paper-32 repo,
`/Volumes/data_bolt/_github_papers/32_2604_malicious_code_generation_safety/` on the older Mac, generated by its
`scripts/prepare_code_abliteration_data.py`). They were never copied to /data (only `configs/heretic_*.toml` and three
scripts were). Without them the code study's refusal directions cannot be recomputed. Their correctness is checked by
re-exporting the code study's originally selected trial (#179) and comparing with its original adapter.
