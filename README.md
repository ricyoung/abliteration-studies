# Abliteration studies: code, run records and labels

This repository holds the code and data of two studies on abliteration, the training-free removal of refusal behaviour from open-weight language models:

- **Paper 33: "Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors"** (`paper33/`). Asks whether refusal is harder to remove from newer model releases. It tests three direction-based methods (DiM, MPOA, SOM; plus J-space on the small pair) and ARA on four model families (Qwen, Gemma, OLMo, Mistral Small) and one small Qwen pair.
- **Paper 30: "The Selection Rule Can Reverse LLM Abliteration Tool Comparisons: A Case Study of Heretic 1.x and 2.0 on Qwen2.5"** (`paper30/`). Compares Heretic 1.x and Heretic 2.0 with community releases and two external tools (OBLITERATUS, jwest33/abliterator) on Qwen2.5 models. It includes a re-export of a 1.x study at the 2.0 selection rule.

Author: Richard J. Young, Department of Information Systems, University of Nevada, Las Vegas.

## What is here

| Path | Contents |
|---|---|
| `paper33/scripts/bench/` | Evaluation harness. It generates held-out responses with vLLM, runs the five capability benchmarks with patched lm-eval, scores refusal with the judges (`score_refusals.py`, `score_refusals_256.py`) and keeps the judge templates. |
| `paper33/scripts/analysis/` | Every statistical analysis in paper 33. Run `run_all.sh` (see its header for the order); the outputs go to `results/analysis/`. `postnorm_gain_scan.py` (Appendix D.1) is run separately: it reads the post-norm gain weights of the four post-norm base models from Hugging Face, so it needs access to those models. |
| `paper33/scripts/figures/` | Figure scripts and styles. |
| `paper33/scripts/*.py, *.sh` | Trial selection, tensor hashing, the dev-set re-score of released exports, and the Heretic 1.x rebuild. |
| `paper33/configs/` | Search configurations (`core_*.toml`). |
| `paper33/runs/<host>/<run>/` | Per run: the configuration as used, `selection.json`, the run log (`output.txt`) and the Optuna study journal (`optuna_journal/*.jsonl`). |
| `paper33/results/analysis/` | All analysis outputs, including the per-response label tables (`responses_judged.csv.gz`). These contain judge, keyword and degeneracy labels and prompt hashes, with no text. |
| `paper33/results/bench/<host>/<model>/` | Capability and refusal summaries, model hashes (SHA-256 per shard) and the Hugging Face repository of each evaluated model. |
| `paper33/results/rescore_dev/` | Dev-set re-score of the released exports. |
| `paper33/data/refusal_sets_meta.json` | The 1,322 held-out items: set, kind, source, AdvBench-overlap flag and the SHA-256 of each prompt. The prompt text is not included. |
| `paper33/env/` | Package lists, toolchain and GPU details of each machine. |
| `paper33/paper/`, `paper30/paper/` | LaTeX sources, tables and figures. |
| `paper30/analysis/` | `assemble_p30.py` and its outputs (`RESULTS.md`, cells, tests, label tables without text). |
| `paper30/scripts/` | Tables, figures and supplementary statistics for paper 30. |
| `paper30/reexport_1x/` | The Heretic 1.x selection-cap re-export: script, version notes, trial records, validation records, provenance and log. |
| `paper30/hf_release/` | Release of the 13 Heretic 1.x models to Hugging Face: per-tensor verification against the evaluated models, the shard-by-shard merge used to rebuild the KL <= 0.1 re-export, card generation, adapter upload, and the upload log. |
| `paper30/external_tools/` | Run scripts and `PROVENANCE.json` (tool commit, preset, runtime patches, base revision, seed, weight hashes) for each OBLITERATUS and jwest33 run. |

## Not included

- **Responses to harmful prompts.** They are kept privately and used only for scoring.
- **Held-out prompt text.** Only identifiers and hashes are included. The public sets can be rebuilt from their sources, listed in `refusal_sets_meta.json`. The code-as-weapon items come from the author's Code-as-Weapon Bank on Hugging Face.
- **Model weights.** The models are on Hugging Face under [richardyoung](https://huggingface.co/richardyoung). Paper 33's models are named `<base>-<METHOD>-heretic`; the SHA-256 of every shard is in `paper33/results/bench/*/model_hash.json`. Paper 30's Heretic 1.x models are named `<base>-heretic1x-<general|code>`, plus `Qwen2.5-32B-Instruct-heretic1x-general-kl0.1`; each repository holds its LoRA adapter and the SHA-256 of every tensor (`tensor_hashes.json`).

## Related code

- Heretic fork with the paper-33 method plug-ins: <https://github.com/ricyoung/heretic> (branches `paper33-plugins` and `paper33-gamma`), under AGPL-3.0.
- Heretic 1.x: heretic-llm 1.4.0, git tag `v1.4.0` (`6ea3b8d`) of upstream Heretic.

## Licence

The author's own code, run records, labels and paper sources are released under the **OpenRAIL++ Licence**, version 1.0 (see `LICENSE`). It carries use-based restrictions. Released models keep the licence and use policy of their base model. Where OpenRAIL++ is also attached to a model, it applies in addition to the base licence.

## Citation

Please cite the papers.

- Paper 33: [arXiv:2611.17846](https://arxiv.org/abs/2611.17846).
- Paper 30: arXiv identifier to be added.
