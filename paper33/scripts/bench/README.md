# bench — capability benchmarks for paper 33

Measures how much a modified model's general ability drops relative to its base model.
Separate uv project (does not use heretic's venv). Canonical copy: this folder. Box copies are disposable.

```
bash run.sh <model_path_or_hf_id> <out_dir> [--limit-mode paper|smoke] \
     [--method M2] [--base-model Qwen/Qwen2.5-7B-Instruct] [--gpu-util 0.85] \
     [--benchmarks mmlu_pro,gsm8k,ifeval,humaneval,mbpp] [--max-model-len 8192] \
     [--trust-remote-code] [--no-hash] [--backend vllm|hf]
python summarize.py runs/ -o capability.csv      # stdlib only; any python3
```

The first run does `uv sync --frozen` (needs uv). Code tasks execute model-written Python
(`HF_ALLOW_CODE_EVAL=1`, set by run.sh): run only on the GPU box / GPU workstation.

## Protocol (PROTOCOL_VERSION 1, in bench.py)
| benchmark | lm-eval task(s) | n paper / smoke | shots | primary metric | max_gen_toks |
|---|---|---|---|---|---|
| MMLU-Pro | `mmlu_pro_<14 subjects>` | 1000 (stratified by subject) / 20 | 5 CoT | exact_match,custom-extract (micro-avg over the 1000) | 2048 |
| GSM8K | `gsm8k` | 500 random of 1319 / 20 | 5 | exact_match,flexible-extract (strict also kept) | 512 |
| IFEval | `ifeval` | 541 (full) / 20 | 0 | prompt_level_strict_acc (+inst/loose) | 1280 |
| HumanEval | `humaneval_instruct` | 164 (full) / 20 | 0 | pass@1 | 1024 |
| MBPP | `mbpp_instruct` | 500 test (full) / 20 | 3 | pass_at_1 | 1024 |

* Subsets: `subsets.json`, made once by `make_subsets.py` (seed 20261104). Indices are positions in
  lm-eval `task.eval_docs`; stored doc ids (question_id / task_id / key / question hash) are re-checked
  before every run, so a changed dataset revision aborts instead of silently shifting the subset.
  Do not regenerate subsets.json mid-study.
* Decoding: greedy (`do_sample=False, temperature=0`), same max_gen_toks for every model.
* Chat template: **on for every model** (`apply_chat_template=True`, few-shot examples as multi-turn
  chat, `fewshot_as_multiturn=True`), using the model's own tokenizer template. Abliterated/ARA models
  keep their base model's template, so base and modified see byte-identical prompts.
  **Thinking disabled** (`enable_thinking=False` passed to the template) for every model: we measure
  non-reasoning ability, and thinking traces would blow the fixed token budgets. Templates without
  that variable ignore it. gpt-oss (harmony format) does not have a no-reasoning switch: compare
  it only against its own base and note it in the paper.
* Seeds: lm-eval random 0, numpy/torch/fewshot 1234, vLLM 1234.
* Backend: vLLM (bf16 = `dtype=auto`, `max_model_len` 8192, batch auto). If vLLM cannot *load* the model
  (bench.py exit 3) run.sh reruns once with HF transformers (bf16, batch 16). vLLM and HF outputs are not
  bit-identical; `backend` is recorded in summary.json — compare base and modified on the same backend.

## Outputs (out_dir)
* `bench.log` — everything (stdout+stderr of run.sh, bench.py, vLLM)
* `summary.json` — per benchmark: metric, score, stderr, n, wall_seconds, max_gen_toks; plus versions
  (lm_eval, vllm, torch, CUDA, driver, GPU, sha256 of uv.lock/bench.py/subsets.json), seeds, decoding,
  model info (HF commit or local path + git commit) and `tree_sha256` of the weight/config/tokenizer files.
  Written after every benchmark, so partial runs keep their results.
* `model_hash.json` — sha256 per file
* `lmeval/<benchmark>/results.json` (lm-eval results + configs) and `samples_<task>.jsonl` (per-sample prompts, outputs, scores)

Env overrides: `BENCH_GPU_UTIL`, `BENCH_MAX_MODEL_LEN`, `BENCH_HF_BATCH`,
`BENCH_VLLM_EXTRA` (JSON of extra vLLM `LLM()` kwargs, e.g. `{"limit_mm_per_prompt": {"image": 0}}`
for multimodal checkpoints, `{"enforce_eager": true}`).

## Machines
* H200 NVL (driver 580, CUDA 13.0): default `--gpu-util 0.85`. On a shared box use less (0.35 in smoke tests).
* RTX 4090 24 GB (driver 615, CUDA 13.4): same lock works (torch 2.14 cu130 wheels; newer driver is
  backward compatible). Free the GPU first (ComfyUI ~11 GB, Ollama, voxtype ~1.8 GB). For ≤9B bf16
  use `--gpu-util 0.9`; vLLM needs that fraction *free* at start. Qwen3.5-9B (~19 GB weights) leaves
  ~2 GB of KV cache: use `--max-model-len 4096` or let it fall back to `--backend hf` — record which.
  Expect roughly 3–5× the H200 wall time.

## Known issues / status (2026-11-04)
* vLLM 0.33's FlashInfer top-k/top-p sampler JIT-compiles with nvcc on first use; boxes without a CUDA
  toolkit fail at engine start ("Could not find nvcc"). run.sh sets `VLLM_USE_FLASHINFER_SAMPLER=0`
  (PyTorch sampler; greedy argmax is the same). Recorded under `versions.env` in summary.json.
* Fixed: in smoke mode lm-eval runs *all* docs of a task that is missing from `samples=`; bench.py now
  passes only the sampled MMLU-Pro subjects.
* Verified: vLLM path end to end on the RTX 4090 (Qwen3-1.7B, smoke, mmlu_pro+gsm8k+ifeval, 131 s incl.
  77 s load). The H200 vLLM smoke test on Qwen2.5-7B-Instruct and the code tasks are still to be run.
* HF fallback is much slower (~2.4 s per generate request at batch 16 on the H200, 7B): about 3 h for the
  MMLU-Pro part alone in paper mode. Treat it as a last resort.
