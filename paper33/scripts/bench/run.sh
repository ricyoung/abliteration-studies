#!/usr/bin/env bash
# Capability benchmark runner (paper 33).
# Usage: run.sh <model_path_or_hf_id> <out_dir> [--limit-mode paper|smoke] [bench.py options]
#   bench.py options: --method M --base-model ID --benchmarks a,b --gpu-util F
#                     --max-model-len N --trust-remote-code --no-hash --backend vllm|hf
# Backend: vLLM; if the model fails to LOAD in vLLM (exit 3) it is retried once with HF transformers.
# Runs model-generated code (HumanEval/MBPP): use only on the rented GPU box / the GPU workstation.
set -uo pipefail
if [ $# -lt 2 ]; then sed -n 2,7p "$0"; exit 2; fi
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$1"; OUT="$2"; shift 2
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
exec > >(tee -a "$OUT/bench.log") 2>&1
echo "=== run.sh $(date -Is) host=$(hostname) model=$MODEL out=$OUT args=$*"

export PATH="$HOME/.local/bin:$PATH"
if [ ! -x "$DIR/.venv/bin/python" ]; then
  (cd "$DIR" && uv sync --frozen) || { echo "uv sync failed"; exit 1; }
fi
export HF_ALLOW_CODE_EVAL=1          # required by lm-eval code_eval (humaneval/mbpp)
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc at first use; rented boxes have no CUDA
# toolkit. Greedy decoding never needs it, so use vLLM's PyTorch sampler (identical argmax).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader || true

# default backend vllm unless the caller passed --backend
BACKEND_ARGS=(--backend vllm)
for a in "$@"; do [ "$a" = "--backend" ] && BACKEND_ARGS=(); done

"$DIR/.venv/bin/python" "$DIR/bench.py" "$MODEL" "$OUT" "${BACKEND_ARGS[@]}" "$@"
rc=$?
if [ $rc -eq 3 ] && [ ${#BACKEND_ARGS[@]} -gt 0 ]; then
  echo "=== vLLM could not load the model; falling back to HF transformers backend"
  mv "$OUT/summary.json" "$OUT/summary.vllm_failed.json" 2>/dev/null
  "$DIR/.venv/bin/python" "$DIR/bench.py" "$MODEL" "$OUT" --backend hf "$@"
  rc=$?
fi
echo "=== run.sh exit $rc $(date -Is)"
exit $rc
