#!/usr/bin/env bash
# 4090 (2026-11-06): evaluate an external-tool merged model (OBLITERATUS / jwest33, /data/heretic_2/models/external/<label>)
# with the paper-33 protocol (same as eval_1x_local.sh minus the adapter rebuild). Never deletes the model.
# Usage: eval_ext_local.sh <label> <model_dir> <base_repo> <base_label>
set -uo pipefail
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder
export HF_HOME=${HF_HOME:-/data/huggingface} PYTHONUNBUFFERED=1 BENCH_GPU_UTIL=${BENCH_GPU_UTIL:-0.72} BENCH_VLLM_EXTRA='{}'
label=$1; MODEL=$2; base_repo=$3; base_label=$4
OUT=$P/results/bench-4090/$label; mkdir -p "$OUT"
REF=$P/results/bench-4090/$base_label/refusal/responses.jsonl
[ -s "$REF" ] || { echo "missing original-model reference answers: $REF"; exit 1; }
[ -s "$MODEL/PROVENANCE.json" ] || { echo "no PROVENANCE.json in $MODEL"; exit 1; }
cp "$MODEL/PROVENANCE.json" "$OUT/push_manifest.json"
/data/heretic_2/heretic/.venv/bin/python $P/scripts/tensor_hash.py "$MODEL" > "$OUT/tensor_hashes.json" || exit 1
bash "$P/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$base_repo" \
     --refusal-sets "$P/scripts/bench/refusal_sets.json" --reference-responses "$REF"
