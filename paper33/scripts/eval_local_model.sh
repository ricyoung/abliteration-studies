#!/usr/bin/env bash
# Run ON a GPU box. Evaluate a refusal-removed model whose weights were PUSHED to the box (e.g. the Heretic 1.x merged models
# from /data/heretic_2/models/merged_models, paper-30 update) with the identical paper-33 protocol; the original model's own
# answers are the teacher-forcing reference. Weights must carry the .pushed manifest (sha256 per shard, verified on arrival).
# Usage: eval_local_model.sh <label> <model_dir> <base_repo> <base_label>
set -uo pipefail
source /workspace/paper33/env.sh
label=$1; MODEL=$2; base_repo=$3; base_label=$4
OUT=$W/results/bench/$label; mkdir -p "$OUT"
SETS=$W/scripts/bench/refusal_sets.json
REF=$W/results/bench/$base_label/refusal/responses.jsonl
[ -s "$REF" ] || { echo "missing original-model reference answers: $REF"; exit 1; }
[ -s "$MODEL/.pushed" ] || { echo "model not (verified) pushed: $MODEL"; exit 1; }
cp "$MODEL/.pushed" "$OUT/push_manifest.json"
export BENCH_VLLM_EXTRA='{}'
bash "$W/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$base_repo" \
     --refusal-sets "$SETS" --reference-responses "$REF"
rc=$?
[ $rc -eq 0 ] && rm -rf "$MODEL"
exit $rc
