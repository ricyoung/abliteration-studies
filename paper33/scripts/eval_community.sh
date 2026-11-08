#!/usr/bin/env bash
# Run ON a GPU box. Evaluate a COMMUNITY refusal-removed model (external HF repo) with the identical paper-33 protocol:
# 5 capability benchmarks + held-out refusal set + teacher forcing against the ORIGINAL model's own answers.
# Usage: eval_community.sh <label> <community_repo> <base_repo> <base_label>
#   label       e.g. qwen3.8-27b__community-hereticorg-ara  (results/bench/<label>/)
#   base_label  e.g. qwen3.8-27b__base  (reference answers: results/bench/<base_label>/refusal/responses.jsonl,
#               or results/ref_import/<base_label>/responses.jsonl when the original was evaluated on another box)
set -uo pipefail
source /workspace/paper33/env.sh
label=$1; repo=$2; base_repo=$3; base_label=$4
OUT=$W/results/bench/$label; mkdir -p "$OUT"
SETS=$W/scripts/bench/refusal_sets.json
REF=$W/results/bench/$base_label/refusal/responses.jsonl
[ -s "$REF" ] || REF=$W/results/ref_import/$base_label/responses.jsonl
[ -s "$REF" ] || { echo "missing original-model reference answers for $base_label"; exit 1; }

# vLLM settings follow the ORIGINAL model's family (identical to eval_one.sh)
export BENCH_VLLM_EXTRA='{}'
case $base_repo in
  google/gemma-4*) export BENCH_VLLM_EXTRA='{"limit_mm_per_prompt": {"image": 0}}';;
  Qwen/Qwen3.8*|Qwen/Qwen3.5*) export BENCH_VLLM_EXTRA='{"max_num_seqs": 256}';;
  mistralai/*|unsloth/Mistral-Small-3.2*) export BENCH_VLLM_EXTRA='{"tokenizer_mode": "hf", "config_format": "hf", "load_format": "safetensors"}';;
  allenai/OLMo-2*) export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1;;
esac
"$W/scripts/bench/.venv/bin/python" - "$repo" > "$OUT/community_repo.json" <<'EOF'
import json, sys
from huggingface_hub import HfApi
i = HfApi().model_info(sys.argv[1], files_metadata=False)
print(json.dumps({"repo": sys.argv[1], "sha": i.sha, "tags": i.tags, "files": sorted(s.rfilename for s in i.siblings)}, indent=1))
EOF
rev=$(python3 -c "import json; print(json.load(open('$OUT/community_repo.json'))['sha'])")
echo "community repo $repo @ $rev (reference: $REF)"
MODEL=$W/tmpmodels/$label; rm -rf "$MODEL"
hf download "$repo" --revision "$rev" --local-dir "$MODEL" > "$OUT/download.log" 2>&1 || { echo "download failed"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "no config.json in $repo"; exit 1; }
bash "$W/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$base_repo" \
     --refusal-sets "$SETS" --reference-responses "$REF"
rc=$?
rm -rf "$MODEL"   # always: external weights are re-downloadable at the pinned revision; keeps the box disk safe
exit $rc
