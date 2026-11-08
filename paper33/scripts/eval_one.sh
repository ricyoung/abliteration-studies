#!/usr/bin/env bash
# Run ON a GPU box. Full evaluation of ONE model (paper 33, plan v2).
# Usage: eval_one.sh <label> <base_repo> [run_name] [--no-publish]
#   no run_name -> original model: capability benchmarks + held-out refusal set (+ its own answers as the
#                  teacher-forcing reference for every derived model)
#   run_name    -> select the trial (fewest keyword refusals with KL <= 0.1, select_trial.py), re-export it as a
#                  merged model -- via heretic's UPLOAD action (public HF repo + reproduce manifest -> Grimoire)
#                  unless --no-publish -- then benchmarks + held-out refusal set + teacher-forced divergence.
# Results: $W/results/bench/<label>/  (summary.json, lmeval/, refusal/responses.jsonl, refusal/teacher_forced.jsonl)
set -uo pipefail
source /workspace/paper33/env.sh
label=$1; repo=$2; run=${3:-}; publish=1
for x in "$@"; do [ "$x" = "--no-publish" ] && publish=0; done
[ "$run" = "--no-publish" ] && run=""
HF_USER=richardyoung
OUT=$W/results/bench/$label; mkdir -p "$OUT"
SETS=$W/scripts/bench/refusal_sets.json
base_label=$(echo "$label" | sed -E 's/__.*//')__base
REF=$W/results/bench/$base_label/refusal/responses.jsonl

export BENCH_VLLM_EXTRA='{}'
case $repo in
  google/gemma-4*) export BENCH_VLLM_EXTRA='{"limit_mm_per_prompt": {"image": 0}}';;
  # hybrid Mamba models: 1 Mamba cache block per seq (default max_num_seqs 1024 > 489 blocks)     
  Qwen/Qwen3.8*|Qwen/Qwen3.5*) export BENCH_VLLM_EXTRA='{"max_num_seqs": 256}';;
  mistralai/*|unsloth/Mistral-Small-3.2*) export BENCH_VLLM_EXTRA='{"tokenizer_mode": "hf", "config_format": "hf", "load_format": "safetensors"}';;
  # OLMo-2 was trained with 4096 positions; keep the protocol's max_model_len 8192 (same as every model) via vLLM's override
  allenai/OLMo-2*) export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1;;
esac

if [ -z "$run" ]; then
  MODEL=$repo
  if [ -f "$OUT/summary.json" ] && python3 -c "import json,sys; b=json.load(open('$OUT/summary.json'))['benchmarks']; sys.exit(0 if len([k for k,v in b.items() if 'score' in v])==5 else 1)"; then
    # capability benchmarks already done (before plan v2): held-out refusal set + self reference only
    bash "$W/scripts/bench/run.sh" "$MODEL" "$OUT/refusal_run" --limit-mode paper --method "$label" --base-model "$repo" \
         --benchmarks none --no-hash --refusal-sets "$SETS" --reference-responses self
    rc=$?; [ $rc -eq 0 ] && ln -sfn refusal_run/refusal "$OUT/refusal"; exit $rc
  fi
  bash "$W/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$repo" \
       --refusal-sets "$SETS" --reference-responses self
  exit $?
fi

[ -s "$REF" ] || { echo "missing original-model reference answers: $REF (run $base_label first)"; exit 1; }
R=$W/runs/$run
"$W/heretic/.venv/bin/python" "$W/scripts/select_trial.py" "$R" --emit-command > "$OUT/selection_emit.txt" 2>&1 || { echo "selector failed"; exit 1; }
cp "$R/selection.json" "$OUT/" 2>/dev/null
sed -n '/^cd \/workspace/,$p' "$OUT/selection_emit.txt" > "$OUT/reexport.sh"
[ -s "$OUT/reexport.sh" ] || { echo "no trial selected (see selection_emit.txt)"; exit 1; }

# plugin methods need their pinned fork checkout: SOM / precomputed (J-space) -> paper33-plugins (heretic-p33);
# gamma-corrected MPOA -> paper33-gamma (heretic-gamma)
PLUG=""
if grep -q "heretic.modifiers.gamma_mpoa" "$R/config.toml"; then PLUG=/workspace/paper33/heretic-gamma/src
elif grep -q -E "heretic.modifiers.(som|precomputed)" "$R/config.toml"; then PLUG=/workspace/paper33/heretic-p33/src; fi
if [ -n "$PLUG" ]; then
  sed -i "s#/workspace/paper33/heretic/.venv/bin/heretic #PYTHONPATH=$PLUG /workspace/paper33/heretic/.venv/bin/python -c 'import sys; from heretic.main import main; sys.exit(main())' #" "$OUT/reexport.sh"
  git -C "${PLUG%/src}" rev-parse HEAD > "$OUT/plugin_commit.txt"
fi

method=$(echo "$label" | sed -E 's/.*__//' | tr a-z A-Z | sed -e 's/^DIM$/DiM/' -e 's/^JSPACE$/JSpace/')
name=$(basename "$repo")-$method-heretic
if [ $publish -eq 1 ]; then
  repro=full; case $method in SOM|JSpace|GMPOA*) repro=none;; esac   # fork-only plugins: upstream Heretic cannot reproduce
  sed -i -e "s#--patch-journal \(.*\) --trial-index#--patch-journal \1 --upload --upload-repro $repro --trial-index#" \
         -e "s#--model-action save --export-strategy MERGE --save-directory [^ ]*#--model-action upload --export-strategy MERGE --upload-repo-id $HF_USER/$name#" "$OUT/reexport.sh"
  grep -q -- "--upload-repo-id" "$OUT/reexport.sh" || { echo "publish rewrite failed"; exit 1; }
fi

echo "=== re-export start $(date -Is) publish=$publish name=$name"; s=$(date +%s)
bash "$OUT/reexport.sh" > "$OUT/reexport.log" 2>&1 || { echo "re-export failed"; tail -30 "$OUT/reexport.log"; exit 1; }
echo "=== re-export done $(( $(date +%s)-s ))s"

if [ $publish -eq 1 ]; then
  MODEL=$W/tmpmodels/$label; rm -rf "$MODEL"
  hf download "$HF_USER/$name" --local-dir "$MODEL" > "$OUT/download.log" 2>&1 || { echo "download back failed"; exit 1; }
  echo "$HF_USER/$name" > "$OUT/hf_repo.txt"
  git -C "$W/heretic-p33" rev-parse HEAD > "$OUT/heretic_p33_commit.txt"
else
  MODEL=$(ls -d "$R"/model_kl0.1_t* | head -1)
fi
[ -f "$MODEL/config.json" ] || { echo "merged model missing: $MODEL"; exit 1; }

bash "$W/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$repo" \
     --refusal-sets "$SETS" --reference-responses "$REF"
rc=$?
[ $rc -eq 0 ] && rm -rf "$MODEL"
exit $rc
