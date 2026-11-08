#!/usr/bin/env bash
# LOCAL (4090 workstation) twin of eval_one.sh for the small pair (Qwen2.5-7B-Instruct <-> Qwen3.5-4B).
# Usage: eval_one_local.sh <label> <base_repo> [run_name] [--no-publish]
# NOTE: this machine's HF token cannot create richardyoung repos (403) -> run with --no-publish; upload later.
# Same protocol as eval_one.sh (selection rule, held-out refusal set, teacher forcing, 5 benchmarks,
# heretic UPLOAD re-export unless --no-publish); only paths differ. Results: results/bench-4090/<label>/
set -uo pipefail
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder
HV=/data/heretic_2/heretic/.venv/bin          # heretic checkout on branch paper33-plugins (built-ins == upstream 0db5c31)
export HF_HOME=${HF_HOME:-/data/huggingface} PYTHONUNBUFFERED=1
export BENCH_GPU_UTIL=${BENCH_GPU_UTIL:-0.72} # desktop apps hold ~5 GB of the 24 GB card
label=$1; repo=$2; run=${3:-}; publish=1
for x in "$@"; do [ "$x" = "--no-publish" ] && publish=0; done
[ "$run" = "--no-publish" ] && run=""
HF_USER=richardyoung
OUT=$P/results/bench-4090/$label; mkdir -p "$OUT"
SETS=$P/scripts/bench/refusal_sets.json
base_label=$(echo "$label" | sed -E 's/__.*//')__base
REF=$P/results/bench-4090/$base_label/refusal/responses.jsonl

export BENCH_VLLM_EXTRA='{}'
case $repo in
  Qwen/Qwen3.8*|Qwen/Qwen3.5*) export BENCH_VLLM_EXTRA='{"max_num_seqs": 256}';;
esac

if [ -z "$run" ]; then
  bash "$P/scripts/bench/run.sh" "$repo" "$OUT" --limit-mode paper --method "$label" --base-model "$repo" \
       --refusal-sets "$SETS" --reference-responses self
  exit $?
fi

[ -s "$REF" ] || { echo "missing original-model reference answers: $REF (run $base_label first)"; exit 1; }
R=$P/runs/local-4090/$run
"$HV/python" "$P/scripts/select_trial.py" "$R" --emit-command > "$OUT/selection_emit.txt" 2>&1 || { echo "selector failed"; exit 1; }
cp "$R/selection.json" "$OUT/" 2>/dev/null
# emitted command uses the box layout -> translate to this machine
sed -n '/^cd \/workspace/,$p' "$OUT/selection_emit.txt" | sed \
    -e "s#/workspace/paper33/runs/#$P/runs/local-4090/#g" \
    -e "s#/workspace/paper33/heretic/.venv/bin/#$HV/#g" \
    -e "s#/workspace/paper33/scripts/#$P/scripts/#g" > "$OUT/reexport.sh"
[ -s "$OUT/reexport.sh" ] || { echo "no trial selected (see selection_emit.txt)"; exit 1; }
grep -q /workspace "$OUT/reexport.sh" && { echo "untranslated box path in reexport.sh"; exit 1; }

method=$(echo "$label" | sed -E 's/.*__//' | tr a-z A-Z | sed -e 's/^DIM$/DiM/' -e 's/^JSPACE$/JSpace/')
name=$(basename "$repo")-$method-heretic
if [ $publish -eq 1 ]; then
  sed -i -e 's#--patch-journal \(.*\) --trial-index#--patch-journal \1 --upload --trial-index#' \
         -e "s#--model-action save --export-strategy MERGE --save-directory [^ ]*#--model-action upload --export-strategy MERGE --upload-repo-id $HF_USER/$name#" "$OUT/reexport.sh"
  grep -q -- "--upload-repo-id" "$OUT/reexport.sh" || { echo "publish rewrite failed"; exit 1; }
fi

echo "=== re-export start $(date -Is) publish=$publish name=$name"; s=$(date +%s)
bash -o pipefail "$OUT/reexport.sh" > "$OUT/reexport.log" 2>&1 || { echo "re-export failed"; tail -30 "$OUT/reexport.log"; exit 1; }
echo "=== re-export done $(( $(date +%s)-s ))s"

if [ $publish -eq 1 ]; then
  MODEL=$P/tmpmodels/$label; rm -rf "$MODEL"
  "$HV/hf" download "$HF_USER/$name" --local-dir "$MODEL" > "$OUT/download.log" 2>&1 || { echo "download back failed"; exit 1; }
  echo "$HF_USER/$name" > "$OUT/hf_repo.txt"
else
  MODEL=$(ls -d "$R"/model_kl0.1_t* | head -1)
fi
git -C /data/heretic_2/heretic rev-parse HEAD > "$OUT/heretic_commit.txt"
[ -f "$MODEL/config.json" ] || { echo "merged model missing: $MODEL"; exit 1; }

bash "$P/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$repo" \
     --refusal-sets "$SETS" --reference-responses "$REF"
rc=$?
# published copies are re-downloadable; unpublished merged models are KEPT for a later upload (exact evaluated weights)
[ $rc -eq 0 ] && [ $publish -eq 1 ] && rm -rf "$MODEL"
exit $rc
