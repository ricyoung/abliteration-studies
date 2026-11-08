#!/usr/bin/env bash
# 4090 chain (2026-11-06): paper-30 Heretic 1.x 7B evals, after the small-pair dev re-score frees the GPU.
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090/logs/chain_1x_7b.log
R=/data/heretic_2/results/heretic; M=/data/heretic_2/models/merged_models
while pgrep -f "heretic/.venv/bin/heretic --model" > /dev/null || pgrep -f "rescore_dev.py jobs_local" > /dev/null; do sleep 60; done
run() { echo "$(date -Is) START $1" >> $L; shift; "$@"; rc=$?; echo "$(date -Is) END rc=$rc" >> $L; }
mkdir -p $P/results/bench-4090/qwen2.5-coder-7b__base
run qwen2.5-coder-7b__base bash $P/scripts/eval_one_local.sh qwen2.5-coder-7b__base Qwen/Qwen2.5-Coder-7B-Instruct > $P/results/bench-4090/qwen2.5-coder-7b__base/eval_one.log 2>&1
for v in general code; do
  for b in "qwen2.5-7b Qwen/Qwen2.5-7B-Instruct Qwen__Qwen2.5-7B-Instruct" "qwen2.5-coder-7b Qwen/Qwen2.5-Coder-7B-Instruct Qwen__Qwen2.5-Coder-7B-Instruct"; do
    set -- $b; lab=$1__heretic1x-$v; mkdir -p $P/results/bench-4090/$lab
    run $lab bash $P/scripts/eval_1x_local.sh $lab $M/$3__$v $R/$3__$v/adapter $2 $1__base > $P/results/bench-4090/$lab/eval.log 2>&1
  done
done
echo "$(date -Is) CHAIN DONE" >> $L
