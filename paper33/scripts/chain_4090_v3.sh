#!/usr/bin/env bash
# 4090 master chain v3 (2026-11-06 ~08:10): replaces chain v2 after its running Qwen2.5-7B general eval.
# Order: Qwen2.5-7B 1.x code -> external tools (OBLITERATUS/jwest33, user decisions: 4B OBLITERATUS = not supported
# (upstream attempt logged), jwest33 as released) -> eval external models -> Coder-7B base + 1.x (after slow download).
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090/logs/chain_1x_7b.log
R=/data/heretic_2/results/heretic; M=/data/heretic_2/models/merged_models; E=/data/heretic_2/models/external
export HF_HOME=/data/huggingface
run() { echo "$(date -Is) START $1" >> $L; shift; "$@"; rc=$?; echo "$(date -Is) END rc=$rc" >> $L; }
echo "$(date -Is) chain v3 start (v2 stopped; its qwen2.5-7b general eval continues to completion)" >> $L
while pgrep -f "eval_1x_local.sh qwen2.5-7b__heretic1x-general" > /dev/null; do sleep 30; done
echo "$(date -Is) qwen2.5-7b__heretic1x-general finished: $(ls $P/results/bench-4090/qwen2.5-7b__heretic1x-general/summary.json 2>/dev/null || echo 'NO summary.json')" >> $L
lab=qwen2.5-7b__heretic1x-code; mkdir -p $P/results/bench-4090/$lab
run $lab bash $P/scripts/eval_1x_local.sh $lab $M/Qwen__Qwen2.5-7B-Instruct__code $R/Qwen__Qwen2.5-7B-Instruct__code/adapter Qwen/Qwen2.5-7B-Instruct qwen2.5-7b__base > $P/results/bench-4090/$lab/eval.log 2>&1
run external_small env NO_WAIT=1 bash $P/scripts/run_external_small.sh
for d in $E/*/; do
  lab=$(basename $d); [ -s $d/PROVENANCE.json ] && ls $d/*.safetensors > /dev/null 2>&1 || continue
  case $lab in qwen2.5-7b__*) base=Qwen/Qwen2.5-7B-Instruct;; qwen3.5-4b__*) base=Qwen/Qwen3.5-4B;; *) continue;; esac
  [ -s $P/results/bench-4090/$lab/summary.json ] && continue
  mkdir -p $P/results/bench-4090/$lab
  run $lab bash $P/scripts/eval_ext_local.sh $lab $d $base ${lab%%__*}__base > $P/results/bench-4090/$lab/eval.log 2>&1
done
until grep -q '^rc=' $P/runs/local-4090/logs/dl_coder7b.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) coder-7b base download: $(tail -1 $P/runs/local-4090/logs/dl_coder7b.log)" >> $L
mkdir -p $P/results/bench-4090/qwen2.5-coder-7b__base
run qwen2.5-coder-7b__base bash $P/scripts/eval_one_local.sh qwen2.5-coder-7b__base Qwen/Qwen2.5-Coder-7B-Instruct > $P/results/bench-4090/qwen2.5-coder-7b__base/eval_one.log 2>&1
for v in general code; do
  lab=qwen2.5-coder-7b__heretic1x-$v; mkdir -p $P/results/bench-4090/$lab
  run $lab bash $P/scripts/eval_1x_local.sh $lab $M/Qwen__Qwen2.5-Coder-7B-Instruct__$v $R/Qwen__Qwen2.5-Coder-7B-Instruct__$v/adapter Qwen/Qwen2.5-Coder-7B-Instruct qwen2.5-coder-7b__base > $P/results/bench-4090/$lab/eval.log 2>&1
done
echo "$(date -Is) CHAIN V3 DONE" >> $L
