#!/usr/bin/env bash
# 4090 chain v2 (2026-11-06 07:30): home internet is ~75 KB/s, so evaluate the two Qwen2.5-7B 1.x models first
# (base + merged models are local), while the Coder-7B base downloads in the background; then Coder-7B base + 1.x.
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090/logs/chain_1x_7b.log
R=/data/heretic_2/results/heretic; M=/data/heretic_2/models/merged_models; HV=/data/heretic_2/heretic/.venv/bin
export HF_HOME=/data/huggingface
echo "$(date -Is) chain v2 start (v1 stopped during Coder-7B base download)" >> $L
( cd /tmp && $HV/python -I -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct'))" > $P/runs/local-4090/logs/dl_coder7b.log 2>&1; echo "rc=$?" >> $P/runs/local-4090/logs/dl_coder7b.log ) &
dl=$!
run() { echo "$(date -Is) START $1" >> $L; shift; "$@"; rc=$?; echo "$(date -Is) END rc=$rc" >> $L; }
for v in general code; do
  lab=qwen2.5-7b__heretic1x-$v; mkdir -p $P/results/bench-4090/$lab
  run $lab bash $P/scripts/eval_1x_local.sh $lab $M/Qwen__Qwen2.5-7B-Instruct__$v $R/Qwen__Qwen2.5-7B-Instruct__$v/adapter Qwen/Qwen2.5-7B-Instruct qwen2.5-7b__base > $P/results/bench-4090/$lab/eval.log 2>&1
done
wait $dl; echo "$(date -Is) coder-7b base download: $(tail -1 $P/runs/local-4090/logs/dl_coder7b.log)" >> $L
mkdir -p $P/results/bench-4090/qwen2.5-coder-7b__base
run qwen2.5-coder-7b__base bash $P/scripts/eval_one_local.sh qwen2.5-coder-7b__base Qwen/Qwen2.5-Coder-7B-Instruct > $P/results/bench-4090/qwen2.5-coder-7b__base/eval_one.log 2>&1
for v in general code; do
  lab=qwen2.5-coder-7b__heretic1x-$v; mkdir -p $P/results/bench-4090/$lab
  run $lab bash $P/scripts/eval_1x_local.sh $lab $M/Qwen__Qwen2.5-Coder-7B-Instruct__$v $R/Qwen__Qwen2.5-Coder-7B-Instruct__$v/adapter Qwen/Qwen2.5-Coder-7B-Instruct qwen2.5-coder-7b__base > $P/results/bench-4090/$lab/eval.log 2>&1
done
echo "$(date -Is) CHAIN DONE" >> $L
