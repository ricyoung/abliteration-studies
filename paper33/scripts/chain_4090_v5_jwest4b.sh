#!/usr/bin/env bash
# 4090 follow-up (2026-11-06): jwest33 on Qwen3.5-4B AS RELEASED (no wrapper hybrid fix) per the user's "as released"
# decision; the existing qwen3.5-4b__jwest33-recommended (wrapper re-enables the tool's skipped hybrid handling) is kept
# as a labelled sensitivity run. Waits for chain v4 (pid $1).
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090/logs/chain_1x_7b.log
T=/data/heretic_2/tools_ext; E=/data/heretic_2/models/external; lab=qwen3.5-4b__jwest33-asreleased
export HF_HOME=/data/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -Is) START $lab (no hybrid fix)" >> $L
[ -s $E/$lab/PROVENANCE.json ] || $T/venv-jwest33/bin/python -I $T/scripts/run_jwest33.py --repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --seed 1 --out $E/$lab --no-hybrid-fix -- --dtype bfloat16 --device cuda --winsorize --null-space \
  >> $P/runs/local-4090/logs/jwest4b_asreleased.log 2>&1
echo "$(date -Is) END $lab rc=$? provenance=$([ -s $E/$lab/PROVENANCE.json ] && echo yes || echo no)" >> $L
if [ -s $E/$lab/PROVENANCE.json ]; then
  mkdir -p $P/results/bench-4090/$lab
  echo "$(date -Is) START eval $lab" >> $L
  bash $P/scripts/eval_ext_local.sh $lab $E/$lab Qwen/Qwen3.5-4B qwen3.5-4b__base > $P/results/bench-4090/$lab/eval.log 2>&1
  echo "$(date -Is) END eval $lab rc=$?" >> $L
fi
echo "$(date -Is) CHAIN V5 DONE" >> $L
