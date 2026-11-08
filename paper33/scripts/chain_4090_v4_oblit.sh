#!/usr/bin/env bash
# 4090 follow-up (2026-11-06 ~09:15): OBLITERATUS presets on Qwen2.5-7B OOMed on the 24 GB card (activation probe ~2 GB
# short). Re-run after chain v3 finishes with OBLITERATUS's own resource knob --gpu-memory-utilization (budget only; the
# admission controller offloads to CPU), trying 0.6 then 0.45; then evaluate each model. som / Qwen3.5-4B stay unavailable.
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090/logs/chain_1x_7b.log
T=/data/heretic_2/tools_ext; E=/data/heretic_2/models/external; W=$1
export HF_HOME=/data/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 "$W" 2>/dev/null; do sleep 60; done
echo "$(date -Is) v4 start: OBLITERATUS 7B re-run with GPU budget knob" >> $L
for preset in basic advanced spectral_cascade; do
  lab=qwen2.5-7b__obliteratus-$preset
  for u in 0.6 0.45; do
    [ -s $E/$lab/PROVENANCE.json ] && break
    echo "$(date -Is) START $lab gpu_mem_util=$u" >> $L
    $T/venv-obliteratus/bin/python -I $T/scripts/run_obliteratus.py --repo Qwen/Qwen2.5-7B-Instruct \
      --revision a09a35458c702b33eeacc393d103063234e8bc28 --preset $preset --seed 1 --device auto \
      --gpu-memory-utilization $u --out $E/$lab >> $P/runs/local-4090/logs/oblit_7b_rerun.log 2>&1
    echo "$(date -Is) END $lab util=$u rc=$? provenance=$([ -s $E/$lab/PROVENANCE.json ] && echo yes || echo no)" >> $L
  done
  if [ -s $E/$lab/PROVENANCE.json ] && [ ! -s $P/results/bench-4090/$lab/summary.json ]; then
    mkdir -p $P/results/bench-4090/$lab
    echo "$(date -Is) START eval $lab" >> $L
    bash $P/scripts/eval_ext_local.sh $lab $E/$lab Qwen/Qwen2.5-7B-Instruct qwen2.5-7b__base > $P/results/bench-4090/$lab/eval.log 2>&1
    echo "$(date -Is) END eval $lab rc=$?" >> $L
  fi
done
echo "$(date -Is) CHAIN V4 DONE" >> $L
