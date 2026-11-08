#!/usr/bin/env bash
# Local 4090 queue (paper 33), seed-parameterised copy of local_queue.sh (SEED env, default 1). Lines: name | model | method(config stem).
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090; Q=${1:-$L/queue.txt}; DONE=$L/logs/done.txt; touch $DONE
H=/data/heretic_2/heretic/.venv/bin/heretic
while true; do
  line=$(grep -v -E '^\s*(#|$)' "$Q" | while IFS='|' read -r n m c; do n=$(echo $n); grep -qx "$n" $DONE || { echo "$n|$(echo $m)|$(echo $c)"; break; }; done)
  [ -z "$line" ] && { sleep 120; continue; }
  IFS='|' read -r n m c <<< "$line"; R=$L/$n; mkdir -p $R; cd $R
  cp $P/configs/$c.toml config.toml; sed -i "s#DIRECTIONS_PATH#$P/directions/${n%%__*}__jspace.pt#" config.toml
  { echo "cmd: heretic --model $m ($c) --seed ${SEED:-1}"; echo "start: $(date -Is)"; echo "host: $(hostname)"; git -C /data/heretic_2/heretic log --oneline -1; } > meta.txt
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > gpu.csv 2>&1 & mon=$!; s=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 $H --model $m --seed ${SEED:-1} --checkpoint-action restart --trial-index 0 --model-action save --export-strategy ADAPTER --save-directory ./adapter_t0 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%T)" "$l"; done > output.log
  rc=${PIPESTATUS[0]}; kill $mon
  { echo "end: $(date -Is)"; echo "exit_code: $rc"; echo "wall_seconds: $(( $(date +%s)-s ))"; } >> meta.txt
  echo "$n" >> $DONE; echo "$(date -Is) $n rc=$rc" >> $L/logs/runs_index.log
done
