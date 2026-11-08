#!/usr/bin/env bash
# Run ON the box in tmux. GPU job queue: one job at a time per GPU, every job via box_run.sh.
# Queue file lines:  <run_name> | <model_repo> | <command ...>
# A job only starts once its model is fully downloaded (OK line in download.log).
# Finished/failed jobs are recorded, so the queue can be restarted safely (skips done ones).
# Append to the queue file at any time — it is re-read before every job.
source /workspace/paper33/env.sh
Q=${1:-$W/queue.txt}; GPU=${2:-0}
export CUDA_VISIBLE_DEVICES=$GPU
DONE=$W/logs/queue_done.txt; touch "$DONE"
echo "$(date -Is) queue start gpu=$GPU" >> "$W/logs/queue.log"
while true; do
  job=$(grep -v -E '^\s*(#|$)' "$Q" | while IFS='|' read -r name repo cmd; do
          name=$(echo $name); grep -qx "$name" "$DONE" || { echo "$name|$(echo $repo)|$cmd"; break; }; done)
  [ -z "$job" ] && { echo "$(date -Is) queue empty" >> "$W/logs/queue.log"; sleep 120; continue; }
  IFS='|' read -r name repo cmd <<< "$job"
  until grep -q "OK $repo " "$W/logs/download.log" 2>/dev/null; do
    echo "$(date -Is) waiting for download $repo ($name)" >> "$W/logs/queue.log"; sleep 120; done
  echo "$(date -Is) START $name" >> "$W/logs/queue.log"
  bash "$W/scripts/box_run.sh" "$name" bash -c "$cmd"; rc=$?
  echo "$name" >> "$DONE"
  echo "$(date -Is) END $name rc=$rc" >> "$W/logs/queue.log"
done
