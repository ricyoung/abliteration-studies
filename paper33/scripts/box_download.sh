#!/usr/bin/env bash
# Run ON the rented box inside tmux: downloads models in priority order, logs each one.
# Usage: bash box_download.sh [models.txt]
source /workspace/paper33/env.sh
LIST=${1:-$W/models.txt}
LOG=$W/logs/download.log
echo "=== download start $(date -Is)" >> "$LOG"
grep -v -E '^\s*(#|$)' "$LIST" | while read -r repo _; do
  # disk guard: never start a model unless >= 120 GB free (protects running jobs)
  while [ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -lt 120 ]; do
    echo "--- $(date -Is) WAIT disk <120G free before $repo" >> "$LOG"; sleep 600; done
  echo "--- $(date -Is) START $repo" >> "$LOG"
  start=$(date +%s)
  for attempt in 1 2 3 4 5; do
    if hf download "$repo" --exclude "*.gguf" --exclude "original/*" --exclude "*.pth" --exclude "*.bin" >> "$LOG" 2>&1 \
       && [ -n "$(find -L "$HF_HOME/hub/models--${repo/\//--}/snapshots" -name '*.safetensors' -size +1M 2>/dev/null | head -1)" ]; then
      echo "--- $(date -Is) OK $repo attempt=$attempt $(( $(date +%s)-start ))s $(du -sh "$HF_HOME/hub/models--${repo/\//--}" | cut -f1)" >> "$LOG"
      break
    fi
    echo "--- $(date -Is) FAIL $repo attempt=$attempt" >> "$LOG"; sleep 60
  done
done
echo "=== download done $(date -Is)" >> "$LOG"
df -h "$W" >> "$LOG"
