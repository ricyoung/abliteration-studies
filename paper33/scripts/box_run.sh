#!/usr/bin/env bash
# Run ON the box. Wraps any experiment so that EVERYTHING is kept:
# full stdout+stderr (timestamped), exit code, wall time, GPU telemetry, config + env.
# Usage: bash box_run.sh <run_name> <command...>
#   e.g. bash box_run.sh qwen3.5-4b__ara__s1 .venv/bin/heretic --model Qwen/Qwen3.5-4B ...
source /workspace/paper33/env.sh
name=$1; shift
R=$W/runs/$name; mkdir -p "$R"
cd "$R"
{ echo "cmd: $*"; echo "start: $(date -Is)"; echo "host: $(hostname)";
  cat "$W/env/heretic_commit.txt"; } > meta.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > gpu.csv 2>&1 &
mon=$!
start=$(date +%s)
set +e
"$@" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done | tee output.log
rc=${PIPESTATUS[0]}
kill $mon 2>/dev/null
cp -f config.toml config.used.toml 2>/dev/null || true
{ echo "end: $(date -Is)"; echo "exit_code: $rc"; echo "wall_seconds: $(( $(date +%s)-start ))";
  echo "peak_vram_mib: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3>m)m=$3} END{print m}' gpu.csv)"; } >> meta.txt
[ $rc -ne 0 ] && echo "$(date -Is) FAILED $name rc=$rc" >> "$W/logs/failures.log"
echo "$(date -Is) $name rc=$rc" >> "$W/logs/runs_index.log"
exit $rc
