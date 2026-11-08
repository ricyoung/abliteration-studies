#!/usr/bin/env bash
# Paper 33 small pair x external abliteration tools, local RTX 4090, run sequentially.
#   models: Qwen/Qwen2.5-7B-Instruct@a09a354 (qwen2.5-7b), Qwen/Qwen3.5-4B@851bf6e (qwen3.5-4b)
#   tools : OBLITERATUS v0.1.4 (461ff08) presets basic / advanced / spectral_cascade / som (preset defaults)
#           jwest33/abliterator 6ca3356, CLI "recommended" preset (= --winsorize --null-space; norm preservation on)
#   NOTE: OBLITERATUS v0.1.4 does not ship its SOM extractor (obliteratus.analysis.som_directions); the `som`
#   preset fails by design in the constructor within seconds ("direction method 'som' is not available"). The
#   job is still attempted so the log records the exact error; it never produces an output.
# = 10 jobs. Output: /data/heretic_2/models/external/<model_short>__<tool>-<preset>/ (HF safetensors bf16,
# tokenizer + chat template, PROVENANCE.json). Installs, pins, fixes: /data/heretic_2/tools_ext/README.md
#
# Behaviour
#   * waits until the GPU is free: no process matching $BUSY_PAT and no non-desktop CUDA process (checked
#     twice, 60 s apart), and re-checks before every job;
#   * skips a job whose output dir already has PROVENANCE.json; never overwrites or deletes an output dir
#     (an output dir without PROVENANCE.json is reported and left alone);
#   * deletes nothing outside its own temp dirs (/data/heretic_2/tools_ext/work/*.tmp<pid>, <out>.staging);
#     a failed job keeps its temp dir for inspection (FAILED.json inside);
#   * log: runs/local-4090/logs/external_small.log (+ one full log per job in logs/external_small/).
#
# Knobs (environment)
#   OBLIT_QWEN35=upstream|generic|skip   default upstream. OBLITERATUS v0.1.4 fails closed on every Qwen3.5
#        size except Qwen3.8-27B ("hybrid surgery remains blocked ... no weights were modified"). upstream runs
#        the first 4B preset to record the real error and marks the other three BLOCKED if it fails that way.
#        generic = opt-in runtime patch (tools_ext/patches/obliteratus_qwen35_generic_contract.py), outside
#        the tool's validated scope, untested on GPU (no free GPU during preparation).
#   JWEST_SQFIX=0|1|both   default 0 = tool as published (square o_proj of Qwen2.5-7B projected on its input
#        side, the known bug); 1 = runtime fix (patches/jwest33_square_writer_fix.py); both = run both 7B variants
#        (extra job qwen2.5-7b__jwest33-recommended-sqfix). Qwen3.5-4B has no square writer; unaffected.
#   JWEST_HYBRID_FIX=1|0   default 1: the jwest33 CLI never runs its hybrid-architecture detection, so the
#        documented default --hybrid-strategy auto (extraction at full-attention layers, skip in_proj_a/b,
#        0.4x on DeltaNet layers) is silently a no-op on Qwen3.5. The wrapper sets config._hybrid_info exactly
#        as the tool's library entry point does. No effect on Qwen2.5-7B. 0 = tool CLI as published.
#   SEED (default 1)   DRY_RUN=1 (print the plan, run nothing)   NO_WAIT=1 (skip the GPU wait)
#   OBLIT_GPU_MEM_UTIL=<fraction>  optional OBLITERATUS --gpu-memory-utilization (resource knob only)
#
# Usage: nohup bash scripts/run_external_small.sh >/dev/null 2>&1 &
set -uo pipefail
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder
T=/data/heretic_2/tools_ext
OUT=/data/heretic_2/models/external
LOG=$P/runs/local-4090/logs/external_small.log
JL=$P/runs/local-4090/logs/external_small
mkdir -p "$JL" "$OUT" "$T/work/tmp"
[ "${DRY_RUN:-0}" = 1 ] && LOG=/dev/stdout

SEED=${SEED:-1}
OBLIT_QWEN35=${OBLIT_QWEN35:-upstream}
JWEST_SQFIX=${JWEST_SQFIX:-0}
JWEST_HYBRID_FIX=${JWEST_HYBRID_FIX:-1}
BUSY_PAT='rescore_dev.py|chain_1x_7b_local|heretic/.venv/bin/heretic|vllm|bench/run.sh'  # chain_1x_7b_local also matches chain_1x_7b_local_v2.sh
DESKTOP_PAT='voxtype|electron|chromium|Proton|filmcraft'

export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 OBLITERATUS_TELEMETRY=0 TOKENIZERS_PARALLELISM=false
export TMPDIR=$T/work/tmp CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TERM=dumb COLUMNS=160
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Q7=Qwen/Qwen2.5-7B-Instruct;  Q7R=a09a35458c702b33eeacc393d103063234e8bc28
Q4=Qwen/Qwen3.5-4B;           Q4R=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
OPY=$T/venv-obliteratus/bin/python
JPY=$T/venv-jwest33/bin/python

log() { echo "$(date -Is) $*" >> "$LOG"; }

# PIDs of this script and its ancestors, so a wrapper command line that mentions a pattern cannot block us
self_tree() { local p=$$; while [ "$p" -gt 1 ]; do echo "$p"; p=$(ps -o ppid= -p "$p" | tr -d ' '); done; }

gpu_busy() {
  local mine procs apps
  mine=$(self_tree | paste -sd'|')
  procs=$(pgrep -f -- "$BUSY_PAT" | grep -v -x -E "${mine:-0}" | head -5 | paste -sd' ')
  if [ -n "$procs" ]; then echo "main-session processes: $(ps -o comm= -p ${procs// /,} 2>/dev/null | paste -sd,) (pids $procs)"; return 0; fi
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -E "$DESKTOP_PAT" | grep -v '^[[:space:]]*$' | head -3 | paste -sd';')
  if [ -n "$apps" ]; then echo "other CUDA processes: $apps"; return 0; fi
  return 1
}

wait_gpu() {
  [ "${NO_WAIT:-0}" = 1 ] && return 0
  local ok=0 last="" n=0 why
  while :; do
    if why=$(gpu_busy); then
      ok=0
      # log on change, and every ~30 min while unchanged
      if [ "$why" != "$last" ] || [ $((n % 15)) -eq 0 ]; then log "WAIT gpu busy: $why"; fi
      last=$why; n=$((n + 1)); sleep 120
    else
      ok=$((ok + 1)); [ $ok -ge 2 ] && { [ -n "$last" ] && log "GPU free"; return 0; }; sleep 60
    fi
  done
}

FAILED=(); DONE=(); SKIPPED=()
run_job() {  # name, then the command
  local name=$1; shift
  local out=$OUT/$name rc t0
  if [ -s "$out/PROVENANCE.json" ]; then log "SKIP $name (PROVENANCE.json exists)"; SKIPPED+=("$name"); return 0; fi
  if [ -e "$out" ]; then log "SKIP $name: $out exists WITHOUT PROVENANCE.json; left untouched, inspect manually"; SKIPPED+=("$name"); return 0; fi
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "[dry-run] $name: $*"; return 0; fi
  wait_gpu
  log "START $name :: $*"
  t0=$(date +%s)
  "$@" > "$JL/$name.log" 2>&1; rc=$?
  log "END   $name rc=$rc $(( ($(date +%s) - t0) / 60 )) min (full log: $JL/$name.log)"
  if [ $rc -ne 0 ]; then
    FAILED+=("$name")
    tail -n 25 "$JL/$name.log" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -v '^[[:space:]]*$' | sed 's/^/    | /' >> "$LOG"
  else
    DONE+=("$name")
  fi
  return $rc
}

oblit() {  # model_short repo rev preset [extra args]
  local ms=$1 repo=$2 rev=$3 preset=$4; shift 4
  local extra=()
  [ -n "${OBLIT_GPU_MEM_UTIL:-}" ] && extra+=(--gpu-memory-utilization "$OBLIT_GPU_MEM_UTIL")
  run_job "${ms}__obliteratus-${preset}" "$OPY" -I "$T/scripts/run_obliteratus.py" --repo "$repo" --revision "$rev" \
    --preset "$preset" --seed "$SEED" --device auto --out "$OUT/${ms}__obliteratus-${preset}" "${extra[@]}" "$@"
}

jwest() {  # model_short repo rev suffix [wrapper args]
  local ms=$1 repo=$2 rev=$3 suf=$4; shift 4
  [ "$JWEST_HYBRID_FIX" = 0 ] && set -- "$@" --no-hybrid-fix
  run_job "${ms}__jwest33-recommended${suf}" "$JPY" -I "$T/scripts/run_jwest33.py" --repo "$repo" --revision "$rev" \
    --seed "$SEED" --out "$OUT/${ms}__jwest33-recommended${suf}" "$@" -- --dtype bfloat16 --device cuda --winsorize --null-space
}

# ---------------------------------------------------------------- preflight (no GPU use)
pre() {
  local bad=0
  for x in "$Q7 $Q7R" "$Q4 $Q4R"; do set -- $x
    [ -s "$HF_HOME/hub/models--${1/\//--}/snapshots/$2/config.json" ] || { log "PREFLIGHT missing base $1@$2 in $HF_HOME"; bad=1; }
  done
  (cd /tmp && CUDA_VISIBLE_DEVICES="" "$OPY" -I -c "import obliteratus.abliterate, fla, causal_conv1d" 2>/dev/null) || { log "PREFLIGHT venv-obliteratus import failed"; bad=1; }
  (cd /tmp && CUDA_VISIBLE_DEVICES="" "$JPY" -I -c "import src.cli, src.abliterate" 2>/dev/null) || { log "PREFLIGHT venv-jwest33 import failed"; bad=1; }
  [ "$(git -C $T/src/obliteratus rev-parse HEAD)" = 461ff086e675d93969bbd7c99c53505762452309 ] || { log "PREFLIGHT OBLITERATUS clone not at v0.1.4"; bad=1; }
  [ "$(git -C $T/src/jwest33-abliterator rev-parse HEAD)" = 6ca3356e493ee43a1e124f50b73e531205b11ceb ] || { log "PREFLIGHT jwest33 clone not at 6ca3356"; bad=1; }
  return $bad
}

log "==== run_external_small.sh start (pid $$; SEED=$SEED OBLIT_QWEN35=$OBLIT_QWEN35 JWEST_SQFIX=$JWEST_SQFIX JWEST_HYBRID_FIX=$JWEST_HYBRID_FIX DRY_RUN=${DRY_RUN:-0})"
pre || { log "PREFLIGHT failed; aborting"; exit 2; }

# ---------------------------------------------------------------- Qwen2.5-7B-Instruct
for preset in basic advanced spectral_cascade som; do oblit qwen2.5-7b "$Q7" "$Q7R" "$preset"; done
case "$JWEST_SQFIX" in
  1)    jwest qwen2.5-7b "$Q7" "$Q7R" "" --fix-square-writers ;;
  both) jwest qwen2.5-7b "$Q7" "$Q7R" ""; jwest qwen2.5-7b "$Q7" "$Q7R" "-sqfix" --fix-square-writers ;;
  *)    jwest qwen2.5-7b "$Q7" "$Q7R" "" ;;
esac

# ---------------------------------------------------------------- Qwen3.5-4B
jwest qwen3.5-4b "$Q4" "$Q4R" ""
case "$OBLIT_QWEN35" in
  skip) log "SKIP qwen3.5-4b OBLITERATUS presets (OBLIT_QWEN35=skip)" ;;
  generic) for preset in basic advanced spectral_cascade som; do oblit qwen3.5-4b "$Q4" "$Q4R" "$preset" --qwen35-generic-contract; done ;;
  *)
    blocked=0
    for preset in basic advanced spectral_cascade som; do
      if [ $blocked = 1 ]; then log "BLOCKED qwen3.5-4b__obliteratus-$preset: same v0.1.4 Qwen3.5 contract error as above (not run; set OBLIT_QWEN35=generic to try the opt-in patch)"; continue; fi
      oblit qwen3.5-4b "$Q4" "$Q4R" "$preset"
      if [ $? -ne 0 ] && grep -q "hybrid surgery remains blocked" "$JL/qwen3.5-4b__obliteratus-$preset.log" 2>/dev/null; then blocked=1; fi
    done ;;
esac

log "==== done. ok: ${DONE[*]:-none} | failed: ${FAILED[*]:-none} | skipped: ${SKIPPED[*]:-none}"

# ---------------------------------------------------------------- prompt overlap (CPU, read-only on the outputs)
if [ "${DRY_RUN:-0}" != 1 ]; then
  ov=$P/runs/local-4090/logs/external_small_prompt_overlap.json
  dirs=$(ls -d $OUT/qwen2.5-7b__* $OUT/qwen3.5-4b__* 2>/dev/null | grep -v '\.staging$')
  if [ -n "$dirs" ]; then
    (cd /tmp && CUDA_VISIBLE_DEVICES="" nice "$OPY" -I "$T/scripts/check_prompt_overlap.py" "$ov" --outputs $dirs) >> "$JL/prompt_overlap.log" 2>&1 \
      && log "prompt overlap vs held-out/AdvBench/dev written to $ov" || log "prompt overlap check failed (see $JL/prompt_overlap.log)"
  fi
fi
