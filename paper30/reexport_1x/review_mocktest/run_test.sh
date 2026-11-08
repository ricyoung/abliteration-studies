#!/usr/bin/env bash
# Run ON the RTX PRO 6000 box (paper-30 v3, 2026-11-06). Re-export the Heretic 1.x (heretic-llm 1.4.0 = git v1.4.0
# 6ea3b8d) Qwen2.5-32B-Instruct studies at the paper-33 selection rule (KL <= 0.1, fewest dev refusals, ties -> lower
# KL) and evaluate them with the paper-33 protocol.
#
#   general: rule trial #96  (dev refusals 50/100, KL 0.0642)   originally exported #84  (89/100, KL 0.0059)
#   code   : rule trial #134 (dev refusals  2/100, KL 0.0717)   originally exported #179 (30/100, KL 0.0092)
#
# Inputs: rsync the local folder /data/heretic_2/paper30/v3_2611/reexport_1x/box_inputs/ to $W/reexport_1x/ (24 MB;
# MANIFEST.sha256 is checked below). The code study also needs the four paper-32 prompt files
#   $W/reexport_1x/datasets/configs/code_abliteration_data/{benign_code_prompts,malicious_code_prompts,benign_code_eval,malicious_code_eval}.txt
# (paper-32 repo on the old Mac: /Volumes/data_bolt/_github_papers/32_2604_malicious_code_generation_safety/configs/
# code_abliteration_data/). Without them the script stops at preflight (SKIP_CODE=1 runs the general study only).
#
# Steps (log: $W/logs/reexport_1x.log; records: $W/results/reexport_1x/):
#   0. preflight: inputs vs MANIFEST.sha256, GPU idle, disk space
#   1. venv $HOME/.venvs/heretic1x-1.4.0 (outside $W, not mirrored home): heretic-llm 1.4.0 (git archive) + the v1.4.0 uv.lock pins (transformers 5.6.2,
#      peft 0.19.1, accelerate 1.13.0, optuna 4.8.0, datasets 4.8.4, ...) + the SAME torch build as the box's working
#      heretic venv (2.14.1+cu130, runs on Blackwell sm_120), from the PyTorch cu130 index (a uv-cache hit on this box).
#      Not --system-site-packages: that exposes the *base interpreter's* site-packages, not another venv's, and layering
#      the 2.0 venv (transformers 5.18, newer peft) under heretic 1.4.0 would silently change the 1.x code path. The
#      lock's torch 2.9.1 is not used: heretic-llm 1.4.0 does not pin torch, so the June version is unknown anyway, and
#      2.14.1+cu130 is the build already proven on this GPU.
#   2. base Qwen/Qwen2.5-32B-Instruct @ 5ede1c97 (HF_HOME=$W/hf; downloaded only if absent)
#   3. GENERAL (one process, export_1x_trial.py): refusal directions recomputed by the 1.4.0 code;
#      job #84 (originally exported trial) = VALIDATION: adapter vs the original adapter, merged per-tensor sha256 vs
#      tensor_hashes_original.json (+ diff stats), dev re-evaluation of ours AND of the original adapter on this GPU
#      (separates hardware noise from reconstruction error) and vs the journal. --gate 84: on failure the process exits 3
#      BEFORE #96 is touched and this script exits 3. Then #96 -> $W/models_1x/qwen2.5-32b__heretic1x-general-kl0.1
#      (merged bf16 + tokenizer + PROVENANCE.json + .pushed), dev re-evaluated vs the journal; if that re-eval is outside
#      tolerance its benchmark is skipped, the other study still runs and the script exits 4 at the end
#      (ALLOW_RULE_MISMATCH=1 to evaluate anyway, also on a re-run without re-exporting); then
#      eval_local_model.sh qwen2.5-32b__heretic1x-general-kl0.1.
#   4. CODE (one process): #179 = VALIDATION of the prompt files + directions (adapter level + dev re-eval; no merge),
#      --gate 179; then #134 -> $W/models_1x/qwen2.5-32b__heretic1x-code-kl0.1, same checks, eval.
#   eval_local_model.sh deletes the model dir after a successful eval; PROVENANCE, adapter and per-tensor hashes are kept
#   in $W/results/reexport_1x/<label>/ and $W/results/bench/<label>/reexport_provenance.json.
#
# Validation criterion (export_1x_trial.py --accept numerical, defaults): same edited modules; refusal-direction cosine
# vs the original adapter >= 0.9995; lora_A/lora_B relative difference <= 0.02; only edited tensors may differ; dev
# refusals of ours within +-MAX_REFUSAL_DIFF (5) and KL within max(0.002, 15%) of the ORIGINAL adapter evaluated on this
# GPU; and within +-MAX_REFUSAL_DIFF_JOURNAL (8) / max(0.002, 30%) of the June journal values. CPU reference (7B general #193): cosine 0.999987, adapter rel. diff
# 0.5%, refusals 65 vs 64 (original adapter, same CPU) vs 67 (journal), KL 0.00423 vs 0.00465 vs 0.00531.
# Bit-identity (needs identical GPU kernels to the June H200 run; not expected on sm_120) is reported as
# "PASS (bit-identical)"; ACCEPT=bit-identical requires it.
#
# Env knobs: SKIP_CODE=1, SKIP_EVAL=1, ACCEPT=numerical|bit-identical, FORCE_GPU=1, ALLOW_RULE_MISMATCH=1,
#   MAX_REFUSAL_DIFF=5, MAX_REFUSAL_DIFF_JOURNAL=8, VENV_1X=<path>.
# Exit codes: 0 ok; 1 error; 2 missing input; 3 validation trial failed (nothing exported); 4 a rule trial exported but
#   its dev re-eval was outside tolerance (benchmark skipped for that study).
# Run it THROUGH the box queue (box_queue.sh line with the usual nvidia-smi idle-wait prefix), not beside it: between
# its steps the GPU is briefly idle and a queued job polling for an idle GPU would start and collide.
set -uo pipefail
source /data/heretic_2/paper30/v3_2611/reexport_1x/review_mocktest/env.sh                     # W=/workspace/paper33, HF_HOME=$W/hf, PATH incl. ~/.local/bin (uv)
R=$W/reexport_1x
L=$W/logs/reexport_1x.log
REC=$W/results/reexport_1x
# [review fix] venv OUTSIDE $W: the box->home mirror (sync_from_box_quiet3.sh / final capture) rsyncs all of
# /workspace/paper33 except a few known venvs; a ~6 GB cu130 venv under $W/reexport_1x would be pulled over the slow
# home link. $HOME is on the same disk (/dev/sda2) and is where uv keeps its Python.
VENV=${VENV_1X:-$HOME/.venvs/heretic1x-1.4.0}
PY=$VENV/bin/python
BASE_REPO=Qwen/Qwen2.5-32B-Instruct
REV=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
GEN=$R/studies/Qwen__Qwen2.5-32B-Instruct__general
CODE=$R/studies/Qwen__Qwen2.5-32B-Instruct__code
ACCEPT=${ACCEPT:-numerical}
mkdir -p "$W/logs" "$REC" "$W/models_1x" "$W/tmpmodels"
exec > >(tee -a "$L") 2>&1
T0=$(date +%s)
say() { echo "[reexport_1x $(date -Is)] $*"; }
die() { say "FATAL: $*"; exit "${2:-1}"; }
say "=== START on $(hostname) (accept=$ACCEPT skip_code=${SKIP_CODE:-0} skip_eval=${SKIP_EVAL:-0})"

# ---------------------------------------------------------------- 0. preflight
cd "$R" || die "missing $R (rsync box_inputs/ there first)"
[ -s MANIFEST.sha256 ] || die "missing $R/MANIFEST.sha256"
sha256sum --quiet -c MANIFEST.sha256 || die "input files do not match MANIFEST.sha256"
say "inputs verified against MANIFEST.sha256"
CODE_FILES=(benign_code_prompts malicious_code_prompts benign_code_eval malicious_code_eval)
if [ "${SKIP_CODE:-0}" != 1 ]; then
  for f in "${CODE_FILES[@]}"; do
    [ -s "$R/datasets/configs/code_abliteration_data/$f.txt" ] || die "code study input missing: $R/datasets/configs/code_abliteration_data/$f.txt (copy it from the paper-32 repo, or SKIP_CODE=1)" 2
  done
  say "code prompt files: $(for f in "${CODE_FILES[@]}"; do printf '%s=%s ' "$f" "$(grep -c . "$R/datasets/configs/code_abliteration_data/$f.txt")"; done)(non-empty lines)"
fi
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader | tee "$REC/nvidia-smi.txt"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${FORCE_GPU:-0}" = 1 ] || [ "${USED:-0}" -lt 8000 ] || die "GPU has ${USED} MiB in use; the 32B model needs ~66 GB (FORCE_GPU=1 to override)"
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
NEED=$(( ${SKIP_CODE:-0} == 1 ? 80 : 150 ))
[ "$FREE_GB" -ge "$NEED" ] || die "only ${FREE_GB} GB free under $W, need >= ${NEED} GB (merged 32B copies are 66 GB each)"
say "disk: ${FREE_GB} GB free; RAM: $(free -g | awk '/Mem:/{print $2" GB total, "$7" GB available"}') (CPU merge of 32B needs ~70 GB)"

# ---------------------------------------------------------------- 1. venv (heretic-llm 1.4.0)
if [ ! -x "$PY" ] || ! "$PY" -c 'import importlib.metadata as m; assert m.version("heretic-llm")=="1.4.0"' 2>/dev/null; then
  say "creating venv $VENV"
  rm -rf "$R/heretic-1x" "$VENV"
  tar -xzf "$R/heretic-1x-v1.4.0.tar.gz" -C "$R" || die "cannot unpack heretic-1x"
  TORCH_VER=$("$W/heretic/.venv/bin/python" -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || echo 2.14.1)
  mkdir -p "$(dirname "$VENV")"
  uv venv --python 3.12 "$VENV" || die "uv venv failed"
  uv pip install --python "$PY" "torch==$TORCH_VER" \
     --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple || die "torch install failed"
  uv pip install --python "$PY" -r "$R/requirements-1x-lock.txt" || die "locked deps install failed"
  uv pip install --python "$PY" --no-deps "$R/heretic-1x" || die "heretic-1x install failed"
fi
"$PY" - <<'PY' || die "venv check failed"
import importlib.metadata as m, torch
v = {p: m.version(p) for p in ["heretic-llm", "transformers", "peft", "accelerate", "optuna", "datasets", "huggingface-hub"]}
assert v["heretic-llm"] == "1.4.0" and v["transformers"] == "5.6.2" and v["peft"] == "0.19.1", v
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()   # sm_120 kernel smoke test
print("venv ok:", v, "torch", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
uv pip freeze --python "$PY" > "$REC/pip_freeze_venv1x.txt" 2>/dev/null

# ---------------------------------------------------------------- 2. base model @ 5ede1c97
BASE_DIR=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_REPO', revision='$REV', local_files_only=True))" 2>/dev/null) || {
  say "base not in HF cache, downloading $BASE_REPO@$REV (~66 GB)"
  BASE_DIR=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_REPO', revision='$REV'))") || die "base download failed"
}
[ "$(basename "$BASE_DIR")" = "$REV" ] || die "unexpected base snapshot $BASE_DIR"
# [review fix] count only shards whose symlink resolves (ls also lists dangling links, e.g. after blob pruning);
# if any is missing, let snapshot_download (online) re-fetch the missing blobs.
count_shards() { find -L "$BASE_DIR" -maxdepth 1 -name 'model-*.safetensors' -type f 2>/dev/null | wc -l; }
NSH=$(count_shards)
if [ "$NSH" -ne 17 ]; then
  say "base snapshot has only $NSH/17 resolvable shards; re-fetching $BASE_REPO@$REV"
  BASE_DIR=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_REPO', revision='$REV'))") || die "base download failed"
  NSH=$(count_shards)
fi
[ "$NSH" -eq 17 ] || die "base snapshot has $NSH/17 shards: $BASE_DIR"
say "base: $BASE_DIR ($NSH shards)"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# [review fix] refusal-count tolerances. The weight-level checks (direction cosine >= 0.9995, adapter rel. diff <= 2%,
# only edited tensors differ) are what catch a wrong reconstruction (wrong prompt files / base / code path). Given those,
# dev-refusal differences are greedy-decoding flips of borderline prompts: the CPU 7B run had 7/100 per-prompt flag
# flips for a net difference of 1, so a net +-3 gate fails by chance ~12-35% of the time (7-10 random flips) and would
# abort the whole run. Defaults widened to 5 (ours vs original adapter, same GPU) and 8 (vs the June journal); a broken
# GPU stack (garbage generations) still moves refusals far beyond that.
COMMON=(--model "$BASE_DIR" --model-revision "$REV" --eval full --eval-ref-adapter --accept "$ACCEPT"
        --max-refusal-diff "${MAX_REFUSAL_DIFF:-5}" --max-refusal-diff-journal "${MAX_REFUSAL_DIFF_JOURNAL:-8}")

# ---------------------------------------------------------------- helpers
report_validation() {  # <record dir of a validation job>
  for f in "$1"/PROVENANCE.json; do
    [ -f "$f" ] && "$PY" - "$f" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); v = p["validation"]; ad = v.get("adapter", {}); mh = v.get("merged_hashes", {})
ev = v.get("eval", {}); eo = v.get("eval_vs_original_adapter_here", {})
print(f"VALIDATION trial {p['trial_index']}: {p['validation_verdict']} {p['validation_reasons']}")
print(f"  adapter: edited {ad.get('n_edited_new')}/{ad.get('n_edited_ref')} bit-identical={ad.get('all_bit_identical')} "
      f"min direction cosine={ad.get('min_direction_cosine')} max rel diff A={ad.get('max_A_rel_diff')} B={ad.get('max_B_rel_diff')}")
if mh:
    print(f"  merged: {mh.get('n_identical')}/{mh.get('n_common')} tensors identical; diff summary {mh.get('diff_summary')}")
print(f"  dev re-eval ours: refusals {ev.get('refusals_reeval')} KL {ev.get('kl_reeval')} | original adapter here: "
      f"refusals {eo.get('refusals_original_adapter')} KL {eo.get('kl_original_adapter')} | journal: refusals {ev.get('refusals_journal')} KL {ev.get('kl_journal')}")
# prompt lists (system prompt + text) of the general study must equal those seen locally (HF cache revs 02c6a92c / 01cead01)
EXP = {"good_prompts": "c8ea9b26d4ba72a38b87a043a12888ff6290cc7aa0d952f4c4439278dfeb710c",
       "bad_prompts": "0db6a62d43acc4dad290251d3d701363b76ea76961a8376daa2db87bda8ddd5d",
       "good_evaluation_prompts": "7e66af3fd7aca0e00f979ebb9888a83a7a53e5375f8d348d2842b016be677c90",
       "bad_evaluation_prompts": "00d1f768a3d5532094c9cd0ea0d0816846f65e778eacb8645227352145228e3c"}
ds = p["datasets"]
if ds["good_prompts"]["journal"] == "mlabonne/harmless_alpaca":
    bad = [k for k, h in EXP.items() if ds[k].get("sha256_prompts") != h]
    print("  prompt lists identical to the local reference" if not bad else f"  WARNING prompt lists differ from the local reference: {bad} (HF dataset revision changed?)")
PY
  done
}
finish_rule() {  # <model_dir> <label> : records, .pushed, rule-trial check
  "$PY" - "$1" "$2" "$REC" <<'PY'
import json, os, sys, shutil
d, label, rec = sys.argv[1:4]
p = json.load(open(os.path.join(d, "PROVENANCE.json")))
keep = os.path.join(rec, label); os.makedirs(keep, exist_ok=True)
for fn in ["PROVENANCE.json", "tensor_hashes.json"]:
    shutil.copy(os.path.join(d, fn), keep)
shutil.copytree(os.path.join(d, "adapter"), os.path.join(keep, "adapter"), dirs_exist_ok=True)
pushed = {"how": "re-exported on box with heretic-llm 1.4.0 (git v1.4.0 6ea3b8d) via export_1x_trial.py: 1.x refusal "
                 "directions recomputed, study trial re-applied, adapter merged with the merge_adapter.py procedure (CPU, bf16)",
          "label": label, "study": p["study_path"], "journal_sha256": p["journal_sha256"],
          "trial_index": p["trial_index"], "trial_direction_index": p["trial_direction_index"], "trial_parameters": p["trial_parameters"],
          "journal_values": p["journal_values"],
          "reevaluated": {k: v for k, v in (p.get("reevaluated") or {}).items() if k != "refusal_flags"},
          "selection_rule": "paper-33: complete trials with dev KL <= 0.1, fewest dev refusals, ties -> lower KL",
          "adapter_sha256": p["adapter_sha256"], "base_snapshot": p["base_model_path"], "base_revision": p["base_model_revision"],
          "refusal_directions_sha256": p["refusal_directions_sha256"],
          "shards_sha256": p.get("merged_shards_sha256"), "tensor_hashes": os.path.join(keep, "tensor_hashes.json"),
          "provenance": os.path.join(keep, "PROVENANCE.json"), "rule_trial_verdict": p.get("validation_verdict")}
# [review fix] written as .pushed.pending; the shell promotes it to .pushed (which eval_local_model.sh requires) only on
# PASS or ALLOW_RULE_MISMATCH=1, so a re-run cannot skip the rule-trial check via the "already exported" branch.
json.dump(pushed, open(os.path.join(d, ".pushed.pending"), "w"), indent=1)
ev = p["validation"].get("eval", {})
print(f"RULE {label}: trial {p['trial_index']} journal refusals {ev.get('refusals_journal')} KL {ev.get('kl_journal'):.5f} | "
      f"re-eval refusals {ev.get('refusals_reeval')} KL {ev.get('kl_reeval'):.5f} | {p.get('validation_verdict')} {p.get('validation_reasons')}")
sys.exit(0 if p.get("validation_verdict", "").startswith("PASS") else 4)
PY
}
run_eval() {  # <model_dir> <label>
  [ "${SKIP_EVAL:-0}" = 1 ] && { say "SKIP_EVAL=1: $2 left in $1"; return 0; }
  say "eval $2"
  bash "$W/scripts/eval_local_model.sh" "$2" "$1" "$BASE_REPO" qwen2.5-32b__base || die "eval failed for $2"
  cp "$REC/$2/PROVENANCE.json" "$W/results/bench/$2/reexport_provenance.json"
  say "eval $2 done"
}
do_study() {  # <name> <study_dir> <val_trial> <rule_trial> <label> <extra args...>
  local name=$1 S=$2 vt=$3 rt=$4 lbl=$5; shift 5
  local OUT=$W/models_1x/$lbl VAL=$W/tmpmodels/reexport_1x_validate_${name}${vt}
  if [ -s "$W/results/bench/$lbl/summary.json" ]; then say "$lbl already evaluated ($W/results/bench/$lbl), skipping"; return 0; fi
  if [ -s "$OUT/.pushed" ] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
    say "$name: $OUT already exported"
  elif [ -s "$OUT/.pushed.pending" ] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
    # exported earlier but the rule-trial dev re-eval was outside tolerance
    if [ "${ALLOW_RULE_MISMATCH:-0}" = 1 ]; then
      say "WARNING: $name rule trial #$rt was outside tolerance; evaluating anyway (ALLOW_RULE_MISMATCH=1)"
      mv "$OUT/.pushed.pending" "$OUT/.pushed"
    else
      say "$name: $OUT exported but rule trial #$rt dev re-eval outside tolerance (see $REC/$lbl/PROVENANCE.json); benchmark NOT run (ALLOW_RULE_MISMATCH=1 to continue)"
      return 4
    fi
  else
    say "$name: validation #$vt then export #$rt"
    rm -rf "$VAL" "$OUT"
    "$PY" "$R/export_1x_trial.py" --study "$S" "${COMMON[@]}" "$@" \
        --job "$vt=$VAL" --job "$rt=$OUT" --ref-adapter "$vt=$S/adapter" --gate "$vt" \
        --directions-file "$REC/$name/refusal_directions_rtx.pt" --record-dir "$REC/$name"
    local rc=$?
    report_validation "$REC/$name/trial${vt}__$(basename "$VAL")"
    [ $rc -eq 3 ] && die "$name VALIDATION FAILED (trial #$vt); rule trial #$rt NOT exported. See $REC/$name/trial${vt}__*/PROVENANCE.json" 3
    [ $rc -eq 0 ] || die "$name export failed rc=$rc"
    rm -f "$VAL"/*.safetensors
    finish_rule "$OUT" "$lbl"
    local rr=$?
    if [ $rr -eq 0 ]; then
      mv "$OUT/.pushed.pending" "$OUT/.pushed"
    elif [ $rr -eq 4 ] && [ "${ALLOW_RULE_MISMATCH:-0}" = 1 ]; then
      say "WARNING: $name rule trial outside tolerance, continuing (ALLOW_RULE_MISMATCH=1)"
      mv "$OUT/.pushed.pending" "$OUT/.pushed"
    elif [ $rr -eq 4 ]; then
      # [review fix] not fatal for the whole script: the other study still runs; the script exits 4 at the end.
      say "$name rule trial #$rt dev re-eval outside tolerance of the journal; model kept in $OUT, benchmark NOT run (ALLOW_RULE_MISMATCH=1 to continue)"
      return 4
    else
      die "$name finish_rule failed rc=$rr"
    fi
  fi
  run_eval "$OUT" "$lbl"
}

# ---------------------------------------------------------------- 3. GENERAL: validate #84 (merge + hashes), export #96
RC_ALL=0
do_study general "$GEN" 84 96 qwen2.5-32b__heretic1x-general-kl0.1 --ref-hashes "84=$GEN/tensor_hashes_original.json" || RC_ALL=$?

# ---------------------------------------------------------------- 4. CODE: validate #179 (adapter level), export #134
if [ "${SKIP_CODE:-0}" = 1 ]; then
  say "code: skipped (SKIP_CODE=1)"
else
  do_study code "$CODE" 179 134 qwen2.5-32b__heretic1x-code-kl0.1 --datasets-root "$R/datasets" --adapter-only 179 || RC_ALL=$?
fi
say "=== DONE in $(( ($(date +%s) - T0) / 60 )) min (rc=$RC_ALL; 4 = a rule trial was exported but not benchmarked)"
exit $RC_ALL
