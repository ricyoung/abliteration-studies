#!/usr/bin/env bash
# 4090 (2026-11-06): evaluate an ORIGINAL Heretic 1.x merged model (paper 30) with the paper-33 protocol, after verifying
# that a fresh CPU rebuild from its adapter (merge_adapter.py, as do_1x.sh on the RTX) is tensor-identical to it.
# Never deletes the original. Usage: eval_1x_local.sh <label> <orig_model_dir> <adapter_dir> <base_repo> <base_label>
set -uo pipefail
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; HV=/data/heretic_2/heretic/.venv/bin
export HF_HOME=${HF_HOME:-/data/huggingface} PYTHONUNBUFFERED=1 BENCH_GPU_UTIL=${BENCH_GPU_UTIL:-0.72} BENCH_VLLM_EXTRA='{}'
label=$1; MODEL=$2; A=$3; base_repo=$4; base_label=$5
OUT=$P/results/bench-4090/$label; mkdir -p "$OUT"
REF=$P/results/bench-4090/$base_label/refusal/responses.jsonl
[ -s "$REF" ] || { echo "missing original-model reference answers: $REF"; exit 1; }
TMP=$P/runs/local-4090/rebuild_tmp/$label; rm -rf "${TMP:?}"
base_dir=$($HV/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$base_repo'))") || exit 1
$HV/python $P/scripts/merge_adapter.py --base "$base_dir" --adapter "$A" --out "$TMP" > "$OUT/merge.log" 2>&1 || { tail -5 "$OUT/merge.log"; exit 1; }
$HV/python $P/scripts/tensor_hash.py "$TMP" > "$OUT/tensor_hashes_rebuilt.json" || exit 1
$HV/python $P/scripts/tensor_hash.py "$MODEL" > "$OUT/tensor_hashes_original.json" || exit 1
python3 - "$OUT/tensor_hashes_original.json" "$OUT/tensor_hashes_rebuilt.json" "$A" "$base_dir" "$MODEL" > "$OUT/push_manifest.json" <<'PY' || { echo "REBUILD != ORIGINAL"; exit 1; }
import json, sys, hashlib, os
o, r = (json.load(open(p)) for p in sys.argv[1:3]); a, base, model = sys.argv[3:6]
bad = sorted(k for k in set(o) | set(r) if o.get(k) != r.get(k))
print(json.dumps({"how": "original 1.x merged model evaluated; verified tensor-identical to a CPU rebuild (merge_adapter.py) of its adapter",
  "model_dir": model, "adapter_sha256": hashlib.sha256(open(os.path.join(a, "adapter_model.safetensors"), "rb").read()).hexdigest(),
  "base_snapshot": base, "n_tensors": len(o), "mismatches": len(bad)}, indent=1))
sys.exit(1 if bad or len(o) != len(r) else 0)
PY
rm -rf "${TMP:?}"
bash "$P/scripts/bench/run.sh" "$MODEL" "$OUT" --limit-mode paper --method "$label" --base-model "$base_repo" \
     --refusal-sets "$P/scripts/bench/refusal_sets.json" --reference-responses "$REF"
