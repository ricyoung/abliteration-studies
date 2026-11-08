#!/usr/bin/env bash
# Run ON the RTX. Rebuild a Heretic 1.x "general" merged model from its LoRA adapter (pushed from /data/heretic_2/results/heretic)
# + the base model (HF cache), record provenance + per-tensor hashes (compared offline with the original 1.x merged model on /data),
# then evaluate it with the identical protocol. Usage: do_1x.sh <label> <base_repo> <base_label>
set -uo pipefail
source /workspace/paper33/env.sh
label=$1; base_repo=$2; base_label=$3
A=$W/adapters_1x/$label; D=$W/models_1x/$label; OUT=$W/results/bench/$label; mkdir -p "$OUT"
base_dir=$($W/heretic/.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$base_repo', local_files_only=True))") || exit 1
rm -rf "$D"
$W/heretic/.venv/bin/python $W/scripts/merge_adapter.py --base "$base_dir" --adapter "$A" --out "$D" > "$OUT/merge.log" 2>&1 || { tail -5 "$OUT/merge.log"; exit 1; }
$W/heretic/.venv/bin/python $W/scripts/tensor_hash.py "$D" > "$OUT/tensor_hashes_rebuilt.json" || exit 1
python3 - "$A" "$base_dir" "$OUT/tensor_hashes_rebuilt.json" > "$D/.pushed" <<'PY'
import hashlib, json, sys, os
a, base, th = sys.argv[1:4]
print(json.dumps({"how": "rebuilt on box: merge_adapter.py (paper32) of the Heretic 1.x adapter into the base",
  "adapter_sha256": hashlib.sha256(open(os.path.join(a, "adapter_model.safetensors"), "rb").read()).hexdigest(),
  "base_snapshot": base, "tensor_hashes": th}, indent=1))
PY
bash $W/scripts/eval_local_model.sh "$label" "$D" "$base_repo" "$base_label"
