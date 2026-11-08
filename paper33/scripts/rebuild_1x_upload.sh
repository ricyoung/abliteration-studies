#!/usr/bin/env bash
# Run ON the RTX (2026-11-06). Rebuild the complete Heretic 1.x Qwen2.5-32B "code" merged model (the /data original has
# only 627/771 tensors), verify every tensor hash against the rebuild that was evaluated on 2026-11-06 (identical
# procedure: do_1x.sh -> merge_adapter.py on CPU), then upload it to a PRIVATE HF repo purely as a fast transfer channel
# to /data (box->home ssh is ~2.5 MB/s). Local copy is deleted only after a verified upload.
set -uo pipefail
source /workspace/paper33/env.sh
label=qwen2.5-32b__heretic1x-code; base_repo=Qwen/Qwen2.5-32B-Instruct
A=$W/adapters_1x/$label; D=$W/models_1x/$label; REF=$W/results/bench/$label/tensor_hashes_rebuilt.json
L=$W/logs/rebuild_1x_upload.log; exec >> "$L" 2>&1
echo "$(date -Is) START rebuild $label"
base_dir=$($W/heretic/.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$base_repo', local_files_only=True))") || exit 1
echo "base snapshot: $base_dir"
rm -rf "$D"
$W/heretic/.venv/bin/python $W/scripts/merge_adapter.py --base "$base_dir" --adapter "$A" --out "$D" || { echo FAILED merge; exit 1; }
$W/heretic/.venv/bin/python $W/scripts/tensor_hash.py "$D" > "$W/results/bench/$label/tensor_hashes_rebuild2.json" || exit 1
python3 - "$REF" "$W/results/bench/$label/tensor_hashes_rebuild2.json" <<'PY' || { echo "FAILED hash check"; exit 1; }
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ta = a.get("tensors", a); tb = b.get("tensors", b)
assert set(ta) == set(tb), f"tensor name sets differ: {len(ta)} vs {len(tb)}"
bad = [k for k in ta if ta[k] != tb[k]]
print(f"tensors {len(ta)}; mismatches {len(bad)}")
sys.exit(1 if bad else 0)
PY
echo "$(date -Is) hash check OK"
cat > "$D/README.md" <<'MD'
Private transfer copy (2026-11-06): Heretic 1.x "code" merge of Qwen/Qwen2.5-32B-Instruct, rebuilt bit-identically from the
original LoRA adapter (sha256 6767b439...) with merge_adapter.py on CPU. Tensor hashes: tensor_hashes_rebuild2.json.
Not a release; kept private for transfer to local storage.
MD
cp "$W/results/bench/$label/tensor_hashes_rebuild2.json" "$D/"
$W/heretic/.venv/bin/python - "$D" <<'PY' || { echo "FAILED upload"; exit 1; }
import sys
from huggingface_hub import HfApi
api = HfApi(); repo = "richardyoung/Qwen2.5-32B-Instruct-heretic1x-code-transfer"
api.create_repo(repo, private=True, exist_ok=True)
info = api.repo_info(repo)
assert info.private, "repo is not private - abort"
api.upload_folder(folder_path=sys.argv[1], repo_id=repo, commit_message="private transfer copy of rebuilt 1.x code merge")
print("uploaded", repo, api.repo_info(repo).sha)
PY
echo "$(date -Is) upload OK"
rm -rf "${D:?}"
echo "$(date -Is) END rebuild+upload rc=0"
