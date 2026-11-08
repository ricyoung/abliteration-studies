#!/usr/bin/env bash
# Publish the small-pair models that were evaluated on the 4090 with --no-publish (this PC's HF token was
# read-only on 2026-11-05 night; a write token was added 06:34).
# For each label: re-export the SAME selected trial (select_trial.py --expect-trial) through heretic's UPLOAD action
# (public repo + heretic model card + tags), then check that every uploaded *.safetensors file (HF LFS sha256) is
# byte-identical to the locally kept merged model that was benchmarked. Mismatch -> exit 1 (repo left as is).
# Reproduce manifest: "full" for upstream built-ins (MPOA, ARA, DiM) -> Heretic Grimoire;
# "none" for SOM / J-space (need the paper33-plugins fork; J-space also reads a local direction file).
# Usage: publish_local.sh [label ...]      (default: all 10 small-pair labels; labels with hf_repo.txt are skipped)
set -uo pipefail
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder
HV=/data/heretic_2/heretic/.venv/bin
export HF_HOME=${HF_HOME:-/data/huggingface} PYTHONUNBUFFERED=1
HF_USER=richardyoung
labels=("$@")
[ ${#labels[@]} -eq 0 ] && for m in qwen3.5-4b qwen2.5-7b; do for x in mpoa ara dim som jspace; do labels+=("${m}__$x"); done; done
nfail=0
for label in "${labels[@]}"; do
  OUT=$P/results/bench-4090/$label
  if [ -s "$OUT/hf_repo.txt" ]; then echo "SKIP $label (already $(cat "$OUT/hf_repo.txt"))"; continue; fi
  [ -f "$OUT/summary.json" ] && [ -s "$OUT/reexport.sh" ] || { echo "FAIL $label: not evaluated yet"; nfail=$((nfail+1)); continue; }
  case $label in qwen3.5-4b__*) repo=Qwen/Qwen3.5-4B;; qwen2.5-7b__*) repo=Qwen/Qwen2.5-7B-Instruct;; *) echo "FAIL $label: unknown base"; nfail=$((nfail+1)); continue;; esac
  method=$(echo "$label" | sed -E 's/.*__//' | tr a-z A-Z | sed -e 's/^DIM$/DiM/' -e 's/^JSPACE$/JSpace/')
  name=$(basename "$repo")-$method-heretic
  repro=full; case $method in SOM|JSpace) repro=none;; esac
  KEPT=$(sed -n 's#.*--save-directory \([^ ]*\).*#\1#p' "$OUT/reexport.sh")
  [ -f "$KEPT/config.json" ] || { echo "FAIL $label: evaluated model missing: $KEPT"; nfail=$((nfail+1)); continue; }
  # same trial, fresh work dir (publish_tN) so the evaluated re-export's files are untouched
  sed -e 's#reexport_t\([0-9]*\)#publish_t\1#g' \
      -e "s#--patch-journal \(.*\) --trial-index#--patch-journal \1 --upload --upload-repro $repro --trial-index#" \
      -e "s#--model-action save --export-strategy MERGE --save-directory [^ ]*#--model-action upload --export-strategy MERGE --upload-repo-id $HF_USER/$name#" \
      -e 's#| tee reexport.log#| tee publish.log#' "$OUT/reexport.sh" > "$OUT/publish.sh"
  grep -q -- "--upload-repo-id $HF_USER/$name" "$OUT/publish.sh" && grep -q -- "--upload-repro $repro" "$OUT/publish.sh" \
    || { echo "FAIL $label: publish rewrite failed"; nfail=$((nfail+1)); continue; }
  echo "=== $label -> $HF_USER/$name (repro=$repro) start $(date -Is)"; s=$(date +%s)
  bash -o pipefail "$OUT/publish.sh" > "$OUT/publish.log" 2>&1 || { echo "FAIL $label: upload failed"; tail -15 "$OUT/publish.log"; nfail=$((nfail+1)); continue; }
  echo "    uploaded in $(( $(date +%s)-s ))s; verifying weights against $KEPT"
  "$HV/python" - "$HF_USER/$name" "$KEPT" "$OUT/publish_check.json" "$repro" <<'EOF'
import hashlib, json, sys, pathlib
from huggingface_hub import HfApi
repo, kept, out, repro = sys.argv[1:5]
info = HfApi().model_info(repo, files_metadata=True)
remote = {f.rfilename: (f.lfs or {}).get("sha256") for f in info.siblings if f.rfilename.endswith(".safetensors")}
local = {}
for p in sorted(pathlib.Path(kept).glob("*.safetensors")):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    local[p.name] = h.hexdigest()
files = [f.rfilename for f in info.siblings]
res = {"repo": repo, "sha": info.sha, "private": info.private, "tags": info.tags, "evaluated_model_dir": kept,
       "match": bool(local) and local == remote, "local": local, "remote": remote,
       "has_reproduce": any(f.startswith("reproduce/") for f in files), "repro_mode": repro}
json.dump(res, open(out, "w"), indent=1)
print(f"    match={res['match']} shards={len(local)} private={info.private} reproduce/={res['has_reproduce']} tags={info.tags}")
ok = res["match"] and not info.private and (res["has_reproduce"] == (repro != "none"))
sys.exit(0 if ok else 1)
EOF
  if [ $? -eq 0 ]; then echo "$HF_USER/$name" > "$OUT/hf_repo.txt"; echo "OK $label $HF_USER/$name"
  else echo "FAIL $label: verification failed (see $OUT/publish_check.json)"; nfail=$((nfail+1)); fi
done
exit $((nfail > 0))
