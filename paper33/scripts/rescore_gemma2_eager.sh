#!/usr/bin/env bash
# RTX (2026-11-06): re-run the 5 Gemma-2-27B dev re-scores. The released exports' config.json lost
# "_attn_implementation": "eager" (saved with transformers 5.18), so transformers loads them with SDPA while the base loads
# eager (Gemma-2 needs eager for attention-logit soft-capping) -> KL ~0.4 artefact. Download each export at its pinned sha,
# set _attn_implementation=eager in the LOCAL copy's config.json only (weights untouched; HF repos untouched), re-score.
set -uo pipefail
source /workspace/paper33/env.sh
G=/workspace/paper33/rescore_dev/gemma2_eager; mkdir -p $G/models
J=/workspace/paper33/rescore_dev/jobs_gemma2_eager.tsv
tail -n +2 $J | while IFS=$'\t' read -r run model method base_repo base_rev merged merged_rev rest; do
  repo=richardyoung/gemma-2-27b-it-$(case $method in dim) echo DiM;; mpoa) echo MPOA;; ara) echo ARA;; som) echo SOM;; gamma_mpoa) echo GMPOA;; esac)-heretic
  $W/heretic/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', revision='$merged_rev', local_dir='$merged')" || exit 1
  $W/heretic/.venv/bin/python - "$merged/config.json" <<'PY' || exit 1
import json, sys
p = sys.argv[1]; c = json.load(open(p)); c["_attn_implementation"] = "eager"; json.dump(c, open(p, "w"), indent=2)
print("patched", p)
PY
done
cd /workspace/paper33/rescore_dev && $W/heretic/.venv/bin/python $W/scripts/rescore_dev.py $J $G/out --heretic=$W/heretic/.venv/bin/heretic
rc=$?
rm -rf "${G:?}/models"
exit $rc
