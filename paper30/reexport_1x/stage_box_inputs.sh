#!/usr/bin/env bash
# Local: (re)build box_inputs/ = everything run_on_rtx.sh needs on the RTX under $W/reexport_1x/ (small files only).
# Then (main session):  rsync -a box_inputs/ mc-rtx:/workspace/paper33/reexport_1x/
# plus the 4 code prompt files into /workspace/paper33/reexport_1x/datasets/configs/code_abliteration_data/.
set -euo pipefail
H=/data/heretic_2/paper30/v3_2611/reexport_1x
B=$H/box_inputs
SRC=/data/heretic_2/results/heretic
mkdir -p "$B/studies" "$B/datasets/configs/code_abliteration_data"
git -C /data/heretic_2/heretic archive --format=tar.gz --prefix=heretic-1x/ -o "$B/heretic-1x-v1.4.0.tar.gz" v1.4.0
for v in general code; do
  s=Qwen__Qwen2.5-32B-Instruct__$v
  mkdir -p "$B/studies/$s/study" "$B/studies/$s/adapter"
  cp -p "$SRC/$s"/study/*.jsonl "$B/studies/$s/study/"
  cp -p "$SRC/$s/result.json" "$B/studies/$s/"
  cp -p "$SRC/$s/adapter/adapter_config.json" "$SRC/$s/adapter/adapter_model.safetensors" "$B/studies/$s/adapter/"
done
cp -p "$SRC/Qwen__Qwen2.5-32B-Instruct__general/tensor_hashes_original.json" "$B/studies/Qwen__Qwen2.5-32B-Instruct__general/"
# 771-tensor hashes of the rebuilt 1.x code merge (the /data original is a broken 627-tensor copy)
cp -p /data/heretic_2/33_2611_does_refusal_removal_get_harder/runs/mc-rtx-1/results/bench/qwen2.5-32b__heretic1x-code/tensor_hashes_rebuilt.json \
      "$B/studies/Qwen__Qwen2.5-32B-Instruct__code/tensor_hashes_rebuilt_771.json"
cp -p "$H/export_1x_trial.py" "$H/journal_1x.py" "$H/run_on_rtx.sh" "$H/requirements-1x-lock.txt" "$H/trials.json" "$H/VERSION.md" "$B/"
# copy the code prompt files too if they have been fetched locally
for f in benign_code_prompts malicious_code_prompts benign_code_eval malicious_code_eval; do
  [ -s "$H/datasets/configs/code_abliteration_data/$f.txt" ] && cp -p "$H/datasets/configs/code_abliteration_data/$f.txt" "$B/datasets/configs/code_abliteration_data/"
done
(cd "$B" && find . -type f ! -name MANIFEST.sha256 ! -path './datasets/*' | sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$B"; wc -l < "$B/MANIFEST.sha256"
