#!/usr/bin/env bash
set -o pipefail
mkdir -p /data/heretic_2/mlx_models
rsync -a --exclude 'DONE' Richard@10.10.0.100:heretic-mac/mlx/ /data/heretic_2/mlx_models/ && rsync -a Richard@10.10.0.100:heretic-mac/mlx/*.test.txt /data/heretic_2/mlx_models/ \
 && echo "pulled $(du -sh /data/heretic_2/mlx_models | cut -f1)" && cd /tmp && HF_HOME=/data/huggingface /data/heretic_2/heretic/.venv/bin/python /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/publish_mlx.py
