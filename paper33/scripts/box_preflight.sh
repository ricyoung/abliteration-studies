#!/usr/bin/env bash
# Run ON the box after bootstrap. Checks the environment and writes env/preflight.txt.
# Exit code != 0 means: do not start runs.
source /workspace/paper33/env.sh
OUT=$W/env/preflight.txt; : > "$OUT"; fail=0
chk(){ if eval "$2" >>"$OUT" 2>&1; then echo "PASS $1" | tee -a "$OUT"; else echo "FAIL $1" | tee -a "$OUT"; fail=1; fi; }
echo "=== preflight $(date -Is)" | tee -a "$OUT"
chk "gpu visible"        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
chk "torch cuda"         "$W/heretic/.venv/bin/python -c 'import torch;assert torch.cuda.is_available();print(torch.__version__,torch.cuda.get_device_name(0))'"
chk "bf16 matmul on gpu" "$W/heretic/.venv/bin/python -c 'import torch;a=torch.randn(4096,4096,device=\"cuda\",dtype=torch.bfloat16);torch.cuda.synchronize();print((a@a).float().abs().mean().item())'"
chk "heretic cli"        "$W/heretic/.venv/bin/heretic --help | head -1"
chk "hf cli"             "hf version"
chk "hf auth (optional)" "hf auth whoami || true"
chk "disk >= 300G free"  "[ \$(df --output=avail -BG $W | tail -1 | tr -dc 0-9) -ge 300 ] && df -h $W"
chk "ram >= 100G"        "[ \$(free -g | awk '/Mem:/{print \$2}') -ge 100 ] && free -g"
chk "internet speed"     "curl -s -o /dev/null -w 'hf download speed: %{speed_download} B/s\n' --max-time 30 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
echo "=== preflight $( [ $fail -eq 0 ] && echo OK || echo FAILED ) $(date -Is)" | tee -a "$OUT"
exit $fail
