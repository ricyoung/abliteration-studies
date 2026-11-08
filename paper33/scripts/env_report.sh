#!/usr/bin/env bash
# Print a reproducibility snapshot of THIS machine (paper 33, Methods / Appendix).
# Usage: bash env_report.sh <heretic_python> [bench_python]   > env_<host>.txt
HP=${1:-python3}; BP=${2:-}
echo "# environment report  host=$(hostname)  date=$(date -Is)"
echo "## OS";      (lsb_release -ds 2>/dev/null || grep PRETTY /etc/os-release); uname -r
echo "## CPU";     lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket" ;
echo "## RAM";     free -g | head -2
echo "## Disk";    df -h / /workspace /data 2>/dev/null | sort -u
echo "## GPU";     nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit,clocks.max.sm --format=csv
nvidia-smi | grep -E "CUDA Version|Driver Version" | head -1
echo "## nvcc";    (nvcc --version 2>/dev/null | tail -1) || echo "no CUDA toolkit (runtime from wheels)"
echo "## heretic python ($HP)"
"$HP" - <<'EOF'
import importlib, platform, sys
print("python", platform.python_version())
import torch
print("torch", torch.__version__, "| CUDA (torch build)", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version(),
      "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "| capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
for m in ["transformers","peft","accelerate","optuna","datasets","bitsandbytes","safetensors","huggingface_hub","numpy","heretic"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "not installed")
EOF
for repo in /workspace/paper33/heretic /workspace/scratch/som/heretic /data/heretic_2/heretic /data/heretic_2/external/jspace-refusal; do
  [ -d "$repo/.git" ] && echo "git $repo: $(git -C "$repo" rev-parse --abbrev-ref HEAD) $(git -C "$repo" rev-parse --short HEAD) $(git -C "$repo" status --porcelain | wc -l) uncommitted"
done
if [ -n "$BP" ] && [ -x "$BP" ]; then
  echo "## bench python ($BP)"
  "$BP" - <<'EOF'
import importlib
for m in ["torch","vllm","lm_eval","transformers","datasets"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "not installed")
EOF
fi
