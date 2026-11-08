#!/usr/bin/env bash
# Run ON the rented GPU box (Ubuntu 22.04 + NVIDIA drivers).
# Sets up the workspace, heretic 2.x with CUDA torch, and logs everything.
# Usage: bash box_bootstrap.sh
set -euo pipefail

W=/workspace/paper33          # everything lives here; synced back to /data/heretic_2/33_*/runs/<box>/
mkdir -p "$W"/{logs,runs,env,hf} 2>/dev/null || { sudo mkdir -p /workspace && sudo chown "$USER" /workspace && mkdir -p "$W"/{logs,runs,env,hf}; }
exec > >(tee -a "$W/logs/bootstrap.log") 2>&1
echo "=== bootstrap $(date -Is) on $(hostname)"

# --- environment snapshot (kept for the paper's reproducibility section)
{ nvidia-smi; nvidia-smi -q; } > "$W/env/nvidia-smi.txt" 2>&1 || true
{ uname -a; cat /etc/os-release; lscpu; free -g; df -h; } > "$W/env/system.txt" 2>&1

# --- tools
sudo apt-get update -qq && sudo apt-get install -y -qq tmux git rsync htop nvtop jq >/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install -q "huggingface_hub[cli,hf_xet]" || true

# --- persistent env for every later shell
cat > "$W/env.sh" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
export W=$W
export HF_HOME=$W/hf
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1
EOF
grep -q "paper33/env.sh" ~/.bashrc || echo "source $W/env.sh" >> ~/.bashrc
source "$W/env.sh"

# --- heretic (working branch from the user's fork; falls back to master)
cd "$W"
[ -d heretic ] || git clone https://github.com/ricyoung/heretic.git
cd heretic
git fetch -q origin
git checkout -q working 2>/dev/null || git checkout -q master
git log -1 --format='heretic %H %s' | tee "$W/env/heretic_commit.txt"
uv sync
TORCH_VER=$(.venv/bin/python -c 'import torch;print(torch.__version__.split("+")[0])')
uv pip install --reinstall --exclude-newer "$(date +%F)" "torch==$TORCH_VER" torchvision \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
.venv/bin/python -c 'import torch;assert torch.cuda.is_available();print("torch",torch.__version__,torch.cuda.get_device_name(0))' | tee -a "$W/env/heretic_commit.txt"
uv pip freeze > "$W/env/pip_freeze.txt"

echo "=== bootstrap done $(date -Is)"
