#!/usr/bin/env bash
# setup_lab.sh — one-time setup on the lab Linux PC (RTX 4060).
# Run from inside the project folder:   bash setup_lab.sh
set -e

echo "== GPU / driver =="
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version --format=csv
else echo "WARNING: nvidia-smi not found -> NVIDIA driver missing; torch will fall back to CPU"; fi

echo "== Python =="
PY=$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)
$PY --version

echo "== virtual env in ./.venv =="
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lab.txt   # Linux torch wheels from PyPI already include CUDA

echo "== check =="
python - <<'EOF'
import torch, numpy, pandas, scipy, sklearn
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
EOF
echo "Done. In every new terminal:  source .venv/bin/activate"
