#!/bin/bash
# RunPod A100 setup script for OmniVoice draft model distillation.
#
# Prerequisites: Start a RunPod GPU Pod with:
#   - GPU: A100 80GB (community cloud ~$0.89-1.19/hr)
#   - Template: RunPod PyTorch 2.x
#   - Disk: 50GB (for model weights + cache)
#
# Usage:
#   1. SSH into RunPod or open Jupyter terminal
#   2. git clone or upload the omnivoice-distill project
#   3. cd omnivoice-distill && bash runpod/setup.sh
#   4. bash runpod/run.sh

set -euo pipefail

echo "=== OmniVoice Draft Distillation — RunPod Setup ==="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -q torch safetensors numpy tqdm pyyaml tensorboard huggingface-hub transformers
echo ""

# Download OmniVoice model weights from HuggingFace
WEIGHTS_DIR="./weights/omnivoice"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Downloading OmniVoice weights from HuggingFace..."
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='k2-fsa/OmniVoice',
    local_dir='$WEIGHTS_DIR',
    allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model'],
)
print('Download complete.')
"
else
    echo "Weights already downloaded at $WEIGHTS_DIR"
fi

# Create cache and checkpoint dirs
mkdir -p ./cache ./checkpoints ./runs

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next: run the full pipeline:"
echo "  bash runpod/run.sh"
echo ""
echo "Or run steps individually:"
echo "  1. Cache teacher data:  python src/cache_teacher_torch.py --weights_dir $WEIGHTS_DIR"
echo "  2. Train draft:         python src/train.py --cache_dir ./cache"
echo "  3. Evaluate:            python src/eval.py --cache_dir ./cache --checkpoint ./checkpoints/best.pt"
