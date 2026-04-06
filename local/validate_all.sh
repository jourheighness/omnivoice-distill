#!/bin/bash
# Quick local validation on Mac Mini M4 Pro.
#
# Proves the full pipeline works end-to-end:
#   1. Cache teacher outputs (synthetic or real)
#   2. Train tiny draft model
#   3. Measure acceptance rate
#
# Takes ~5-10 minutes on M4 Pro with 24GB RAM.
#
# Usage:
#   cd omnivoice-distill
#   bash local/validate_all.sh [--weights /path/to/mlx/weights]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

WEIGHTS_PATH="${1:-}"
NUM_SAMPLES=20
TARGET_LEN=50
HIDDEN=256
LAYERS=2

echo "=== OmniVoice Draft Distillation — Local Validation ==="
echo "  Project:  $PROJECT_DIR"
echo "  Samples:  $NUM_SAMPLES"
echo "  Target:   ${TARGET_LEN} frames (${TARGET_LEN}x40ms = $((TARGET_LEN * 40))ms audio)"
echo "  Draft:    ${LAYERS}L / ${HIDDEN}H (tiny, for validation only)"
echo ""

# Phase 1: Cache teacher data
echo "=== Phase 1: Cache teacher outputs ==="

if [ -n "$WEIGHTS_PATH" ]; then
    echo "Using real OmniVoice MLX model at: $WEIGHTS_PATH"
    cd local
    python3 cache_teacher.py \
        --weights_path "$WEIGHTS_PATH" \
        --output_dir ../cache_local \
        --num_samples "$NUM_SAMPLES" \
        --target_len "$TARGET_LEN" \
        --synthetic
    cd ..
else
    echo "No weights path provided — using synthetic data."
    echo "(For real validation: bash local/validate_all.sh /path/to/mlx/weights)"
    cd local
    python3 cache_teacher.py \
        --weights_path dummy \
        --output_dir ../cache_local \
        --num_samples "$NUM_SAMPLES" \
        --target_len "$TARGET_LEN" \
        --synthetic
    cd ..
fi

echo ""

# Phase 2: Train draft model
echo "=== Phase 2: Train tiny draft model ==="
cd local
python3 train_local.py \
    --cache_dir ../cache_local \
    --num_epochs 3 \
    --batch_size 4 \
    --lr 1e-3 \
    --hidden_size "$HIDDEN" \
    --num_layers "$LAYERS" \
    --save_path ../checkpoints_local/draft.safetensors
cd ..

echo ""

# Phase 3: Test speculative decode (if weights available)
if [ -n "$WEIGHTS_PATH" ]; then
    echo "=== Phase 3: Test speculative decode ==="
    cd local
    python3 test_speculative.py \
        --weights_path "$WEIGHTS_PATH" \
        --draft_path ../checkpoints_local/draft.safetensors \
        --target_len "$TARGET_LEN" \
        --num_tests 5 \
        --draft_hidden "$HIDDEN" \
        --draft_layers "$LAYERS"
    cd ..
else
    echo "=== Phase 3: Skipped (no teacher weights for speculative decode test) ==="
    echo "The training pipeline validated successfully."
    echo "To test speculative decode, re-run with weights path."
fi

echo ""
echo "=== Validation Complete ==="
echo ""
echo "If the training loss decreased and accuracy improved across epochs,"
echo "the pipeline is working. Next steps:"
echo ""
echo "  1. If you have MLX weights, re-run with:"
echo "     bash local/validate_all.sh /path/to/mlx/weights"
echo ""
echo "  2. When ready for full training, set up RunPod:"
echo "     bash runpod/setup.sh"
echo "     bash runpod/run.sh"
