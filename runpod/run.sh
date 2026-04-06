#!/bin/bash
# Full training pipeline for RunPod.
#
# Runs all three phases:
#   1. Cache teacher outputs (~1-4 hours depending on data)
#   2. Train draft model (~15-30 min)
#   3. Evaluate acceptance rate and speed
#
# Budget: ~$5-15 for 4-10 hours on A100 community cloud
#
# Usage:
#   bash runpod/run.sh [--num_samples 1000] [--epochs 5]

set -euo pipefail

NUM_SAMPLES="${1:-500}"
EPOCHS="${2:-5}"
HIDDEN=512
LAYERS=6
HEADS=8
BATCH=64
LR=3e-4

WEIGHTS_DIR="./weights/omnivoice"
CACHE_DIR="./cache"
CKPT_DIR="./checkpoints"

echo "=== OmniVoice Draft Distillation Pipeline ==="
echo "  Samples:    $NUM_SAMPLES"
echo "  Epochs:     $EPOCHS"
echo "  Draft:      ${LAYERS}L / ${HIDDEN}H / ${HEADS}heads"
echo "  Batch size: $BATCH"
echo ""

# Phase 1: Cache teacher outputs
echo "=== Phase 1: Caching teacher outputs ==="
echo "This is the slow part — teacher runs iterative unmasking per sample."
echo ""

python3 src/cache_teacher_torch.py \
    --weights_dir "$WEIGHTS_DIR" \
    --output_dir "$CACHE_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --target_len 75 \
    --num_step 8

echo ""
echo "=== Phase 2: Training draft model ==="
echo ""

python3 src/train.py \
    --cache_dir "$CACHE_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --hidden_size "$HIDDEN" \
    --num_layers "$LAYERS" \
    --num_heads "$HEADS" \
    --batch_size "$BATCH" \
    --num_epochs "$EPOCHS" \
    --lr "$LR"

echo ""
echo "=== Phase 3: Evaluation ==="
echo ""

python3 src/eval.py \
    --cache_dir "$CACHE_DIR" \
    --checkpoint "$CKPT_DIR/best.pt" \
    --num_tests 50

echo ""
echo "=== Pipeline complete ==="
echo "Checkpoints: $CKPT_DIR/"
echo "Logs:        ./runs/ (view with: tensorboard --logdir ./runs)"
echo ""
echo "To download the draft model:"
echo "  scp runpod:$CKPT_DIR/best.pt ./local_checkpoints/"
