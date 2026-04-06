"""Convert PyTorch draft model checkpoint to MLX safetensors."""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from draft_mlx import DraftMLXConfig, DraftModelMLX


def convert(pt_path: str, output_path: str):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = ckpt["model_state_dict"]

    print(f"PyTorch checkpoint: epoch={ckpt['epoch']}, eval_loss={ckpt['eval_loss']:.4f}, eval_acc={ckpt['eval_acc']:.1%}")
    print(f"Config: {cfg}")
    print(f"Keys: {len(state)}")

    # Convert to MLX arrays (skip RoPE buffers — computed at runtime)
    mlx_weights = {}
    for k, v in state.items():
        if "inv_freq" in k:
            continue
        arr = v.float().numpy()
        mlx_weights[k] = mx.array(arr)

    # Map PyTorch names to MLX names
    # PyTorch: layers.0.attn.qkv.weight -> MLX: layers.0.attn.qkv.weight (same)
    # The architectures match so names should be 1:1

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), mlx_weights)
    print(f"Saved MLX weights to {output} ({output.stat().st_size / 1e6:.1f} MB)")

    # Verify: load into MLX model
    mlx_config = DraftMLXConfig(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    )
    model = DraftModelMLX(mlx_config)
    model.load_weights(str(output))
    mx.eval(model.parameters())

    num_params = sum(p.size for _, p in model.parameters().items() if isinstance(p, mx.array))
    print(f"MLX model loaded: {num_params:,} parameters")
    print("Conversion verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", default="../checkpoints/best.pt")
    parser.add_argument("--output", default="../checkpoints/draft_mlx.safetensors")
    args = parser.parse_args()
    convert(args.pt_path, args.output)
