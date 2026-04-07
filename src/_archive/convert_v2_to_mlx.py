"""Convert v2 PyTorch draft checkpoint + teacher embeddings to MLX."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_mlx_v2 import DraftV2Config, DraftModelV2MLX


def convert(pt_path, teacher_safetensors, output_path):
    # Load PT checkpoint (trainable weights only — no teacher embeddings)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    pt_state = ckpt["model_state_dict"]

    print(f"PT checkpoint: epoch={ckpt['epoch']}, eval_acc={ckpt['eval_acc']:.1%}")
    print(f"Config: {cfg}")
    print(f"Trainable keys: {len(pt_state)}")

    # Load teacher embedding weights
    from safetensors.torch import load_file
    teacher_weights = load_file(teacher_safetensors)

    # Build MLX weights dict
    mlx_weights = {}

    # Teacher embeddings
    mlx_weights["text_embed.weight"] = mx.array(teacher_weights["llm.embed_tokens.weight"].float().numpy())
    mlx_weights["audio_embed.weight"] = mx.array(teacher_weights["audio_embeddings.weight"].float().numpy())
    print(f"Teacher text embed: {mlx_weights['text_embed.weight'].shape}")
    print(f"Teacher audio embed: {mlx_weights['audio_embed.weight'].shape}")

    # Trainable weights (skip inv_freq buffers)
    for k, v in pt_state.items():
        if "inv_freq" in k:
            continue
        mlx_weights[k] = mx.array(v.float().numpy())

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), mlx_weights)
    print(f"Saved to {output} ({output.stat().st_size / 1e6:.1f} MB)")

    # Verify load
    config = DraftV2Config(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    )
    model = DraftModelV2MLX(config)
    model.load_weights(str(output))
    mx.eval(model.parameters())
    print("MLX model loaded and verified!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", default="../checkpoints/best.pt")
    parser.add_argument("--teacher_safetensors", required=True)
    parser.add_argument("--output", default="../checkpoints/draft_v2_mlx.safetensors")
    args = parser.parse_args()
    convert(args.pt_path, args.teacher_safetensors, args.output)
