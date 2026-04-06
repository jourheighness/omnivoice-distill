"""Cache teacher outputs using the existing OmniVoice MLX model.

Runs on Mac (Apple Silicon). Generates audio token sequences from the
teacher model and saves codebook-0 tokens + conditioning hidden states
for training the AR draft model.

Usage:
    python local/cache_teacher.py \
        --weights_path /path/to/mlx/weights \
        --output_dir ./cache_local \
        --num_samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Add parent paths so we can import the existing OmniVoice MLX code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))
from omnivoice_mlx.generate import (
    OmniVoiceMLXConfig,
    OmniVoiceMLXModel,
    generate_iterative,
)
from omnivoice_mlx.vocoder import AudioTokenizerDecoder


def extract_conditioning_hidden_states(
    model: OmniVoiceMLXModel,
    input_ids: mx.array,   # (C, L_total)
    audio_mask: mx.array,  # (L_total,)
    target_len: int,
) -> mx.array:
    """Run teacher forward and extract hidden states at conditioning positions.

    Returns: (1, L_cond, hidden_size) hidden states for the conditioning prefix.
    These are what the draft model will use as its conditioning input.
    """
    C = model.config.num_audio_codebook
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    # Prepare inputs as the model expects
    ids = mx.expand_dims(input_ids, axis=0)  # (1, C, L)
    mask = mx.expand_dims(audio_mask, axis=0)  # (1, L)
    attn = mx.ones((1, 1, L_total, L_total), dtype=mx.bool_)

    # Get embeddings
    inputs_embeds = model._prepare_embed_inputs(ids, mask)

    # Run through LLM backbone to get hidden states
    hidden_states = model.llm(inputs_embeds, mask=attn)  # (1, L, H)

    # Extract conditioning region only
    cond_hidden = hidden_states[:, :L_cond, :]  # (1, L_cond, H)
    mx.eval(cond_hidden)
    return cond_hidden


def generate_and_cache(
    model: OmniVoiceMLXModel,
    input_ids: mx.array,
    audio_mask: mx.array,
    target_len: int,
    num_step: int = 8,
    guidance_scale: float = 3.0,
) -> dict:
    """Generate tokens from teacher and cache everything needed for training.

    Returns dict with:
        - cond_hidden: (L_cond, H) conditioning hidden states
        - cb0_tokens: (target_len,) codebook-0 token sequence
        - all_tokens: (C, target_len) all codebook tokens
        - target_len: int
    """
    # Generate all codebook tokens via iterative unmasking
    tokens = generate_iterative(
        model, input_ids, audio_mask, target_len,
        num_step=num_step, guidance_scale=guidance_scale,
    )
    mx.eval(tokens)

    # Extract conditioning hidden states
    cond_hidden = extract_conditioning_hidden_states(
        model, input_ids, audio_mask, target_len,
    )

    return {
        "cond_hidden": np.array(cond_hidden[0]),       # (L_cond, H)
        "cb0_tokens": np.array(tokens[0]),              # (target_len,) codebook 0
        "all_tokens": np.array(tokens),                 # (C, target_len)
        "target_len": target_len,
    }


# --- Synthetic test data for validation ---

def make_synthetic_inputs(config: OmniVoiceMLXConfig, target_len: int = 50):
    """Create synthetic inputs for testing the cache pipeline.

    In production you'd use real text + reference audio. For validation,
    we just need the shapes to be right so we can test the full pipeline.
    """
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    # Simulate: [style(10) | text(20) | ref_audio(30) | target(target_len)]
    style_len = 10
    text_len = 20
    ref_len = 30
    L_total = style_len + text_len + ref_len + target_len

    # Style + text: random text tokens in codebook-0 range, broadcast to C codebooks
    text_tokens = mx.random.randint(0, 1000, (text_len + style_len,))
    text_ids = mx.broadcast_to(text_tokens.reshape(1, -1), (C, text_len + style_len))

    # Ref audio: random audio tokens per codebook
    ref_ids = mx.random.randint(0, 1024, (C, ref_len))

    # Target: all masks (will be filled by generation)
    target_ids = mx.full((C, target_len), mask_id, dtype=mx.int32)

    input_ids = mx.concatenate([text_ids, ref_ids, target_ids], axis=1)

    # Audio mask: False for text, True for audio regions
    audio_mask = mx.concatenate([
        mx.zeros(text_len + style_len, dtype=mx.bool_),
        mx.ones(ref_len + target_len, dtype=mx.bool_),
    ])

    return input_ids, audio_mask


def main():
    parser = argparse.ArgumentParser(description="Cache teacher outputs for draft model training")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to OmniVoice MLX weights dir")
    parser.add_argument("--output_dir", type=str, default="./cache_local")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--target_len", type=int, default=50, help="Target audio length in frames (25fps)")
    parser.add_argument("--num_step", type=int, default=8, help="Unmasking steps")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic inputs (for pipeline validation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OmniVoice MLX model from {args.weights_path}...")
    config = OmniVoiceMLXConfig()
    model = OmniVoiceMLXModel(config)

    weights_path = Path(args.weights_path)
    model_weights = mx.load(str(weights_path / "model.safetensors"))
    model.load_weights(list(model_weights.items()))
    mx.eval(model.parameters())
    print("Model loaded.")

    manifest = []

    for i in range(args.num_samples):
        t0 = time.perf_counter()

        if args.synthetic:
            input_ids, audio_mask = make_synthetic_inputs(config, args.target_len)
        else:
            raise NotImplementedError("Real data loading not yet implemented — use --synthetic for validation")

        cached = generate_and_cache(
            model, input_ids, audio_mask, args.target_len,
            num_step=args.num_step,
        )

        # Save as npz
        fname = f"sample_{i:04d}.npz"
        np.savez_compressed(
            output_dir / fname,
            cond_hidden=cached["cond_hidden"],
            cb0_tokens=cached["cb0_tokens"],
            all_tokens=cached["all_tokens"],
        )

        elapsed = time.perf_counter() - t0
        manifest.append({
            "file": fname,
            "target_len": args.target_len,
            "cond_len": cached["cond_hidden"].shape[0],
        })
        print(f"  [{i+1}/{args.num_samples}] {fname} | {elapsed:.1f}s | "
              f"cond_len={cached['cond_hidden'].shape[0]} target_len={args.target_len}")

    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(manifest)} samples to {output_dir}/")
    print(f"Total conditioning hidden dim: {config.hidden_size}")


if __name__ == "__main__":
    main()
