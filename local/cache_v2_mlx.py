"""Cache v2 format using MLX teacher — targets match inference.

Saves raw token IDs (framework-independent conditioning) + MLX-generated
cb0 tokens (matching the inference teacher). This is the combination that
solves both the conditioning gap AND the target gap.

Usage:
    python local/cache_v2_mlx.py \
        --teacher_weights ~/path/to/omnivoice_mlx \
        --data_dir data/libritts_local \
        --output_dir ./cache_v2_mlx \
        --num_samples 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--data_dir", default="data/libritts_local")
    parser.add_argument("--output_dir", default="./cache_v2_mlx")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_step", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    with open(data_dir / "manifest.json") as f:
        data_manifest = json.load(f)

    num_samples = min(args.num_samples, len(data_manifest))
    print(f"Processing {num_samples} samples from {data_dir}")

    # Step 1: Encode unique speakers via PyTorch
    print("\nStep 1: Encoding speaker references (PyTorch)...")
    from omnivoice import OmniVoice as OmniVoicePT
    pt_model = OmniVoicePT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)

    speaker_refs = {}
    for entry in data_manifest[:num_samples]:
        sid = entry["speaker_id"]
        if sid in speaker_refs:
            continue
        try:
            prompt = pt_model.create_voice_clone_prompt(
                ref_audio=str(data_dir / Path(entry["audio_path"]).name),
                ref_text=entry["text"][:100],
                preprocess_prompt=True,
            )
            speaker_refs[sid] = {
                "tokens": prompt.ref_audio_tokens.cpu().numpy(),
                "text": prompt.ref_text,
            }
        except Exception as e:
            print(f"  Speaker {sid}: FAILED ({e})")

    print(f"  Encoded {len(speaker_refs)} speakers")
    del pt_model
    if hasattr(torch, 'mps'):
        torch.mps.empty_cache()

    # Step 2: Load MLX teacher
    print("\nStep 2: Loading MLX teacher...")
    from omnivoice_mlx.generate import OmniVoiceMLXConfig, OmniVoiceMLXModel
    from generate_deterministic import generate_deterministic

    config = OmniVoiceMLXConfig()
    teacher = OmniVoiceMLXModel(config)
    wp = Path(args.teacher_weights)
    teacher.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
    mx.eval(teacher.parameters())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(
        mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids))
    )

    # Step 3: Generate and cache in v2 format
    print(f"\nStep 3: Caching {num_samples} samples (v2 format, MLX targets)...")
    manifest = []
    errors = 0
    t_start = time.perf_counter()

    for i in range(num_samples):
        entry = data_manifest[i]
        sid = entry["speaker_id"]
        if sid not in speaker_refs:
            errors += 1
            continue

        ref = speaker_refs[sid]
        ref_audio_tokens = mx.array(ref["tokens"], dtype=mx.int32)
        ref_text = ref["text"]
        text = entry["text"]

        # Build tokens
        text_str = f"<|text_start|>{ref_text} {text}<|text_end|>"
        text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
        text_tokens = mx.broadcast_to(
            mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids))
        )

        chars_per_frame = max(1, len(ref_text) / ref_audio_tokens.shape[1])
        target_len = max(25, min(150, int(len(text) / chars_per_frame)))

        target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)
        input_ids = mx.concatenate([style_tokens, text_tokens, ref_audio_tokens, target_masks], axis=1)
        L_total = input_ids.shape[1]
        L_cond = L_total - target_len

        audio_mask = mx.concatenate([
            mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
            mx.ones(ref_audio_tokens.shape[1] + target_len, dtype=mx.bool_),
        ])

        try:
            # Generate with MLX teacher
            tokens = generate_deterministic(
                teacher, input_ids, audio_mask, target_len,
                num_step=args.num_step,
            )
            mx.eval(tokens)

            # Save v2 format: raw token IDs + MLX-generated cb0
            fname = f"sample_{i:04d}.npz"
            np.savez_compressed(
                output_dir / fname,
                cond_ids=np.array(input_ids[:, :L_cond]),     # (C, L_cond) int
                audio_mask=np.array(audio_mask[:L_cond]),     # (L_cond,) bool
                cb0_tokens=np.array(tokens[0]),                # (target_len,) int
                all_tokens=np.array(tokens),                   # (C, target_len) int
            )
            manifest.append({
                "file": fname,
                "target_len": target_len,
                "cond_len": L_cond,
                "speaker_id": sid,
            })
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = len(manifest) / elapsed
            eta = (num_samples - i - 1) / max(rate, 0.01)
            speakers = len(set(m["speaker_id"] for m in manifest))
            print(f"  [{i+1}/{num_samples}] {len(manifest)} cached | "
                  f"{speakers} speakers | {rate:.1f}/s | ETA {eta:.0f}s")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = time.perf_counter() - t_start
    speakers = len(set(m["speaker_id"] for m in manifest))
    print(f"\nDone: {len(manifest)} samples, {speakers} speakers, {total:.0f}s ({errors} errors)")
    print(f"Format: v2 (token IDs + MLX-generated targets)")


if __name__ == "__main__":
    main()
