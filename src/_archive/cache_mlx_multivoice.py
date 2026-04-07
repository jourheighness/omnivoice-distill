"""Cache MLX teacher outputs using diverse LibriTTS voices.

Each sample uses a DIFFERENT speaker's audio as reference,
giving the draft model diverse conditioning to learn from.

Usage:
    python local/cache_mlx_multivoice.py \
        --teacher_weights ~/path/to/omnivoice_mlx \
        --data_dir data/libritts_local \
        --output_dir ./cache_mlx_diverse \
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
    parser.add_argument("--output_dir", default="./cache_mlx_diverse")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_step", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Load manifest
    with open(data_dir / "manifest.json") as f:
        data_manifest = json.load(f)
    print(f"Loaded {len(data_manifest)} audio samples from {data_dir}")

    num_samples = min(args.num_samples, len(data_manifest))

    # Step 1: Encode ALL unique speakers' reference audio via PyTorch
    print("\nStep 1: Encoding reference audio for each speaker (PyTorch)...")
    from omnivoice import OmniVoice as OmniVoicePT

    pt_model = OmniVoicePT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)

    # Group samples by speaker, encode one ref per speaker
    speaker_refs = {}  # speaker_id -> (ref_audio_tokens, ref_text)
    speakers_seen = set()

    for entry in data_manifest[:num_samples]:
        sid = entry["speaker_id"]
        if sid in speakers_seen:
            continue
        speakers_seen.add(sid)

        try:
            prompt = pt_model.create_voice_clone_prompt(
                ref_audio=entry["audio_path"],
                ref_text=entry["text"][:100],
                preprocess_prompt=True,
            )
            ref_tokens = prompt.ref_audio_tokens.cpu().numpy()
            speaker_refs[sid] = {
                "tokens": ref_tokens,
                "text": prompt.ref_text,
            }
            print(f"  Speaker {sid}: {ref_tokens.shape[1]} frames")
        except Exception as e:
            print(f"  Speaker {sid}: FAILED ({e})")
            continue

    print(f"  Encoded {len(speaker_refs)} unique speakers")

    del pt_model
    if hasattr(torch, 'mps'):
        torch.mps.empty_cache()

    # Step 2: Load MLX teacher
    print("\nStep 2: Loading MLX teacher...")
    from omnivoice_mlx.generate import (
        OmniVoiceMLXConfig, OmniVoiceMLXModel, generate_iterative,
    )

    config = OmniVoiceMLXConfig()
    teacher = OmniVoiceMLXModel(config)
    wp = Path(args.teacher_weights)
    teacher.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
    mx.eval(teacher.parameters())
    print("  Teacher loaded.")

    # Setup tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    style_text = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_text, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(
        mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids))
    )

    # Step 3: Generate and cache
    print(f"\nStep 3: Caching {num_samples} samples...")
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

        # Build text tokens
        text_str = f"<|text_start|>{ref_text} {text}<|text_end|>"
        text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
        text_tokens = mx.broadcast_to(
            mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids))
        )

        # Estimate target length
        chars_per_frame = max(1, len(ref_text) / ref_audio_tokens.shape[1])
        target_len = max(25, min(150, int(len(text) / chars_per_frame)))

        # Build input
        target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)
        input_ids = mx.concatenate([style_tokens, text_tokens, ref_audio_tokens, target_masks], axis=1)
        L_total = input_ids.shape[1]
        L_cond = L_total - target_len

        audio_mask = mx.concatenate([
            mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
            mx.ones(ref_audio_tokens.shape[1] + target_len, dtype=mx.bool_),
        ])

        try:
            # Generate tokens
            mx.random.seed(i)
            tokens = generate_iterative(
                teacher, input_ids, audio_mask, target_len,
                num_step=args.num_step, guidance_scale=3.0,
            )
            mx.eval(tokens)

            # Extract conditioning
            ids_batch = mx.expand_dims(input_ids, 0)
            mask_batch = mx.expand_dims(audio_mask, 0)
            attn = mx.ones((1, 1, L_total, L_total), dtype=mx.bool_)

            embeds = teacher._prepare_embed_inputs(ids_batch, mask_batch)
            hidden = teacher.llm(embeds, mask=attn)
            cond_hidden = hidden[:, :L_cond, :]
            mx.eval(cond_hidden)

            # Save
            fname = f"sample_{i:04d}.npz"
            np.savez_compressed(
                output_dir / fname,
                cond_hidden=np.array(cond_hidden[0]),
                cb0_tokens=np.array(tokens[0]),
                all_tokens=np.array(tokens),
            )
            manifest.append({
                "file": fname,
                "target_len": target_len,
                "cond_len": L_cond,
                "speaker_id": sid,
                "text": text[:80],
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on sample {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  [{i+1}/{num_samples}] {rate:.1f}/s | "
                  f"{len(manifest)} cached | {errors} errors | ETA: {eta:.0f}s")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = time.perf_counter() - t_start
    speakers = set(m["speaker_id"] for m in manifest)
    print(f"\nCached {len(manifest)} samples ({len(speakers)} speakers) "
          f"to {output_dir}/ in {total:.0f}s ({errors} errors)")


if __name__ == "__main__":
    main()
