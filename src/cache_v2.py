"""Cache teacher outputs as raw tokens (v2 — framework independent).

Saves input token IDs + audio mask + generated cb0 tokens.
No hidden states — the draft model embeds tokens itself.

Can run on either PyTorch (RunPod) or MLX (Mac) — output is identical
since it's just integer token IDs.

Usage:
    python src/cache_v2.py \
        --data_manifest data/libritts/manifest.json \
        --output_dir ./cache_v2 \
        --num_samples 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--output_dir", default="./cache_v2")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_step", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_manifest) as f:
        data_manifest = json.load(f)

    num_samples = min(args.num_samples, len(data_manifest))
    print(f"Processing {num_samples} samples")

    # Load teacher
    from omnivoice import OmniVoice
    print("Loading OmniVoice...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float16)
    model.eval()

    config = model.config
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    manifest = []
    errors = 0

    for i in tqdm(range(num_samples)):
        entry = data_manifest[i]

        try:
            # Create voice clone prompt (encodes ref audio)
            prompt = model.create_voice_clone_prompt(
                ref_audio=entry["audio_path"],
                ref_text=entry["text"][:100],
                preprocess_prompt=True,
            )

            ref_audio_tokens = prompt.ref_audio_tokens  # (C, T_ref) on device
            ref_text = prompt.ref_text

            # Build style tokens
            style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
            style_ids = tokenizer(style_str, return_tensors="pt").input_ids[0].to(args.device)
            style_tokens = style_ids.unsqueeze(0).expand(C, -1)

            # Build text tokens
            text_str = f"<|text_start|>{ref_text} {entry['text']}<|text_end|>"
            text_ids = tokenizer(text_str, return_tensors="pt").input_ids[0].to(args.device)
            text_tokens = text_ids.unsqueeze(0).expand(C, -1)

            # Estimate target length
            chars_per_frame = max(1, len(ref_text) / ref_audio_tokens.shape[1])
            target_len = max(25, min(150, int(len(entry["text"]) / chars_per_frame)))

            # Build full input: style | text | ref_audio | target_masks
            target_masks = torch.full((C, target_len), mask_id, dtype=torch.long, device=args.device)
            input_ids = torch.cat([style_tokens, text_tokens, ref_audio_tokens, target_masks], dim=1)
            input_ids = input_ids.unsqueeze(0)  # (1, C, L)

            L_total = input_ids.shape[-1]
            L_cond = L_total - target_len

            audio_mask = torch.cat([
                torch.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=torch.bool, device=args.device),
                torch.ones(ref_audio_tokens.shape[1] + target_len, dtype=torch.bool, device=args.device),
            ]).unsqueeze(0)

            # Generate tokens
            with torch.no_grad():
                inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
                # Use the model's internal generation
                try:
                    tokens = model._generate_iterative(
                        input_ids=input_ids, audio_mask=audio_mask,
                        target_len=target_len, num_step=args.num_step,
                    )
                except TypeError:
                    # Fallback: manual unmasking
                    from cache_teacher_real import _run_unmasking_loop
                    tokens = _run_unmasking_loop(
                        model, input_ids, audio_mask, target_len, num_step=args.num_step,
                    )

            if isinstance(tokens, torch.Tensor):
                tokens_np = tokens.cpu().numpy()
            else:
                tokens_np = np.array(tokens)
            if tokens_np.ndim == 3:
                tokens_np = tokens_np[0]

            # Save: raw token IDs (framework independent!)
            cond_ids = input_ids[0, :, :L_cond].cpu().numpy()  # (C, L_cond)
            cond_mask = audio_mask[0, :L_cond].cpu().numpy()   # (L_cond,)

            fname = f"sample_{i:04d}.npz"
            np.savez_compressed(
                output_dir / fname,
                cond_ids=cond_ids,            # (C, L_cond) int — framework independent
                audio_mask=cond_mask,         # (L_cond,) bool
                cb0_tokens=tokens_np[0],      # (target_len,) int
                all_tokens=tokens_np,         # (C, target_len) int
            )
            manifest.append({
                "file": fname,
                "target_len": target_len,
                "cond_len": int(L_cond),
                "text": entry["text"][:80],
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  Error on sample {i}: {e}")
            if errors > 50:
                print("Too many errors, stopping.")
                break
            continue

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(manifest)} samples to {output_dir}/ ({errors} errors)")


if __name__ == "__main__":
    main()
