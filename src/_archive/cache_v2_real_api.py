"""Cache v2 using the REAL OmniVoice _generate_iterative API.

Uses GenerationTask + OmniVoiceGenerationConfig to match actual inference.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--output_dir", default="./cache_v2_real")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_manifest) as f:
        data_manifest = json.load(f)

    print(f"Processing {min(args.num_samples, len(data_manifest))} samples")

    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
    from transformers import AutoTokenizer

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    gen_config = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=3.0,
        position_temperature=5.0,
        class_temperature=0.0,
    )

    C, mask_id = 8, 1024
    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="pt").input_ids[0].to(args.device)
    style_tokens = style_ids.unsqueeze(0).expand(C, -1)

    manifest = []
    errors = 0

    for i in tqdm(range(min(args.num_samples, len(data_manifest)))):
        entry = data_manifest[i]
        text = entry["text"]

        try:
            # Encode ref audio
            prompt = model.create_voice_clone_prompt(
                ref_audio=entry["audio_path"],
                ref_text=text[:100],
                preprocess_prompt=True,
            )
            ref_tokens = prompt.ref_audio_tokens

            # Target length
            chars_per_frame = max(1, len(prompt.ref_text) / ref_tokens.shape[1])
            target_len = max(25, min(150, int(len(text) / chars_per_frame)))

            # Generate using REAL API
            task = GenerationTask(
                batch_size=1,
                texts=[text],
                target_lens=[target_len],
                langs=["English"],
                instructs=["None"],
                ref_audio_tokens=[ref_tokens],
                ref_texts=[prompt.ref_text],
                ref_rms=[getattr(prompt, "ref_rms", 0.1)],
            )

            with torch.no_grad():
                token_list = model._generate_iterative(task, gen_config)
                tokens = token_list[0]  # (C, T)

            tokens_np = tokens.cpu().numpy()

            # Build conditioning token IDs (same format as before)
            text_str = f"<|text_start|>{prompt.ref_text} {text}<|text_end|>"
            text_ids = tokenizer(text_str, return_tensors="pt").input_ids[0].to(args.device)
            text_toks = text_ids.unsqueeze(0).expand(C, -1)

            target_masks = torch.full((C, target_len), mask_id, dtype=torch.long, device=args.device)
            input_ids = torch.cat([style_tokens, text_toks, ref_tokens, target_masks], dim=1)
            L_cond = input_ids.shape[1] - target_len

            audio_mask = torch.cat([
                torch.zeros(style_tokens.shape[1] + text_toks.shape[1], dtype=torch.bool, device=args.device),
                torch.ones(ref_tokens.shape[1] + target_len, dtype=torch.bool, device=args.device),
            ])

            fname = f"sample_{i:04d}.npz"
            np.savez_compressed(
                output_dir / fname,
                cond_ids=input_ids[:, :L_cond].cpu().numpy(),
                audio_mask=audio_mask[:L_cond].cpu().numpy(),
                cb0_tokens=tokens_np[0],
                all_tokens=tokens_np,
            )
            manifest.append({
                "file": fname,
                "target_len": target_len,
                "cond_len": int(L_cond),
            })

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  Error {i}: {e}")
            continue

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(manifest)} samples to {output_dir}/ ({errors} errors)")


if __name__ == "__main__":
    main()
