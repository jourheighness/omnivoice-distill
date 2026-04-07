"""Cache v2 with SOFT targets — save teacher's top-k logits per position.

Instead of hard token IDs, saves the teacher's probability distribution
so the draft can learn via KL divergence. This handles high-entropy
stochastic generation where point prediction fails.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--output_dir", default="./cache_v2_soft")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=64, help="Save top-k logits per position")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_manifest) as f:
        data_manifest = json.load(f)

    from omnivoice import OmniVoice
    from transformers import AutoTokenizer

    print("Loading teacher...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    C, mask_id = 8, 1024
    V = 1025  # audio vocab size per codebook

    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="pt").input_ids[0].to(args.device)
    style_tokens = style_ids.unsqueeze(0).expand(C, -1)

    manifest = []
    errors = 0
    num = min(args.num_samples, len(data_manifest))

    for i in tqdm(range(num)):
        entry = data_manifest[i]
        text = entry["text"]

        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=entry["audio_path"], ref_text=text[:100],
                preprocess_prompt=True,
            )
            ref_tokens = prompt.ref_audio_tokens

            chars_per_frame = max(1, len(prompt.ref_text) / ref_tokens.shape[1])
            target_len = max(25, min(150, int(len(text) / chars_per_frame)))

            text_str = f"<|text_start|>{prompt.ref_text} {text}<|text_end|>"
            text_ids = tokenizer(text_str, return_tensors="pt").input_ids[0].to(args.device)
            text_toks = text_ids.unsqueeze(0).expand(C, -1)

            target_masks = torch.full((C, target_len), mask_id, dtype=torch.long, device=args.device)
            input_ids = torch.cat([style_tokens, text_toks, ref_tokens, target_masks], dim=1).unsqueeze(0)
            L_total = input_ids.shape[-1]
            L_cond = L_total - target_len

            audio_mask = torch.cat([
                torch.zeros(style_tokens.shape[1] + text_toks.shape[1], dtype=torch.bool, device=args.device),
                torch.ones(ref_tokens.shape[1] + target_len, dtype=torch.bool, device=args.device),
            ]).unsqueeze(0)

            # Get teacher's logits via forward pass on fully masked target
            with torch.no_grad():
                inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
                llm_out = model.llm(inputs_embeds=inputs_embeds, return_dict=True)
                hidden = llm_out.last_hidden_state  # (1, L, H)

                # Audio heads: (1, L, C*V) -> (1, C, T, V)
                all_logits = model.audio_heads(hidden)  # (1, L, C*V)
                target_logits = all_logits[:, -target_len:, :]  # (1, T, C*V)
                target_logits = target_logits.reshape(1, target_len, C, V)
                target_logits = target_logits.permute(0, 2, 1, 3)  # (1, C, T, V)

                # Get cb0 logits and compute soft targets
                cb0_logits = target_logits[0, 0, :, :].float()  # (T, V)

                # Top-k: save indices and log-probs
                cb0_probs = F.softmax(cb0_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(cb0_probs, args.top_k, dim=-1)

                # Also get hard target (argmax)
                cb0_hard = cb0_logits.argmax(dim=-1)

            # Save
            fname = f"sample_{i:04d}.npz"
            np.savez_compressed(
                output_dir / fname,
                cond_ids=input_ids[0, :, :L_cond].cpu().numpy(),
                audio_mask=audio_mask[0, :L_cond].cpu().numpy(),
                cb0_tokens=cb0_hard.cpu().numpy(),          # hard targets
                topk_indices=topk_indices.cpu().numpy(),     # (T, top_k)
                topk_probs=topk_probs.cpu().half().numpy(),  # (T, top_k) float16
            )
            manifest.append({
                "file": fname,
                "target_len": target_len,
                "cond_len": int(L_cond),
                "top_k": args.top_k,
            })

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  Error {i}: {e}")
            continue

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(manifest)} samples (soft targets, top-{args.top_k}) to {output_dir}/ ({errors} errors)")


if __name__ == "__main__":
    main()
