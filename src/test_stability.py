"""Test position stability during OmniVoice iterative unmasking.

Question: do token predictions at each position stabilize before the
final step? If so, we can start streaming audio from stable positions
before the full 8-step process completes.

Tracks per-position cb0 token values across all 8 steps to find when
each position "locks in" to its final value.
"""

import math
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def _get_time_steps(t_start, t_end, num_step, t_shift):
    t = torch.linspace(t_start, t_end, num_step)
    t = t_shift * t / (1 + (t_shift - 1) * t)
    return t


def _gumbel_sample(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    return logits / temperature + gumbel_noise


def generate_tracking_stability(model, task, gen_config, seed=42):
    """Run _generate_iterative and track per-position token predictions at every step."""
    torch.manual_seed(seed)

    B = task.batch_size
    assert B == 1, "Stability tracking only supports batch_size=1"

    inputs_list = [
        model._prepare_inference_inputs(
            task.texts[i], task.target_lens[i], task.ref_texts[i],
            task.ref_audio_tokens[i], task.langs[i], task.instructs[i],
            gen_config.denoise,
        )
        for i in range(B)
    ]

    c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
    max_c_len = max(c_lens)
    pad_id = model.config.audio_mask_id
    C = model.config.num_audio_codebook

    batch_input_ids = torch.full(
        (2 * B, C, max_c_len), pad_id, dtype=torch.long, device=model.device,
    )
    batch_audio_mask = torch.zeros(
        (2 * B, max_c_len), dtype=torch.bool, device=model.device,
    )
    batch_attention_mask = torch.zeros(
        (2 * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=model.device,
    )

    for i, inp in enumerate(inputs_list):
        c_len, u_len = c_lens[i], task.target_lens[i]
        batch_input_ids[i, :, :c_len] = inp["input_ids"]
        batch_audio_mask[i, :c_len] = inp["audio_mask"]
        batch_attention_mask[i, :, :c_len, :c_len] = True
        batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
        batch_audio_mask[B + i, :u_len] = inp["audio_mask"][..., -u_len:]
        batch_attention_mask[B + i, :, :u_len, :u_len] = True
        if max_c_len > u_len:
            pad_diag = torch.arange(u_len, max_c_len, device=model.device)
            batch_attention_mask[B + i, :, pad_diag, pad_diag] = True

    t_len = task.target_lens[0]
    tokens = torch.full(
        (B, C, t_len), model.config.audio_mask_id, dtype=torch.long, device=model.device,
    )

    timesteps = _get_time_steps(0.0, 1.0, gen_config.num_step + 1, gen_config.t_shift).tolist()
    total_mask = t_len * C
    rem = total_mask
    schedule = []
    for step in range(gen_config.num_step):
        num = (
            rem if step == gen_config.num_step - 1
            else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        )
        schedule.append(int(num))
        rem -= int(num)

    layer_ids = torch.arange(C, device=model.device).view(1, -1, 1)

    # Track: what does the model PREDICT at each position at each step
    # (regardless of whether it's unmasked yet)
    # Shape: (num_steps, C, t_len) — predicted token at each step
    all_predictions = torch.full(
        (gen_config.num_step, C, t_len), -1, dtype=torch.long, device=model.device,
    )
    # Also track what's actually committed (unmasked) at each step
    committed_tokens = torch.full(
        (gen_config.num_step, C, t_len), model.config.audio_mask_id,
        dtype=torch.long, device=model.device,
    )

    c_len = c_lens[0]

    for step in range(gen_config.num_step):
        batch_logits = model(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)

        k = schedule[step]

        c_logits = batch_logits[0:1, :, c_len - t_len:c_len, :]
        u_logits = batch_logits[1:2, :, :t_len, :]

        pred_tokens, scores = model._predict_tokens_with_scoring(
            c_logits, u_logits, gen_config,
        )

        # Store ALL predictions (even for already-unmasked positions)
        all_predictions[step] = pred_tokens[0]  # (C, t_len)

        # Standard unmasking logic
        scores = scores - (layer_ids * gen_config.layer_penalty_factor)
        if gen_config.position_temperature > 0.0:
            scores = _gumbel_sample(scores, gen_config.position_temperature)

        sample_tokens = tokens[0:1, :, :t_len]
        scores.masked_fill_(sample_tokens != model.config.audio_mask_id, -float("inf"))

        if k > 0:
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

        tokens[0:1, :, :t_len] = sample_tokens
        batch_input_ids[0:1, :, c_len - t_len:c_len] = sample_tokens
        batch_input_ids[1:2, :, :t_len] = sample_tokens

        committed_tokens[step] = sample_tokens[0]

    return tokens[0, :, :t_len], all_predictions.cpu(), committed_tokens.cpu()


def analyze_stability(all_predictions, committed_tokens, mask_id, num_steps):
    """Analyze when positions stabilize.

    A position "stabilizes" at step S if:
    - The model's prediction at step S matches the final committed value
    - AND the prediction doesn't change after step S
    """
    C, T = all_predictions.shape[1], all_predictions.shape[2]
    final_tokens = committed_tokens[-1]  # (C, T) — final output

    # For cb0 specifically (most important for audio quality)
    cb0_final = final_tokens[0]  # (T,)
    cb0_preds = all_predictions[:, 0, :]  # (num_steps, T)

    # When does each position first predict its final value AND stay there?
    stability_step = torch.full((T,), num_steps - 1, dtype=torch.long)

    for pos in range(T):
        final_val = cb0_final[pos]
        if final_val == mask_id:
            continue
        # Find earliest step where prediction matches final and never changes
        for s in range(num_steps):
            if cb0_preds[s, pos] == final_val:
                # Check if it stays
                stays = True
                for s2 in range(s + 1, num_steps):
                    if cb0_preds[s2, pos] != final_val:
                        stays = False
                        break
                if stays:
                    stability_step[pos] = s
                    break

    # When does each position get COMMITTED (unmasked)?
    commit_step = torch.full((T,), num_steps - 1, dtype=torch.long)
    for pos in range(T):
        for s in range(num_steps):
            if committed_tokens[s, 0, pos] != mask_id:
                commit_step[pos] = s
                break

    return {
        "cb0_stability_step": stability_step,
        "cb0_commit_step": commit_step,
        "cb0_predictions_per_step": cb0_preds,
        "cb0_final": cb0_final,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", default="barth_ref.wav")
    parser.add_argument("--seeds", default="42,123,456", help="Seeds to test")
    args = parser.parse_args()

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe you would do something like that! After everything we've been through together.",
        "Welcome to the annual science conference. Today we'll explore the fascinating world of quantum computing.",
    ]

    print("Loading OmniVoice...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    model.eval()

    gen_config = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=3.0,
        position_temperature=5.0,
        class_temperature=0.0,
    )

    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio, ref_text="Reference audio.", preprocess_prompt=True,
    )

    seeds = [int(s) for s in args.seeds.split(",")]
    num_steps = gen_config.num_step
    mask_id = model.config.audio_mask_id

    for text_idx, text in enumerate(test_texts):
        print(f"\n{'='*70}")
        print(f"Text {text_idx}: {text[:80]}")

        ref_len = prompt.ref_audio_tokens.shape[1]
        chars_per_frame = max(1, len(prompt.ref_text) / ref_len)
        target_len = max(25, min(150, int(len(text) / chars_per_frame)))

        task = GenerationTask(
            batch_size=1,
            texts=[text],
            target_lens=[target_len],
            langs=["English"],
            instructs=["None"],
            ref_audio_tokens=[prompt.ref_audio_tokens],
            ref_texts=[prompt.ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )

        for seed in seeds:
            print(f"\n  seed={seed}")

            with torch.no_grad():
                final_tokens, all_preds, committed = generate_tracking_stability(
                    model, task, gen_config, seed=seed,
                )

            result = analyze_stability(all_preds, committed, mask_id, num_steps)
            stab = result["cb0_stability_step"]
            commit = result["cb0_commit_step"]
            T = len(stab)

            # Distribution: what fraction of positions stabilize at each step?
            print(f"  target_len={T}")
            print(f"  CB0 prediction stability (step at which prediction matches final & stays):")
            for s in range(num_steps):
                count = (stab == s).sum().item()
                pct = count / T * 100
                bar = "#" * int(pct / 2)
                print(f"    step {s}: {count:3d}/{T} ({pct:5.1f}%) {bar}")

            print(f"  CB0 commit step (when actually unmasked):")
            for s in range(num_steps):
                count = (commit == s).sum().item()
                pct = count / T * 100
                bar = "#" * int(pct / 2)
                print(f"    step {s}: {count:3d}/{T} ({pct:5.1f}%) {bar}")

            # Key metric: how many positions are stable AND committed by step N?
            print(f"  Positions stable AND committed (streamable) by step:")
            for s in range(num_steps):
                stable_and_committed = ((stab <= s) & (commit <= s)).sum().item()
                pct = stable_and_committed / T * 100
                print(f"    after step {s}: {stable_and_committed:3d}/{T} ({pct:5.1f}%)")

            # Spatial analysis: are early positions stable earlier?
            quarters = [
                ("1st quarter", 0, T // 4),
                ("2nd quarter", T // 4, T // 2),
                ("3rd quarter", T // 2, 3 * T // 4),
                ("4th quarter", 3 * T // 4, T),
            ]
            print(f"  Mean stability step by position:")
            for name, start, end in quarters:
                mean_stab = stab[start:end].float().mean().item()
                mean_commit = commit[start:end].float().mean().item()
                print(f"    {name}: stability={mean_stab:.1f} commit={mean_commit:.1f}")

            # Do predictions "flip-flop" or converge monotonically?
            cb0_preds = result["cb0_predictions_per_step"]  # (num_steps, T)
            flips = 0
            for pos in range(T):
                for s in range(1, num_steps - 1):
                    if (cb0_preds[s, pos] != cb0_preds[s-1, pos] and
                        cb0_preds[s, pos] != cb0_preds[s+1, pos] and
                        cb0_preds[s, pos] != mask_id and
                        cb0_preds[s-1, pos] != mask_id):
                        flips += 1
            print(f"  Prediction flip-flops (changed then changed back): {flips}")


if __name__ == "__main__":
    main()
