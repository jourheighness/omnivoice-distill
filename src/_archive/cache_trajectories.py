"""Cache teacher trajectories for SDTT step distillation.

For each training sample, runs the teacher's 8-step generation and saves:
- Per-step token states (the partially-unmasked sequence after each step)
- Per-step logits (the model's predictions at each step, after CFG)
- Conditioning inputs (input_ids, audio_mask, attention_mask)

These trajectories let us train a student to skip steps:
  Student sees state_at_step_k, trained to produce logits_at_step_{k+2}
"""

import math
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


def _get_time_steps(t_start, t_end, num_step, t_shift):
    t = torch.linspace(t_start, t_end, num_step)
    t = t_shift * t / (1 + (t_shift - 1) * t)
    return t


def cache_trajectory(model, task, gen_config, seed=None):
    """Run 8-step generation, return per-step states and logits."""
    if seed is not None:
        torch.manual_seed(seed)

    B = task.batch_size
    assert B == 1
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id

    inp = model._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0],
        task.ref_audio_tokens[0], task.langs[0], task.instructs[0],
        gen_config.denoise,
    )

    # inp has batch dim: input_ids (1, C, c_len), audio_mask (1, c_len)
    inp_ids = inp["input_ids"].squeeze(0)  # (C, c_len)
    inp_amask = inp["audio_mask"].squeeze(0)  # (c_len,)
    c_len = inp_ids.size(1)
    t_len = task.target_lens[0]

    # Build batched inputs for CFG (cond + uncond)
    batch_input_ids = torch.full(
        (2, C, c_len), mask_id, dtype=torch.long, device=model.device,
    )
    batch_audio_mask = torch.zeros(
        (2, c_len), dtype=torch.bool, device=model.device,
    )
    batch_attention_mask = torch.zeros(
        (2, 1, c_len, c_len), dtype=torch.bool, device=model.device,
    )

    batch_input_ids[0, :, :c_len] = inp_ids
    batch_audio_mask[0, :c_len] = inp_amask
    batch_attention_mask[0, :, :c_len, :c_len] = True

    u_len = t_len
    batch_input_ids[1, :, :u_len] = inp_ids[:, -u_len:]
    batch_audio_mask[1, :u_len] = inp_amask[-u_len:]
    batch_attention_mask[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pad_diag = torch.arange(u_len, c_len, device=model.device)
        batch_attention_mask[1, :, pad_diag, pad_diag] = True

    tokens = torch.full(
        (C, t_len), mask_id, dtype=torch.long, device=model.device,
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

    layer_ids = torch.arange(C, device=model.device).view(-1, 1)

    # Storage for trajectory
    step_states = []   # token state BEFORE each step's forward pass
    step_logits = []   # CFG-combined logits at each step (target region only)

    for step in range(gen_config.num_step):
        # Save state before this step
        step_states.append(tokens.clone().cpu())

        # Update batch inputs
        batch_input_ids[0, :, c_len - t_len:c_len] = tokens
        batch_input_ids[1, :, :t_len] = tokens

        with torch.no_grad():
            batch_logits = model(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

        c_logits = batch_logits[0:1, :, c_len - t_len:c_len, :]  # (1, C, T, V)
        u_logits = batch_logits[1:2, :, :t_len, :]  # (1, C, T, V)

        # Compute CFG-combined log-probs (same as _predict_tokens_with_scoring)
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        log_probs[..., mask_id] = -float("inf")

        # Save the CFG logits (as float16 to save space)
        step_logits.append(log_probs[0].cpu().half())  # (C, T, V)

        # Predict tokens and scores for unmasking
        pred_tokens = log_probs.argmax(dim=-1)[0]  # (C, T)
        scores = log_probs.max(dim=-1)[0][0]  # (C, T)

        scores = scores - (layer_ids * gen_config.layer_penalty_factor)

        if gen_config.position_temperature > 0.0:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores) + 1e-20) + 1e-20)
            scores = scores / gen_config.position_temperature + gumbel_noise

        is_masked = tokens == mask_id
        scores[~is_masked] = float("-inf")

        k = schedule[step]
        if k > 0:
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            tokens = flat_tokens.view(C, t_len)

    # Final state
    step_states.append(tokens.clone().cpu())

    return {
        "step_states": step_states,   # list of (C, T) tensors, len=num_step+1
        "step_logits": step_logits,   # list of (C, T, V) tensors, len=num_step
        "cond_ids": inp_ids.cpu(),       # (C, c_len)
        "audio_mask": inp_amask.cpu(),  # (c_len,)
        "c_len": c_len,
        "t_len": t_len,
        "schedule": schedule,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--output_dir", default="./cache_trajectories")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_manifest) as f:
        data_manifest = json.load(f)

    print(f"Caching {min(args.num_samples, len(data_manifest))} trajectories")

    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
    from transformers import AutoTokenizer

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float16)
    model.eval()

    gen_config = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=3.0,
        position_temperature=5.0,
        class_temperature=0.0,
    )

    manifest = []
    errors = 0

    for i in tqdm(range(min(args.num_samples, len(data_manifest)))):
        entry = data_manifest[i]
        text = entry["text"]

        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=entry["audio_path"],
                ref_text=text[:100],
                preprocess_prompt=True,
            )

            ref_len = prompt.ref_audio_tokens.shape[1]
            chars_per_frame = max(1, len(prompt.ref_text) / ref_len)
            target_len = max(25, min(150, int(len(text) / chars_per_frame)))

            task = GenerationTask(
                batch_size=1, texts=[text], target_lens=[target_len],
                langs=["English"], instructs=["None"],
                ref_audio_tokens=[prompt.ref_audio_tokens],
                ref_texts=[prompt.ref_text],
                ref_rms=[getattr(prompt, "ref_rms", 0.1)],
            )

            with torch.no_grad():
                traj = cache_trajectory(model, task, gen_config, seed=i)

            # Save trajectory — states and logits separately to manage size
            # Logits are huge (C * T * V * num_steps), so save only target tokens
            # and the logits for training pairs
            fname = f"traj_{i:04d}.pt"
            save_data = {
                "step_states": torch.stack(traj["step_states"]),  # (9, C, T)
                "cond_ids": traj["cond_ids"],          # (C, c_len)
                "audio_mask": traj["audio_mask"],      # (c_len,)
                "c_len": traj["c_len"],
                "t_len": traj["t_len"],
            }

            # Save logits for step pairs: (0→2), (2→4), (4→6), (6→8)
            # These are the training targets for 8→4 distillation
            # Step k logits tell the student what to predict when at state k
            # to match what the teacher produces 2 steps later
            pair_logits = {}
            for src_step in [0, 2, 4, 6]:
                tgt_step = min(src_step + 2, 7)
                # Save target step logits (the teacher's predictions at that step)
                pair_logits[f"logits_step{tgt_step}"] = traj["step_logits"][tgt_step]

            save_data["pair_logits"] = pair_logits
            torch.save(save_data, output_dir / fname)

            manifest.append({
                "file": fname,
                "t_len": traj["t_len"],
                "c_len": traj["c_len"],
                "text": text[:100],
            })

            if i % 50 == 0 and i > 0:
                print(f"  {i} samples cached, {errors} errors")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  Error {i}: {e}")
            continue

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(manifest)} trajectories to {output_dir}/ ({errors} errors)")

    # Print size stats
    import os
    total_size = sum(
        os.path.getsize(output_dir / m["file"])
        for m in manifest
        if (output_dir / m["file"]).exists()
    )
    print(f"Total size: {total_size / 1e9:.1f} GB")
    if manifest:
        avg_size = total_size / len(manifest)
        print(f"Avg per sample: {avg_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
