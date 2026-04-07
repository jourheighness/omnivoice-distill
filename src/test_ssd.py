"""Test Self-Speculative Decoding (SSD) for OmniVoice.

Instead of fixed 8-step schedule, use adaptive draft-verify loop:
1. Draft: predict all masked positions in one pass
2. Verify: re-run with draft tokens, check which predictions match
3. Accept matching positions, re-mask rejected ones
4. Repeat until converged or max rounds

Compare: quality (WER), speed, and acceptance rates vs standard 8-step.
"""

import math
import time
import json
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path


def generate_standard(model, task, gen_config, seed=42):
    """Standard 8-step generation (baseline)."""
    torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        tokens_list = model._generate_iterative(task, gen_config)
    elapsed = time.time() - t0
    return tokens_list[0], elapsed


def generate_ssd(model, task, gen_config, max_rounds=8, seed=42):
    """Self-speculative decoding: draft all, verify, accept matches."""
    torch.manual_seed(seed)

    B = task.batch_size
    assert B == 1
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id

    # Prepare inputs (same as _generate_iterative)
    inp = model._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0],
        task.ref_audio_tokens[0], task.langs[0], task.instructs[0],
        gen_config.denoise,
    )

    c_len = inp["input_ids"].size(2)
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

    # Cond
    batch_input_ids[0, :, :c_len] = inp["input_ids"]
    batch_audio_mask[0, :c_len] = inp["audio_mask"]
    batch_attention_mask[0, :, :c_len, :c_len] = True

    # Uncond
    u_len = t_len
    batch_input_ids[1, :, :u_len] = inp["input_ids"][..., -u_len:]
    batch_audio_mask[1, :u_len] = inp["audio_mask"][..., -u_len:]
    batch_attention_mask[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pad_diag = torch.arange(u_len, c_len, device=model.device)
        batch_attention_mask[1, :, pad_diag, pad_diag] = True

    tokens = torch.full(
        (C, t_len), mask_id, dtype=torch.long, device=model.device,
    )

    layer_ids = torch.arange(C, device=model.device).view(-1, 1)
    round_stats = []

    t0 = time.time()
    forward_passes = 0

    for round_idx in range(max_rounds):
        n_masked = (tokens == mask_id).sum().item()
        if n_masked == 0:
            break

        # --- DRAFT: predict all masked positions ---
        batch_input_ids[0, :, c_len - t_len:c_len] = tokens
        batch_input_ids[1, :, :t_len] = tokens

        with torch.no_grad():
            batch_logits = model(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)
        forward_passes += 1

        c_logits = batch_logits[0:1, :, c_len - t_len:c_len, :]
        u_logits = batch_logits[1:2, :, :t_len, :]

        draft_tokens, draft_scores = model._predict_tokens_with_scoring(
            c_logits, u_logits, gen_config,
        )
        draft_tokens = draft_tokens[0]  # (C, t_len)
        draft_scores = draft_scores[0]  # (C, t_len)

        # Apply layer penalty to scores for ranking
        draft_scores_ranked = draft_scores - (layer_ids * gen_config.layer_penalty_factor)

        # Fill ALL masked positions with draft predictions
        is_masked = tokens == mask_id
        draft_full = tokens.clone()
        draft_full[is_masked] = draft_tokens[is_masked]

        # --- VERIFY: re-run with draft tokens, check consistency ---
        batch_input_ids[0, :, c_len - t_len:c_len] = draft_full
        batch_input_ids[1, :, :t_len] = draft_full

        with torch.no_grad():
            verify_logits = model(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)
        forward_passes += 1

        v_c_logits = verify_logits[0:1, :, c_len - t_len:c_len, :]
        v_u_logits = verify_logits[1:2, :, :t_len, :]

        verify_tokens, verify_scores = model._predict_tokens_with_scoring(
            v_c_logits, v_u_logits, gen_config,
        )
        verify_tokens = verify_tokens[0]  # (C, t_len)

        # --- ACCEPT/REJECT ---
        # Accept positions where draft == verify (model confirms its own prediction)
        matches = (draft_full == verify_tokens) & is_masked
        n_accepted = matches.sum().item()
        n_was_masked = is_masked.sum().item()
        accept_rate = n_accepted / max(n_was_masked, 1)

        # Accept matched positions
        tokens[matches] = draft_full[matches]

        # For rejected positions that were masked: keep them masked
        # (they stay as mask_id since we only wrote to draft_full)

        # Also: if acceptance is low, accept at least the top-k most confident
        # to guarantee progress (avoid infinite loops)
        rejected_mask = is_masked & ~matches
        n_rejected = rejected_mask.sum().item()

        if n_rejected > 0 and accept_rate < 0.5:
            # Force-accept the most confident 25% of rejected positions
            force_k = max(1, n_rejected // 4)
            rejected_scores = draft_scores_ranked.clone()
            rejected_scores[~rejected_mask] = float("-inf")
            _, force_idx = torch.topk(rejected_scores.flatten(), force_k)
            flat_tokens = tokens.flatten()
            flat_draft = draft_full.flatten()
            flat_tokens[force_idx] = flat_draft[force_idx]
            tokens = flat_tokens.view(C, t_len)
            n_accepted += force_k

        round_stats.append({
            "round": round_idx,
            "masked_before": n_was_masked,
            "accepted": n_accepted,
            "accept_rate": round(accept_rate, 3),
            "remaining_masked": (tokens == mask_id).sum().item(),
        })

    elapsed = time.time() - t0

    # If any positions still masked after max_rounds, fill with last draft
    still_masked = tokens == mask_id
    if still_masked.any():
        tokens[still_masked] = draft_full[still_masked]

    return tokens, elapsed, forward_passes, round_stats


def generate_ssd_topk(model, task, gen_config, accept_top_pct=0.5, max_rounds=8, seed=42):
    """SSD variant: instead of binary accept/reject, accept top-K% most
    confident predictions each round (more like standard OmniVoice but adaptive)."""
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

    c_len = inp["input_ids"].size(2)
    t_len = task.target_lens[0]

    batch_input_ids = torch.full(
        (2, C, c_len), mask_id, dtype=torch.long, device=model.device,
    )
    batch_audio_mask = torch.zeros(
        (2, c_len), dtype=torch.bool, device=model.device,
    )
    batch_attention_mask = torch.zeros(
        (2, 1, c_len, c_len), dtype=torch.bool, device=model.device,
    )

    batch_input_ids[0, :, :c_len] = inp["input_ids"]
    batch_audio_mask[0, :c_len] = inp["audio_mask"]
    batch_attention_mask[0, :, :c_len, :c_len] = True

    u_len = t_len
    batch_input_ids[1, :, :u_len] = inp["input_ids"][..., -u_len:]
    batch_audio_mask[1, :u_len] = inp["audio_mask"][..., -u_len:]
    batch_attention_mask[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pad_diag = torch.arange(u_len, c_len, device=model.device)
        batch_attention_mask[1, :, pad_diag, pad_diag] = True

    tokens = torch.full(
        (C, t_len), mask_id, dtype=torch.long, device=model.device,
    )

    layer_ids = torch.arange(C, device=model.device).view(-1, 1)
    round_stats = []

    t0 = time.time()
    forward_passes = 0

    for round_idx in range(max_rounds):
        n_masked = (tokens == mask_id).sum().item()
        if n_masked == 0:
            break

        batch_input_ids[0, :, c_len - t_len:c_len] = tokens
        batch_input_ids[1, :, :t_len] = tokens

        with torch.no_grad():
            batch_logits = model(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)
        forward_passes += 1

        c_logits = batch_logits[0:1, :, c_len - t_len:c_len, :]
        u_logits = batch_logits[1:2, :, :t_len, :]

        pred_tokens, scores = model._predict_tokens_with_scoring(
            c_logits, u_logits, gen_config,
        )
        pred_tokens = pred_tokens[0]
        scores = scores[0]

        scores = scores - (layer_ids * gen_config.layer_penalty_factor)

        # Apply Gumbel noise for position selection
        if gen_config.position_temperature > 0.0:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores) + 1e-20) + 1e-20)
            scores = scores / gen_config.position_temperature + gumbel_noise

        # Only score masked positions
        is_masked = tokens == mask_id
        scores[~is_masked] = float("-inf")

        # Accept top-K% of masked positions (adaptive K)
        k = max(1, int(n_masked * accept_top_pct))
        # On last round, accept everything
        if round_idx == max_rounds - 1:
            k = n_masked

        _, topk_idx = torch.topk(scores.flatten(), k)
        flat_tokens = tokens.flatten()
        flat_pred = pred_tokens.flatten()
        flat_tokens[topk_idx] = flat_pred[topk_idx]
        tokens = flat_tokens.view(C, t_len)

        remaining = (tokens == mask_id).sum().item()
        round_stats.append({
            "round": round_idx,
            "masked_before": n_masked,
            "accepted": k,
            "remaining_masked": remaining,
        })

    elapsed = time.time() - t0
    return tokens, elapsed, forward_passes, round_stats


def tokens_to_audio(model, tokens):
    with torch.no_grad():
        result = model.audio_tokenizer.decode(tokens.unsqueeze(0))
    return result.audio_values.squeeze(0).squeeze(0).cpu().float()


def transcribe(audio_path, whisper_model):
    segments, _ = whisper_model.transcribe(str(audio_path), language="en")
    return " ".join(s.text.strip() for s in segments)


def main():
    output_dir = Path("./test_ssd")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("Encoding reference voice...")
    prompt = model.create_voice_clone_prompt(
        ref_audio="barth_ref.wav", ref_text="Reference audio.", preprocess_prompt=True,
    )

    print("Loading Whisper...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel("base.en", device="cuda", compute_type="float16")

    for text_idx, text in enumerate(test_texts):
        print(f"\n{'='*70}")
        print(f"Text {text_idx}: {text[:80]}")

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

        # 1. Standard 8-step baseline
        print(f"\n  STANDARD (8 steps):")
        std_tokens, std_time = generate_standard(model, task, gen_config, seed=42)
        std_audio = tokens_to_audio(model, std_tokens)
        std_path = output_dir / f"text{text_idx}_standard.wav"
        torchaudio.save(str(std_path), std_audio.unsqueeze(0), 24000)
        std_transcript = transcribe(std_path, whisper)
        std_unique = len(np.unique(std_tokens[0].cpu().numpy()))
        print(f"    time={std_time:.3f}s  16 fwd passes  unique_cb0={std_unique}")
        print(f"    transcript: {std_transcript[:120]}")

        # 2. SSD (draft-verify)
        print(f"\n  SSD (draft-verify):")
        ssd_tokens, ssd_time, ssd_fwd, ssd_stats = generate_ssd(
            model, task, gen_config, max_rounds=8, seed=42,
        )
        ssd_audio = tokens_to_audio(model, ssd_tokens)
        ssd_path = output_dir / f"text{text_idx}_ssd.wav"
        torchaudio.save(str(ssd_path), ssd_audio.unsqueeze(0), 24000)
        ssd_transcript = transcribe(ssd_path, whisper)
        ssd_unique = len(np.unique(ssd_tokens[0].cpu().numpy()))

        cb0_match = (std_tokens[0].cpu() == ssd_tokens[0].cpu()).float().mean().item()
        print(f"    time={ssd_time:.3f}s  {ssd_fwd} fwd passes  unique_cb0={ssd_unique}")
        print(f"    cb0 match vs standard: {cb0_match:.1%}")
        print(f"    transcript: {ssd_transcript[:120]}")
        print(f"    rounds:")
        for s in ssd_stats:
            print(f"      round {s['round']}: {s['masked_before']} masked → accepted {s['accepted']} ({s['accept_rate']:.0%}) → {s['remaining_masked']} remaining")

        # 3. SSD top-k (50% per round)
        print(f"\n  SSD-TOPK (50% per round):")
        topk_tokens, topk_time, topk_fwd, topk_stats = generate_ssd_topk(
            model, task, gen_config, accept_top_pct=0.5, max_rounds=8, seed=42,
        )
        topk_audio = tokens_to_audio(model, topk_tokens)
        topk_path = output_dir / f"text{text_idx}_ssd_topk50.wav"
        torchaudio.save(str(topk_path), topk_audio.unsqueeze(0), 24000)
        topk_transcript = transcribe(topk_path, whisper)
        topk_unique = len(np.unique(topk_tokens[0].cpu().numpy()))

        cb0_match = (std_tokens[0].cpu() == topk_tokens[0].cpu()).float().mean().item()
        print(f"    time={topk_time:.3f}s  {topk_fwd} fwd passes  unique_cb0={topk_unique}")
        print(f"    cb0 match vs standard: {cb0_match:.1%}")
        print(f"    transcript: {topk_transcript[:120]}")
        print(f"    rounds:")
        for s in topk_stats:
            print(f"      round {s['round']}: {s['masked_before']} masked → accepted {s['accepted']} → {s['remaining_masked']} remaining")

        # 4. SSD top-k (33% per round — 3 rounds to finish)
        print(f"\n  SSD-TOPK (33% per round):")
        topk33_tokens, topk33_time, topk33_fwd, topk33_stats = generate_ssd_topk(
            model, task, gen_config, accept_top_pct=0.34, max_rounds=8, seed=42,
        )
        topk33_audio = tokens_to_audio(model, topk33_tokens)
        topk33_path = output_dir / f"text{text_idx}_ssd_topk33.wav"
        torchaudio.save(str(topk33_path), topk33_audio.unsqueeze(0), 24000)
        topk33_transcript = transcribe(topk33_path, whisper)
        topk33_unique = len(np.unique(topk33_tokens[0].cpu().numpy()))

        cb0_match = (std_tokens[0].cpu() == topk33_tokens[0].cpu()).float().mean().item()
        print(f"    time={topk33_time:.3f}s  {topk33_fwd} fwd passes  unique_cb0={topk33_unique}")
        print(f"    cb0 match vs standard: {cb0_match:.1%}")
        print(f"    transcript: {topk33_transcript[:120]}")
        print(f"    rounds:")
        for s in topk33_stats:
            print(f"      round {s['round']}: {s['masked_before']} masked → accepted {s['accepted']} → {s['remaining_masked']} remaining")

        # 5. Fewer standard steps (4-step and 2-step baselines)
        for n_steps in [4, 2]:
            print(f"\n  STANDARD ({n_steps} steps):")
            few_config = OmniVoiceGenerationConfig(
                num_step=n_steps,
                guidance_scale=3.0,
                position_temperature=5.0,
                class_temperature=0.0,
            )
            few_tokens, few_time = generate_standard(model, task, few_config, seed=42)
            few_audio = tokens_to_audio(model, few_tokens)
            few_path = output_dir / f"text{text_idx}_standard_{n_steps}step.wav"
            torchaudio.save(str(few_path), few_audio.unsqueeze(0), 24000)
            few_transcript = transcribe(few_path, whisper)
            few_unique = len(np.unique(few_tokens[0].cpu().numpy()))
            cb0_match = (std_tokens[0].cpu() == few_tokens[0].cpu()).float().mean().item()
            print(f"    time={few_time:.3f}s  {n_steps*2} fwd passes  unique_cb0={few_unique}")
            print(f"    cb0 match vs 8-step: {cb0_match:.1%}")
            print(f"    transcript: {few_transcript[:120]}")

    print(f"\n{'='*70}")
    print(f"Audio files saved to {output_dir}/")


if __name__ == "__main__":
    main()
