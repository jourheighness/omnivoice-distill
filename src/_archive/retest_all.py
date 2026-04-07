"""Re-test all approaches with CORRECT ref_text conditioning.

Previous tests used ref_text="Reference audio." which caused the model
to try to speak those words. Now using actual Whisper transcript of
the ref audio.
"""

import math
import time
import json
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r: return 0.0 if not h else 1.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            c = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(r)][len(h)] / len(r)


def tokens_to_audio(model, tokens):
    with torch.no_grad():
        result = model.audio_tokenizer.decode(tokens.unsqueeze(0))
    return result.audio_values.squeeze(0).squeeze(0).cpu().float()


def transcribe(audio_path, whisper_model):
    segments, _ = whisper_model.transcribe(str(audio_path), language="en")
    return " ".join(s.text.strip() for s in segments)


def _gumbel_sample(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    return logits / temperature + gumbel_noise


# ============================================================
# TEST 1: SSD (self-speculative decoding)
# ============================================================
def test_ssd(model, task, gen_config, seed=42):
    """Draft all masked positions, verify, accept matches."""
    torch.manual_seed(seed)
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id

    inp = model._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0],
        task.ref_audio_tokens[0], task.langs[0], task.instructs[0],
        gen_config.denoise,
    )
    inp_ids = inp["input_ids"].squeeze(0)
    inp_amask = inp["audio_mask"].squeeze(0)
    c_len = inp_ids.size(1)
    t_len = task.target_lens[0]

    batch_input_ids = torch.full((2, C, c_len), mask_id, dtype=torch.long, device=model.device)
    batch_audio_mask = torch.zeros((2, c_len), dtype=torch.bool, device=model.device)
    batch_attention_mask = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=model.device)

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

    tokens = torch.full((C, t_len), mask_id, dtype=torch.long, device=model.device)
    layer_ids = torch.arange(C, device=model.device).view(-1, 1)

    t0 = time.time()
    fwd = 0
    round_stats = []

    for round_idx in range(8):
        n_masked = (tokens == mask_id).sum().item()
        if n_masked == 0:
            break

        # Draft
        batch_input_ids[0, :, c_len - t_len:c_len] = tokens
        batch_input_ids[1, :, :t_len] = tokens
        with torch.no_grad():
            logits = model(input_ids=batch_input_ids, audio_mask=batch_audio_mask, attention_mask=batch_attention_mask).logits.to(torch.float32)
        fwd += 1

        c_logits = logits[0:1, :, c_len-t_len:c_len, :]
        u_logits = logits[1:2, :, :t_len, :]
        draft_tokens, draft_scores = model._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
        draft_tokens, draft_scores = draft_tokens[0], draft_scores[0]

        is_masked = tokens == mask_id
        draft_full = tokens.clone()
        draft_full[is_masked] = draft_tokens[is_masked]

        # Verify
        batch_input_ids[0, :, c_len-t_len:c_len] = draft_full
        batch_input_ids[1, :, :t_len] = draft_full
        with torch.no_grad():
            v_logits = model(input_ids=batch_input_ids, audio_mask=batch_audio_mask, attention_mask=batch_attention_mask).logits.to(torch.float32)
        fwd += 1

        v_c = v_logits[0:1, :, c_len-t_len:c_len, :]
        v_u = v_logits[1:2, :, :t_len, :]
        verify_tokens, _ = model._predict_tokens_with_scoring(v_c, v_u, gen_config)
        verify_tokens = verify_tokens[0]

        matches = (draft_full == verify_tokens) & is_masked
        n_accepted = matches.sum().item()
        accept_rate = n_accepted / max(is_masked.sum().item(), 1)
        tokens[matches] = draft_full[matches]

        # Force-accept top confident if acceptance low
        rejected = is_masked & ~matches
        n_rej = rejected.sum().item()
        if n_rej > 0 and accept_rate < 0.5:
            force_k = max(1, n_rej // 4)
            scores_ranked = draft_scores - (layer_ids * gen_config.layer_penalty_factor)
            scores_ranked[~rejected] = float("-inf")
            _, force_idx = torch.topk(scores_ranked.flatten(), force_k)
            flat = tokens.flatten()
            flat[force_idx] = draft_full.flatten()[force_idx]
            tokens = flat.view(C, t_len)
            n_accepted += force_k

        remaining = (tokens == mask_id).sum().item()
        round_stats.append({"round": round_idx, "masked": is_masked.sum().item(), "accepted": n_accepted, "rate": round(accept_rate, 3), "remaining": remaining})

    still_masked = tokens == mask_id
    if still_masked.any():
        tokens[still_masked] = draft_full[still_masked]

    return tokens, time.time() - t0, fwd, round_stats


# ============================================================
# TEST 2: Stability tracking
# ============================================================
def test_stability(model, task, gen_config, seed=42):
    """Track per-position prediction stability across steps."""
    torch.manual_seed(seed)
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id

    inp = model._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0],
        task.ref_audio_tokens[0], task.langs[0], task.instructs[0],
        gen_config.denoise,
    )
    inp_ids = inp["input_ids"].squeeze(0)
    inp_amask = inp["audio_mask"].squeeze(0)
    c_len = inp_ids.size(1)
    t_len = task.target_lens[0]
    num_steps = gen_config.num_step

    batch_input_ids = torch.full((2, C, c_len), mask_id, dtype=torch.long, device=model.device)
    batch_audio_mask = torch.zeros((2, c_len), dtype=torch.bool, device=model.device)
    batch_attention_mask = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=model.device)

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

    tokens = torch.full((C, t_len), mask_id, dtype=torch.long, device=model.device)
    layer_ids = torch.arange(C, device=model.device).view(-1, 1)

    from omnivoice.models.omnivoice import _get_time_steps
    timesteps = _get_time_steps(0.0, 1.0, num_steps + 1, gen_config.t_shift).tolist()
    total_mask = t_len * C
    rem = total_mask
    schedule = []
    for step in range(num_steps):
        num = rem if step == num_steps - 1 else min(math.ceil(total_mask * (timesteps[step+1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    all_cb0_preds = torch.full((num_steps, t_len), -1, dtype=torch.long)

    for step in range(num_steps):
        batch_input_ids[0, :, c_len-t_len:c_len] = tokens
        batch_input_ids[1, :, :t_len] = tokens

        with torch.no_grad():
            logits = model(input_ids=batch_input_ids, audio_mask=batch_audio_mask, attention_mask=batch_attention_mask).logits.to(torch.float32)

        c_logits = logits[0:1, :, c_len-t_len:c_len, :]
        u_logits = logits[1:2, :, :t_len, :]
        pred_tokens, scores = model._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
        pred_tokens, scores = pred_tokens[0], scores[0]

        all_cb0_preds[step] = pred_tokens[0]  # cb0 predictions

        scores = scores - (layer_ids * gen_config.layer_penalty_factor)
        if gen_config.position_temperature > 0:
            scores = _gumbel_sample(scores, gen_config.position_temperature)
        is_masked = tokens == mask_id
        scores[~is_masked] = float("-inf")

        k = schedule[step]
        if k > 0:
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = tokens.flatten()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            tokens = flat.view(C, t_len)

    # Analyze: when does each cb0 position stabilize?
    final_cb0 = tokens[0]
    stability_step = torch.full((t_len,), num_steps - 1, dtype=torch.long)
    for pos in range(t_len):
        fv = final_cb0[pos]
        if fv == mask_id:
            continue
        for s in range(num_steps):
            if all_cb0_preds[s, pos] == fv:
                stays = all(all_cb0_preds[s2, pos] == fv for s2 in range(s+1, num_steps))
                if stays:
                    stability_step[pos] = s
                    break

    # Count flip-flops
    flips = 0
    for pos in range(t_len):
        for s in range(1, num_steps - 1):
            if (all_cb0_preds[s, pos] != all_cb0_preds[s-1, pos] and
                all_cb0_preds[s, pos] != all_cb0_preds[s+1, pos] and
                all_cb0_preds[s, pos] != -1 and all_cb0_preds[s-1, pos] != -1):
                flips += 1

    return tokens, stability_step, flips


def main():
    output_dir = Path("./retest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OmniVoice...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
    from faster_whisper import WhisperModel

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    model.eval()
    whisper = WhisperModel("base.en", device="cuda", compute_type="float16")

    # Correct ref_text
    segs, _ = whisper.transcribe("barth_ref.wav", language="en")
    actual_ref = " ".join(s.text.strip() for s in segs)
    print(f"Ref text: {actual_ref[:80]}...")

    prompt = model.create_voice_clone_prompt(
        ref_audio="barth_ref.wav", ref_text=actual_ref, preprocess_prompt=True,
    )

    gen_config = OmniVoiceGenerationConfig(
        num_step=8, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
    )

    texts = [
        "I can not believe you would do something like that. After everything we have been through together, you just threw it all away.",
        "Welcome to the annual science conference. Today we will explore the fascinating world of quantum computing and its implications for artificial intelligence.",
        "Once upon a time, in a land far far away, there lived a brave knight who feared nothing. But one dark night, everything changed.",
    ]

    for ti, text in enumerate(texts):
        rl = prompt.ref_audio_tokens.shape[1]
        cpf = max(1, len(prompt.ref_text) / rl)
        tl = max(25, min(200, int(len(text) / cpf)))
        task = GenerationTask(
            batch_size=1, texts=[text], target_lens=[tl], langs=["English"], instructs=["None"],
            ref_audio_tokens=[prompt.ref_audio_tokens], ref_texts=[prompt.ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )

        print(f"\n{'='*70}")
        print(f"Text {ti}: {text[:80]}...")
        print(f"target_len={tl}")

        # Baseline 8-step
        torch.manual_seed(42)
        t0 = time.time()
        with torch.no_grad():
            std_tokens = model._generate_iterative(task, gen_config)[0]
        std_time = time.time() - t0
        std_audio = tokens_to_audio(model, std_tokens)
        fp = output_dir / f"text{ti}_8step.wav"
        torchaudio.save(str(fp), std_audio.unsqueeze(0), 24000)
        std_t = transcribe(fp, whisper)
        std_w = wer(text, std_t)
        std_u = len(np.unique(std_tokens[0].cpu().numpy()))
        print(f"\n  8-STEP BASELINE: WER={std_w:.0%} unique={std_u} time={std_time:.3f}s")
        print(f"    {std_t[:120]}")

        # SSD
        ssd_tokens, ssd_time, ssd_fwd, ssd_stats = test_ssd(model, task, gen_config, seed=42)
        ssd_audio = tokens_to_audio(model, ssd_tokens)
        fp = output_dir / f"text{ti}_ssd.wav"
        torchaudio.save(str(fp), ssd_audio.unsqueeze(0), 24000)
        ssd_t = transcribe(fp, whisper)
        ssd_w = wer(text, ssd_t)
        ssd_u = len(np.unique(ssd_tokens[0].cpu().numpy()))
        cb0_match = (std_tokens[0].cpu() == ssd_tokens[0].cpu()).float().mean().item()
        print(f"\n  SSD: WER={ssd_w:.0%} unique={ssd_u} time={ssd_time:.3f}s fwd={ssd_fwd} cb0_match={cb0_match:.0%}")
        print(f"    {ssd_t[:120]}")
        for s in ssd_stats:
            print(f"      round {s['round']}: {s['masked']} masked -> accepted {s['accepted']} ({s['rate']:.0%}) -> {s['remaining']} left")

        # Stability
        _, stab_step, flips = test_stability(model, task, gen_config, seed=42)
        print(f"\n  STABILITY: flip-flops={flips}")
        for s in range(8):
            count = (stab_step == s).sum().item()
            pct = count / tl * 100
            print(f"    step {s}: {count:3d}/{tl} ({pct:5.1f}%) stable")

    print(f"\nAudio files saved to {output_dir}/")


if __name__ == "__main__":
    main()
