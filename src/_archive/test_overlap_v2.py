"""Improved overlap conditioning: energy-based boundary detection."""

import math
import re
import time
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import (
    GenerationTask, OmniVoiceGenerationConfig, _get_time_steps,
)
from faster_whisper import WhisperModel


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / len(r)


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def presplit_text(text, target_words=40, min_words=20, max_words=55):
    sentences = split_sentences(text)
    groups = []
    current = []
    current_words = 0
    for sent in sentences:
        sw = len(sent.split())
        if current_words + sw > max_words and current:
            groups.append(current)
            current = [sent]
            current_words = sw
        else:
            current.append(sent)
            current_words += sw
    if current:
        groups.append(current)
    if len(groups) > 1:
        tail_words = sum(len(s.split()) for s in groups[-1])
        if tail_words < min_words:
            prev_words = sum(len(s.split()) for s in groups[-2])
            if prev_words + tail_words <= max_words + 15:
                groups[-2].extend(groups[-1])
                groups.pop()
    return groups  # list of list of sentences


def rms_normalize(audio, target_rms):
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def generate_cfg_scheduled(model, text, target_frames, prompt, cfg_schedule, seed=42):
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    num_steps = len(cfg_schedule)

    torch.manual_seed(seed)
    inp = model._prepare_inference_inputs(
        text, target_frames, prompt.ref_text, prompt.ref_audio_tokens,
        "English", "None", True,
    )
    inp_ids = inp["input_ids"].squeeze(0)
    inp_amask = inp["audio_mask"].squeeze(0)
    c_len = inp_ids.size(1)

    b2_ids = torch.full((2, C, c_len), mask_id, dtype=torch.long, device=model.device)
    b2_amask = torch.zeros((2, c_len), dtype=torch.bool, device=model.device)
    b2_attn = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=model.device)
    b2_ids[0, :, :c_len] = inp_ids
    b2_amask[0, :c_len] = inp_amask
    b2_attn[0, :, :c_len, :c_len] = True
    u_len = target_frames
    b2_ids[1, :, :u_len] = inp_ids[:, -u_len:]
    b2_amask[1, :u_len] = inp_amask[-u_len:]
    b2_attn[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pd = torch.arange(u_len, c_len, device=model.device)
        b2_attn[1, :, pd, pd] = True
    b1_ids = b2_ids[0:1].clone()
    b1_amask = b2_amask[0:1].clone()
    b1_attn = b2_attn[0:1].clone()

    tokens = torch.full((C, target_frames), mask_id, dtype=torch.long, device=model.device)
    layer_ids = torch.arange(C, device=model.device).view(-1, 1)

    timesteps = _get_time_steps(0.0, 1.0, num_steps + 1, 0.1).tolist()
    total_mask = target_frames * C
    rem = total_mask
    schedule = []
    for step in range(num_steps):
        num = rem if step == num_steps - 1 else min(
            math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    for step in range(num_steps):
        cfg = cfg_schedule[step]
        if cfg > 0:
            b2_ids[0, :, c_len - target_frames:c_len] = tokens
            b2_ids[1, :, :target_frames] = tokens
            with torch.no_grad():
                logits = model(input_ids=b2_ids, audio_mask=b2_amask,
                               attention_mask=b2_attn).logits.to(torch.float32)
            c_log = F.log_softmax(logits[0:1, :, c_len - target_frames:c_len, :], dim=-1)
            u_log = F.log_softmax(logits[1:2, :, :target_frames, :], dim=-1)
            log_probs = torch.log_softmax(c_log + cfg * (c_log - u_log), dim=-1)
        else:
            b1_ids[0, :, c_len - target_frames:c_len] = tokens
            with torch.no_grad():
                logits = model(input_ids=b1_ids, audio_mask=b1_amask,
                               attention_mask=b1_attn).logits.to(torch.float32)
            log_probs = F.log_softmax(logits[0:1, :, c_len - target_frames:c_len, :], dim=-1)

        log_probs[..., mask_id] = float("-inf")
        pred_tokens = log_probs.argmax(dim=-1)[0]
        scores = log_probs.max(dim=-1)[0][0]
        scores = scores - (layer_ids * 5.0)
        g = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
        scores = scores / 5.0 + g
        is_masked = tokens == mask_id
        scores[~is_masked] = float("-inf")
        k = schedule[step]
        if k > 0:
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = tokens.flatten()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            tokens = flat.view(C, target_frames)

    return tokens


def decode_tokens(model, tokens):
    return model.audio_tokenizer.decode(
        tokens.unsqueeze(0)
    ).audio_values.squeeze().cpu().float()


def find_energy_valley(audio, estimated_frame, search_window_ms=200):
    """Find the lowest energy point near the estimated split frame.

    Works on decoded audio (24kHz). Searches +-search_window_ms around
    the estimated position for the point with lowest short-term energy.
    """
    sr = 24000
    window_samples = int(sr * search_window_ms / 1000)
    # Convert frame to audio sample (25fps codec = 960 samples per frame)
    est_sample = estimated_frame * (sr // 25)
    est_sample = min(est_sample, len(audio) - 1)

    search_start = max(0, est_sample - window_samples)
    search_end = min(len(audio), est_sample + window_samples)

    if search_end <= search_start:
        return est_sample

    # Short-term energy with 10ms window
    hop = int(sr * 0.005)  # 5ms hop
    win = int(sr * 0.010)  # 10ms window
    min_energy = float("inf")
    best_pos = est_sample

    for pos in range(search_start, search_end - win, hop):
        energy = torch.mean(audio[pos:pos + win] ** 2).item()
        if energy < min_energy:
            min_energy = energy
            best_pos = pos + win // 2

    return best_pos


def assemble(chunks, silence_ms=60, crossfade_ms=40):
    if len(chunks) == 1:
        return chunks[0]
    rms_vals = [torch.sqrt(torch.mean(a ** 2)).item() for a in chunks]
    trms = np.mean(rms_vals)
    chunks = [rms_normalize(a, trms) for a in chunks]
    silence = torch.zeros(int(24000 * silence_ms / 1000))
    xfade = int(24000 * crossfade_ms / 1000)
    result = chunks[0]
    for chunk in chunks[1:]:
        result = torch.cat([result, silence])
        if len(result) >= xfade and len(chunk) >= xfade:
            fo = torch.linspace(1, 0, xfade)
            fi = torch.linspace(0, 1, xfade)
            ov = result[-xfade:] * fo + chunk[:xfade] * fi
            result = torch.cat([result[:-xfade], ov, chunk[xfade:]])
        else:
            result = torch.cat([result, chunk])
    return result


def main():
    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    segs, _ = whisper_model.transcribe("barth_ref.wav", language="en")
    actual_ref = " ".join(s.text.strip() for s in segs)
    prompt = model.create_voice_clone_prompt(
        ref_audio="barth_ref.wav", ref_text=actual_ref, preprocess_prompt=True
    )

    cfg_schedule = [0, 0, 3, 3, 3, 3, 3, 3]
    rl = prompt.ref_audio_tokens.shape[1]
    cpf = max(1, len(prompt.ref_text) / rl)

    out = Path("./overlap_v2_output")
    out.mkdir(parents=True, exist_ok=True)

    text = (
        "The old lighthouse keeper had not spoken to another human being in three years. "
        "Every morning he climbed the one hundred and forty seven steps to the lamp room. "
        "Every evening he descended them again, his joints aching with each step. "
        "The sea was his only companion, and it was not always a kind one. "
        "On stormy nights, the waves would crash against the rocks with such fury that the whole tower trembled. "
        "But tonight was different. "
        "Tonight the sea was calm, the stars were bright, and somewhere in the distance, he could hear music. "
        "It was faint at first, barely distinguishable from the wind. "
        "But as the minutes passed, it grew louder and more distinct. "
        "Someone was playing a violin on the shore below."
    )

    groups_sents = presplit_text(text)
    groups = [" ".join(g) for g in groups_sents]
    group_last_sent = [g[-1] for g in groups_sents]

    print(f"Groups ({len(groups)}):")
    for gi, g in enumerate(groups):
        print(f"  {gi}: {len(g.split())}w | {g[:70]}")

    # === Baseline (no overlap, 60ms gap) ===
    print("\n--- Baseline ---")
    base_audios = []
    for g in groups:
        tf = max(60, min(450, int(len(g) / cpf)))
        toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule)
        base_audios.append(decode_tokens(model, toks))
    base_audio = assemble(base_audios, silence_ms=60, crossfade_ms=40)
    torchaudio.save(str(out / "baseline.wav"), base_audio.unsqueeze(0), 24000)

    # === Overlap v2: energy-based boundary detection ===
    print("\n--- Overlap v2 (energy-based cut) ---")
    ov_audios = []
    for gi, g in enumerate(groups):
        if gi == 0:
            tf = max(60, min(450, int(len(g) / cpf)))
            toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule)
            ov_audios.append(decode_tokens(model, toks))
        else:
            # Prepend last sentence of previous group
            overlap_sent = group_last_sent[gi - 1]
            gen_text = overlap_sent + " " + g
            tf = max(60, min(450, int(len(gen_text) / cpf)))
            toks = generate_cfg_scheduled(model, gen_text, tf, prompt, cfg_schedule)

            # Decode full audio first
            full_audio = decode_tokens(model, toks)

            # Find the boundary: estimate where overlap ends
            overlap_ratio = len(overlap_sent) / len(gen_text)
            estimated_frame = int(tf * overlap_ratio)

            # Find energy valley near the estimated boundary
            cut_sample = find_energy_valley(full_audio, estimated_frame)

            # Keep only the audio after the cut
            keep_audio = full_audio[cut_sample:]
            if len(keep_audio) > 0:
                ov_audios.append(keep_audio)
            else:
                ov_audios.append(full_audio)

    ov_audio = assemble(ov_audios, silence_ms=60, crossfade_ms=40)
    torchaudio.save(str(out / "overlap_v2_energy.wav"), ov_audio.unsqueeze(0), 24000)

    # === Overlap v2 with tighter gap (40ms) ===
    ov_audio_tight = assemble(ov_audios, silence_ms=40, crossfade_ms=30)
    torchaudio.save(str(out / "overlap_v2_tight.wav"), ov_audio_tight.unsqueeze(0), 24000)

    # === No-overlap with same tight gap for comparison ===
    base_tight = assemble(base_audios, silence_ms=40, crossfade_ms=30)
    torchaudio.save(str(out / "baseline_tight.wav"), base_tight.unsqueeze(0), 24000)

    # WER check
    for name, fp in [
        ("baseline_60ms", out / "baseline.wav"),
        ("baseline_tight", out / "baseline_tight.wav"),
        ("overlap_v2_energy", out / "overlap_v2_energy.wav"),
        ("overlap_v2_tight", out / "overlap_v2_tight.wav"),
    ]:
        s, _ = whisper_model.transcribe(str(fp), language="en")
        t = " ".join(seg.text.strip() for seg in s)
        w = wer(text, t)
        print(f"  {name:>25s}: WER={w:.0%}")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
