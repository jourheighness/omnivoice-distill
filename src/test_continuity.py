"""Test continuity techniques at group boundaries.

A: Overlap as conditioning context (generate with overlap text, discard overlap audio)
B: End-of-group reference update (use tail audio as ref for next group)
C: Silence gap tuning (sweep 40-150ms)
D: Combined best
"""

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
    return [" ".join(g) for g in groups]


def rms_normalize(audio, target_rms):
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def generate_cfg_scheduled(model, text, target_frames, prompt, cfg_schedule, seed=42):
    """Generate with CFG scheduling. Returns tokens."""
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


def assemble(chunks, silence_ms=100, crossfade_ms=50):
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

    out = Path("./continuity_output")
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

    groups = presplit_text(text)
    print(f"Groups ({len(groups)}):")
    for gi, g in enumerate(groups):
        print(f"  {gi}: {len(g.split())}w | {g[:70]}")

    # Get last sentence of each group for overlap context
    group_last_sents = []
    for g in groups:
        sents = split_sentences(g)
        group_last_sents.append(sents[-1] if sents else "")

    # === A: Baseline (no continuity tricks) ===
    print("\n--- A: Baseline ---")
    a_audios = []
    for gi, g in enumerate(groups):
        tf = max(60, min(450, int(len(g) / cpf)))
        toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule)
        a_audios.append(decode_tokens(model, toks))

    for silence_ms in [40, 80, 120]:
        audio = assemble(a_audios, silence_ms=silence_ms, crossfade_ms=50)
        fp = out / f"A_baseline_gap{silence_ms}ms.wav"
        torchaudio.save(str(fp), audio.unsqueeze(0), 24000)
        s, _ = whisper_model.transcribe(str(fp), language="en")
        t = " ".join(seg.text.strip() for seg in s)
        w = wer(text, t)
        print(f"  gap={silence_ms}ms: WER={w:.0%}")

    # === B: Overlap as conditioning context ===
    # Generate group N+1 with "last_sent_of_N + group_N+1_text" as input,
    # then discard the frames corresponding to last_sent_of_N
    print("\n--- B: Overlap conditioning (discard overlap audio) ---")
    b_audios = []
    for gi, g in enumerate(groups):
        if gi == 0:
            gen_text = g
            tf = max(60, min(450, int(len(g) / cpf)))
            toks = generate_cfg_scheduled(model, gen_text, tf, prompt, cfg_schedule)
            b_audios.append(decode_tokens(model, toks))
        else:
            # Prepend last sentence of previous group as context
            overlap_text = group_last_sents[gi - 1]
            gen_text = overlap_text + " " + g
            tf = max(60, min(450, int(len(gen_text) / cpf)))
            toks = generate_cfg_scheduled(model, gen_text, tf, prompt, cfg_schedule)

            # Discard frames for the overlap text
            overlap_frames = int(tf * len(overlap_text) / len(gen_text))
            keep_toks = toks[:, overlap_frames:]
            b_audios.append(decode_tokens(model, keep_toks))

    audio = assemble(b_audios, silence_ms=80, crossfade_ms=50)
    fp = out / "B_overlap_context.wav"
    torchaudio.save(str(fp), audio.unsqueeze(0), 24000)
    s, _ = whisper_model.transcribe(str(fp), language="en")
    t = " ".join(seg.text.strip() for seg in s)
    w = wer(text, t)
    print(f"  WER={w:.0%}")

    # === C: End-of-group reference update ===
    # After generating group N, use last ~3s of generated audio as ref for group N+1
    print("\n--- C: End-of-group reference update ---")
    c_audios = []
    current_prompt = prompt
    for gi, g in enumerate(groups):
        tf = max(60, min(450, int(len(g) / cpf)))
        toks = generate_cfg_scheduled(model, g, tf, current_prompt, cfg_schedule)
        audio_chunk = decode_tokens(model, toks)
        c_audios.append(audio_chunk)

        # Update reference: use last 3 seconds of this chunk's audio
        tail_samples = min(int(24000 * 3), len(audio_chunk))
        tail_audio = audio_chunk[-tail_samples:]
        # Re-encode tail audio as new reference
        tail_wav = tail_audio.unsqueeze(0).to(model.device)
        try:
            new_prompt = model.create_voice_clone_prompt(
                ref_audio=tail_wav, ref_text=g[-100:], preprocess_prompt=True
            )
            current_prompt = new_prompt
        except Exception as e:
            print(f"    Ref update failed for group {gi}: {e}")

    audio = assemble(c_audios, silence_ms=80, crossfade_ms=50)
    fp = out / "C_ref_update.wav"
    torchaudio.save(str(fp), audio.unsqueeze(0), 24000)
    s, _ = whisper_model.transcribe(str(fp), language="en")
    t = " ".join(seg.text.strip() for seg in s)
    w = wer(text, t)
    print(f"  WER={w:.0%}")

    # === D: Combined (overlap context + ref update + tuned silence) ===
    print("\n--- D: Combined (overlap context + ref update) ---")
    d_audios = []
    current_prompt = prompt
    for gi, g in enumerate(groups):
        if gi == 0:
            gen_text = g
        else:
            gen_text = group_last_sents[gi - 1] + " " + g

        tf = max(60, min(450, int(len(gen_text) / cpf)))
        toks = generate_cfg_scheduled(model, gen_text, tf, current_prompt, cfg_schedule)

        if gi == 0:
            audio_chunk = decode_tokens(model, toks)
        else:
            overlap_frames = int(tf * len(group_last_sents[gi - 1]) / len(gen_text))
            keep_toks = toks[:, overlap_frames:]
            audio_chunk = decode_tokens(model, keep_toks)

        d_audios.append(audio_chunk)

        # Update reference
        tail_samples = min(int(24000 * 3), len(audio_chunk))
        tail_audio = audio_chunk[-tail_samples:]
        tail_wav = tail_audio.unsqueeze(0).to(model.device)
        try:
            new_prompt = model.create_voice_clone_prompt(
                ref_audio=tail_wav, ref_text=g[-100:], preprocess_prompt=True
            )
            current_prompt = new_prompt
        except Exception as e:
            print(f"    Ref update failed: {e}")

    audio = assemble(d_audios, silence_ms=100, crossfade_ms=50)
    fp = out / "D_combined.wav"
    torchaudio.save(str(fp), audio.unsqueeze(0), 24000)
    s, _ = whisper_model.transcribe(str(fp), language="en")
    t = " ".join(seg.text.strip() for seg in s)
    w = wer(text, t)
    print(f"  WER={w:.0%}")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
