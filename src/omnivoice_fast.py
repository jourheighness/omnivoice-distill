"""OmniVoice Fast — all speedups combined.

1. Correct ref_text (Whisper transcription of ref audio)
2. CFG scheduling [0,0,3,3,3,3,3,3] — skip CFG on first 2 steps
3. Full-text generation for prosody context
4. Sentence-aware split decode with merge logic (60-450 frame window)
5. RMS normalization + silence padding + crossfade at boundaries
"""

import math
import re
import time
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path


def split_sentences(text):
    """Split on sentence boundaries (.!?) keeping punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def estimate_frames(text, cpf):
    """Estimate number of audio frames for text."""
    return max(15, int(len(text) / cpf))


def plan_split_points(text, cpf, min_frames=60, max_frames=450):
    """Plan where to split decoded audio at sentence boundaries.

    Returns list of (sentence_group_text, estimated_frames) tuples.
    Groups short sentences together to meet min_frames.
    Splits happen at sentence boundaries only.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [(text, estimate_frames(text, cpf))]

    # Build groups by merging short sentences
    groups = []
    current_text = ""
    current_frames = 0

    for sent in sentences:
        sent_frames = estimate_frames(sent, cpf)

        if not current_text:
            current_text = sent
            current_frames = sent_frames
            continue

        combined_frames = estimate_frames(current_text + " " + sent, cpf)

        if combined_frames <= max_frames:
            # Can merge — but should we?
            if current_frames < min_frames:
                # Current is too short, must merge
                current_text = current_text + " " + sent
                current_frames = combined_frames
            elif combined_frames <= 200:
                # Both fit comfortably in sweet spot, merge
                current_text = current_text + " " + sent
                current_frames = combined_frames
            else:
                # Current is fine, start new group
                groups.append((current_text, current_frames))
                current_text = sent
                current_frames = sent_frames
        else:
            # Would exceed max, flush current
            groups.append((current_text, current_frames))
            current_text = sent
            current_frames = sent_frames

    if current_text:
        groups.append((current_text, current_frames))

    # Final pass: merge any remaining too-short tail with previous
    if len(groups) > 1 and groups[-1][1] < min_frames:
        prev_text, prev_frames = groups[-2]
        tail_text, tail_frames = groups[-1]
        merged = prev_text + " " + tail_text
        merged_frames = estimate_frames(merged, cpf)
        if merged_frames <= max_frames:
            groups[-2] = (merged, merged_frames)
            groups.pop()

    return groups


def rms_normalize(audio, target_rms):
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def generate_fast(model, text, prompt, gen_config_or_steps=8,
                  cfg_schedule=None, voice_seed=42,
                  crossfade_ms=30, silence_ms=25,
                  instruct="None", min_chunk_frames=60,
                  max_chunk_frames=450):
    """Full pipeline: generate with full text, split-decode at sentence boundaries.

    Returns: (full_audio, chunk_audios, chunk_info)
    where chunk_audios can be played sequentially for streaming.
    """
    from omnivoice.models.omnivoice import (
        GenerationTask, OmniVoiceGenerationConfig, _get_time_steps,
    )

    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id

    rl = prompt.ref_audio_tokens.shape[1]
    cpf = max(1, len(prompt.ref_text) / rl)

    # Plan sentence splits
    groups = plan_split_points(text, cpf, min_chunk_frames, max_chunk_frames)
    total_frames = sum(est for _, est in groups)

    # Clamp to model limits
    total_frames = max(25, min(500, total_frames))

    # Build CFG schedule
    if isinstance(gen_config_or_steps, int):
        num_steps = gen_config_or_steps
    else:
        num_steps = gen_config_or_steps.num_step

    if cfg_schedule is None:
        cfg_schedule = [0, 0] + [3.0] * (num_steps - 2)

    # --- GENERATE full token sequence with full text context ---
    torch.manual_seed(voice_seed)

    inp = model._prepare_inference_inputs(
        text, total_frames, prompt.ref_text, prompt.ref_audio_tokens,
        "English", instruct, True,
    )
    inp_ids = inp["input_ids"].squeeze(0)
    inp_amask = inp["audio_mask"].squeeze(0)
    c_len = inp_ids.size(1)

    # Build cond+uncond batches
    b2_ids = torch.full((2, C, c_len), mask_id, dtype=torch.long, device=model.device)
    b2_amask = torch.zeros((2, c_len), dtype=torch.bool, device=model.device)
    b2_attn = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=model.device)
    b2_ids[0, :, :c_len] = inp_ids
    b2_amask[0, :c_len] = inp_amask
    b2_attn[0, :, :c_len, :c_len] = True
    u_len = total_frames
    b2_ids[1, :, :u_len] = inp_ids[:, -u_len:]
    b2_amask[1, :u_len] = inp_amask[-u_len:]
    b2_attn[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pd = torch.arange(u_len, c_len, device=model.device)
        b2_attn[1, :, pd, pd] = True
    b1_ids = b2_ids[0:1].clone()
    b1_amask = b2_amask[0:1].clone()
    b1_attn = b2_attn[0:1].clone()

    tokens = torch.full((C, total_frames), mask_id, dtype=torch.long, device=model.device)
    layer_ids = torch.arange(C, device=model.device).view(-1, 1)

    timesteps = _get_time_steps(0.0, 1.0, num_steps + 1, 0.1).tolist()
    total_mask = total_frames * C
    rem = total_mask
    schedule = []
    for step in range(num_steps):
        num = rem if step == num_steps - 1 else min(
            math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    t_gen_start = time.time()

    for step in range(num_steps):
        cfg = cfg_schedule[step] if step < len(cfg_schedule) else 0.0

        if cfg > 0:
            b2_ids[0, :, c_len - total_frames:c_len] = tokens
            b2_ids[1, :, :total_frames] = tokens
            with torch.no_grad():
                logits = model(input_ids=b2_ids, audio_mask=b2_amask,
                               attention_mask=b2_attn).logits.to(torch.float32)
            c_log = F.log_softmax(logits[0:1, :, c_len - total_frames:c_len, :], dim=-1)
            u_log = F.log_softmax(logits[1:2, :, :total_frames, :], dim=-1)
            log_probs = torch.log_softmax(c_log + cfg * (c_log - u_log), dim=-1)
        else:
            b1_ids[0, :, c_len - total_frames:c_len] = tokens
            with torch.no_grad():
                logits = model(input_ids=b1_ids, audio_mask=b1_amask,
                               attention_mask=b1_attn).logits.to(torch.float32)
            log_probs = F.log_softmax(logits[0:1, :, c_len - total_frames:c_len, :], dim=-1)

        log_probs[..., mask_id] = float("-inf")
        pred_tokens = log_probs.argmax(dim=-1)[0]
        scores = log_probs.max(dim=-1)[0][0]
        scores = scores - (layer_ids * 5.0)
        if 5.0 > 0:
            g = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
            scores = scores / 5.0 + g
        is_masked = tokens == mask_id
        scores[~is_masked] = float("-inf")
        k = schedule[step]
        if k > 0:
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = tokens.flatten()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            tokens = flat.view(C, total_frames)

    gen_time = time.time() - t_gen_start

    # --- SPLIT DECODE at sentence boundaries ---
    # Distribute frames proportional to character length
    char_lens = [len(g[0]) for g in groups]
    total_chars = sum(char_lens)
    frame_splits = []
    used = 0
    for i, cl in enumerate(char_lens):
        if i == len(char_lens) - 1:
            frames = total_frames - used
        else:
            frames = int(total_frames * cl / total_chars)
        frame_splits.append(frames)
        used += frames

    # Decode each chunk
    chunk_audios = []
    chunk_info = []
    pos = 0
    for ci, (group_text, est_frames) in enumerate(groups):
        n_frames = frame_splits[ci]
        chunk_toks = tokens[:, pos:pos + n_frames]

        t0 = time.time()
        audio = model.audio_tokenizer.decode(
            chunk_toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        decode_time = time.time() - t0

        chunk_audios.append(audio)
        chunk_info.append({
            "text": group_text,
            "frames": n_frames,
            "decode_ms": decode_time * 1000,
            "duration_s": len(audio) / 24000,
        })
        pos += n_frames

    # --- ASSEMBLE with silence + crossfade ---
    if len(chunk_audios) == 1:
        full_audio = chunk_audios[0]
    else:
        # RMS normalize
        rms_values = [torch.sqrt(torch.mean(a ** 2)).item() for a in chunk_audios]
        target_rms = np.mean(rms_values)
        chunk_audios_norm = [rms_normalize(a, target_rms) for a in chunk_audios]

        silence_samples = int(24000 * silence_ms / 1000)
        xfade_samples = int(24000 * crossfade_ms / 1000)
        silence = torch.zeros(silence_samples)

        full_audio = chunk_audios_norm[0]
        for chunk in chunk_audios_norm[1:]:
            full_audio = torch.cat([full_audio, silence])
            if len(full_audio) >= xfade_samples and len(chunk) >= xfade_samples:
                fo = torch.linspace(1, 0, xfade_samples)
                fi = torch.linspace(0, 1, xfade_samples)
                overlap = full_audio[-xfade_samples:] * fo + chunk[:xfade_samples] * fi
                full_audio = torch.cat([full_audio[:-xfade_samples], overlap, chunk[xfade_samples:]])
            else:
                full_audio = torch.cat([full_audio, chunk])

    return {
        "audio": full_audio,
        "chunk_audios": chunk_audios,
        "chunk_info": chunk_info,
        "gen_time_ms": gen_time * 1000,
        "total_frames": total_frames,
        "num_chunks": len(groups),
        "cfg_schedule": cfg_schedule[:num_steps],
    }


def main():
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
    from faster_whisper import WhisperModel

    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    # Correct ref_text
    segs, _ = whisper_model.transcribe("barth_ref.wav", language="en")
    actual_ref = " ".join(s.text.strip() for s in segs)
    prompt = model.create_voice_clone_prompt(
        ref_audio="barth_ref.wav", ref_text=actual_ref, preprocess_prompt=True
    )

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

    # Also generate baseline for comparison
    gc = OmniVoiceGenerationConfig(
        num_step=8, guidance_scale=3.0,
        position_temperature=5.0, class_temperature=0.0,
    )

    texts = [
        "I can not believe you would do something like that. After everything we have been through together, you just threw it all away.",
        "Welcome to the annual science conference. Today we will explore the fascinating world of quantum computing and its implications for artificial intelligence.",
        "Hello. My name is Johannes. I am testing sentence chunking for streaming text to speech synthesis. This is the last sentence.",
        "Stop. Just stop. I have heard enough. You need to leave right now and never come back.",
        "The weather today is beautiful. Clear skies, warm breeze, and the sun is shining. Perfect day for a walk in the park. I think I will go outside.",
        "As the sun slowly sets over the quiet and distant rolling hills, the birds begin their soft and gentle evening song, filling the warm summer air with a melody that echoes across the peaceful valley below, while the last rays of golden light dance upon the surface of the still lake.",
    ]

    out = Path("./fast_output")
    out.mkdir(parents=True, exist_ok=True)

    for ti, text in enumerate(texts):
        print(f"\nText {ti}: {text[:80]}...")

        # Baseline: standard 8-step, full CFG
        rl = prompt.ref_audio_tokens.shape[1]
        cpf = max(1, len(prompt.ref_text) / rl)
        tl = max(50, min(500, int(len(text) / cpf)))
        task = GenerationTask(
            batch_size=1, texts=[text], target_lens=[tl],
            langs=["English"], instructs=["None"],
            ref_audio_tokens=[prompt.ref_audio_tokens],
            ref_texts=[prompt.ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )
        torch.manual_seed(42)
        t0 = time.time()
        with torch.no_grad():
            base_toks = model._generate_iterative(task, gc)[0]
        base_time = (time.time() - t0) * 1000
        base_audio = model.audio_tokenizer.decode(
            base_toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        torchaudio.save(str(out / f"baseline_{ti}.wav"), base_audio.unsqueeze(0), 24000)
        s2, _ = whisper_model.transcribe(str(out / f"baseline_{ti}.wav"), language="en")
        base_t = " ".join(s.text.strip() for s in s2)
        base_w = wer(text, base_t)
        print(f"  BASELINE (8step cfg=3 full): {base_time:.0f}ms WER={base_w:.0%} | {base_t[:90]}")

        # Fast: CFG schedule + split decode
        result = generate_fast(model, text, prompt, gen_config_or_steps=8)
        torchaudio.save(str(out / f"fast_{ti}.wav"), result["audio"].unsqueeze(0), 24000)
        s3, _ = whisper_model.transcribe(str(out / f"fast_{ti}.wav"), language="en")
        fast_t = " ".join(s.text.strip() for s in s3)
        fast_w = wer(text, fast_t)

        ttfa = result["gen_time_ms"] + result["chunk_info"][0]["decode_ms"]
        print(
            f"  FAST (cfg_sched+split): gen={result['gen_time_ms']:.0f}ms "
            f"ttfa={ttfa:.0f}ms WER={fast_w:.0%} "
            f"({result['num_chunks']} chunks) | {fast_t[:90]}"
        )
        for ci, info in enumerate(result["chunk_info"]):
            print(f"    chunk {ci}: {info['frames']}f {info['duration_s']:.1f}s decode={info['decode_ms']:.0f}ms | {info['text'][:60]}")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
