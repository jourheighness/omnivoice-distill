"""Two-level chunking: pre-split text into groups, generate each, split-decode for streaming."""

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
    """Split text into groups of ~target_words at sentence boundaries."""
    sentences = split_sentences(text)
    groups = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current:
            groups.append(" ".join(current))
            current = [sent]
            current_words = sent_words
        else:
            current.append(sent)
            current_words += sent_words

    if current:
        groups.append(" ".join(current))

    # Merge short tail with previous
    if len(groups) > 1 and len(groups[-1].split()) < min_words:
        combined = groups[-2] + " " + groups[-1]
        if len(combined.split()) <= max_words + 10:  # slight overflow ok for tail
            groups[-2] = combined
            groups.pop()

    return groups


def rms_normalize(audio, target_rms):
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def generate_group(model, text, prompt, cfg_schedule, voice_seed=42):
    """Generate one group with CFG scheduling."""
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    num_steps = len(cfg_schedule)

    rl = prompt.ref_audio_tokens.shape[1]
    cpf = max(1, len(prompt.ref_text) / rl)
    total_frames = max(25, min(450, int(len(text) / cpf)))

    torch.manual_seed(voice_seed)

    inp = model._prepare_inference_inputs(
        text, total_frames, prompt.ref_text, prompt.ref_audio_tokens,
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

    for step in range(num_steps):
        cfg = cfg_schedule[step]
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

    return tokens, total_frames


def split_decode_group(model, tokens, group_text, cpf):
    """Split a group's tokens at sentence boundaries and decode each."""
    sentences = split_sentences(group_text)
    total_frames = tokens.shape[1]

    if len(sentences) <= 1:
        audio = model.audio_tokenizer.decode(
            tokens.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        return [audio], sentences

    # Proportional frame allocation
    char_lens = [len(s) for s in sentences]
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

    audios = []
    pos = 0
    for n_frames in frame_splits:
        chunk_toks = tokens[:, pos:pos + n_frames]
        audio = model.audio_tokenizer.decode(
            chunk_toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        audios.append(audio)
        pos += n_frames

    return audios, sentences


def assemble_audio(audio_chunks, crossfade_ms=30, silence_ms=25):
    """Assemble audio chunks with RMS normalization, silence, and crossfade."""
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    rms_values = [torch.sqrt(torch.mean(a ** 2)).item() for a in audio_chunks]
    target_rms = np.mean(rms_values)
    chunks = [rms_normalize(a, target_rms) for a in audio_chunks]

    silence_samples = int(24000 * silence_ms / 1000)
    xfade_samples = int(24000 * crossfade_ms / 1000)
    silence = torch.zeros(silence_samples)

    result = chunks[0]
    for chunk in chunks[1:]:
        result = torch.cat([result, silence])
        if len(result) >= xfade_samples and len(chunk) >= xfade_samples:
            fo = torch.linspace(1, 0, xfade_samples)
            fi = torch.linspace(0, 1, xfade_samples)
            overlap = result[-xfade_samples:] * fo + chunk[:xfade_samples] * fi
            result = torch.cat([result[:-xfade_samples], overlap, chunk[xfade_samples:]])
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

    out = Path("./two_level_output")
    out.mkdir(parents=True, exist_ok=True)

    # Long test texts
    texts = {
        "monologue": (
            "I have been thinking about this for a long time now. "
            "The truth is, I never wanted things to end up this way. "
            "When we first started working together, I believed we could change the world. "
            "We had the talent, the drive, and most importantly, the vision. "
            "But somewhere along the way, things went wrong. "
            "The compromises started small, almost imperceptible at first. "
            "A minor shortcut here, a small omission there. "
            "Before we knew it, we had become everything we once despised. "
            "And now here we are, standing in the ruins of what could have been something truly magnificent."
        ),
        "technical": (
            "The transformer architecture has fundamentally changed how we approach sequence modeling. "
            "At its core, the self-attention mechanism allows each position in a sequence to attend to every other position. "
            "This creates a rich representation that captures long-range dependencies far more effectively than recurrent networks. "
            "The key innovation was the multi-head attention mechanism, which projects queries, keys, and values into multiple subspaces. "
            "Combined with positional encodings and layer normalization, this architecture has proven remarkably versatile. "
            "It now serves as the backbone for language models, speech synthesis systems, protein folding prediction, and many other applications."
        ),
        "story": (
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
        ),
    }

    gc_baseline = OmniVoiceGenerationConfig(
        num_step=8, guidance_scale=3.0,
        position_temperature=5.0, class_temperature=0.0,
    )

    for name, text in texts.items():
        n_words = len(text.split())
        print(f"\n{'='*70}")
        print(f"[{name}] {n_words} words: {text[:80]}...")

        # --- BASELINE: single full generation ---
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
            base_toks = model._generate_iterative(task, gc_baseline)[0]
        base_gen = (time.time() - t0) * 1000
        base_audio = model.audio_tokenizer.decode(
            base_toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        bp = out / f"{name}_baseline.wav"
        torchaudio.save(str(bp), base_audio.unsqueeze(0), 24000)
        s2, _ = whisper_model.transcribe(str(bp), language="en")
        base_t = " ".join(s.text.strip() for s in s2)
        base_w = wer(text, base_t)
        base_dur = len(base_audio) / 24000
        print(f"  BASELINE: gen={base_gen:.0f}ms dur={base_dur:.1f}s frames={tl} WER={base_w:.0%}")
        print(f"    {base_t[:120]}")

        # --- TWO-LEVEL: presplit + generate each + split decode ---
        groups = presplit_text(text)
        print(f"  Groups ({len(groups)}):")
        for gi, g in enumerate(groups):
            gw = len(g.split())
            gf = int(len(g) / cpf)
            print(f"    {gi}: {gw}w ~{gf}f | {g[:70]}")

        all_audio_chunks = []
        group_times = []
        group_ttfa = None

        for gi, group_text in enumerate(groups):
            t0 = time.time()
            toks, n_frames = generate_group(
                model, group_text, prompt, cfg_schedule, voice_seed=42
            )
            gen_ms = (time.time() - t0) * 1000
            group_times.append(gen_ms)

            # Split decode within group
            sub_audios, sub_sents = split_decode_group(model, toks, group_text, cpf)
            all_audio_chunks.extend(sub_audios)

            if gi == 0:
                group_ttfa = gen_ms + 5  # ~5ms decode

        # Assemble
        final_audio = assemble_audio(all_audio_chunks)
        fp = out / f"{name}_twolevel.wav"
        torchaudio.save(str(fp), final_audio.unsqueeze(0), 24000)
        s3, _ = whisper_model.transcribe(str(fp), language="en")
        fast_t = " ".join(s.text.strip() for s in s3)
        fast_w = wer(text, fast_t)
        fast_dur = len(final_audio) / 24000
        total_gen = sum(group_times)

        print(f"  TWO-LEVEL: total_gen={total_gen:.0f}ms ttfa={group_ttfa:.0f}ms "
              f"dur={fast_dur:.1f}s WER={fast_w:.0%} ({len(groups)} groups, {len(all_audio_chunks)} chunks)")
        print(f"    group_times: {['%.0fms' % t for t in group_times]}")
        print(f"    {fast_t[:120]}")
        print(f"  SPEEDUP: {base_gen/total_gen:.1f}x gen, TTFA={group_ttfa:.0f}ms vs {base_gen:.0f}ms")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
