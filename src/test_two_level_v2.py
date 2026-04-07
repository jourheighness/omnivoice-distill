"""Two-level v2: overlapping groups with crossfade in the overlap region."""

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


def presplit_with_overlap(text, target_words=40, min_words=20, max_words=55):
    """Split text into groups with 1-sentence overlap between adjacent groups.

    Returns list of (group_text, overlap_sentence_or_None) tuples.
    overlap_sentence is the last sentence that overlaps with the next group.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [(text, None)]

    # First pass: assign sentences to groups
    groups_sents = []
    current = []
    current_words = 0

    for sent in sentences:
        sw = len(sent.split())
        if current_words + sw > max_words and current:
            groups_sents.append(current)
            current = [sent]
            current_words = sw
        else:
            current.append(sent)
            current_words += sw
    if current:
        groups_sents.append(current)

    # Merge short tail
    if len(groups_sents) > 1 and sum(len(s.split()) for s in groups_sents[-1]) < min_words:
        combined_words = sum(len(s.split()) for s in groups_sents[-2]) + sum(len(s.split()) for s in groups_sents[-1])
        if combined_words <= max_words + 15:
            groups_sents[-2].extend(groups_sents[-1])
            groups_sents.pop()

    # Build groups with overlap: each group includes the first sentence of the next group
    result = []
    for gi in range(len(groups_sents)):
        sents = list(groups_sents[gi])
        overlap_sent = None

        if gi < len(groups_sents) - 1:
            # Add first sentence of next group as overlap
            overlap_sent = groups_sents[gi + 1][0]
            sents.append(overlap_sent)

        group_text = " ".join(sents)
        result.append((group_text, overlap_sent))

    return result


def rms_normalize(audio, target_rms):
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def generate_group(model, text, prompt, cfg_schedule, voice_seed=42):
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


def decode_tokens(model, tokens):
    """Decode tokens to audio."""
    return model.audio_tokenizer.decode(
        tokens.unsqueeze(0)
    ).audio_values.squeeze().cpu().float()


def split_audio_by_sentences(model, tokens, group_text, cpf):
    """Split tokens proportionally by sentence character length, decode each."""
    sentences = split_sentences(group_text)
    total_frames = tokens.shape[1]

    if len(sentences) <= 1:
        return [(decode_tokens(model, tokens), group_text)]

    char_lens = [len(s) for s in sentences]
    total_chars = sum(char_lens)
    parts = []
    pos = 0
    for i, cl in enumerate(char_lens):
        if i == len(char_lens) - 1:
            n = total_frames - pos
        else:
            n = int(total_frames * cl / total_chars)
        audio = decode_tokens(model, tokens[:, pos:pos + n])
        parts.append((audio, sentences[i]))
        pos += n

    return parts


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

    out = Path("./two_level_v2_output")
    out.mkdir(parents=True, exist_ok=True)

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

    CROSSFADE_MS = 60
    SILENCE_MS = 40

    for name, text in texts.items():
        n_words = len(text.split())
        print(f"\n{'='*70}")
        print(f"[{name}] {n_words} words")

        # Build overlapping groups
        groups = presplit_with_overlap(text)
        print(f"  Groups ({len(groups)}):")
        for gi, (gtxt, overlap) in enumerate(groups):
            gw = len(gtxt.split())
            print(f"    {gi}: {gw}w | {gtxt[:70]}...")
            if overlap:
                print(f"       overlap: {overlap[:50]}")

        # Generate each group
        group_audios = []  # list of list of (audio, sentence_text)
        group_times = []

        for gi, (gtxt, overlap) in enumerate(groups):
            t0 = time.time()
            toks, nf = generate_group(model, gtxt, prompt, cfg_schedule)
            gen_ms = (time.time() - t0) * 1000
            group_times.append(gen_ms)

            parts = split_audio_by_sentences(model, toks, gtxt, cpf)
            group_audios.append(parts)

        # Assemble with overlap crossfade
        # For each pair of adjacent groups, the last sentence-audio of group N
        # and the first sentence-audio of group N+1 cover the same text.
        # Crossfade between them.

        final_parts = []  # flat list of audio chunks to assemble

        for gi in range(len(group_audios)):
            parts = group_audios[gi]

            if gi == 0:
                # First group: take all parts except the overlap tail
                if groups[gi][1] is not None:
                    # Has overlap — take all but last part
                    for audio, sent in parts[:-1]:
                        final_parts.append(audio)
                    # The overlap sentence: crossfade with next group's first sentence
                    overlap_audio_from_this = parts[-1][0]
                    overlap_audio_from_next = group_audios[gi + 1][0][0]

                    # Crossfade the two versions of the overlap sentence
                    xfade_len = min(
                        len(overlap_audio_from_this),
                        len(overlap_audio_from_next),
                    )
                    # Full-length crossfade over the overlap sentence
                    fade_out = torch.linspace(1, 0, xfade_len)
                    fade_in = torch.linspace(0, 1, xfade_len)

                    # Trim to same length
                    a1 = overlap_audio_from_this[:xfade_len]
                    a2 = overlap_audio_from_next[:xfade_len]

                    # RMS match before crossfade
                    rms1 = torch.sqrt(torch.mean(a1 ** 2))
                    rms2 = torch.sqrt(torch.mean(a2 ** 2))
                    if rms2 > 0:
                        a2 = a2 * (rms1 / rms2)

                    blended = a1 * fade_out + a2 * fade_in
                    final_parts.append(blended)
                else:
                    for audio, sent in parts:
                        final_parts.append(audio)

            elif gi < len(group_audios) - 1:
                # Middle group: skip first part (overlap from previous), take middle,
                # crossfade last with next
                for audio, sent in parts[1:-1]:
                    final_parts.append(audio)

                overlap_audio_from_this = parts[-1][0]
                overlap_audio_from_next = group_audios[gi + 1][0][0]
                xfade_len = min(len(overlap_audio_from_this), len(overlap_audio_from_next))
                fade_out = torch.linspace(1, 0, xfade_len)
                fade_in = torch.linspace(0, 1, xfade_len)
                a1 = overlap_audio_from_this[:xfade_len]
                a2 = overlap_audio_from_next[:xfade_len]
                rms1 = torch.sqrt(torch.mean(a1 ** 2))
                rms2 = torch.sqrt(torch.mean(a2 ** 2))
                if rms2 > 0:
                    a2 = a2 * (rms1 / rms2)
                blended = a1 * fade_out + a2 * fade_in
                final_parts.append(blended)
            else:
                # Last group: skip first part (overlap from previous), take rest
                for audio, sent in parts[1:]:
                    final_parts.append(audio)

        # Assemble final audio with silence between non-overlap parts
        silence = torch.zeros(int(24000 * SILENCE_MS / 1000))
        xfade_samples = int(24000 * CROSSFADE_MS / 1000)

        result = final_parts[0]
        for part in final_parts[1:]:
            result = torch.cat([result, silence])
            if len(result) >= xfade_samples and len(part) >= xfade_samples:
                fo = torch.linspace(1, 0, xfade_samples)
                fi = torch.linspace(0, 1, xfade_samples)
                overlap = result[-xfade_samples:] * fo + part[:xfade_samples] * fi
                result = torch.cat([result[:-xfade_samples], overlap, part[xfade_samples:]])
            else:
                result = torch.cat([result, part])

        fp = out / f"{name}_v2.wav"
        torchaudio.save(str(fp), result.unsqueeze(0), 24000)

        s2, _ = whisper_model.transcribe(str(fp), language="en")
        transcript = " ".join(s.text.strip() for s in s2)
        w = wer(text, transcript)
        dur = len(result) / 24000
        ttfa = group_times[0]

        print(f"  V2 OVERLAP: gen={sum(group_times):.0f}ms ttfa={ttfa:.0f}ms dur={dur:.1f}s WER={w:.0%}")
        print(f"    group_times: {['%.0fms' % t for t in group_times]}")
        print(f"    {transcript[:120]}")

        # Also save v1 (no overlap) for comparison
        # Re-run with non-overlapping groups
        groups_v1 = []
        sentences = split_sentences(text)
        current = []
        current_words = 0
        for sent in sentences:
            sw = len(sent.split())
            if current_words + sw > 55 and current:
                groups_v1.append(" ".join(current))
                current = [sent]
                current_words = sw
            else:
                current.append(sent)
                current_words += sw
        if current:
            groups_v1.append(" ".join(current))

        v1_audios = []
        for gtxt in groups_v1:
            toks, nf = generate_group(model, gtxt, prompt, cfg_schedule)
            audio = decode_tokens(model, toks)
            v1_audios.append(audio)

        # RMS normalize v1
        rms_vals = [torch.sqrt(torch.mean(a ** 2)).item() for a in v1_audios]
        trms = np.mean(rms_vals)
        v1_audios = [rms_normalize(a, trms) for a in v1_audios]

        v1_result = v1_audios[0]
        for part in v1_audios[1:]:
            v1_result = torch.cat([v1_result, silence])
            if len(v1_result) >= xfade_samples and len(part) >= xfade_samples:
                fo = torch.linspace(1, 0, xfade_samples)
                fi = torch.linspace(0, 1, xfade_samples)
                ov = v1_result[-xfade_samples:] * fo + part[:xfade_samples] * fi
                v1_result = torch.cat([v1_result[:-xfade_samples], ov, part[xfade_samples:]])
            else:
                v1_result = torch.cat([v1_result, part])

        fp1 = out / f"{name}_v1.wav"
        torchaudio.save(str(fp1), v1_result.unsqueeze(0), 24000)
        s3, _ = whisper_model.transcribe(str(fp1), language="en")
        t1 = " ".join(s.text.strip() for s in s3)
        w1 = wer(text, t1)
        print(f"  V1 NO-OVERLAP: WER={w1:.0%}")
        print(f"    {t1[:120]}")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
