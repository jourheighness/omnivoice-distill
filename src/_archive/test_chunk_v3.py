"""V3 sentence chunking: merge short sentences to minimum frame count."""

import torch
import torchaudio
import time
import numpy as np
import re
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
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


def merge_short_chunks(sentences, cpf, min_frames=65, max_frames=150):
    """Merge consecutive short sentences until each chunk >= min_frames."""
    chunks = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        est_frames = int(len(candidate) / cpf)
        if not current:
            current = sent
        elif est_frames <= max_frames:
            current = candidate
        else:
            # Current chunk is big enough, start new
            chunks.append(current)
            current = sent
    if current:
        chunks.append(current)

    # Second pass: merge any remaining short chunks with previous neighbor
    merged = []
    for chunk in chunks:
        est = int(len(chunk) / cpf)
        if merged and est < min_frames:
            prev_est = int(len(merged[-1]) / cpf)
            if prev_est + est <= max_frames:
                merged[-1] = merged[-1] + " " + chunk
                continue
        merged.append(chunk)

    return merged


def rms_normalize(audio, target_rms):
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def generate_chunked_v3(model, full_text, prompt, gc,
                        min_frames=65, crossfade_ms=30,
                        silence_ms=25, voice_seed=42,
                        instruct="Measured pace, clear diction"):
    sentences = split_sentences(full_text)
    rl = prompt.ref_audio_tokens.shape[1]
    cpf = max(1, len(prompt.ref_text) / rl)

    chunks = merge_short_chunks(sentences, cpf, min_frames=min_frames)

    est_frames = [int(len(c) / cpf) for c in chunks]
    print(f"    Chunks ({len(chunks)}): {[c[:40] for c in chunks]}")
    print(f"    Est frames: {est_frames}")

    audio_chunks = []
    chunk_times = []

    for ci, chunk_text in enumerate(chunks):
        tl = max(min_frames, min(150, int(len(chunk_text) / cpf)))

        torch.manual_seed(voice_seed)

        task = GenerationTask(
            batch_size=1, texts=[chunk_text], target_lens=[tl],
            langs=["English"], instructs=[instruct],
            ref_audio_tokens=[prompt.ref_audio_tokens],
            ref_texts=[prompt.ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )

        t0 = time.time()
        with torch.no_grad():
            toks = model._generate_iterative(task, gc)[0]
        chunk_times.append(time.time() - t0)

        audio = model.audio_tokenizer.decode(
            toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        audio_chunks.append(audio)

    if len(audio_chunks) == 1:
        return audio_chunks[0], chunk_times, chunks

    # RMS normalize
    rms_values = [torch.sqrt(torch.mean(a ** 2)).item() for a in audio_chunks]
    target_rms = np.mean(rms_values)
    audio_chunks = [rms_normalize(a, target_rms) for a in audio_chunks]

    # Concatenate with silence + crossfade
    silence_samples = int(24000 * silence_ms / 1000)
    crossfade_samples = int(24000 * crossfade_ms / 1000)
    silence = torch.zeros(silence_samples)

    result = audio_chunks[0]
    for chunk in audio_chunks[1:]:
        result = torch.cat([result, silence])
        if len(result) >= crossfade_samples and len(chunk) >= crossfade_samples:
            fade_out = torch.linspace(1, 0, crossfade_samples)
            fade_in = torch.linspace(0, 1, crossfade_samples)
            overlap = (
                result[-crossfade_samples:] * fade_out
                + chunk[:crossfade_samples] * fade_in
            )
            result = torch.cat(
                [result[:-crossfade_samples], overlap, chunk[crossfade_samples:]]
            )
        else:
            result = torch.cat([result, chunk])

    return result, chunk_times, chunks


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
    ]

    for ti, text in enumerate(texts):
        print(f"\nText {ti}: {text[:80]}")

        rl = prompt.ref_audio_tokens.shape[1]
        cpf = max(1, len(prompt.ref_text) / rl)

        # Full baseline
        tl = max(50, min(200, int(len(text) / cpf)))
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
            full_toks = model._generate_iterative(task, gc)[0]
        full_time = time.time() - t0
        full_audio = model.audio_tokenizer.decode(
            full_toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        torchaudio.save(f"/tmp/v3_full_{ti}.wav", full_audio.unsqueeze(0), 24000)
        s2, _ = whisper_model.transcribe(f"/tmp/v3_full_{ti}.wav", language="en")
        ft = " ".join(s.text.strip() for s in s2)
        fw = wer(text, ft)
        print(f"  FULL: {full_time*1000:.0f}ms WER={fw:.0%} | {ft[:100]}")

        # V3 chunked
        v3_audio, v3_times, v3_chunks = generate_chunked_v3(
            model, text, prompt, gc
        )
        torchaudio.save(f"/tmp/v3_chunked_{ti}.wav", v3_audio.unsqueeze(0), 24000)
        s3, _ = whisper_model.transcribe(f"/tmp/v3_chunked_{ti}.wav", language="en")
        ct = " ".join(s.text.strip() for s in s3)
        cw = wer(text, ct)
        ttfa = v3_times[0] * 1000
        total = sum(v3_times) * 1000
        print(
            f"  V3: total={total:.0f}ms ttfa={ttfa:.0f}ms WER={cw:.0%} "
            f"({len(v3_chunks)} chunks) | {ct[:100]}"
        )


if __name__ == "__main__":
    main()
