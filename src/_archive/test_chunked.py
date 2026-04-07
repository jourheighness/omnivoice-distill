"""Test Approach B: chunked generation with full text context.

Generates audio in chunks, feeding previous chunk tokens as additional
reference audio. Each chunk sees the FULL text — preserving the model's
ability to plan prosody/emotion across the whole utterance.

Compares: full utterance (baseline) vs chunked+concatenated.
"""

import math
import time
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path


def generate_full(model, text, prompt, gen_config, target_len, seed=42):
    """Generate full utterance in one shot (baseline)."""
    from omnivoice.models.omnivoice import GenerationTask
    torch.manual_seed(seed)

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

    t0 = time.time()
    with torch.no_grad():
        tokens_list = model._generate_iterative(task, gen_config)
    elapsed = time.time() - t0
    return tokens_list[0], elapsed  # (C, T)


def generate_chunked(model, text, prompt, gen_config, total_target_len,
                     chunk_size=50, overlap=15, seed=42):
    """Generate in overlapping chunks with full text context.

    Each chunk:
    - Sees the FULL text
    - Gets previous chunks' tokens appended to reference audio
    - Generates chunk_size new frames
    - Overlap region is cross-faded with previous chunk
    """
    from omnivoice.models.omnivoice import GenerationTask

    C = model.config.num_audio_codebook
    device = model.device

    all_chunks = []  # list of (tokens, start_frame, end_frame)
    generated_tokens = None  # accumulated tokens from previous chunks

    # Calculate chunk boundaries
    chunks = []
    pos = 0
    while pos < total_target_len:
        end = min(pos + chunk_size, total_target_len)
        actual_size = end - pos
        chunks.append((pos, end, actual_size))
        pos = end - overlap  # next chunk starts with overlap
        if end >= total_target_len:
            break

    print(f"    Chunk plan: {len(chunks)} chunks, size={chunk_size}, overlap={overlap}")
    for i, (start, end, size) in enumerate(chunks):
        print(f"      chunk {i}: frames {start}-{end} (size={size})")

    t0 = time.time()

    for chunk_idx, (start, end, actual_size) in enumerate(chunks):
        torch.manual_seed(seed + chunk_idx)

        # Build reference: original ref + all previously generated tokens
        if generated_tokens is not None:
            # Append generated tokens to reference
            ref_tokens = torch.cat([
                prompt.ref_audio_tokens,  # (C, T_ref)
                generated_tokens,  # (C, T_generated)
            ], dim=1)
            # Use original ref text + marker that we have continuation
            ref_text = prompt.ref_text
        else:
            ref_tokens = prompt.ref_audio_tokens
            ref_text = prompt.ref_text

        task = GenerationTask(
            batch_size=1,
            texts=[text],
            target_lens=[actual_size],
            langs=["English"],
            instructs=["None"],
            ref_audio_tokens=[ref_tokens],
            ref_texts=[ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )

        with torch.no_grad():
            chunk_tokens_list = model._generate_iterative(task, gen_config)
        chunk_tokens = chunk_tokens_list[0]  # (C, actual_size)

        all_chunks.append({
            "tokens": chunk_tokens,
            "start": start,
            "end": end,
        })

        # Accumulate: for next chunk's reference, include non-overlap portion
        if generated_tokens is None:
            # First chunk: keep everything except the overlap tail
            keep = actual_size - overlap if chunk_idx < len(chunks) - 1 else actual_size
            generated_tokens = chunk_tokens[:, :keep]
        else:
            # Subsequent chunks: skip the overlap head, keep the rest
            new_part = chunk_tokens[:, overlap:] if chunk_idx < len(chunks) - 1 else chunk_tokens[:, overlap:]
            generated_tokens = torch.cat([generated_tokens, new_part], dim=1)

    elapsed = time.time() - t0

    # Assemble final tokens with cross-fade in overlap regions
    # For discrete tokens, "cross-fade" means: in the overlap region,
    # use the chunk whose tokens have higher confidence (or just use prev chunk's)
    # Simplest: just use previous chunk in overlap region (hard cut)
    final_tokens = torch.full(
        (C, total_target_len), model.config.audio_mask_id,
        dtype=torch.long, device=device,
    )

    for chunk_info in all_chunks:
        t = chunk_info["tokens"]
        s, e = chunk_info["start"], chunk_info["end"]
        actual_len = t.shape[1]
        # Only write to positions not yet filled (first chunk wins in overlap)
        for pos in range(actual_len):
            frame = s + pos
            if frame < total_target_len:
                if final_tokens[0, frame] == model.config.audio_mask_id:
                    final_tokens[:, frame] = t[:, pos]

    return final_tokens, elapsed, len(chunks)


def generate_chunked_blend(model, text, prompt, gen_config, total_target_len,
                           chunk_size=50, overlap=15, seed=42):
    """Same as generate_chunked but in overlap regions, use LATER chunk's tokens.

    Rationale: the later chunk has more context (it sees all previous tokens).
    """
    from omnivoice.models.omnivoice import GenerationTask

    C = model.config.num_audio_codebook
    device = model.device

    all_chunks = []
    generated_tokens = None

    chunks = []
    pos = 0
    while pos < total_target_len:
        end = min(pos + chunk_size, total_target_len)
        actual_size = end - pos
        chunks.append((pos, end, actual_size))
        pos = end - overlap
        if end >= total_target_len:
            break

    t0 = time.time()

    for chunk_idx, (start, end, actual_size) in enumerate(chunks):
        torch.manual_seed(seed + chunk_idx)

        if generated_tokens is not None:
            ref_tokens = torch.cat([
                prompt.ref_audio_tokens,
                generated_tokens,
            ], dim=1)
        else:
            ref_tokens = prompt.ref_audio_tokens

        task = GenerationTask(
            batch_size=1,
            texts=[text],
            target_lens=[actual_size],
            langs=["English"],
            instructs=["None"],
            ref_audio_tokens=[ref_tokens],
            ref_texts=[prompt.ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )

        with torch.no_grad():
            chunk_tokens_list = model._generate_iterative(task, gen_config)
        chunk_tokens = chunk_tokens_list[0]

        all_chunks.append({
            "tokens": chunk_tokens,
            "start": start,
            "end": end,
        })

        if generated_tokens is None:
            keep = actual_size - overlap if chunk_idx < len(chunks) - 1 else actual_size
            generated_tokens = chunk_tokens[:, :keep]
        else:
            new_part = chunk_tokens[:, overlap:]
            generated_tokens = torch.cat([generated_tokens, new_part], dim=1)

    elapsed = time.time() - t0

    # Later chunk wins in overlap (overwrite)
    final_tokens = torch.full(
        (C, total_target_len), model.config.audio_mask_id,
        dtype=torch.long, device=device,
    )

    for chunk_info in all_chunks:
        t = chunk_info["tokens"]
        s = chunk_info["start"]
        actual_len = t.shape[1]
        for pos in range(actual_len):
            frame = s + pos
            if frame < total_target_len:
                final_tokens[:, frame] = t[:, pos]  # overwrite = later wins

    return final_tokens, elapsed, len(chunks)


def tokens_to_audio(model, tokens):
    with torch.no_grad():
        result = model.audio_tokenizer.decode(tokens.unsqueeze(0))
        audio = result.audio_values
    return audio.squeeze(0).squeeze(0).cpu().float()


def transcribe(audio_path, whisper_model):
    segments, _ = whisper_model.transcribe(str(audio_path), language="en")
    return " ".join(s.text.strip() for s in segments)


def main():
    output_dir = Path("./test_chunked")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        {
            "text": "I can't believe you would do something like that! After everything we've been through together, you just threw it all away.",
            "name": "emotional",
        },
        {
            "text": "Welcome to the annual science conference. Today we'll explore the fascinating world of quantum computing and its implications for artificial intelligence research.",
            "name": "long_formal",
        },
        {
            "text": "Once upon a time, in a land far far away, there lived a brave knight who feared nothing. But one dark night, everything changed.",
            "name": "narrative",
        },
    ]

    print("Loading OmniVoice...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

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

    results = []

    for case in test_cases:
        text = case["text"]
        name = case["name"]
        print(f"\n{'='*70}")
        print(f"[{name}] {text[:80]}...")

        ref_len = prompt.ref_audio_tokens.shape[1]
        chars_per_frame = max(1, len(prompt.ref_text) / ref_len)
        target_len = max(50, min(200, int(len(text) / chars_per_frame)))
        print(f"  target_len={target_len}")

        # 1. Full utterance baseline
        print(f"\n  FULL (baseline):")
        full_tokens, full_time = generate_full(model, text, prompt, gen_config, target_len)
        full_audio = tokens_to_audio(model, full_tokens)
        full_path = output_dir / f"{name}_full.wav"
        torchaudio.save(str(full_path), full_audio.unsqueeze(0), 24000)
        full_transcript = transcribe(full_path, whisper)
        print(f"    time={full_time:.2f}s transcript: {full_transcript[:120]}")

        # 2. Chunked (first chunk wins overlap)
        for chunk_size, overlap in [(50, 15), (40, 10), (30, 8)]:
            print(f"\n  CHUNKED first-wins (chunk={chunk_size}, overlap={overlap}):")
            chunked_tokens, chunked_time, n_chunks = generate_chunked(
                model, text, prompt, gen_config, target_len,
                chunk_size=chunk_size, overlap=overlap,
            )
            chunked_audio = tokens_to_audio(model, chunked_tokens)
            chunked_path = output_dir / f"{name}_chunked_{chunk_size}_{overlap}.wav"
            torchaudio.save(str(chunked_path), chunked_audio.unsqueeze(0), 24000)
            chunked_transcript = transcribe(chunked_path, whisper)
            print(f"    {n_chunks} chunks, time={chunked_time:.2f}s")
            print(f"    transcript: {chunked_transcript[:120]}")

            # Token overlap comparison (how many cb0 tokens match between full and chunked?)
            min_len = min(full_tokens.shape[1], chunked_tokens.shape[1])
            cb0_match = (full_tokens[0, :min_len] == chunked_tokens[0, :min_len]).float().mean().item()
            print(f"    cb0 match vs full: {cb0_match:.1%}")

        # 3. Chunked (later chunk wins overlap)
        print(f"\n  CHUNKED later-wins (chunk=50, overlap=15):")
        blend_tokens, blend_time, n_chunks = generate_chunked_blend(
            model, text, prompt, gen_config, target_len,
            chunk_size=50, overlap=15,
        )
        blend_audio = tokens_to_audio(model, blend_tokens)
        blend_path = output_dir / f"{name}_blend_50_15.wav"
        torchaudio.save(str(blend_path), blend_audio.unsqueeze(0), 24000)
        blend_transcript = transcribe(blend_path, whisper)
        print(f"    {n_chunks} chunks, time={blend_time:.2f}s")
        print(f"    transcript: {blend_transcript[:120]}")

        min_len = min(full_tokens.shape[1], blend_tokens.shape[1])
        cb0_match = (full_tokens[0, :min_len] == blend_tokens[0, :min_len]).float().mean().item()
        print(f"    cb0 match vs full: {cb0_match:.1%}")

    print(f"\n{'='*70}")
    print(f"Audio files saved to {output_dir}/")
    print("Listen to compare: full vs chunked vs blend")


if __name__ == "__main__":
    main()
