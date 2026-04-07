"""OmniVoice Fast MLX — port of omnivoice_fast.py to Apple Silicon.

Pipeline:
1. Voice calibration (cached chars_per_sec per voice)
2. Frame-budget splitting at sentence boundaries
3. Tight frame allocation (padding=0.95)
4. Adaptive steps: 8 for first group (fast TTFA), 16 for rest
5. CFG scheduling [0,0,3,3,...] — skip CFG on first 2 steps
6. Split-decode at sentence boundaries
7. Assembly: RMS normalize + silence + crossfade + trailing silence trim
"""

import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Add voice-service to path for omnivoice_mlx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))

from omnivoice_mlx.generate import (
    OmniVoiceMLXConfig,
    OmniVoiceMLXModel,
    generate_iterative,
)
from omnivoice_mlx.vocoder import AudioTokenizerDecoder


# ---------------------------------------------------------------------------
# Voice calibration data (cached from Whisper analysis on GPU)
# ---------------------------------------------------------------------------

VOICE_CALIBRATIONS = {
    "barth": {"chars_per_sec": 17.4, "words_per_sec": 3.5},
    "astarion": {"chars_per_sec": 13.0, "words_per_sec": 2.6},
    "vesper": {"chars_per_sec": 18.1, "words_per_sec": 4.2},
}

# Default fallback for unknown voices
_DEFAULT_CALIBRATION = {"chars_per_sec": 15.0, "words_per_sec": 3.0}


def calibrate_from_ref(ref_text: str, num_ref_frames: int, voice_name: str = "", fps: int = 25) -> dict:
    """Auto-calibrate voice speaking rate from ref audio metadata.

    Uses Whisper-validated calibrations when available (more accurate —
    they exclude pauses). Falls back to naive estimation from ref_text/frames
    with a 0.8x correction factor (ref audio typically has pauses that inflate
    the naive rate).
    """
    # Prefer Whisper-validated calibrations
    for key in VOICE_CALIBRATIONS:
        if key in voice_name.lower():
            return dict(VOICE_CALIBRATIONS[key])

    if not ref_text or num_ref_frames <= 0:
        return dict(_DEFAULT_CALIBRATION)

    duration_sec = num_ref_frames / fps
    chars = len(ref_text.strip())
    words = len(ref_text.strip().split())
    # 0.8x correction: naive calc overestimates because ref audio has pauses
    # that the text doesn't account for. Better to allocate too many frames
    # (slight silence) than too few (clipping/popping).
    return {
        "chars_per_sec": chars / max(duration_sec, 0.1) * 0.8,
        "words_per_sec": words / max(duration_sec, 0.1) * 0.8,
    }


# ---------------------------------------------------------------------------
# Text splitting (same as GPU version)
# ---------------------------------------------------------------------------

def split_sentences(text):
    """Split on sentence boundaries (.!?) keeping punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def estimate_frames(text, cpf):
    """Estimate number of audio frames for text."""
    return max(15, int(len(text) / cpf))


def plan_split_points(text, cpf, min_frames=60, max_frames=280, first_max_frames=100):
    """Plan where to split decoded audio at sentence boundaries.

    Returns list of (sentence_group_text, estimated_frames) tuples.
    Groups short sentences together to meet min_frames.

    first_max_frames: cap on the first group for fast TTFA. Once the first
    group is flushed, subsequent groups use max_frames.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [(text, estimate_frames(text, cpf))]

    groups = []
    current_text = ""
    current_frames = 0

    for sent in sentences:
        sent_frames = estimate_frames(sent, cpf)
        # First group uses tighter cap for fast TTFA
        cap = first_max_frames if not groups else max_frames
        merge_sweet = min(cap, 200)

        if not current_text:
            current_text = sent
            current_frames = sent_frames
            # If this single sentence already exceeds the cap, flush it
            if current_frames >= cap and len(sentences) > 1:
                groups.append((current_text, current_frames))
                current_text = ""
                current_frames = 0
            continue

        combined_frames = estimate_frames(current_text + " " + sent, cpf)

        if combined_frames <= cap:
            if current_frames < min_frames:
                current_text = current_text + " " + sent
                current_frames = combined_frames
            elif combined_frames <= merge_sweet:
                current_text = current_text + " " + sent
                current_frames = combined_frames
            else:
                groups.append((current_text, current_frames))
                current_text = sent
                current_frames = sent_frames
        else:
            groups.append((current_text, current_frames))
            current_text = sent
            current_frames = sent_frames

    if current_text:
        groups.append((current_text, current_frames))

    # Merge too-short tail with previous
    if len(groups) > 1 and groups[-1][1] < min_frames:
        prev_text, prev_frames = groups[-2]
        tail_text, tail_frames = groups[-1]
        merged = prev_text + " " + tail_text
        merged_frames = estimate_frames(merged, cpf)
        if merged_frames <= max_frames:
            groups[-2] = (merged, merged_frames)
            groups.pop()

    return groups


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _assemble_chunks(chunk_audios, silence_ms=40, crossfade_ms=30):
    """RMS normalize, add silence + crossfade between chunks."""
    if len(chunk_audios) == 1:
        return chunk_audios[0]

    rms_values = [np.sqrt(np.mean(a ** 2)) for a in chunk_audios]
    target_rms = np.mean(rms_values)
    normed = [a * (target_rms / rms) if rms > 1e-6 else a for a, rms in zip(chunk_audios, rms_values)]

    silence_samples = int(24000 * silence_ms / 1000)
    xfade_samples = int(24000 * crossfade_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    full_audio = normed[0]
    for chunk in normed[1:]:
        full_audio = np.concatenate([full_audio, silence])
        if len(full_audio) >= xfade_samples and len(chunk) >= xfade_samples:
            fo = np.linspace(1, 0, xfade_samples, dtype=np.float32)
            fi = np.linspace(0, 1, xfade_samples, dtype=np.float32)
            overlap = full_audio[-xfade_samples:] * fo + chunk[:xfade_samples] * fi
            full_audio = np.concatenate([full_audio[:-xfade_samples], overlap, chunk[xfade_samples:]])
        else:
            full_audio = np.concatenate([full_audio, chunk])
    return full_audio


def build_inputs(
    text: str,
    ref_text: str,
    ref_audio_tokens: mx.array,  # (C, N_ref)
    tokenizer,
    target_len: int,
    config: OmniVoiceMLXConfig,
):
    """Build input_ids and audio_mask for generation."""
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    # Style prefix
    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(
        mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids))
    )

    # Text: ref_text + target_text
    text_str = f"<|text_start|>{ref_text} {text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
    text_tokens = mx.broadcast_to(
        mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids))
    )

    # Target masks
    target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)

    # Concatenate: style | text | ref_audio | target
    input_ids = mx.concatenate([style_tokens, text_tokens, ref_audio_tokens, target_masks], axis=1)

    # Audio mask: True for ref_audio + target
    text_region = style_tokens.shape[1] + text_tokens.shape[1]
    audio_region = ref_audio_tokens.shape[1] + target_len
    audio_mask = mx.concatenate([
        mx.zeros(text_region, dtype=mx.bool_),
        mx.ones(audio_region, dtype=mx.bool_),
    ])

    return input_ids, audio_mask


def generate_fast(
    model: OmniVoiceMLXModel,
    vocoder: AudioTokenizerDecoder,
    text: str,
    ref_text: str,
    ref_audio_tokens: mx.array,
    tokenizer,
    voice_name: str = "barth",
    padding: float = 0.95,
    num_steps: int = 0,  # 0=auto-scale with frame count
    cfg_skip: int = 2,
    cfg_val: float = 3.0,
    crossfade_ms: int = 30,
    silence_ms: int = 40,
    seed: int = 42,
    min_frames: int = 60,
    max_frames: int = 280,
    first_max_frames: int = 100,
):
    """Full-context generation with split-decode for streaming.

    Generates ALL tokens at once (full text context for prosody),
    then split-decodes at sentence boundaries. First chunk is small
    for fast TTFA; later chunks are larger.

    TTFA = gen_time + first_chunk_decode_time.
    """
    config = model.config

    # Voice calibration
    cal = VOICE_CALIBRATIONS.get(voice_name, {"chars_per_sec": 15.0})
    fps = 25
    cpf = cal["chars_per_sec"] / fps

    # Plan sentence groups (first group capped for fast TTFA)
    groups = plan_split_points(text, cpf, min_frames, max_frames, first_max_frames)

    # Total frames with tight padding
    total_frames = sum(max(15, int(est * padding)) for _, est in groups)
    total_frames = max(25, min(500, total_frames))

    # Build input with full text context
    input_ids, audio_mask = build_inputs(
        text, ref_text, ref_audio_tokens, tokenizer, total_frames, config,
    )

    mx.random.seed(seed)

    # --- Generate all tokens at once ---
    # Auto-scale steps: short sequences need fewer, long ones need more
    if num_steps == 0:  # auto
        num_steps = 12
    cfg_schedule = [0.0] * min(cfg_skip, num_steps) + [cfg_val] * max(0, num_steps - cfg_skip)

    t_gen_start = time.perf_counter()
    tokens = generate_iterative(
        model, input_ids, audio_mask, total_frames,
        num_step=num_steps,
        cfg_schedule=cfg_schedule,
    )
    mx.eval(tokens)
    gen_time = time.perf_counter() - t_gen_start

    # --- Split-decode at sentence boundaries ---
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

    chunk_audios = []
    chunk_info = []
    pos = 0
    for ci, (group_text, est_frames) in enumerate(groups):
        n_frames = frame_splits[ci]
        chunk_toks = tokens[:, pos:pos + n_frames]

        t0 = time.perf_counter()
        codes = mx.expand_dims(chunk_toks, axis=0)
        audio = vocoder(codes)
        mx.eval(audio)
        decode_time = time.perf_counter() - t0

        audio_np = np.array(audio[0, :, 0], dtype=np.float32)
        chunk_audios.append(audio_np)
        chunk_info.append({
            "text": group_text,
            "frames": n_frames,
            "gen_ms": gen_time * 1000 if ci == 0 else 0,
            "decode_ms": decode_time * 1000,
            "duration_s": len(audio_np) / 24000,
            "steps": num_steps,
        })
        pos += n_frames

    # --- Assemble with silence + crossfade ---
    full_audio = _assemble_chunks(chunk_audios, silence_ms, crossfade_ms)

    # Trim trailing silence (-40dB threshold)
    threshold = 10 ** (-40 / 20)
    above = np.where(np.abs(full_audio) > threshold)[0]
    if len(above) > 0:
        end = min(len(full_audio), above[-1] + int(24000 * 0.1))
        full_audio = full_audio[:end]

    total_decode = sum(c["decode_ms"] for c in chunk_info)
    ttfa = gen_time * 1000 + chunk_info[0]["decode_ms"]

    return {
        "audio": full_audio,
        "chunk_audios": chunk_audios,
        "chunk_info": chunk_info,
        "total_time_ms": gen_time * 1000 + total_decode,
        "gen_time_ms": gen_time * 1000,
        "ttfa_ms": ttfa,
        "total_frames": total_frames,
        "num_chunks": len(groups),
        "num_steps": num_steps,
    }


def generate_hybrid(
    model: OmniVoiceMLXModel,
    vocoder: AudioTokenizerDecoder,
    text: str,
    ref_text: str,
    ref_audio_tokens: mx.array,
    tokenizer,
    voice_name: str = "barth",
    padding: float = 1.05,
    first_padding: float = 0.95,
    first_steps: int = 8,
    full_steps: int = 0,  # 0=auto
    cfg_skip: int = 2,
    cfg_val: float = 3.0,
    rep_penalty: float = 0.0,
    rep_window: int = 3,
    crossfade_ms: int = 30,
    silence_ms: int = 40,
    seed: int = 42,
    min_frames: int = 60,
    max_frames: int = 280,
    first_max_frames: int = 100,
):
    """Hybrid: fast first chunk + full-context continuation.

    1. Generate first sentence group at first_steps with first_padding (tight, no repeats)
    2. Generate FULL text at full_steps with padding (loose, natural pace)
    3. Take chunks 1+ from the full generation, crossfade at boundary

    TTFA = first_chunk_gen + first_chunk_decode.
    """
    config = model.config

    cal = VOICE_CALIBRATIONS.get(voice_name, {"chars_per_sec": 15.0})
    fps = 25
    cpf = cal["chars_per_sec"] / fps

    groups = plan_split_points(text, cpf, min_frames, max_frames, first_max_frames)

    # Single group — no hybrid needed, just generate directly
    if len(groups) == 1:
        return generate_fast(
            model, vocoder, text, ref_text, ref_audio_tokens, tokenizer,
            voice_name=voice_name, padding=padding, cfg_skip=cfg_skip,
            cfg_val=cfg_val, seed=seed, min_frames=min_frames,
            max_frames=max_frames, first_max_frames=first_max_frames,
        )

    mx.random.seed(seed)

    # --- Phase 1: Fast first chunk (tight padding to avoid repeats) ---
    first_text = groups[0][0]
    first_frames = max(25, int(groups[0][1] * first_padding))
    first_cfg = [0.0] * min(cfg_skip, first_steps) + [cfg_val] * max(0, first_steps - cfg_skip)

    first_ids, first_mask = build_inputs(
        first_text, ref_text, ref_audio_tokens, tokenizer, first_frames, config,
    )

    t0 = time.perf_counter()
    first_tokens = generate_iterative(
        model, first_ids, first_mask, first_frames,
        num_step=first_steps, cfg_schedule=first_cfg,
        rep_penalty=rep_penalty, rep_window=rep_window,
    )
    mx.eval(first_tokens)
    first_gen_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    first_audio = np.array(
        vocoder(mx.expand_dims(first_tokens, 0))[0, :, 0], dtype=np.float32
    )
    mx.eval(first_audio) if hasattr(first_audio, 'eval') else None
    first_decode_ms = (time.perf_counter() - t0) * 1000
    ttfa = first_gen_ms + first_decode_ms

    # --- Phase 2: Full-context generation for remaining chunks ---
    total_frames = sum(max(15, int(est * padding)) for _, est in groups)
    total_frames = max(25, min(500, total_frames))

    if full_steps == 0:
        if total_frames <= 100:
            full_steps = 4
        elif total_frames <= 200:
            full_steps = 6
        else:
            full_steps = 8
    full_cfg = [0.0] * min(cfg_skip, full_steps) + [cfg_val] * max(0, full_steps - cfg_skip)

    full_ids, full_mask = build_inputs(
        text, ref_text, ref_audio_tokens, tokenizer, total_frames, config,
    )

    mx.random.seed(seed)  # same seed for consistency
    t0 = time.perf_counter()
    full_tokens = generate_iterative(
        model, full_ids, full_mask, total_frames,
        num_step=full_steps, cfg_schedule=full_cfg,
        rep_penalty=rep_penalty, rep_window=rep_window,
    )
    mx.eval(full_tokens)
    full_gen_ms = (time.perf_counter() - t0) * 1000

    # Split-decode chunks 1+ from full generation
    char_lens = [len(g[0]) for g in groups]
    total_chars = sum(char_lens)
    frame_splits = []
    used = 0
    for i, cl in enumerate(char_lens):
        if i == len(char_lens) - 1:
            frame_splits.append(total_frames - used)
        else:
            frame_splits.append(int(total_frames * cl / total_chars))
            used += frame_splits[-1]

    chunk_audios = [first_audio]  # chunk 0 from fast generation
    chunk_info = [{
        "text": first_text,
        "frames": first_frames,
        "gen_ms": first_gen_ms,
        "decode_ms": first_decode_ms,
        "duration_s": len(first_audio) / 24000,
        "steps": first_steps,
    }]

    pos = frame_splits[0]  # skip chunk 0's frames in the full generation
    for ci in range(1, len(groups)):
        n_frames = frame_splits[ci]
        chunk_toks = full_tokens[:, pos:pos + n_frames]

        t0 = time.perf_counter()
        audio = vocoder(mx.expand_dims(chunk_toks, 0))
        mx.eval(audio)
        decode_ms = (time.perf_counter() - t0) * 1000

        audio_np = np.array(audio[0, :, 0], dtype=np.float32)
        chunk_audios.append(audio_np)
        chunk_info.append({
            "text": groups[ci][0],
            "frames": n_frames,
            "gen_ms": full_gen_ms if ci == 1 else 0,
            "decode_ms": decode_ms,
            "duration_s": len(audio_np) / 24000,
            "steps": full_steps,
        })
        pos += n_frames

    full_audio = _assemble_chunks(chunk_audios, silence_ms, crossfade_ms)

    # Trim trailing silence
    threshold = 10 ** (-40 / 20)
    above = np.where(np.abs(full_audio) > threshold)[0]
    if len(above) > 0:
        end = min(len(full_audio), above[-1] + int(24000 * 0.1))
        full_audio = full_audio[:end]

    total_time = first_gen_ms + first_decode_ms + full_gen_ms + sum(c["decode_ms"] for c in chunk_info[1:])

    return {
        "audio": full_audio,
        "chunk_audios": chunk_audios,
        "chunk_info": chunk_info,
        "total_time_ms": total_time,
        "gen_time_ms": first_gen_ms + full_gen_ms,
        "ttfa_ms": ttfa,
        "total_frames": total_frames,
        "num_chunks": len(groups),
        "num_steps": f"{first_steps}/{full_steps}",
    }


def generate_fast_adaptive(
    model: OmniVoiceMLXModel,
    vocoder: AudioTokenizerDecoder,
    text: str,
    ref_text: str,
    ref_audio_tokens: mx.array,
    tokenizer,
    voice_name: str = "barth",
    padding: float = 0.95,
    first_steps: int = 8,
    later_steps: int = 16,
    cfg_skip: int = 2,
    cfg_val: float = 3.0,
    crossfade_ms: int = 30,
    silence_ms: int = 40,
    seed: int = 42,
    min_frames: int = 60,
    max_frames: int = 280,
    first_max_frames: int = 100,
):
    """Fast pipeline with adaptive steps — generate each group separately.

    Group 0: first_steps (fast TTFA), capped at first_max_frames
    Groups 1+: later_steps (best quality), up to max_frames

    For streaming: yield chunk_audios as they're generated.
    """
    config = model.config

    # Voice calibration
    cal = VOICE_CALIBRATIONS.get(voice_name, {"chars_per_sec": 15.0})
    fps = 25
    cpf = cal["chars_per_sec"] / fps

    # Plan groups — first group capped for fast TTFA
    groups = plan_split_points(text, cpf, min_frames, max_frames, first_max_frames)

    mx.random.seed(seed)

    chunk_audios = []
    chunk_info = []
    t_total_start = time.perf_counter()

    for gi, (group_text, est_frames) in enumerate(groups):
        target_len = max(25, min(500, int(est_frames * padding)))
        num_steps = first_steps if gi == 0 else later_steps
        cfg_schedule = [0.0] * min(cfg_skip, num_steps) + [cfg_val] * max(0, num_steps - cfg_skip)

        input_ids, audio_mask = build_inputs(
            group_text, ref_text, ref_audio_tokens, tokenizer, target_len, config,
        )

        t0 = time.perf_counter()
        tokens = generate_iterative(
            model, input_ids, audio_mask, target_len,
            num_step=num_steps,
            cfg_schedule=cfg_schedule,
        )
        mx.eval(tokens)
        gen_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        codes = mx.expand_dims(tokens, axis=0)
        audio = vocoder(codes)
        mx.eval(audio)
        decode_ms = (time.perf_counter() - t1) * 1000

        audio_np = np.array(audio[0, :, 0], dtype=np.float32)
        chunk_audios.append(audio_np)
        chunk_info.append({
            "text": group_text,
            "frames": target_len,
            "gen_ms": gen_ms,
            "decode_ms": decode_ms,
            "duration_s": len(audio_np) / 24000,
            "steps": num_steps,
        })

    total_time = time.perf_counter() - t_total_start

    full_audio = _assemble_chunks(chunk_audios, silence_ms, crossfade_ms)

    # Trim trailing silence
    threshold = 10 ** (-40 / 20)
    above = np.where(np.abs(full_audio) > threshold)[0]
    if len(above) > 0:
        end = min(len(full_audio), above[-1] + int(24000 * 0.1))
        full_audio = full_audio[:end]

    return {
        "audio": full_audio,
        "chunk_audios": chunk_audios,
        "chunk_info": chunk_info,
        "total_time_ms": total_time * 1000,
        "num_chunks": len(groups),
    }


# ---------------------------------------------------------------------------
# Voice encoding + caching
# ---------------------------------------------------------------------------

def encode_voice(ref_audio_path: str, ref_text: str, output_path: str | None = None):
    """Encode reference audio using PyTorch OmniVoice, save as .npz for MLX.

    This is a one-time cost per voice. The fast pipeline loads cached tokens.
    """
    import torch
    from omnivoice import OmniVoice

    print(f"Encoding {ref_audio_path}...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path, ref_text=ref_text, preprocess_prompt=True,
    )

    tokens = prompt.ref_audio_tokens.cpu().numpy()
    actual_ref_text = prompt.ref_text

    if output_path is None:
        output_path = str(Path(ref_audio_path).with_suffix(".npz"))

    np.savez(output_path,
             ref_audio_tokens=tokens,
             ref_text=np.array(actual_ref_text))
    print(f"Saved to {output_path}: tokens={tokens.shape}, ref_text='{actual_ref_text[:60]}...'")
    del model
    return output_path


def load_cached_voice(npz_path: str) -> tuple[mx.array, str]:
    """Load pre-encoded voice tokens."""
    d = np.load(npz_path, allow_pickle=True)
    tokens = mx.array(d["ref_audio_tokens"].astype(np.int32))
    ref_text = str(d["ref_text"])
    return tokens, ref_text


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_benchmark(model, vocoder, ref_text, ref_audio_tokens, tokenizer, voice_name, args):
    """Benchmark across step/CFG configurations."""
    import soundfile as sf

    text = "I can not believe you would do something like that. After everything we have been through together, you just threw it all away."
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Warmup — eliminates first-run compilation overhead
    print("Warmup...")
    _ = generate_fast_adaptive(
        model, vocoder, text, ref_text, ref_audio_tokens, tokenizer,
        voice_name=voice_name, first_steps=4, later_steps=4,
    )

    configs = [
        # (first_steps, later_steps, cfg_skip, cfg_val, label)
        (8, 16, 2, 3.0, "8/16 skip2 cfg3 (baseline)"),
        (8, 16, 2, 3.0, "8/16 skip2 cfg3 (warm)"),
        (6, 12, 2, 3.0, "6/12 skip2 cfg3"),
        (4, 8,  2, 3.0, "4/8  skip2 cfg3"),
        (8, 16, 6, 7.0, "8/16 skip6 cfg7 (aggressive)"),
        (6, 12, 4, 7.0, "6/12 skip4 cfg7"),
        (4, 8,  2, 5.0, "4/8  skip2 cfg5"),
    ]

    print(f"\n{'Config':<35} {'TTFA':>6} {'Total':>7} {'Audio':>6} {'RTF':>5} {'Fwd0':>5}")
    print("-" * 75)

    for first_s, later_s, skip, cfg, label in configs:
        result = generate_fast_adaptive(
            model, vocoder, text, ref_text, ref_audio_tokens, tokenizer,
            voice_name=voice_name, first_steps=first_s, later_steps=later_s,
            cfg_skip=skip, cfg_val=cfg,
        )

        audio_s = len(result["audio"]) / 24000
        ttfa = result["chunk_info"][0]["gen_ms"] + result["chunk_info"][0]["decode_ms"]
        total = result["total_time_ms"]
        fwd0 = 2 * first_s - skip  # approximate fwd passes for first chunk

        sf.write(str(out / f"bench_{label.split()[0].replace('/', '_')}.wav"), result["audio"], 24000)

        print(f"{label:<35} {ttfa:>5.0f}ms {total:>6.0f}ms {audio_s:>5.1f}s {audio_s/(total/1000):>4.1f}x {fwd0:>4}")

    print(f"\nAudio saved to {out}/")


def main():
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="OmniVoice Fast MLX")
    parser.add_argument("--voice", default="barth", help="Voice name (barth/astarion/vesper)")
    parser.add_argument("--voice_npz", help="Path to cached voice .npz (overrides --voice)")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parent.parent.parent / "voice-service/weights/omnivoice_mlx"))
    parser.add_argument("--text", default="I can not believe you would do something like that. After everything we have been through together, you just threw it all away.")
    parser.add_argument("--mode", choices=["fast", "adaptive"], default="adaptive")
    parser.add_argument("--output_dir", default="./fast_mlx_output")
    parser.add_argument("--quantize", choices=["4", "8"], help="Quantize LLM to N-bit (4 or 8)")
    parser.add_argument("--compile", action="store_true", help="Compile LLM __call__ with mx.compile")
    parser.add_argument("--first_steps", type=int, default=4, help="Steps for first group (TTFA)")
    parser.add_argument("--later_steps", type=int, default=8, help="Steps for later groups")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--cfg_skip", type=int, default=2, help="Skip CFG on first N steps")
    parser.add_argument("--bench", action="store_true", help="Run benchmark across configs")
    parser.add_argument("--encode", help="Path to ref audio to encode (one-time setup)")
    parser.add_argument("--encode_text", help="Whisper transcript of ref audio (required with --encode)")
    args = parser.parse_args()

    # One-time voice encoding
    if args.encode:
        if not args.encode_text:
            print("ERROR: --encode_text required with --encode (Whisper transcript of ref audio)")
            sys.exit(1)
        encode_voice(args.encode, args.encode_text)
        return

    # Load model + vocoder
    print("Loading MLX model + vocoder...")
    t0 = time.perf_counter()
    config = OmniVoiceMLXConfig()
    model = OmniVoiceMLXModel(config)
    vocoder = AudioTokenizerDecoder(
        num_quantizers=config.num_quantizers, hidden_size=config.vq_hidden_size,
        codebook_dim=config.codebook_dim, codebook_size=config.codebook_size,
        semantic_hidden_size=config.semantic_hidden_size, dac_input_dim=config.dac_input_dim,
        dac_hidden_dim=config.dac_hidden_dim, dac_upsampling_ratios=config.dac_upsampling_ratios,
    )
    wp = Path(args.weights)
    model.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
    vocoder.load_weights(list(mx.load(str(wp / "vocoder.safetensors")).items()))
    mx.eval(model.parameters(), vocoder.parameters())

    if args.quantize:
        print(f"  Quantizing LLM to {args.quantize}-bit...")
        bits = int(args.quantize)
        nn.quantize(model.llm, bits=bits)
        mx.eval(model.parameters())

    if args.compile:
        print("  Compiling LLM backbone...")
        # Compile the Qwen3 __call__ (transformer layers + norm).
        # Can't compile the full OmniVoice model (dynamic embedding/slicing).
        # Can't replace the module itself (breaks attribute access to embed_tokens).
        model.llm.__call__ = mx.compile(model.llm.__call__, shapeless=True)

    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    # Load voice
    if args.voice_npz:
        ref_audio_tokens, ref_text = load_cached_voice(args.voice_npz)
        voice_name = Path(args.voice_npz).stem
    else:
        # Try to find cached voice .npz
        voice_npz = Path(__file__).parent.parent / f"{args.voice}_ref.npz"
        if voice_npz.exists():
            ref_audio_tokens, ref_text = load_cached_voice(str(voice_npz))
            voice_name = args.voice
        else:
            print(f"ERROR: No cached voice found at {voice_npz}")
            print(f"Run: python {__file__} --encode {args.voice}_ref.wav --encode_text 'whisper transcript'")
            sys.exit(1)

    print(f"Voice: {voice_name} | ref_text: '{ref_text[:60]}...' | ref_frames: {ref_audio_tokens.shape[1]}")

    if args.bench:
        run_benchmark(model, vocoder, ref_text, ref_audio_tokens, tokenizer, voice_name, args)
        return

    # Test texts
    texts = [
        args.text,
        "Welcome to the annual science conference. Today we will explore the fascinating world of quantum computing and its implications for artificial intelligence.",
        "Stop. Just stop. I have heard enough. You need to leave right now and never come back.",
        "The weather today is beautiful. Clear skies, warm breeze, and the sun is shining. Perfect day for a walk in the park.",
    ]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen_fn = generate_fast_adaptive if args.mode == "adaptive" else generate_fast

    for ti, text in enumerate(texts):
        print(f"\n--- Text {ti}: {text[:80]}... ---")

        result = gen_fn(
            model, vocoder, text, ref_text, ref_audio_tokens, tokenizer,
            voice_name=voice_name,
            first_steps=args.first_steps, later_steps=args.later_steps,
        )

        # Save audio
        sf.write(str(out / f"{args.mode}_{ti}.wav"), result["audio"], 24000)

        # Print stats
        total_audio_s = len(result["audio"]) / 24000
        if args.mode == "adaptive":
            ttfa = result["chunk_info"][0]["gen_ms"] + result["chunk_info"][0]["decode_ms"]
            total_ms = result["total_time_ms"]
            print(f"  TTFA: {ttfa:.0f}ms | Total: {total_ms:.0f}ms | Audio: {total_audio_s:.1f}s | RTF: {total_audio_s/(total_ms/1000):.1f}x")
            for ci, info in enumerate(result["chunk_info"]):
                print(f"    chunk {ci}: {info['frames']}f {info['duration_s']:.1f}s gen={info['gen_ms']:.0f}ms dec={info['decode_ms']:.0f}ms {info['steps']}step | {info['text'][:60]}")
        else:
            gen_ms = result["gen_time_ms"]
            print(f"  Gen: {gen_ms:.0f}ms | Audio: {total_audio_s:.1f}s | RTF: {total_audio_s/(gen_ms/1000):.1f}x")
            for ci, info in enumerate(result["chunk_info"]):
                print(f"    chunk {ci}: {info['frames']}f {info['duration_s']:.1f}s dec={info['decode_ms']:.0f}ms | {info['text'][:60]}")

    print(f"\nAudio saved to {out}/")


if __name__ == "__main__":
    main()
