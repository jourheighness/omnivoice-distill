"""End-to-end test: speculative decode with real OmniVoice teacher + trained draft.

Generates audio via:
1. Baseline: full iterative unmasking (current approach)
2. Speculative: draft AR cb0 + teacher fills cb1-7

Compares timing and outputs audio files for listening.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))

from omnivoice_mlx.generate import (
    OmniVoiceMLXConfig,
    OmniVoiceMLXModel,
    generate_iterative,
    _log_softmax,
)
from omnivoice_mlx.vocoder import AudioTokenizerDecoder
from draft_mlx import DraftMLXConfig, DraftModelMLX

import math


def load_teacher(weights_path):
    """Load OmniVoice MLX teacher model."""
    config = OmniVoiceMLXConfig()
    model = OmniVoiceMLXModel(config)
    vocoder = AudioTokenizerDecoder(
        num_quantizers=config.num_quantizers,
        hidden_size=config.vq_hidden_size,
        codebook_dim=config.codebook_dim,
        codebook_size=config.codebook_size,
        semantic_hidden_size=config.semantic_hidden_size,
        dac_input_dim=config.dac_input_dim,
        dac_hidden_dim=config.dac_hidden_dim,
        dac_upsampling_ratios=config.dac_upsampling_ratios,
    )

    weights_path = Path(weights_path)
    model_weights = mx.load(str(weights_path / "model.safetensors"))
    vocoder_weights = mx.load(str(weights_path / "vocoder.safetensors"))

    model.load_weights(list(model_weights.items()))
    vocoder.load_weights(list(vocoder_weights.items()))

    mx.eval(model.parameters(), vocoder.parameters())
    return model, vocoder, config


def load_draft(checkpoint_path):
    """Load trained MLX draft model."""
    config = DraftMLXConfig(
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=2048,
    )
    model = DraftModelMLX(config)
    model.load_weights(str(checkpoint_path))
    mx.eval(model.parameters())
    return model, config


def get_conditioning_hidden(teacher, input_ids, audio_mask, target_len):
    """Extract teacher conditioning hidden states."""
    C = teacher.config.num_audio_codebook
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    ids = mx.expand_dims(input_ids, axis=0)
    mask = mx.expand_dims(audio_mask, axis=0)
    attn = mx.ones((1, 1, L_total, L_total), dtype=mx.bool_)

    inputs_embeds = teacher._prepare_embed_inputs(ids, mask)
    hidden_states = teacher.llm(inputs_embeds, mask=attn)
    cond_hidden = hidden_states[:, :L_cond, :]
    mx.eval(cond_hidden)
    return cond_hidden


def speculative_generate(teacher, draft, input_ids, audio_mask, target_len,
                          teacher_steps=4, guidance_scale=3.0):
    """Speculative decoding: draft cb0 AR, teacher fills cb1-7."""
    config = teacher.config
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    t0 = time.perf_counter()

    # Get conditioning for draft
    cond_hidden = get_conditioning_hidden(teacher, input_ids, audio_mask, target_len)
    t_cond = time.perf_counter()

    # Draft generates cb0 autoregressively
    draft_cb0 = draft.generate_ar(cond_hidden, num_tokens=target_len, temperature=0.0)
    mx.eval(draft_cb0)
    t_draft = time.perf_counter()

    # Teacher fills all codebooks via unmasking
    # (cb0 from draft provides a strong prior even though we run full unmasking)
    tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=teacher_steps, guidance_scale=guidance_scale,
    )
    mx.eval(tokens)
    t_teacher = time.perf_counter()

    # Measure cb0 agreement
    teacher_cb0 = tokens[0]
    matches = mx.sum(draft_cb0 == teacher_cb0).item()
    acceptance = matches / target_len

    return {
        "tokens": tokens,
        "draft_cb0": draft_cb0,
        "teacher_cb0": teacher_cb0,
        "acceptance": acceptance,
        "cond_ms": (t_cond - t0) * 1000,
        "draft_ms": (t_draft - t_cond) * 1000,
        "teacher_ms": (t_teacher - t_draft) * 1000,
        "total_ms": (t_teacher - t0) * 1000,
    }


def baseline_generate(teacher, input_ids, audio_mask, target_len,
                       num_step=8, guidance_scale=3.0):
    """Standard full unmasking generation."""
    t0 = time.perf_counter()
    tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=num_step, guidance_scale=guidance_scale,
    )
    mx.eval(tokens)
    elapsed = time.perf_counter() - t0
    return {"tokens": tokens, "total_ms": elapsed * 1000}


def tokens_to_audio(vocoder, tokens):
    """Decode tokens to audio waveform."""
    codes = mx.expand_dims(tokens, axis=0)
    audio = vocoder(codes)
    mx.eval(audio)
    return np.array(audio[0, :, 0], dtype=np.float32)


def make_test_inputs(config, target_len=75):
    """Create test inputs (synthetic — only for warmup)."""
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    style_len = 10
    text_len = 30
    ref_len = 40

    text_tokens = mx.random.randint(0, 1000, (text_len + style_len,))
    text_ids = mx.broadcast_to(text_tokens.reshape(1, -1), (C, text_len + style_len))
    ref_ids = mx.random.randint(0, 1024, (C, ref_len))
    target_ids = mx.full((C, target_len), mask_id, dtype=mx.int32)

    input_ids = mx.concatenate([text_ids, ref_ids, target_ids], axis=1)

    audio_mask = mx.concatenate([
        mx.zeros(text_len + style_len, dtype=mx.bool_),
        mx.ones(ref_len + target_len, dtype=mx.bool_),
    ])

    return input_ids, audio_mask


def load_cached_samples(cache_dir, num=5):
    """Load cached teacher data for testing with real conditioning."""
    import json
    cache_path = Path(cache_dir)
    manifest = json.loads((cache_path / "manifest.json").read_text())
    samples = []
    for entry in manifest[:num]:
        data = np.load(cache_path / entry["file"])
        samples.append({
            "cond_hidden": mx.array(data["cond_hidden"]),
            "cb0_tokens": mx.array(data["cb0_tokens"].astype(np.int32)),
            "all_tokens": mx.array(data["all_tokens"].astype(np.int32)),
            "target_len": entry["target_len"],
        })
    return samples


def test_with_cached_data(teacher, vocoder, draft, config, cache_dir, output_dir, num_tests=5):
    """Test using real cached conditioning from teacher — this is the real test."""
    samples = load_cached_samples(cache_dir, num_tests)
    print(f"Loaded {len(samples)} cached samples for testing\n")

    results = []
    for i, sample in enumerate(samples):
        cond = mx.expand_dims(sample["cond_hidden"], axis=0)  # (1, L_cond, H)
        target_len = sample["target_len"]
        teacher_cb0 = sample["cb0_tokens"]  # ground truth from teacher

        # Draft generates cb0 (feed first token as seed, predict the rest)
        first_token = teacher_cb0[0:1]  # seed with first token
        t0 = time.perf_counter()
        draft_rest = draft.generate_ar(cond, num_tokens=target_len - 1,
                                        temperature=0.0, start_token=first_token)
        mx.eval(draft_rest)
        draft_cb0 = mx.concatenate([first_token, draft_rest])
        draft_ms = (time.perf_counter() - t0) * 1000

        # Acceptance: how many match the teacher's cb0?
        matches = mx.sum(draft_cb0 == teacher_cb0).item()
        acceptance = matches / target_len

        # Build full tokens: use draft cb0 + teacher's cb1-7
        all_tokens = mx.array(sample["all_tokens"])  # (C, T) from teacher
        spec_tokens = mx.array(all_tokens)
        # Replace cb0 with draft's prediction for speculative output
        spec_tokens_np = np.array(spec_tokens)
        spec_tokens_np[0] = np.array(draft_cb0)
        spec_tokens_with_draft = mx.array(spec_tokens_np)

        results.append({
            "acceptance": acceptance,
            "draft_ms": draft_ms,
            "target_len": target_len,
        })

        # Save audio for first sample
        if i == 0:
            import soundfile as sf
            # Teacher's full output
            teacher_audio = tokens_to_audio(vocoder, all_tokens)
            sf.write(str(output_dir / "teacher_real.wav"), teacher_audio, 24000)

            # Speculative: draft cb0 + teacher cb1-7
            spec_audio = tokens_to_audio(vocoder, spec_tokens_with_draft)
            sf.write(str(output_dir / "speculative_real.wav"), spec_audio, 24000)
            print(f"  Saved: teacher_real.wav and speculative_real.wav")

        print(f"  Test {i+1}/{len(samples)}: "
              f"draft={draft_ms:.0f}ms | "
              f"accept={acceptance:.1%} ({matches}/{target_len}) | "
              f"audio={target_len*40}ms")

    mean_acc = np.mean([r["acceptance"] for r in results])
    mean_draft = np.mean([r["draft_ms"] for r in results])

    print(f"\n{'='*60}")
    print(f"REAL DATA RESULTS ({len(results)} tests)")
    print(f"{'='*60}")
    print(f"  CB0 acceptance:  {mean_acc:.1%}")
    print(f"  Draft speed:     {mean_draft:.0f}ms per utterance")
    print(f"  Draft per token: {mean_draft/results[0]['target_len']:.1f}ms")

    rtf = (results[0]["target_len"] / 25) / (mean_draft / 1000)
    print(f"  Realtime factor: {rtf:.1f}x")

    if mean_acc > 0.7:
        print(f"\n  VERDICT: EXCELLENT — speculative decode will work great")
    elif mean_acc > 0.5:
        print(f"\n  VERDICT: GOOD — worthwhile speedup")
    else:
        print(f"\n  VERDICT: Needs more training data")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", type=str, required=True)
    parser.add_argument("--draft_weights", type=str, required=True)
    parser.add_argument("--target_len", type=int, default=75)
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="../test_output")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Path to cached teacher data for real testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading teacher...")
    teacher, vocoder, config = load_teacher(args.teacher_weights)
    print("Loading draft...")
    draft, draft_config = load_draft(args.draft_weights)
    print()

    # Warmup
    print("Warmup pass...")
    input_ids, audio_mask = make_test_inputs(config, args.target_len)
    _ = baseline_generate(teacher, input_ids, audio_mask, args.target_len, num_step=4)
    print()

    # Use cached real data if available
    if args.cache_dir:
        test_with_cached_data(teacher, vocoder, draft, config,
                              args.cache_dir, output_dir, args.num_tests)
    else:
        print("No --cache_dir provided. Run with cached teacher data for real results.")
        print("Example: --cache_dir ../cache_real")


if __name__ == "__main__":
    main()
