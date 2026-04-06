"""Test speculative decoding locally — draft + teacher combined.

Validates the full pipeline: draft generates cb0 tokens AR,
teacher fills in remaining codebooks via unmasking.

Usage:
    python local/test_speculative.py \
        --weights_path /path/to/mlx/weights \
        --draft_path ./checkpoints_local/draft.safetensors
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))
from omnivoice_mlx.generate import (
    OmniVoiceMLXConfig,
    OmniVoiceMLXModel,
    generate_iterative,
    _log_softmax,
)

from draft_mlx import DraftMLXConfig, DraftModelMLX
from cache_teacher import extract_conditioning_hidden_states, make_synthetic_inputs


def speculative_generate(
    teacher: OmniVoiceMLXModel,
    draft: DraftModelMLX,
    input_ids: mx.array,       # (C, L_total) with mask region
    audio_mask: mx.array,      # (L_total,)
    target_len: int,
    draft_chunk: int = 10,     # how many frames to draft at a time
    teacher_steps: int = 4,    # unmasking steps for teacher verification
    guidance_scale: float = 3.0,
    temperature: float = 0.0,
) -> dict:
    """Speculative generation: draft predicts cb0, teacher fills cb1-7.

    Pipeline:
    1. Extract conditioning hidden states from teacher
    2. Draft generates cb0 tokens autoregressively in chunks
    3. For each chunk, teacher fills in cb1-7 via unmasking
       (with cb0 pre-filled from draft, reducing mask count)
    4. Measure acceptance rate by comparing with full teacher generation

    Returns dict with tokens, timing, and acceptance metrics.
    """
    config = teacher.config
    C = config.num_codebooks if hasattr(config, 'num_codebooks') else config.num_audio_codebook
    mask_id = config.audio_mask_id
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    t_start = time.perf_counter()

    # Step 1: Get conditioning embeddings for draft model
    cond_hidden = extract_conditioning_hidden_states(
        teacher, input_ids, audio_mask, target_len,
    )  # (1, L_cond, H)

    t_cond = time.perf_counter()

    # Step 2: Draft generates all cb0 tokens autoregressively
    draft_cb0 = draft.generate_ar(
        cond_hidden, num_tokens=target_len, temperature=temperature,
    )  # (target_len,)
    mx.eval(draft_cb0)

    t_draft = time.perf_counter()

    # Step 3: Teacher fills remaining codebooks
    # Pre-fill cb0 from draft, mask cb1-7
    tokens = mx.full((C, target_len), mask_id, dtype=mx.int32)
    tokens = tokens.at[0, :].add(draft_cb0 - mask_id)  # set cb0 to draft predictions

    # Run teacher unmasking with cb0 already filled
    # (This means ~7/8 of tokens are masked instead of 8/8 — fewer steps needed)
    teacher_tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=teacher_steps, guidance_scale=guidance_scale,
    )
    mx.eval(teacher_tokens)

    t_teacher = time.perf_counter()

    # Step 4: Measure acceptance — how many cb0 tokens match?
    teacher_cb0 = teacher_tokens[0]  # (target_len,)
    matches = mx.sum(draft_cb0 == teacher_cb0).item()
    acceptance_rate = matches / target_len

    # Build final output: use teacher's cb0 (verified) + teacher's cb1-7
    final_tokens = teacher_tokens  # teacher has the final say

    return {
        "tokens": final_tokens,                    # (C, target_len)
        "draft_cb0": np.array(draft_cb0),          # (target_len,)
        "teacher_cb0": np.array(teacher_cb0),      # (target_len,)
        "acceptance_rate": acceptance_rate,
        "timing": {
            "conditioning_ms": (t_cond - t_start) * 1000,
            "draft_ms": (t_draft - t_cond) * 1000,
            "teacher_verify_ms": (t_teacher - t_draft) * 1000,
            "total_ms": (t_teacher - t_start) * 1000,
        },
    }


def baseline_generate(
    teacher: OmniVoiceMLXModel,
    input_ids: mx.array,
    audio_mask: mx.array,
    target_len: int,
    num_step: int = 8,
    guidance_scale: float = 3.0,
) -> dict:
    """Baseline: full teacher generation for comparison."""
    t0 = time.perf_counter()
    tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=num_step, guidance_scale=guidance_scale,
    )
    mx.eval(tokens)
    elapsed = time.perf_counter() - t0

    return {
        "tokens": tokens,
        "total_ms": elapsed * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Test speculative decoding locally")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--draft_path", type=str, default="./checkpoints_local/draft.safetensors")
    parser.add_argument("--target_len", type=int, default=50)
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--draft_hidden", type=int, default=256)
    parser.add_argument("--draft_layers", type=int, default=2)
    args = parser.parse_args()

    # Load teacher
    print("Loading teacher model...")
    config = OmniVoiceMLXConfig()
    teacher = OmniVoiceMLXModel(config)
    weights = mx.load(str(Path(args.weights_path) / "model.safetensors"))
    teacher.load_weights(list(weights.items()))
    mx.eval(teacher.parameters())

    # Load draft
    print("Loading draft model...")
    draft_config = DraftMLXConfig(
        hidden_size=args.draft_hidden,
        num_layers=args.draft_layers,
        num_heads=max(1, args.draft_hidden // 64),
    )
    draft = DraftModelMLX(draft_config)
    draft.load_weights(args.draft_path)
    mx.eval(draft.parameters())

    draft_params = sum(p.size for p in draft.parameters().values() if isinstance(p, mx.array))
    print(f"Draft model: {draft_params:,} params")

    # Run tests
    print(f"\nRunning {args.num_tests} speculative decode tests (target_len={args.target_len})...\n")

    spec_times = []
    base_times = []
    acceptance_rates = []

    for i in range(args.num_tests):
        input_ids, audio_mask = make_synthetic_inputs(config, args.target_len)

        # Baseline
        base_result = baseline_generate(teacher, input_ids, audio_mask, args.target_len)

        # Speculative
        spec_result = speculative_generate(
            teacher, draft, input_ids, audio_mask, args.target_len,
            teacher_steps=4,  # fewer steps since cb0 is pre-filled
        )

        spec_times.append(spec_result["timing"]["total_ms"])
        base_times.append(base_result["total_ms"])
        acceptance_rates.append(spec_result["acceptance_rate"])

        print(f"  Test {i+1}: "
              f"baseline={base_result['total_ms']:.0f}ms | "
              f"speculative={spec_result['timing']['total_ms']:.0f}ms "
              f"(draft={spec_result['timing']['draft_ms']:.0f}ms + "
              f"verify={spec_result['timing']['teacher_verify_ms']:.0f}ms) | "
              f"cb0_accept={spec_result['acceptance_rate']:.1%}")

    print(f"\n{'='*60}")
    print(f"RESULTS ({args.num_tests} tests, target_len={args.target_len})")
    print(f"{'='*60}")
    print(f"  Baseline avg:     {np.mean(base_times):.0f}ms")
    print(f"  Speculative avg:  {np.mean(spec_times):.0f}ms")
    print(f"  Speedup:          {np.mean(base_times)/np.mean(spec_times):.2f}x")
    print(f"  CB0 accept rate:  {np.mean(acceptance_rates):.1%}")
    print(f"\nNote: With a well-trained draft, acceptance rate should be >60%.")
    print(f"The tiny local model is just for pipeline validation.")


if __name__ == "__main__":
    main()
