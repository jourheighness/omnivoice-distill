"""Test v2 draft on real speech — the real test."""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))

from draft_mlx_v2 import DraftV2Config, DraftModelV2MLX
from omnivoice_mlx.generate import OmniVoiceMLXConfig, OmniVoiceMLXModel, generate_iterative
from omnivoice_mlx.vocoder import AudioTokenizerDecoder


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--draft_weights", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--text", default="Hello there, how are you doing today?")
    args = parser.parse_args()

    # Step 1: Encode ref audio via PyTorch
    print("Step 1: Encoding ref audio (PyTorch)...")
    from omnivoice import OmniVoice as PT
    pt = PT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)
    prompt = pt.create_voice_clone_prompt(ref_audio=args.ref_audio, ref_text="Reference audio.", preprocess_prompt=True)
    ref_tokens_np = prompt.ref_audio_tokens.cpu().numpy()
    ref_text = prompt.ref_text
    del pt; torch.mps.empty_cache() if hasattr(torch, 'mps') else None

    # Step 2: Load MLX teacher + vocoder
    print("Step 2: Loading MLX teacher...")
    config = OmniVoiceMLXConfig()
    teacher = OmniVoiceMLXModel(config)
    vocoder = AudioTokenizerDecoder(
        num_quantizers=config.num_quantizers, hidden_size=config.vq_hidden_size,
        codebook_dim=config.codebook_dim, codebook_size=config.codebook_size,
        semantic_hidden_size=config.semantic_hidden_size,
        dac_input_dim=config.dac_input_dim, dac_hidden_dim=config.dac_hidden_dim,
        dac_upsampling_ratios=config.dac_upsampling_ratios,
    )
    wp = Path(args.teacher_weights)
    teacher.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
    vocoder.load_weights(list(mx.load(str(wp / "vocoder.safetensors")).items()))
    mx.eval(teacher.parameters(), vocoder.parameters())

    # Step 3: Load v2 draft
    print("Step 3: Loading v2 draft...")
    draft_config = DraftV2Config(hidden_size=512, num_layers=6, num_heads=8)
    draft = DraftModelV2MLX(draft_config)
    draft.load_weights(str(args.draft_weights))
    mx.eval(draft.parameters())

    # Step 4: Build input
    print("Step 4: Building input...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    C = config.num_audio_codebook
    mask_id = config.audio_mask_id
    ref_audio_tokens = mx.array(ref_tokens_np, dtype=mx.int32)

    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids)))

    text_str = f"<|text_start|>{ref_text} {args.text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
    text_tokens = mx.broadcast_to(mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids)))

    chars_per_frame = max(1, len(ref_text) / ref_audio_tokens.shape[1])
    target_len = max(30, int(len(args.text) / chars_per_frame))

    target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)
    input_ids = mx.concatenate([style_tokens, text_tokens, ref_audio_tokens, target_masks], axis=1)
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    audio_mask = mx.concatenate([
        mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
        mx.ones(ref_audio_tokens.shape[1] + target_len, dtype=mx.bool_),
    ])

    print(f"  Target: {target_len} frames ({target_len/25:.1f}s), L_cond={L_cond}")

    # Step 5: Baseline
    print("\nStep 5: Baseline (8-step unmasking)...")
    mx.random.seed(42)
    t0 = time.perf_counter()
    base_tokens = generate_iterative(teacher, input_ids, audio_mask, target_len, num_step=8, guidance_scale=3.0)
    mx.eval(base_tokens)
    base_ms = (time.perf_counter() - t0) * 1000

    # Step 6: Draft generates cb0 using raw tokens as conditioning
    print("Step 6: Speculative (v2 draft)...")
    cond_ids = mx.expand_dims(input_ids[:, :L_cond], 0)  # (1, C, L_cond)
    cond_mask = mx.expand_dims(audio_mask[:L_cond], 0)    # (1, L_cond)

    t0 = time.perf_counter()
    draft_cb0 = draft.generate_ar(cond_ids, cond_mask, num_tokens=target_len)
    mx.eval(draft_cb0)
    draft_ms = (time.perf_counter() - t0) * 1000

    # Acceptance
    teacher_cb0 = base_tokens[0]
    matches = int(mx.sum(draft_cb0 == teacher_cb0).item())
    acceptance = matches / target_len

    # Decode audio
    import soundfile as sf
    out = Path("../test_output/v2_real")
    out.mkdir(parents=True, exist_ok=True)

    base_audio = np.array(vocoder(mx.expand_dims(base_tokens, 0))[0, :, 0])
    sf.write(str(out / "baseline.wav"), base_audio, 24000)

    # Speculative: draft cb0 + teacher cb1-7
    spec_tokens = np.array(base_tokens)
    spec_tokens[0] = np.array(draft_cb0)
    spec_audio = np.array(vocoder(mx.expand_dims(mx.array(spec_tokens), 0))[0, :, 0])
    sf.write(str(out / "speculative_v2.wav"), spec_audio, 24000)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"  Baseline:    {base_ms:.0f}ms")
    print(f"  Draft:       {draft_ms:.0f}ms")
    print(f"  CB0 accept:  {acceptance:.1%} ({matches}/{target_len})")
    print(f"  Audio: {out}/")


if __name__ == "__main__":
    main()
