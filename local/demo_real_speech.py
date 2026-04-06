"""Generate real speech: baseline vs speculative, with an actual voice.

Uses:
- PyTorch OmniVoice to encode reference audio (one-time)
- MLX OmniVoice teacher for baseline generation
- MLX draft model for speculative cb0 prediction
- MLX vocoder for audio output
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))

from draft_mlx import DraftMLXConfig, DraftModelMLX

# Voice and text to synthesize
REF_AUDIO = "/Users/johannescarlsten/bartholomew/source/voice-service/voices/barth-v01/ref_audio.wav"
REF_TEXT = "This is a reference audio sample for voice cloning."
SYNTH_TEXT = "The speculative decoding approach generates audio tokens much faster than the baseline method."


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--draft_weights", required=True)
    parser.add_argument("--ref_audio", default=REF_AUDIO)
    parser.add_argument("--ref_text", default=REF_TEXT)
    parser.add_argument("--text", default=SYNTH_TEXT)
    parser.add_argument("--output_dir", default="../test_output/real_speech")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Encode reference audio via PyTorch model
    print("Step 1: Encoding reference audio (PyTorch)...")
    from omnivoice import OmniVoice as OmniVoicePT
    pt_model = OmniVoicePT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)

    voice_prompt = pt_model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        preprocess_prompt=True,
    )

    ref_audio_tokens = mx.array(voice_prompt.ref_audio_tokens.cpu().numpy(), dtype=mx.int32)
    ref_text = voice_prompt.ref_text
    ref_rms = voice_prompt.ref_rms
    print(f"  Ref audio tokens: {ref_audio_tokens.shape}")
    print(f"  Ref text: '{ref_text}'")

    # Free PyTorch model
    del pt_model
    torch.mps.empty_cache() if hasattr(torch, 'mps') else None

    # Step 2: Load MLX teacher + vocoder
    print("\nStep 2: Loading MLX teacher + vocoder...")
    from omnivoice_mlx.generate import (
        OmniVoiceMLXConfig, OmniVoiceMLXModel, generate_iterative,
    )
    from omnivoice_mlx.vocoder import AudioTokenizerDecoder

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
    print("  Teacher loaded.")

    # Step 3: Load draft model
    print("\nStep 3: Loading draft model...")
    draft_config = DraftMLXConfig(hidden_size=512, num_layers=6, num_heads=8, max_seq_len=2048)
    draft = DraftModelMLX(draft_config)
    draft.load_weights(str(args.draft_weights))
    mx.eval(draft.parameters())
    print("  Draft loaded.")

    # Step 4: Build input sequence
    print("\nStep 4: Building input sequence...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    # Style tokens
    style_text = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_text, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(
        mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids))
    )

    # Text tokens
    text_str = f"<|text_start|>{ref_text} {args.text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
    text_tokens = mx.broadcast_to(
        mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids))
    )

    # Estimate target length
    chars_per_frame = len(ref_text) / ref_audio_tokens.shape[1]
    target_len = max(30, int(len(args.text) / chars_per_frame))
    print(f"  Target length: {target_len} frames ({target_len/25:.1f}s)")

    # Target masks
    target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)

    # Full input: style | text | ref_audio | target
    input_ids = mx.concatenate([style_tokens, text_tokens, ref_audio_tokens, target_masks], axis=1)
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    audio_mask = mx.concatenate([
        mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
        mx.ones(ref_audio_tokens.shape[1] + target_len, dtype=mx.bool_),
    ])

    print(f"  Input shape: {input_ids.shape}, L_cond={L_cond}, target={target_len}")

    # Step 5: Baseline generation (full unmasking, 8 steps)
    print("\nStep 5: Baseline generation (8 steps)...")
    mx.random.seed(42)
    t0 = time.perf_counter()
    base_tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=8, guidance_scale=3.0,
    )
    mx.eval(base_tokens)
    base_time = time.perf_counter() - t0

    base_audio = np.array(vocoder(mx.expand_dims(base_tokens, 0))[0, :, 0])
    mx.eval(base_audio)

    import soundfile as sf
    sf.write(str(output_dir / "baseline.wav"), np.array(base_audio), 24000)
    print(f"  Baseline: {base_time*1000:.0f}ms | {len(base_audio)/24000:.1f}s audio")

    # Step 6: Extract conditioning for draft
    print("\nStep 6: Speculative generation...")
    ids_batch = mx.expand_dims(input_ids, 0)
    mask_batch = mx.expand_dims(audio_mask, 0)
    attn = mx.ones((1, 1, L_total, L_total), dtype=mx.bool_)

    embeds = teacher._prepare_embed_inputs(ids_batch, mask_batch)
    cond_hidden = teacher.llm(embeds, mask=attn)[:, :L_cond, :]
    mx.eval(cond_hidden)

    # Draft generates cb0
    t0 = time.perf_counter()
    first_tok = base_tokens[0, 0:1]  # seed with first token from baseline
    draft_cb0 = mx.concatenate([
        first_tok,
        draft.generate_ar(cond_hidden, num_tokens=target_len - 1, temperature=0.0, start_token=first_tok),
    ])
    mx.eval(draft_cb0)
    draft_time = time.perf_counter() - t0

    # Teacher fills cb1-7 (4 steps instead of 8)
    t1 = time.perf_counter()
    mx.random.seed(42)
    teacher_tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=4, guidance_scale=3.0,
    )
    mx.eval(teacher_tokens)
    verify_time = time.perf_counter() - t1

    # Build speculative output: draft cb0 + teacher cb1-7
    spec_tokens_np = np.array(teacher_tokens)
    spec_tokens_np[0] = np.array(draft_cb0)
    spec_tokens = mx.array(spec_tokens_np)

    spec_audio = np.array(vocoder(mx.expand_dims(spec_tokens, 0))[0, :, 0])

    sf.write(str(output_dir / "speculative.wav"), np.array(spec_audio), 24000)

    # Acceptance
    matches = int(mx.sum(draft_cb0 == base_tokens[0]).item())
    accept = matches / target_len

    spec_total = draft_time + verify_time
    print(f"  Draft: {draft_time*1000:.0f}ms | Verify: {verify_time*1000:.0f}ms | Total: {spec_total*1000:.0f}ms")
    print(f"  CB0 acceptance: {accept:.1%} ({matches}/{target_len})")
    print(f"  Speedup: {base_time/spec_total:.2f}x")

    print(f"\n{'='*50}")
    print(f"Audio saved to {output_dir}/")
    print(f"  baseline.wav    — full 8-step unmasking ({base_time*1000:.0f}ms)")
    print(f"  speculative.wav — draft + 4-step verify ({spec_total*1000:.0f}ms)")
    print(f"  Acceptance: {accept:.1%}")


if __name__ == "__main__":
    main()
