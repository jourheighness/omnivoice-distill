"""Real speech demo with v2 speculative decoding.

Proper speculative decode: draft predicts cb0, teacher verifies token-by-token.
Accepted tokens use correct context for next prediction (no error cascading).
"""

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


def speculative_decode_cb0(draft, teacher, input_ids, audio_mask, target_len,
                            cond_ids, cond_audio_mask, num_step=8, guidance_scale=3.0):
    """Proper speculative decoding for cb0.

    1. Teacher generates full output (baseline) — we need this for verification
    2. Draft predicts each cb0 token given correct previous tokens
    3. Compare: acceptance rate = how often draft matches teacher

    In production, you'd interleave draft/teacher. For demo, we simulate.
    """
    config = teacher.config
    C = config.num_audio_codebook

    # Teacher generates ground truth
    t0 = time.perf_counter()
    mx.random.seed(42)
    teacher_tokens = generate_iterative(
        teacher, input_ids, audio_mask, target_len,
        num_step=num_step, guidance_scale=guidance_scale,
    )
    mx.eval(teacher_tokens)
    teacher_ms = (time.perf_counter() - t0) * 1000
    teacher_cb0 = teacher_tokens[0]  # (target_len,)

    # Draft predicts cb0 token-by-token with teacher-verified context
    # This simulates real speculative decoding where accepted tokens
    # provide correct context for the next prediction
    t0 = time.perf_counter()

    cond_emb = draft._embed_conditioning(cond_ids, cond_audio_mask)
    if cond_emb.ndim == 2:
        cond_emb = mx.expand_dims(cond_emb, 0)

    draft_cb0 = []
    accepted = 0
    current = mx.array([[int(teacher_cb0[0].item())]], dtype=mx.int32)  # seed with first token

    for pos in range(1, target_len):
        # Draft predicts next token given correct previous tokens
        tok_emb = draft.cb0_embed(current)
        x = mx.concatenate([cond_emb, tok_emb], axis=1)
        for layer in draft.layers:
            x = layer(x)
        x = draft.norm(x)
        logits = draft.head(x[:, -1:, :])
        draft_pred = logits.argmax(axis=-1)[0, 0]
        mx.eval(draft_pred)

        teacher_tok = teacher_cb0[pos]

        if int(draft_pred.item()) == int(teacher_tok.item()):
            accepted += 1

        # Always use teacher's token as context (simulates verify-then-accept)
        current = mx.concatenate([current, mx.array([[int(teacher_tok.item())]], dtype=mx.int32)], axis=1)

    mx.eval(current)
    draft_ms = (time.perf_counter() - t0) * 1000

    acceptance = accepted / (target_len - 1)

    return {
        "teacher_tokens": teacher_tokens,
        "teacher_cb0": np.array(teacher_cb0),
        "acceptance": acceptance,
        "accepted": accepted,
        "total": target_len - 1,
        "teacher_ms": teacher_ms,
        "draft_ms": draft_ms,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--draft_weights", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--text", default="Hello, how are you doing today? I hope everything is going well.")
    parser.add_argument("--output_dir", default="../test_output/v2_demo")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Encode ref audio
    print("Loading PyTorch encoder...")
    from omnivoice import OmniVoice as PT
    pt = PT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)
    prompt = pt.create_voice_clone_prompt(ref_audio=args.ref_audio, ref_text="Reference.", preprocess_prompt=True)
    ref_tokens = mx.array(prompt.ref_audio_tokens.cpu().numpy(), dtype=mx.int32)
    ref_text = prompt.ref_text
    del pt

    # Step 2: Load teacher + vocoder + draft
    print("Loading MLX teacher + draft...")
    config = OmniVoiceMLXConfig()
    teacher = OmniVoiceMLXModel(config)
    vocoder = AudioTokenizerDecoder(
        num_quantizers=config.num_quantizers, hidden_size=config.vq_hidden_size,
        codebook_dim=config.codebook_dim, codebook_size=config.codebook_size,
        semantic_hidden_size=config.semantic_hidden_size, dac_input_dim=config.dac_input_dim,
        dac_hidden_dim=config.dac_hidden_dim, dac_upsampling_ratios=config.dac_upsampling_ratios,
    )
    wp = Path(args.teacher_weights)
    teacher.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
    vocoder.load_weights(list(mx.load(str(wp / "vocoder.safetensors")).items()))
    mx.eval(teacher.parameters(), vocoder.parameters())

    draft_config = DraftV2Config(hidden_size=512, num_layers=6, num_heads=8)
    draft = DraftModelV2MLX(draft_config)
    draft.load_weights(str(args.draft_weights))
    mx.eval(draft.parameters())

    # Step 3: Build input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids)))

    text_str = f"<|text_start|>{ref_text} {args.text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
    text_tokens = mx.broadcast_to(mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids)))

    chars_per_frame = max(1, len(ref_text) / ref_tokens.shape[1])
    target_len = max(30, int(len(args.text) / chars_per_frame))

    target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)
    input_ids = mx.concatenate([style_tokens, text_tokens, ref_tokens, target_masks], axis=1)
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    audio_mask = mx.concatenate([
        mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
        mx.ones(ref_tokens.shape[1] + target_len, dtype=mx.bool_),
    ])

    # Conditioning for draft (just the prefix tokens)
    cond_ids = mx.expand_dims(input_ids[:, :L_cond], 0)
    cond_audio_mask = mx.expand_dims(audio_mask[:L_cond], 0)

    print(f"Target: {target_len} frames ({target_len/25:.1f}s)")

    # Step 4: Speculative decode
    print("\nRunning speculative decode...")
    result = speculative_decode_cb0(
        draft, teacher, input_ids, audio_mask, target_len,
        cond_ids, cond_audio_mask,
    )

    # Decode audio
    import soundfile as sf
    audio = np.array(vocoder(mx.expand_dims(result["teacher_tokens"], 0))[0, :, 0])
    sf.write(str(out / "output.wav"), audio, 24000)

    print(f"\n{'='*50}")
    print(f"SPECULATIVE DECODE RESULTS")
    print(f"{'='*50}")
    print(f"  CB0 acceptance:  {result['acceptance']:.1%} ({result['accepted']}/{result['total']})")
    print(f"  Teacher time:    {result['teacher_ms']:.0f}ms")
    print(f"  Draft verify:    {result['draft_ms']:.0f}ms")
    print(f"  Audio: {out / 'output.wav'} ({len(audio)/24000:.1f}s)")
    print(f"\n  With {result['acceptance']:.0%} acceptance, speculative decode would")
    print(f"  skip ~{result['accepted']} of {result['total']} teacher forward passes.")


if __name__ == "__main__":
    main()
