"""Full PT speculative decoding demo — same framework, no gaps.

Runs entirely on PyTorch/CUDA:
1. Encode ref audio
2. Teacher generates baseline (full unmasking)
3. Draft predicts cb0 (teacher-forced verification)
4. Measure acceptance rate + timing
5. Decode to audio

Usage:
    python src/demo_pt.py --ref_audio barth_ref.wav --text "Hello there"
"""

import argparse
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from safetensors.torch import load_file

from draft_model_v2 import DraftModelV2
from cache_teacher_real import _run_unmasking_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--text", default="Hello there, how are you doing today? The weather is beautiful outside.")
    parser.add_argument("--checkpoint", default="checkpoints_v2_stochastic/best.pt")
    parser.add_argument("--target_len", type=int, default=75)
    parser.add_argument("--output_dir", default="test_output_pt")
    args = parser.parse_args()

    device = "cuda:0"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load OmniVoice teacher
    print("Loading teacher...")
    from omnivoice import OmniVoice
    teacher = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float16)
    teacher.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    # Load draft
    print("Loading draft...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    draft = DraftModelV2(
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], dropout=0.0,
    ).to(device)

    # Load teacher embeddings into draft
    teacher_weights_path = None
    import glob
    for p in glob.glob("/workspace/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/*/model.safetensors"):
        teacher_weights_path = p
        break
    if teacher_weights_path:
        draft.load_teacher_embeddings(teacher_weights_path, device=device)

    # Load trained weights
    draft.load_state_dict(ckpt["model_state_dict"], strict=False)
    draft.eval()
    print(f"  Draft: {draft.count_params():,} trainable params")

    # Encode reference audio
    print("Encoding ref audio...")
    prompt = teacher.create_voice_clone_prompt(
        ref_audio=args.ref_audio, ref_text="Reference audio.",
        preprocess_prompt=True,
    )
    ref_tokens = prompt.ref_audio_tokens  # (C, T) on device

    # Build input
    C = 8
    mask_id = 1024
    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="pt").input_ids[0].to(device)
    style_tokens = style_ids.unsqueeze(0).expand(C, -1)

    text_str = f"<|text_start|>{prompt.ref_text} {args.text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="pt").input_ids[0].to(device)
    text_tokens = text_ids.unsqueeze(0).expand(C, -1)

    target_masks = torch.full((C, args.target_len), mask_id, dtype=torch.long, device=device)
    input_ids = torch.cat([style_tokens, text_tokens, ref_tokens, target_masks], dim=1).unsqueeze(0)
    L_total = input_ids.shape[-1]
    L_cond = L_total - args.target_len

    audio_mask = torch.cat([
        torch.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=torch.bool, device=device),
        torch.ones(ref_tokens.shape[1] + args.target_len, dtype=torch.bool, device=device),
    ]).unsqueeze(0)

    print(f"Input: L_cond={L_cond}, target={args.target_len}")

    # === BASELINE: full teacher generation ===
    print("\n--- Baseline (teacher, 8 steps) ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        baseline_tokens = _run_unmasking_loop(teacher, input_ids, audio_mask, args.target_len, num_step=8)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000
    baseline_cb0 = baseline_tokens[0].cpu().numpy()  # (target_len,)
    print(f"  {baseline_ms:.0f}ms | cb0 unique: {len(set(baseline_cb0.tolist()))}")

    # Decode baseline audio
    baseline_all = baseline_tokens.unsqueeze(0)  # (1, C, T)
    with torch.no_grad():
        baseline_audio_tensor = teacher.audio_tokenizer.decode(baseline_all)
    if hasattr(baseline_audio_tensor, 'audio_values'):
        baseline_audio = baseline_audio_tensor.audio_values.cpu().float().numpy().flatten()
    else:
        baseline_audio = baseline_audio_tensor.cpu().float().numpy().flatten()
    sf.write(str(out / "baseline.wav"), baseline_audio, 24000)
    print(f"  Audio: {len(baseline_audio)/24000:.1f}s saved")

    # === SPECULATIVE: draft predicts, teacher verifies ===
    print("\n--- Speculative decode ---")

    # Draft: teacher-forced prediction (simulates spec decode)
    cond_ids = input_ids[:, :, :L_cond]  # (1, C, L_cond)
    cond_mask = audio_mask[:, :L_cond]   # (1, L_cond)

    t0 = time.perf_counter()
    with torch.no_grad():
        cb0_input = torch.from_numpy(baseline_cb0[:-1].astype(np.int64)).unsqueeze(0).to(device)
        logits = draft(cb0_input, cond_ids=cond_ids, audio_mask=cond_mask)
        preds = logits.argmax(dim=-1)[0].cpu().numpy()
    torch.cuda.synchronize()
    draft_ms = (time.perf_counter() - t0) * 1000

    # Acceptance
    match = int((preds == baseline_cb0[1:]).sum())
    total = len(baseline_cb0) - 1
    acceptance = match / total

    print(f"  Draft: {draft_ms:.0f}ms")
    print(f"  Acceptance: {match}/{total} ({acceptance:.1%})")
    print(f"  Teacher cb0: {baseline_cb0[:15]}")
    print(f"  Draft preds:  {preds[:15]}")

    # === RESULTS ===
    print(f"\n{'='*50}")
    print(f"SPECULATIVE DECODE (PyTorch, same framework)")
    print(f"{'='*50}")
    print(f"  Baseline:   {baseline_ms:.0f}ms")
    print(f"  Draft:      {draft_ms:.0f}ms")
    print(f"  Acceptance: {acceptance:.1%}")
    print(f"  Audio:      {out}/baseline.wav")

    if acceptance > 0.7:
        print(f"\n  WORKS! {acceptance:.0%} of tokens accepted.")
        print(f"  In production: ~{acceptance:.0%} fewer teacher forward passes.")
    elif acceptance > 0.3:
        print(f"\n  PARTIAL — {acceptance:.0%} acceptance gives some speedup.")
    else:
        print(f"\n  LOW — needs more training data or different approach.")


if __name__ == "__main__":
    main()
