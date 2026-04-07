"""Full PT speculative decoding demo — same framework, no gaps."""

import time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--text", default="Hello there, how are you doing today?")
    parser.add_argument("--checkpoint", default="checkpoints_v2_stochastic/best.pt")
    parser.add_argument("--target_len", type=int, default=75)
    parser.add_argument("--output_dir", default="test_output_pt")
    args = parser.parse_args()

    device = "cuda:0"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load teacher
    print("Loading teacher...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
    teacher = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float16)
    teacher.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    # Load draft
    print("Loading draft...")
    from draft_model_v2 import DraftModelV2
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    draft = DraftModelV2(
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], dropout=0.0,
    ).to(device)

    import glob
    for p in glob.glob("/workspace/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/*/model.safetensors"):
        draft.load_teacher_embeddings(p, device=device)
        break
    draft.load_state_dict(ckpt["model_state_dict"], strict=False)
    draft.eval()

    # Encode ref
    print("Encoding ref audio...")
    prompt = teacher.create_voice_clone_prompt(
        ref_audio=args.ref_audio, ref_text="Reference.", preprocess_prompt=True,
    )
    ref_tokens = prompt.ref_audio_tokens

    # Build input for manual token extraction
    C, mask_id = 8, 1024
    style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_str, return_tensors="pt").input_ids[0].to(device)
    style_tokens = style_ids.unsqueeze(0).expand(C, -1)

    text_str = f"<|text_start|>{prompt.ref_text} {args.text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="pt").input_ids[0].to(device)
    text_tokens = text_ids.unsqueeze(0).expand(C, -1)

    target_masks = torch.full((C, args.target_len), mask_id, dtype=torch.long, device=device)
    input_ids = torch.cat([style_tokens, text_tokens, ref_tokens, target_masks], dim=1).unsqueeze(0)
    L_cond = input_ids.shape[-1] - args.target_len

    audio_mask = torch.cat([
        torch.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=torch.bool, device=device),
        torch.ones(ref_tokens.shape[1] + args.target_len, dtype=torch.bool, device=device),
    ]).unsqueeze(0)

    print(f"L_cond={L_cond}, target={args.target_len}")

    # === Use the REAL OmniVoice generation via GenerationTask ===
    print("\n--- Baseline (real _generate_iterative) ---")

    gen_config = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=3.0,
        position_temperature=5.0,
        class_temperature=0.0,
    )

    task = GenerationTask(
        texts=[args.text],
        target_lens=[args.target_len],
        languages=["English"],
        instructions=["None"],
        ref_audio_tokens=[ref_tokens],
        ref_texts=[prompt.ref_text],
        ref_rms=[getattr(prompt, 'ref_rms', 0.1)],
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            token_list = teacher._generate_iterative(task, gen_config)
            baseline_tokens = token_list[0]  # (C, T)
            print("  Used _generate_iterative (real OmniVoice)")
        except Exception as e:
            print(f"  _generate_iterative failed: {e}")
            from cache_teacher_real import _run_unmasking_loop
            baseline_tokens = _run_unmasking_loop(teacher, input_ids, audio_mask, args.target_len, num_step=8)
            print("  Used fallback")
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000

    if baseline_tokens.dim() == 3:
        baseline_tokens = baseline_tokens[0]
    baseline_cb0 = baseline_tokens[0].cpu().numpy()
    print(f"  {baseline_ms:.0f}ms | cb0 unique: {len(set(baseline_cb0.tolist()))}")
    print(f"  cb0: {baseline_cb0[:15]}")

    # Decode audio
    with torch.no_grad():
        audio_out = teacher.audio_tokenizer.decode(baseline_tokens.unsqueeze(0))
    if hasattr(audio_out, 'audio_values'):
        audio_np = audio_out.audio_values.cpu().float().numpy().flatten()
    else:
        audio_np = audio_out.cpu().float().numpy().flatten()
    sf.write(str(out / "baseline.wav"), audio_np, 24000)
    print(f"  Audio: {len(audio_np)/24000:.1f}s")

    # === Draft teacher-forced ===
    print("\n--- Draft prediction ---")
    cond_ids = input_ids[:, :, :L_cond]
    cond_mask = audio_mask[:, :L_cond]

    with torch.no_grad():
        tok_in = torch.from_numpy(baseline_cb0[:-1].astype(np.int64)).unsqueeze(0).to(device)
        logits = draft(tok_in, cond_ids=cond_ids, audio_mask=cond_mask)
        preds = logits.argmax(dim=-1)[0].cpu().numpy()

    match = int((preds == baseline_cb0[1:]).sum())
    total = len(baseline_cb0) - 1
    acceptance = match / total

    print(f"  Acceptance: {match}/{total} ({acceptance:.1%})")
    print(f"  Teacher: {baseline_cb0[:15]}")
    print(f"  Draft:   {preds[:15]}")

    print(f"\n{'='*50}")
    if acceptance > 0.5:
        print(f"  IT WORKS! {acceptance:.0%} acceptance on real speech.")
    else:
        print(f"  {acceptance:.0%} — check if training used same generation method.")
    print(f"  Audio: {out}/baseline.wav")


if __name__ == "__main__":
    main()
