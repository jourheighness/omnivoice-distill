"""Cache real teacher outputs using OmniVoice PyTorch on GPU.

Uses OmniVoice's public generate API, then separately extracts
conditioning hidden states for draft model training.

Usage:
    python src/cache_teacher_real.py --data_manifest data/libritts/manifest.json --output_dir ./cache --num_samples 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def load_teacher(device="cuda:0"):
    """Load OmniVoice model."""
    from omnivoice import OmniVoice
    print("Loading OmniVoice teacher model...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map=device, dtype=torch.float16,
    )
    model.eval()

    # Inspect available methods for debugging
    public_methods = [m for m in dir(model) if not m.startswith('__') and callable(getattr(model, m, None))]
    print(f"  Model loaded on {device}")
    print(f"  Available methods: {[m for m in public_methods if 'generat' in m.lower() or 'prepare' in m.lower() or 'encode' in m.lower() or 'clone' in m.lower()]}")
    return model


def load_audio_mono(audio_path, target_sr=24000):
    """Load audio file as mono waveform at target sample rate."""
    wav, sr = torchaudio.load(audio_path)
    # Convert to mono first
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # (1, T)


def generate_and_cache_sample(model, text, ref_audio_path, target_sr=24000):
    """Generate audio tokens and extract conditioning hidden states.

    Strategy:
    1. Load + prep reference audio (mono, 24kHz)
    2. Use model's generate to get output audio tokens
    3. Separately extract conditioning hidden states from the LLM
    """
    device = next(model.parameters()).device
    config = model.config
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    # Load reference audio as mono
    ref_wav = load_audio_mono(ref_audio_path, target_sr).to(device)
    # Ensure shape is (1, T) — batch dim for encoder
    if ref_wav.dim() == 1:
        ref_wav = ref_wav.unsqueeze(0)

    # Limit length to ~10 seconds to avoid OOM
    max_samples = target_sr * 10
    if ref_wav.shape[-1] > max_samples:
        ref_wav = ref_wav[:, :max_samples]

    # Use create_voice_clone_prompt which handles encoding properly
    # or encode manually with the right API
    try:
        prompt = model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=text[:100])
        ref_codes = prompt.ref_audio_tokens if hasattr(prompt, 'ref_audio_tokens') else prompt.audio_tokens
        if isinstance(ref_codes, torch.Tensor) and ref_codes.dim() == 3:
            ref_codes = ref_codes.squeeze(0)
    except Exception as e_prompt:
        # Direct encode fallback
        with torch.no_grad():
            # Some tokenizers expect (batch, channels, time) = (1, 1, T)
            wav_input = ref_wav.unsqueeze(0) if ref_wav.dim() == 2 else ref_wav
            ref_codes = model.audio_tokenizer.encode(wav_input)
        if hasattr(ref_codes, 'audio_codes'):
            ref_codes = ref_codes.audio_codes
        while ref_codes.dim() > 2:
            ref_codes = ref_codes.squeeze(0)  # -> (C, T_ref)

    # Estimate target length from text (rough: ~3 chars per frame at 25Hz)
    target_len = max(25, min(200, len(text) // 3))

    # Build the input sequence manually:
    # [text_tokens | ref_audio_tokens | target_mask_tokens]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    text_str = f"<|text_start|>{text}<|text_end|>"
    text_ids = tokenizer(text_str, return_tensors="pt").input_ids[0].to(device)
    text_len = text_ids.shape[0]

    # Broadcast text tokens across codebooks
    text_tokens = text_ids.unsqueeze(0).expand(C, -1)  # (C, text_len)

    # Target: all mask tokens
    target_tokens = torch.full((C, target_len), mask_id, dtype=torch.long, device=device)

    # Full input: text | ref_audio | target
    input_ids = torch.cat([text_tokens, ref_codes, target_tokens], dim=1)  # (C, L_total)
    input_ids = input_ids.unsqueeze(0)  # (1, C, L_total)

    L_total = input_ids.shape[-1]
    L_cond = L_total - target_len

    # Audio mask: False for text, True for audio (ref + target)
    audio_mask = torch.cat([
        torch.zeros(text_len, dtype=torch.bool, device=device),
        torch.ones(ref_codes.shape[1] + target_len, dtype=torch.bool, device=device),
    ]).unsqueeze(0)  # (1, L_total)

    # Extract conditioning hidden states
    with torch.no_grad():
        inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
        llm_out = model.llm(inputs_embeds=inputs_embeds, return_dict=True)
        hidden_states = llm_out.last_hidden_state  # (1, L_total, H)

    cond_hidden = hidden_states[:, :L_cond, :].cpu().float().numpy()[0]  # (L_cond, H)

    # Generate tokens via iterative unmasking
    # Try to find the right internal method
    tokens = None
    with torch.no_grad():
        # Try various method signatures
        for method_name in ['_generate_iterative', 'generate_iterative', '_iterative_generate']:
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                tokens = method(input_ids=input_ids, audio_mask=audio_mask,
                                target_len=target_len, num_step=8)
                break
            except TypeError:
                try:
                    tokens = method(input_ids, audio_mask, target_len, num_step=8)
                    break
                except Exception:
                    continue

        # If internal methods failed, use the full generate and intercept
        if tokens is None:
            # Use the public generate method with ref audio
            # This generates waveform, but we need tokens
            # Fall back to running the unmasking loop ourselves
            tokens = _run_unmasking_loop(model, input_ids, audio_mask, target_len, num_step=8)

    if isinstance(tokens, torch.Tensor):
        tokens_np = tokens.cpu().numpy()
    else:
        tokens_np = np.array(tokens)

    # Ensure shape is (C, target_len)
    if tokens_np.ndim == 3:
        tokens_np = tokens_np[0]  # remove batch dim

    return {
        "cond_hidden": cond_hidden,         # (L_cond, H)
        "cb0_tokens": tokens_np[0],          # (target_len,)
        "all_tokens": tokens_np,             # (C, target_len)
        "target_len": target_len,
    }


def _run_unmasking_loop(model, input_ids, audio_mask, target_len, num_step=8):
    """Run iterative unmasking manually if internal methods aren't accessible."""
    import math

    device = input_ids.device
    config = model.config
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id
    L_total = input_ids.shape[-1]
    L_cond = L_total - target_len

    # Initialize target region with masks
    tokens = torch.full((C, target_len), mask_id, dtype=torch.long, device=device)

    # Timestep schedule
    t = torch.linspace(0, 1, num_step + 2)[1:-1]  # skip 0 and 1
    total_mask = target_len * C

    for step in range(num_step):
        # Build full sequence
        full_ids = torch.cat([input_ids[0, :, :L_cond], tokens], dim=1).unsqueeze(0)

        with torch.no_grad():
            inputs_embeds = model._prepare_embed_inputs(full_ids, audio_mask)
            llm_out = model.llm(inputs_embeds=inputs_embeds, return_dict=True)
            hidden = llm_out.last_hidden_state  # (1, L_total, H)

            # Get logits from audio heads
            logits = model.audio_heads(hidden)  # (1, L_total, C*V)
            V = config.audio_vocab_size
            logits = logits[:, -target_len:, :]  # target region
            logits = logits.reshape(1, target_len, C, V).permute(0, 2, 1, 3)  # (1, C, T, V)

        # Mask out the mask token
        logits[:, :, :, mask_id] = float('-inf')

        # Predict tokens
        pred_tokens = logits.argmax(dim=-1)[0]  # (C, T)

        # Confidence scores
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        scores = log_probs.max(dim=-1).values[0]  # (C, T)

        # Layer penalty
        layer_ids = torch.arange(C, device=device).unsqueeze(1)
        scores = scores - layer_ids * 5.0

        # Only score masked positions
        is_masked = tokens == mask_id
        scores = torch.where(is_masked, scores, torch.tensor(float('-inf'), device=device))

        # How many to unmask this step
        k = math.ceil(total_mask * (step + 1) / num_step) - math.ceil(total_mask * step / num_step)
        k = min(k, is_masked.sum().item())
        if k <= 0:
            continue

        # Top-k selection
        flat_scores = scores.reshape(-1)
        flat_pred = pred_tokens.reshape(-1)
        flat_tokens = tokens.reshape(-1)

        topk_indices = flat_scores.topk(k).indices
        flat_tokens[topk_indices] = flat_pred[topk_indices]
        tokens = flat_tokens.reshape(C, target_len)

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Cache real teacher outputs")
    parser.add_argument("--data_manifest", type=str, default="data/libritts/manifest.json")
    parser.add_argument("--output_dir", type=str, default="./cache")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_step", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_manifest) as f:
        data_manifest = json.load(f)

    num_samples = min(args.num_samples, len(data_manifest))
    print(f"Processing {num_samples} samples from {args.data_manifest}")

    model = load_teacher(args.device)

    # Pre-load tokenizer once
    from transformers import AutoTokenizer
    _ = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    manifest = []
    errors = 0

    for i in tqdm(range(num_samples)):
        entry = data_manifest[i]

        try:
            if i == 0:
                # Debug first sample
                _wav = load_audio_mono(entry["audio_path"])
                print(f"  DEBUG sample 0: wav shape={_wav.shape}, dtype={_wav.dtype}, text='{entry['text'][:60]}'")
                del _wav

            cached = generate_and_cache_sample(
                model,
                text=entry["text"],
                ref_audio_path=entry["audio_path"],
            )

            fname = f"sample_{i:04d}.npz"
            np.savez_compressed(
                output_dir / fname,
                cond_hidden=cached["cond_hidden"],
                cb0_tokens=cached["cb0_tokens"],
                all_tokens=cached["all_tokens"],
            )
            manifest.append({
                "file": fname,
                "target_len": cached["target_len"],
                "cond_len": cached["cond_hidden"].shape[0],
                "text": entry["text"][:100],
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  Error on sample {i}: {e}")
            if errors > 50:
                print("Too many errors, stopping.")
                break
            continue

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(manifest)} samples to {output_dir}/ ({errors} errors)")


if __name__ == "__main__":
    main()
