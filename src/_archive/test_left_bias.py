"""Quick verification: does left-biased position selection in OmniVoice
produce acceptable audio without retraining?

Modifies the position scoring in _generate_iterative to favor earlier
positions, then generates audio at several bias levels and evaluates
with Whisper transcription accuracy.
"""

import math
import time
import json
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path


def _get_time_steps(t_start, t_end, num_step, t_shift):
    """Reproduce OmniVoice's shifted timestep schedule."""
    t = torch.linspace(t_start, t_end, num_step)
    # Apply shift: r = tau*t / (1 + (tau-1)*t)
    t = t_shift * t / (1 + (t_shift - 1) * t)
    return t


def _gumbel_sample(logits, temperature):
    """Gumbel-softmax sampling for position selection."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    return logits / temperature + gumbel_noise


def generate_with_left_bias(
    model, task, gen_config, position_bias=0.0, seed=42,
):
    """Run _generate_iterative with added left-to-right position bias.

    position_bias: factor multiplied by normalized position index (0-1)
                   and subtracted from confidence scores.
                   0 = standard OmniVoice, higher = more left-biased.

    Returns: (tokens_list, step_info) where step_info tracks which
             positions were unmasked at each step.
    """
    torch.manual_seed(seed)

    B = task.batch_size
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    # Prepare inputs (same as _generate_iterative)
    inputs_list = [
        model._prepare_inference_inputs(
            task.texts[i],
            task.target_lens[i],
            task.ref_texts[i],
            task.ref_audio_tokens[i],
            task.langs[i],
            task.instructs[i],
            gen_config.denoise,
        )
        for i in range(B)
    ]

    c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
    max_c_len = max(c_lens)
    pad_id = model.config.audio_mask_id

    batch_input_ids = torch.full(
        (2 * B, model.config.num_audio_codebook, max_c_len),
        pad_id, dtype=torch.long, device=model.device,
    )
    batch_audio_mask = torch.zeros(
        (2 * B, max_c_len), dtype=torch.bool, device=model.device,
    )
    batch_attention_mask = torch.zeros(
        (2 * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=model.device,
    )

    for i, inp in enumerate(inputs_list):
        c_len = c_lens[i]
        u_len = task.target_lens[i]

        batch_input_ids[i, :, :c_len] = inp["input_ids"]
        batch_audio_mask[i, :c_len] = inp["audio_mask"]
        batch_attention_mask[i, :, :c_len, :c_len] = True

        batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
        batch_audio_mask[B + i, :u_len] = inp["audio_mask"][..., -u_len:]
        batch_attention_mask[B + i, :, :u_len, :u_len] = True
        if max_c_len > u_len:
            pad_diag = torch.arange(u_len, max_c_len, device=model.device)
            batch_attention_mask[B + i, :, pad_diag, pad_diag] = True

    tokens = torch.full(
        (B, model.config.num_audio_codebook, max(task.target_lens)),
        model.config.audio_mask_id, dtype=torch.long, device=model.device,
    )

    timesteps = _get_time_steps(0.0, 1.0, gen_config.num_step + 1, gen_config.t_shift).tolist()
    schedules = []
    for t_len in task.target_lens:
        total_mask = t_len * model.config.num_audio_codebook
        rem = total_mask
        sched = []
        for step in range(gen_config.num_step):
            num = (
                rem if step == gen_config.num_step - 1
                else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
            )
            sched.append(int(num))
            rem -= int(num)
        schedules.append(sched)

    layer_ids = torch.arange(model.config.num_audio_codebook, device=model.device).view(1, -1, 1)

    step_info = []  # track unmasking order

    for step in range(gen_config.num_step):
        batch_logits = model(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)

        for i in range(B):
            k = schedules[i][step]
            if k <= 0:
                continue

            c_len = c_lens[i]
            t_len = task.target_lens[i]

            c_logits = batch_logits[i:i+1, :, c_len - t_len:c_len, :]
            u_logits = batch_logits[B+i:B+i+1, :, :t_len, :]

            pred_tokens, scores = model._predict_tokens_with_scoring(
                c_logits, u_logits, gen_config,
            )

            # Standard layer penalty
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)

            # === LEFT BIAS: subtract position_bias * normalized_position ===
            if position_bias > 0:
                pos_indices = torch.arange(t_len, device=model.device, dtype=torch.float32)
                pos_indices = pos_indices / max(t_len - 1, 1)  # normalize to [0, 1]
                # Shape: (1, 1, t_len) — broadcast across batch and codebooks
                pos_penalty = pos_indices.view(1, 1, -1) * position_bias
                scores = scores - pos_penalty

            # Gumbel noise for position selection
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)

            sample_tokens = tokens[i:i+1, :, :t_len]
            scores.masked_fill_(sample_tokens != model.config.audio_mask_id, -float("inf"))

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

            tokens[i:i+1, :, :t_len] = sample_tokens
            batch_input_ids[i:i+1, :, c_len - t_len:c_len] = sample_tokens
            batch_input_ids[B+i:B+i+1, :, :t_len] = sample_tokens

            # Track which positions are unmasked after this step
            unmasked = (sample_tokens[0] != model.config.audio_mask_id)  # (C, T)
            # Fraction of positions unmasked per time position (across all codebooks)
            pos_unmasked_frac = unmasked.float().mean(dim=0).cpu().numpy()  # (T,)
            step_info.append({
                "step": step,
                "bias": position_bias,
                "k": k,
                "unmasked_positions": pos_unmasked_frac.tolist(),
                "mean_unmasked_first_quarter": float(pos_unmasked_frac[:t_len//4].mean()),
                "mean_unmasked_last_quarter": float(pos_unmasked_frac[3*t_len//4:].mean()),
            })

    result_tokens = [tokens[i, :, :task.target_lens[i]] for i in range(B)]
    return result_tokens, step_info


def tokens_to_audio(model, tokens):
    """Decode audio tokens to waveform using OmniVoice's codec."""
    with torch.no_grad():
        # tokens shape: (C, T)
        result = model.audio_tokenizer.decode(tokens.unsqueeze(0))
        audio = result.audio_values  # (1, 1, T)
    return audio.squeeze(0).squeeze(0).cpu().float()


def transcribe(audio_path, whisper_model):
    """Transcribe audio using faster-whisper."""
    segments, info = whisper_model.transcribe(str(audio_path), language="en")
    text = " ".join(s.text.strip() for s in segments)
    return text


def word_error_rate(ref, hyp):
    """Simple WER calculation."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", default="barth_ref.wav")
    parser.add_argument("--output_dir", default="./test_left_bias")
    parser.add_argument("--biases", default="0,0.5,1,2,5,10", help="Comma-separated bias values")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bias_values = [float(b) for b in args.biases.split(",")]

    # Test texts — varying length and emotion
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe you would do something like that! After everything we've been through together.",
        "Welcome to the annual science conference. Today we'll explore the fascinating world of quantum computing and its implications for artificial intelligence.",
    ]

    print("Loading OmniVoice...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    model.eval()

    gen_config = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=3.0,
        position_temperature=5.0,
        class_temperature=0.0,
    )

    # Prepare voice clone prompt
    print("Encoding reference voice...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio, ref_text="Reference audio.", preprocess_prompt=True,
    )

    print("Loading Whisper...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    results = []

    for text_idx, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Text {text_idx}: {text[:80]}...")

        # Estimate target length
        ref_len = prompt.ref_audio_tokens.shape[1]
        chars_per_frame = max(1, len(prompt.ref_text) / ref_len)
        target_len = max(25, min(150, int(len(text) / chars_per_frame)))

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

        for bias in bias_values:
            torch.manual_seed(args.seed)
            print(f"\n  bias={bias:.1f} ... ", end="", flush=True)

            t0 = time.time()
            with torch.no_grad():
                tokens_list, step_info = generate_with_left_bias(
                    model, task, gen_config,
                    position_bias=bias, seed=args.seed,
                )
            gen_time = time.time() - t0

            # Decode to audio
            audio = tokens_to_audio(model, tokens_list[0])

            # Save audio
            fname = f"text{text_idx}_bias{bias:.1f}.wav"
            fpath = output_dir / fname
            torchaudio.save(str(fpath), audio.unsqueeze(0), 24000)

            # Transcribe
            transcript = transcribe(fpath, whisper_model)
            wer = word_error_rate(text, transcript)

            # Token stats
            cb0 = tokens_list[0][0].cpu().numpy()
            n_unique = len(np.unique(cb0))

            # Unmasking progression: how quickly do first-quarter positions resolve?
            first_q_prog = [s["mean_unmasked_first_quarter"] for s in step_info]
            last_q_prog = [s["mean_unmasked_last_quarter"] for s in step_info]

            result = {
                "text_idx": text_idx,
                "text": text,
                "bias": bias,
                "gen_time_s": round(gen_time, 3),
                "wer": round(wer, 3),
                "transcript": transcript,
                "n_unique_cb0": int(n_unique),
                "target_len": target_len,
                "first_quarter_progression": [round(x, 3) for x in first_q_prog],
                "last_quarter_progression": [round(x, 3) for x in last_q_prog],
            }
            results.append(result)

            print(f"WER={wer:.1%} unique_cb0={n_unique} time={gen_time:.2f}s")
            print(f"    transcript: {transcript[:100]}")
            print(f"    1st-Q unmask: {' → '.join(f'{x:.0%}' for x in first_q_prog)}")
            print(f"    last-Q unmask: {' → '.join(f'{x:.0%}' for x in last_q_prog)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Group by bias
    for bias in bias_values:
        bias_results = [r for r in results if r["bias"] == bias]
        avg_wer = np.mean([r["wer"] for r in bias_results])
        avg_unique = np.mean([r["n_unique_cb0"] for r in bias_results])
        print(f"  bias={bias:>5.1f}: avg_WER={avg_wer:.1%}  avg_unique_cb0={avg_unique:.0f}")

    # Save full results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {output_dir}/results.json")
    print(f"Audio files:  {output_dir}/text*_bias*.wav")


if __name__ == "__main__":
    main()
