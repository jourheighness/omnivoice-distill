"""Evaluate step-distilled OmniVoice: 4-step distilled vs 8-step teacher.

Loads the LoRA checkpoint, runs generation with 4 steps, compares
quality (Whisper WER, token diversity) against standard 8-step baseline.
"""

import time
import torch
import torchaudio
import numpy as np
from pathlib import Path


def transcribe(audio_path, whisper_model):
    segments, _ = whisper_model.transcribe(str(audio_path), language="en")
    return " ".join(s.text.strip() for s in segments)


def word_error_rate(ref, hyp):
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
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


def tokens_to_audio(model, tokens):
    with torch.no_grad():
        result = model.audio_tokenizer.decode(tokens.unsqueeze(0))
    return result.audio_values.squeeze(0).squeeze(0).cpu().float()


def generate(model, text, prompt, gen_config, seed=42):
    from omnivoice.models.omnivoice import GenerationTask
    torch.manual_seed(seed)

    ref_len = prompt.ref_audio_tokens.shape[1]
    chars_per_frame = max(1, len(prompt.ref_text) / ref_len)
    target_len = max(25, min(200, int(len(text) / chars_per_frame)))

    task = GenerationTask(
        batch_size=1, texts=[text], target_lens=[target_len],
        langs=["English"], instructs=["None"],
        ref_audio_tokens=[prompt.ref_audio_tokens],
        ref_texts=[prompt.ref_text],
        ref_rms=[getattr(prompt, "ref_rms", 0.1)],
    )

    t0 = time.time()
    with torch.no_grad():
        tokens_list = model._generate_iterative(task, gen_config)
    elapsed = time.time() - t0
    return tokens_list[0], elapsed, target_len


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", default="./checkpoints_distill/lora_epoch3")
    parser.add_argument("--ref_audio", default="barth_ref.wav")
    parser.add_argument("--output_dir", default="./eval_distill")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe you would do something like that! After everything we've been through together.",
        "Welcome to the annual science conference. Today we'll explore the fascinating world of quantum computing and its implications for artificial intelligence.",
        "Once upon a time, in a land far far away, there lived a brave knight who feared nothing.",
    ]

    print("Loading OmniVoice (baseline)...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    model.eval()

    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio, ref_text="Reference audio.", preprocess_prompt=True,
    )

    print("Loading Whisper...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel("base.en", device="cuda", compute_type="float16")

    # Test configs
    configs = {
        "8step_baseline": OmniVoiceGenerationConfig(
            num_step=8, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
        ),
        "4step_baseline": OmniVoiceGenerationConfig(
            num_step=4, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
        ),
        "2step_baseline": OmniVoiceGenerationConfig(
            num_step=2, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
        ),
    }

    # Run baselines first
    results = {}
    for config_name, gen_config in configs.items():
        print(f"\n--- {config_name} ---")
        results[config_name] = []
        for i, text in enumerate(test_texts):
            tokens, elapsed, target_len = generate(model, text, prompt, gen_config, seed=42)
            audio = tokens_to_audio(model, tokens)
            fpath = output_dir / f"{config_name}_text{i}.wav"
            torchaudio.save(str(fpath), audio.unsqueeze(0), 24000)
            transcript = transcribe(fpath, whisper)
            wer = word_error_rate(text, transcript)
            unique = len(np.unique(tokens[0].cpu().numpy()))
            results[config_name].append({
                "wer": wer, "unique_cb0": unique, "time": elapsed, "transcript": transcript,
            })
            print(f"  text{i}: WER={wer:.0%} unique={unique} time={elapsed:.3f}s")
            print(f"    {transcript[:120]}")

    # Now load LoRA and test distilled model
    print(f"\nLoading LoRA from {args.lora_dir}...")
    from peft import PeftModel
    model.llm = PeftModel.from_pretrained(model.llm, args.lora_dir)
    model.llm = model.llm.merge_and_unload()  # merge for inference speed
    model.eval()

    # Test distilled model at different step counts
    distill_configs = {
        "4step_distilled": OmniVoiceGenerationConfig(
            num_step=4, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
        ),
        "2step_distilled": OmniVoiceGenerationConfig(
            num_step=2, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
        ),
        "8step_distilled": OmniVoiceGenerationConfig(
            num_step=8, guidance_scale=3.0, position_temperature=5.0, class_temperature=0.0,
        ),
    }

    for config_name, gen_config in distill_configs.items():
        print(f"\n--- {config_name} ---")
        results[config_name] = []
        for i, text in enumerate(test_texts):
            tokens, elapsed, target_len = generate(model, text, prompt, gen_config, seed=42)
            audio = tokens_to_audio(model, tokens)
            fpath = output_dir / f"{config_name}_text{i}.wav"
            torchaudio.save(str(fpath), audio.unsqueeze(0), 24000)
            transcript = transcribe(fpath, whisper)
            wer = word_error_rate(text, transcript)
            unique = len(np.unique(tokens[0].cpu().numpy()))
            results[config_name].append({
                "wer": wer, "unique_cb0": unique, "time": elapsed, "transcript": transcript,
            })
            print(f"  text{i}: WER={wer:.0%} unique={unique} time={elapsed:.3f}s")
            print(f"    {transcript[:120]}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for config_name in results:
        r = results[config_name]
        avg_wer = np.mean([x["wer"] for x in r])
        avg_unique = np.mean([x["unique_cb0"] for x in r])
        avg_time = np.mean([x["time"] for x in r])
        print(f"  {config_name:>20s}: WER={avg_wer:.0%}  unique_cb0={avg_unique:.0f}  time={avg_time:.3f}s")


if __name__ == "__main__":
    main()
