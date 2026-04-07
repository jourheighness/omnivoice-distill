"""Test clean best pipeline with multiple voices + benchmark."""

import math
import re
import time
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import (
    GenerationTask, OmniVoiceGenerationConfig, _get_time_steps,
)
from faster_whisper import WhisperModel

import sys; sys.path.insert(0, "src")
from test_overlap_v2 import (
    generate_cfg_scheduled, decode_tokens, rms_normalize,
    split_sentences, presplit_text, assemble, wer,
)


def run_clean_pipeline(model, text, prompt, cfg_schedule, cpf, voice_seed=42):
    """Full clean pipeline: presplit, generate groups, assemble."""
    groups_sents = presplit_text(text)
    groups = [" ".join(g) for g in groups_sents]

    audios = []
    gen_times = []

    for gi, g in enumerate(groups):
        tf = max(60, min(450, int(len(g) / cpf)))
        torch.cuda.synchronize()
        t0 = time.time()
        toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule, seed=voice_seed)
        torch.cuda.synchronize()
        gen_times.append(time.time() - t0)
        audios.append(decode_tokens(model, toks))

    audio = assemble(audios, silence_ms=40, crossfade_ms=30)
    return audio, gen_times, groups


def main():
    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    cfg_schedule = [0, 0, 3, 3, 3, 3, 3, 3]

    out = Path("./voices_output")
    out.mkdir(parents=True, exist_ok=True)

    voices = {
        "barth": "barth_ref.wav",
        "astarion": "astarion_ref.wav",
        "vesper": "vesper_ref.wav",
    }

    texts = {
        "story": (
            "The old lighthouse keeper had not spoken to another human being in three years. "
            "Every morning he climbed the one hundred and forty seven steps to the lamp room. "
            "Every evening he descended them again, his joints aching with each step. "
            "The sea was his only companion, and it was not always a kind one. "
            "On stormy nights, the waves would crash against the rocks with such fury that the whole tower trembled. "
            "But tonight was different. "
            "Tonight the sea was calm, the stars were bright, and somewhere in the distance, he could hear music. "
            "It was faint at first, barely distinguishable from the wind. "
            "But as the minutes passed, it grew louder and more distinct. "
            "Someone was playing a violin on the shore below."
        ),
        "emotional": (
            "I have been thinking about this for a long time now. "
            "The truth is, I never wanted things to end up this way. "
            "When we first started working together, I believed we could change the world. "
            "We had the talent, the drive, and most importantly, the vision. "
            "But somewhere along the way, things went wrong."
        ),
        "short": "Welcome to the annual science conference. Today we will explore the fascinating world of quantum computing and its implications for artificial intelligence.",
    }

    results = []

    for voice_name, ref_path in voices.items():
        print(f"\n{'='*70}")
        print(f"Voice: {voice_name}")

        # Transcribe ref audio for correct ref_text
        segs, _ = whisper_model.transcribe(ref_path, language="en")
        actual_ref = " ".join(s.text.strip() for s in segs)
        print(f"  ref_text: {actual_ref[:80]}...")

        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_path, ref_text=actual_ref, preprocess_prompt=True
        )
        rl = prompt.ref_audio_tokens.shape[1]
        cpf = max(1, len(prompt.ref_text) / rl)
        print(f"  ref_tokens: {rl} frames, cpf={cpf:.2f}")

        for text_name, text in texts.items():
            n_words = len(text.split())

            audio, gen_times, groups = run_clean_pipeline(
                model, text, prompt, cfg_schedule, cpf
            )

            fname = f"{voice_name}_{text_name}.wav"
            torchaudio.save(str(out / fname), audio.unsqueeze(0), 24000)

            s2, _ = whisper_model.transcribe(str(out / fname), language="en")
            transcript = " ".join(seg.text.strip() for seg in s2)
            w = wer(text, transcript)
            dur = len(audio) / 24000
            ttfa = gen_times[0] * 1000
            total_gen = sum(gen_times) * 1000

            results.append({
                "voice": voice_name,
                "text": text_name,
                "words": n_words,
                "groups": len(groups),
                "wer": w,
                "ttfa_ms": ttfa,
                "total_gen_ms": total_gen,
                "dur_s": dur,
            })

            print(f"  [{text_name}] {n_words}w {len(groups)}g: "
                  f"gen={total_gen:.0f}ms ttfa={ttfa:.0f}ms "
                  f"WER={w:.0%} dur={dur:.1f}s")
            print(f"    {transcript[:100]}")

    # Benchmark: run the short text 10 times per voice for stable timing
    print(f"\n{'='*70}")
    print("BENCHMARK (short text, 10 runs each)")
    short_text = texts["short"]

    for voice_name, ref_path in voices.items():
        segs, _ = whisper_model.transcribe(ref_path, language="en")
        actual_ref = " ".join(s.text.strip() for s in segs)
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_path, ref_text=actual_ref, preprocess_prompt=True
        )
        rl = prompt.ref_audio_tokens.shape[1]
        cpf = max(1, len(prompt.ref_text) / rl)

        # Warmup
        run_clean_pipeline(model, short_text, prompt, cfg_schedule, cpf)

        bench_times = []
        for i in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            audio, gt, _ = run_clean_pipeline(model, short_text, prompt, cfg_schedule, cpf, voice_seed=42+i)
            torch.cuda.synchronize()
            bench_times.append(time.time() - t0)

        dur = len(audio) / 24000
        avg = np.mean(bench_times) * 1000
        p50 = np.percentile(bench_times, 50) * 1000
        p95 = np.percentile(bench_times, 95) * 1000
        rtf = (np.mean(bench_times)) / dur

        print(f"  {voice_name}: avg={avg:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms "
              f"dur={dur:.1f}s RTF={rtf:.3f}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['voice']:>10s} {r['text']:>10s}: "
              f"{r['words']:3d}w {r['groups']}g "
              f"WER={r['wer']:3.0%} ttfa={r['ttfa_ms']:4.0f}ms "
              f"gen={r['total_gen_ms']:4.0f}ms dur={r['dur_s']:.1f}s")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
