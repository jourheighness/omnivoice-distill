"""Final test: calibrated voice rate + frame-budget splitting + all speedups."""

import torch
import torchaudio
import numpy as np
import time
import re
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
from faster_whisper import WhisperModel

import sys; sys.path.insert(0, "src")
from test_overlap_v2 import generate_cfg_scheduled, decode_tokens, split_sentences, assemble, wer
from voice_calibration import calibrate_voice, estimate_target_len


def frame_budget_split(text, voice_cal, min_frames=80, max_frames=280, fps=25, padding=1.08):
    """Split text into groups where each group fits within frame budget.

    Uses unclamped frame estimates for splitting decisions, then clamps
    the final target_len when generating.
    """
    sentences = split_sentences(text)
    cps = voice_cal["chars_per_sec"]

    def est_unclamped(txt):
        """Estimate frames WITHOUT clamping to max — for split decisions."""
        n_chars = len(txt.strip())
        duration_sec = n_chars / cps
        return int(duration_sec * fps * padding)

    groups = []
    current_sents = []
    current_frames = 0

    for sent in sentences:
        sent_frames = est_unclamped(sent)

        if current_sents and current_frames + sent_frames > max_frames:
            groups.append(" ".join(current_sents))
            current_sents = [sent]
            current_frames = sent_frames
        else:
            current_sents.append(sent)
            current_frames += sent_frames

    if current_sents:
        groups.append(" ".join(current_sents))

    # Merge tiny tail
    if len(groups) > 1 and est_unclamped(groups[-1]) < min_frames:
        merged = groups[-2] + " " + groups[-1]
        if est_unclamped(merged) <= max_frames + 50:  # slight overflow ok for tail
            groups[-2] = merged
            groups.pop()

    return groups


def main():
    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    cfg_schedule = [0, 0, 3, 3, 3, 3, 3, 3]
    out = Path("./calibrated_output")
    out.mkdir(parents=True, exist_ok=True)

    voices = {
        "barth": {
            "path": "barth_ref.wav",
            "ref_text": (
                "He's established fail-safe protocols to wipe the memory "
                "if there's any attempt to access certain files. "
                "Only about six people in the world could program "
                "safeguards like that. I invented them."
            ),
        },
        "astarion": {
            "path": "astarion_ref.wav",
            "ref_text": (
                "Excuse me, while I go and regurgitate the sap "
                "wrangling in my throat. You have a type, "
                "don't you, elven prostitutes."
            ),
        },
        "vesper": {
            "path": "vesper_ref.wav",
            "ref_text": None,
        },
    }

    text = (
        "The old lighthouse keeper had not spoken to another human being in three years. "
        "Every morning he climbed the one hundred and forty seven steps to the lamp room. "
        "Every evening he descended them again, his joints aching with each step. "
        "The sea was his only companion, and it was not always a kind one. "
        "On stormy nights, the waves would crash against the rocks with such fury "
        "that the whole tower trembled. "
        "But tonight was different. "
        "Tonight the sea was calm, the stars were bright, and somewhere in the distance, "
        "he could hear music. "
        "It was faint at first, barely distinguishable from the wind. "
        "But as the minutes passed, it grew louder and more distinct. "
        "Someone was playing a violin on the shore below."
    )

    for voice_name, vc in voices.items():
        # Calibrate voice
        cal = calibrate_voice(vc["path"], whisper_model)

        # Set ref_text
        ref_text = vc["ref_text"] if vc["ref_text"] else cal["transcript"]
        prompt = model.create_voice_clone_prompt(
            ref_audio=vc["path"], ref_text=ref_text, preprocess_prompt=True
        )

        print(f"\n{'='*70}")
        print(f"{voice_name}: {cal['chars_per_sec']:.1f} c/s, "
              f"{cal['words_per_sec']:.1f} w/s, gap={cal.get('avg_gap_ms', 0):.0f}ms")

        # Split with calibrated frame budget
        groups = frame_budget_split(text, cal)

        audios = []
        gen_times = []

        for gi, g in enumerate(groups):
            tf = estimate_target_len(g, cal)
            gw = len(g.split())
            density = gw / tf

            torch.cuda.synchronize()
            t0 = time.time()
            toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule)
            torch.cuda.synchronize()
            gen_times.append(time.time() - t0)

            audio = decode_tokens(model, toks)
            torch.cuda.empty_cache()
            audios.append(audio)

            print(f"  group {gi}: {gw}w {tf}f density={density:.3f}w/f "
                  f"dur={len(audio)/24000:.1f}s")

        audio = assemble(audios, silence_ms=40, crossfade_ms=30)
        fp = out / f"{voice_name}.wav"
        torchaudio.save(str(fp), audio.unsqueeze(0), 24000)

        s2, _ = whisper_model.transcribe(str(fp), language="en")
        transcript = " ".join(seg.text.strip() for seg in s2)
        w = wer(text, transcript)
        dur = len(audio) / 24000
        ttfa = gen_times[0] * 1000
        total = sum(gen_times) * 1000

        print(f"  RESULT: {len(groups)} groups gen={total:.0f}ms ttfa={ttfa:.0f}ms "
              f"dur={dur:.1f}s WER={w:.0%}")
        print(f"  {transcript[:120]}")

    print(f"\nAudio in {out}/")


if __name__ == "__main__":
    main()
