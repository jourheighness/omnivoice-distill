"""Per-chunk WER analysis: understand why Astarion was near-perfect."""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
from faster_whisper import WhisperModel
import re

import sys; sys.path.insert(0, "src")
from test_overlap_v2 import generate_cfg_scheduled, decode_tokens, split_sentences, wer


def main():
    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    cfg_schedule = [0, 0, 3, 3, 3, 3, 3, 3]
    SPEED_MULT = 1.3
    STANDARD_CPF = 0.75

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
    }

    for voice_name, vc in voices.items():
        prompt = model.create_voice_clone_prompt(
            ref_audio=vc["path"], ref_text=vc["ref_text"], preprocess_prompt=True
        )
        rl = prompt.ref_audio_tokens.shape[1]
        cpf_actual = len(prompt.ref_text) / rl
        cpf = 0.7 * STANDARD_CPF + 0.3 * cpf_actual

        def est_frames(txt):
            return int(len(txt) / cpf * SPEED_MULT)

        # Rebuild groups
        sentences = split_sentences(text)
        groups = []
        current_sents = []
        for sent in sentences:
            combined = est_frames(" ".join(current_sents + [sent])) if current_sents else est_frames(sent)
            if current_sents and combined > 280:
                groups.append(" ".join(current_sents))
                current_sents = [sent]
            else:
                current_sents.append(sent)
        if current_sents:
            groups.append(" ".join(current_sents))
        if len(groups) > 1 and est_frames(groups[-1]) < 80:
            merged = groups[-2] + " " + groups[-1]
            if est_frames(merged) <= 280:
                groups[-2] = merged
                groups.pop()

        print(f"\n{'='*70}")
        print(f"{voice_name} (cpf_actual={cpf_actual:.2f}, cpf_blended={cpf:.2f})")
        print(f"  ref_tokens={rl} frames, speed={SPEED_MULT}x")
        print()

        total_wer_sum = 0
        total_words = 0

        for gi, g in enumerate(groups):
            gw = len(g.split())
            tf = max(80, min(280, est_frames(g)))
            density = gw / tf  # words per frame
            chars_density = len(g) / tf  # chars per frame

            toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule)
            audio = decode_tokens(model, toks)
            torch.cuda.empty_cache()

            unique = len(np.unique(toks[0].cpu().numpy()))
            fp = f"/tmp/chunk_{voice_name}_{gi}.wav"
            torchaudio.save(fp, audio.unsqueeze(0), 24000)
            segs, _ = whisper_model.transcribe(fp, language="en")
            transcript = " ".join(s.text.strip() for s in segs)
            w = wer(g, transcript)
            dur = len(audio) / 24000

            # WER breakdown
            ref_words = g.lower().split()
            hyp_words = transcript.lower().split()
            total_words += len(ref_words)
            total_wer_sum += w * len(ref_words)

            status = "OK" if w < 0.1 else "WARN" if w < 0.2 else "BAD"

            print(f"  [{status}] group {gi}: {gw}w {tf}f "
                  f"density={density:.3f}w/f ({chars_density:.2f}c/f) "
                  f"dur={dur:.1f}s unique={unique} WER={w:.0%}")
            print(f"    IN:  {g[:90]}")
            print(f"    OUT: {transcript[:90]}")

            if w > 0.05:
                # Show specific word differences
                max_len = max(len(ref_words), len(hyp_words))
                diffs = []
                for i in range(min(len(ref_words), len(hyp_words))):
                    if ref_words[i] != hyp_words[i]:
                        diffs.append(f"'{ref_words[i]}' -> '{hyp_words[i]}'")
                if len(ref_words) > len(hyp_words):
                    for i in range(len(hyp_words), len(ref_words)):
                        diffs.append(f"'{ref_words[i]}' -> DROPPED")
                elif len(hyp_words) > len(ref_words):
                    for i in range(len(ref_words), len(hyp_words)):
                        diffs.append(f"ADDED '{hyp_words[i]}'")
                if diffs:
                    print(f"    DIFFS: {', '.join(diffs[:8])}")
            print()

        avg_wer = total_wer_sum / total_words if total_words else 0
        print(f"  OVERALL: {len(groups)} groups, weighted WER={avg_wer:.0%}")
        print()


if __name__ == "__main__":
    main()
