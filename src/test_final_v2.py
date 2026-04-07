"""Final v2: fixed cpf, cleaned ref_text, word-based frame count."""

import torch
import torchaudio
import numpy as np
import time
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
from faster_whisper import WhisperModel

import sys; sys.path.insert(0, "src")
from test_overlap_v2 import (
    generate_cfg_scheduled, decode_tokens, rms_normalize,
    split_sentences, presplit_text, assemble, wer,
)


def main():
    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    out = Path("./final_v2_output")
    out.mkdir(parents=True, exist_ok=True)

    cfg_schedule = [0, 0, 3, 3, 3, 3, 3, 3]
    SPEED_MULT = 1.3

    # Clean ref_text — manually verified, no truncation
    voice_configs = {
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
            "ref_text": None,  # auto-transcribe
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

    for voice_name, vc in voice_configs.items():
        ref_text = vc["ref_text"]
        if ref_text is None:
            segs, _ = whisper_model.transcribe(vc["path"], language="en")
            ref_text = " ".join(s.text.strip() for s in segs)
            print(f"  auto ref_text: {ref_text[:80]}")
        prompt = model.create_voice_clone_prompt(
            ref_audio=vc["path"], ref_text=ref_text, preprocess_prompt=True
        )
        rl = prompt.ref_audio_tokens.shape[1]
        cpf_actual = len(prompt.ref_text) / rl
        STANDARD_CPF = 0.75
        cpf = 0.7 * STANDARD_CPF + 0.3 * cpf_actual

        MIN_FRAMES = 80
        TARGET_FRAMES = 200
        MAX_FRAMES = 280

        def est_frames(txt):
            return int(len(txt) / cpf * SPEED_MULT)

        # Frame-budget-aware splitting: fill groups to TARGET, flush at MAX
        sentences = split_sentences(text)
        groups = []
        current_sents = []
        current_frames = 0

        for sent in sentences:
            sent_frames = est_frames(sent)
            combined = est_frames(" ".join(current_sents + [sent])) if current_sents else sent_frames

            if current_sents and combined > MAX_FRAMES:
                # Would exceed max — must flush
                groups.append(" ".join(current_sents))
                current_sents = [sent]
                current_frames = sent_frames
            else:
                # Keep adding — even if past target, up to max
                current_sents.append(sent)
                current_frames = combined

        if current_sents:
            groups.append(" ".join(current_sents))

        # Merge tiny tail
        if len(groups) > 1 and est_frames(groups[-1]) < MIN_FRAMES:
            merged = groups[-2] + " " + groups[-1]
            if est_frames(merged) <= MAX_FRAMES:
                groups[-2] = merged
                groups.pop()

        print(f"{voice_name} (cpf_actual={cpf_actual:.2f}, cpf_blended={cpf:.2f}):")

        audios = []
        gen_times = []
        for gi, g in enumerate(groups):
            gw = len(g.split())
            tf = est_frames(g)
            tf = max(MIN_FRAMES, min(MAX_FRAMES, tf))
            torch.cuda.synchronize()
            t0 = time.time()
            toks = generate_cfg_scheduled(model, g, tf, prompt, cfg_schedule)
            torch.cuda.synchronize()
            gen_times.append(time.time() - t0)
            with torch.no_grad():
                audio_chunk = decode_tokens(model, toks)
            del toks
            torch.cuda.empty_cache()
            audios.append(audio_chunk)
            print(f"  group {gi}: {gw}w -> {tf} frames")

        audio = assemble(audios, silence_ms=40, crossfade_ms=30)
        fp = out / f"{voice_name}_final.wav"
        torchaudio.save(str(fp), audio.unsqueeze(0), 24000)

        s2, _ = whisper_model.transcribe(str(fp), language="en")
        t = " ".join(seg.text.strip() for seg in s2)
        w = wer(text, t)
        dur = len(audio) / 24000
        ttfa = gen_times[0] * 1000
        total = sum(gen_times) * 1000

        print(f"  gen={total:.0f}ms ttfa={ttfa:.0f}ms dur={dur:.1f}s WER={w:.0%}")
        print(f"  {t[:120]}")
        print()

    print(f"Audio in {out}/")


if __name__ == "__main__":
    main()
