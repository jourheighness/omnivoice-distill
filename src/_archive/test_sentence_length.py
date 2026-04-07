"""Test sentence length sweet spot for split decode chunking."""

import torch
import torchaudio
import time
import numpy as np
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
from faster_whisper import WhisperModel


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / len(r)


def main():
    print("Loading...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.eval()
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    segs, _ = whisper_model.transcribe("barth_ref.wav", language="en")
    actual_ref = " ".join(s.text.strip() for s in segs)
    prompt = model.create_voice_clone_prompt(
        ref_audio="barth_ref.wav", ref_text=actual_ref, preprocess_prompt=True
    )

    gc = OmniVoiceGenerationConfig(
        num_step=12, guidance_scale=3.0,
        position_temperature=5.0, class_temperature=0.0,
    )

    test_sentences = {
        "4w": "The sun is setting.",
        "6w": "The sun sets over the distant hills.",
        "8w": "The sun slowly sets over the quiet distant hills.",
        "10w": "The sun slowly sets over the quiet and distant rolling hills.",
        "12w": "As the sun slowly sets over the quiet and distant hills, birds sing.",
        "15w": "As the sun slowly sets over the quiet and distant hills, the birds begin their evening song.",
        "20w": "As the sun slowly sets over the quiet and distant rolling hills, the birds begin their soft and gentle evening song.",
        "25w": "As the sun slowly sets over the quiet and distant rolling hills, the birds begin their soft and gentle evening song, filling the air.",
    }

    rl = prompt.ref_audio_tokens.shape[1]
    cpf = max(1, len(prompt.ref_text) / rl)

    print(f"\ncpf={cpf:.2f} (chars per frame)\n")

    for label, text in test_sentences.items():
        n_words = len(text.split())
        n_chars = len(text)
        tl = max(15, min(200, int(n_chars / cpf)))

        task = GenerationTask(
            batch_size=1, texts=[text], target_lens=[tl],
            langs=["English"], instructs=["None"],
            ref_audio_tokens=[prompt.ref_audio_tokens],
            ref_texts=[prompt.ref_text],
            ref_rms=[getattr(prompt, "ref_rms", 0.1)],
        )

        torch.manual_seed(42)
        t0 = time.time()
        with torch.no_grad():
            toks = model._generate_iterative(task, gc)[0]
        elapsed = time.time() - t0

        audio = model.audio_tokenizer.decode(
            toks.unsqueeze(0)
        ).audio_values.squeeze().cpu().float()
        unique = len(np.unique(toks[0].cpu().numpy()))

        fpath = f"/tmp/sentlen_{label}.wav"
        torchaudio.save(fpath, audio.unsqueeze(0), 24000)

        s2, _ = whisper_model.transcribe(fpath, language="en")
        transcript = " ".join(s.text.strip() for s in s2)
        w = wer(text, transcript)

        dur_s = len(audio) / 24000
        print(
            f"{label:>4s} ({n_words:2d}w {n_chars:3d}c): "
            f"frames={tl:3d} WER={w:3.0%} unique={unique:3d} "
            f"dur={dur_s:.1f}s gen={elapsed*1000:.0f}ms | {transcript[:80]}"
        )

    print("\nAudio files: /tmp/sentlen_*.wav")


if __name__ == "__main__":
    main()
