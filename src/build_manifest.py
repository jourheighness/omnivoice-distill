"""Build manifest from extracted LibriTTS-R files on disk."""

import json
import os
from pathlib import Path

base = Path("/workspace/LibriTTS_R/train-clean-360")
out = Path("data/libritts_big")
out.mkdir(parents=True, exist_ok=True)

wavs = sorted(base.rglob("*.wav"))
print(f"Found {len(wavs)} wav files")

manifest = []
for i, wav in enumerate(wavs[:3000]):
    txt_path = str(wav).rsplit(".wav", 1)[0] + ".normalized.txt"
    text = ""
    if os.path.exists(txt_path):
        text = open(txt_path).read().strip()

    # Speaker ID from path: .../SPEAKER/CHAPTER/file.wav
    parts = wav.parts
    speaker = 0
    for p in parts:
        if p.isdigit():
            speaker = int(p)
            break

    manifest.append({
        "id": i,
        "audio_path": str(wav),
        "text": text,
        "speaker_id": speaker,
        "sr": 24000,
    })

    if (i + 1) % 500 == 0:
        speakers = len(set(m["speaker_id"] for m in manifest))
        print(f"  {i+1}/3000 ({speakers} speakers)")

json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
speakers = len(set(m["speaker_id"] for m in manifest))
print(f"Done: {len(manifest)} samples, {speakers} speakers")
print(f"Sample: '{manifest[0]['text'][:80]}'")
