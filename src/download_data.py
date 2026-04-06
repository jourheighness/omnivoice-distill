"""Download LibriTTS dataset — stream mode to avoid torchcodec entirely."""

import os
import io
import json
import soundfile as sf
from datasets import load_dataset

os.makedirs("data/libritts", exist_ok=True)

print("Downloading LibriTTS clean-100 (streaming mode)...")
ds = load_dataset(
    "parler-tts/libritts_r_filtered", "clean",
    split="train.clean.100",
    streaming=True,
)

manifest = []
for i, sample in enumerate(ds):
    if i >= 500:
        break

    audio_path = f"data/libritts/sample_{i:04d}.wav"

    audio = sample["audio"]
    if isinstance(audio, dict):
        if "array" in audio and audio["array"] is not None:
            sf.write(audio_path, audio["array"], audio["sampling_rate"])
            sr = audio["sampling_rate"]
        elif "bytes" in audio and audio["bytes"]:
            data, sr = sf.read(io.BytesIO(audio["bytes"]))
            sf.write(audio_path, data, sr)
        elif "path" in audio and audio["path"] and os.path.exists(audio["path"]):
            import shutil
            shutil.copy(audio["path"], audio_path)
            _, sr = sf.read(audio_path)
        else:
            print(f"  Skipping {i}: unknown audio format {list(audio.keys())}")
            continue
    else:
        print(f"  Skipping {i}: unexpected type {type(audio)}")
        continue

    manifest.append({
        "id": i,
        "audio_path": audio_path,
        "text": sample.get("text_normalized", sample.get("text", "")),
        "speaker_id": sample.get("speaker_id", 0),
        "sr": int(sr),
    })
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/500 saved")

with open("data/libritts/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Downloaded {len(manifest)} samples")
if manifest:
    print(f"Sample text: {manifest[0]['text'][:100]}")
