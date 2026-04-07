"""Extract LibriTTS-R train-clean-360 for large-scale training."""

import json
import tarfile
from pathlib import Path

tar_path = "/workspace/train-clean-360.tar.gz"
out = Path("data/libritts_big")
out.mkdir(parents=True, exist_ok=True)

print(f"Extracting from {tar_path}...")

with tarfile.open(tar_path, "r:gz") as tar:
    members = tar.getmembers()
    wavs = {}
    txts = {}
    for m in members:
        if m.name.endswith(".wav"):
            wavs[m.name.rsplit(".wav", 1)[0]] = m
        elif m.name.endswith(".normalized.txt"):
            txts[m.name.rsplit(".normalized.txt", 1)[0]] = m

    paired = sorted(set(wavs.keys()) & set(txts.keys()))
    print(f"Found {len(paired)} paired samples")

    manifest = []
    for i, stem in enumerate(paired[:3000]):
        parts = stem.split("/")
        speaker = 0
        for p in parts:
            if p.isdigit():
                speaker = int(p)
                break
        fname = f"sample_{i:04d}.wav"
        fpath = out / fname
        wf = tar.extractfile(wavs[stem])
        if not wf:
            continue
        fpath.write_bytes(wf.read())
        tf = tar.extractfile(txts[stem])
        text = tf.read().decode("utf-8").strip() if tf else ""
        manifest.append({
            "id": i,
            "audio_path": str(fpath),
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
