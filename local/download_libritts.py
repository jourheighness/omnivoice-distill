"""Download LibriTTS-R dev-clean directly from OpenSLR.

~300MB download, ~30 speakers, high quality 24kHz audio.
"""

import json
import os
import subprocess
import tarfile
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/libritts_local")
    parser.add_argument("--num_samples", type=int, default=2000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tar_path = out / "dev-clean.tar.gz"

    # Download from OpenSLR
    if not tar_path.exists():
        url = "https://www.openslr.org/resources/141/dev_clean.tar.gz"
        print(f"Downloading LibriTTS-R dev-clean from OpenSLR (~300MB)...")
        subprocess.run(["curl", "-L", "-o", str(tar_path), url], check=True)
    else:
        print(f"Using cached {tar_path}")

    # Extract
    print("Extracting...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        members = tar.getmembers()
        wav_members = sorted([m for m in members if m.name.endswith(".wav")], key=lambda m: m.name)
        txt_lookup = {}
        for m in members:
            if m.name.endswith(".normalized.txt"):
                txt_lookup[m.name] = m

        print(f"  Found {len(wav_members)} wav files")

        manifest = []
        for i, wm in enumerate(wav_members):
            if i >= args.num_samples:
                break

            # Matching transcript: same name but .normalized.txt
            txt_name = wm.name.rsplit(".wav", 1)[0] + ".normalized.txt"
            txt_member = txt_lookup.get(txt_name)

            # Extract speaker from path: LibriTTS_R/dev-clean/SPEAKER/CHAPTER/file.wav
            parts = wm.name.split("/")
            speaker_id = 0
            for p in parts:
                if p.isdigit():
                    speaker_id = int(p)
                    break

            # Extract wav
            fname = f"sample_{i:04d}.wav"
            fpath = out / fname
            wf = tar.extractfile(wm)
            if wf is None:
                continue
            fpath.write_bytes(wf.read())

            # Extract text
            text = ""
            if txt_member:
                tf = tar.extractfile(txt_member)
                if tf:
                    text = tf.read().decode("utf-8").strip()

            if not text:
                text = f"Sample audio number {i} from speaker {speaker_id}."

            manifest.append({
                "id": i,
                "audio_path": str(fpath),
                "text": text,
                "speaker_id": speaker_id,
                "sr": 24000,
            })

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{min(args.num_samples, len(wav_members))} extracted")

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    speakers = set(m["speaker_id"] for m in manifest)
    print(f"\nSaved {len(manifest)} samples, {len(speakers)} unique speakers to {out}/")
    print(f"Avg text length: {sum(len(m['text']) for m in manifest)/len(manifest):.0f} chars")


if __name__ == "__main__":
    main()
