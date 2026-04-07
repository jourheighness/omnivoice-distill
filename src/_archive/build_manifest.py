"""Build manifest from extracted LibriTTS-R, sampling evenly across speakers."""

import json
import os
import random
from collections import defaultdict
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/workspace/LibriTTS_R/train-clean-360")
    parser.add_argument("--output_dir", default="data/libritts_big")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--max_per_speaker", type=int, default=20)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by speaker
    speaker_files = defaultdict(list)
    for wav in base.rglob("*.wav"):
        # Path: base/SPEAKER/CHAPTER/file.wav
        parts = wav.relative_to(base).parts
        if len(parts) >= 2 and parts[0].isdigit():
            speaker_files[int(parts[0])].append(wav)

    print(f"Found {sum(len(v) for v in speaker_files.values())} wavs across {len(speaker_files)} speakers")

    # Sample evenly: up to max_per_speaker from each, then fill remaining
    random.seed(42)
    manifest = []
    remaining_budget = args.num_samples

    # First pass: take up to max_per_speaker from each
    speakers = sorted(speaker_files.keys())
    samples_per = min(args.max_per_speaker, args.num_samples // len(speakers) + 1)

    for sid in speakers:
        files = speaker_files[sid]
        random.shuffle(files)
        take = min(samples_per, len(files), remaining_budget)
        for wav in files[:take]:
            txt_path = str(wav).rsplit(".wav", 1)[0] + ".normalized.txt"
            text = ""
            if os.path.exists(txt_path):
                text = open(txt_path).read().strip()
            manifest.append({
                "id": len(manifest),
                "audio_path": str(wav),
                "text": text,
                "speaker_id": sid,
                "sr": 24000,
            })
            remaining_budget -= 1
            if remaining_budget <= 0:
                break
        if remaining_budget <= 0:
            break

    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    speakers_in = len(set(m["speaker_id"] for m in manifest))
    print(f"Built manifest: {len(manifest)} samples, {speakers_in} speakers")
    print(f"  ~{len(manifest)//max(speakers_in,1)} samples per speaker")
    if manifest:
        print(f"  Sample: '{manifest[0]['text'][:80]}'")


if __name__ == "__main__":
    main()
