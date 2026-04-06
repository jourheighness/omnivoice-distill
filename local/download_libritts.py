"""Extract LibriTTS-R dev-clean with proper transcripts and speaker IDs."""

import json
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
    if not tar_path.exists():
        print("No tar found. Download first.")
        return

    print(f"Extracting from {tar_path}...")

    # First pass: index all files by stem
    wavs = {}    # stem -> TarInfo
    txts = {}    # stem -> TarInfo

    with tarfile.open(str(tar_path), "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith(".wav"):
                stem = m.name.rsplit(".wav", 1)[0]
                wavs[stem] = m
            elif m.name.endswith(".normalized.txt"):
                stem = m.name.rsplit(".normalized.txt", 1)[0]
                txts[stem] = m

        # Match wav+txt pairs
        paired_stems = sorted(set(wavs.keys()) & set(txts.keys()))
        print(f"  Found {len(wavs)} wavs, {len(txts)} txts, {len(paired_stems)} paired")

        n = min(args.num_samples, len(paired_stems))
        manifest = []

        # Clean old files
        for old in out.glob("sample_*.wav"):
            old.unlink()

        for i, stem in enumerate(paired_stems[:n]):
            # Extract speaker from path: .../SPEAKER_ID/CHAPTER/file
            parts = stem.split("/")
            speaker_id = 0
            for p in parts:
                if p.isdigit():
                    speaker_id = int(p)
                    break

            # Extract wav
            fname = f"sample_{i:04d}.wav"
            fpath = out / fname
            wf = tar.extractfile(wavs[stem])
            if wf is None:
                continue
            fpath.write_bytes(wf.read())

            # Extract transcript
            tf = tar.extractfile(txts[stem])
            if tf is None:
                continue
            text = tf.read().decode("utf-8").strip()

            manifest.append({
                "id": i,
                "audio_path": str(fpath),
                "text": text,
                "speaker_id": speaker_id,
                "sr": 24000,
            })

            if (i + 1) % 200 == 0:
                speakers_so_far = len(set(m["speaker_id"] for m in manifest))
                print(f"  {i+1}/{n} extracted ({speakers_so_far} speakers)")

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    speakers = set(m["speaker_id"] for m in manifest)
    avg_text = sum(len(m["text"]) for m in manifest) / max(len(manifest), 1)
    print(f"\nDone: {len(manifest)} samples, {len(speakers)} speakers")
    print(f"Avg text: {avg_text:.0f} chars")
    print(f"Sample: '{manifest[0]['text'][:100]}'")


if __name__ == "__main__":
    main()
