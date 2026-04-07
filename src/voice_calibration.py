"""Voice calibration: measure speaking rate from reference audio.

Uses Whisper word-level timestamps to compute chars-per-second for
each voice, then uses that to set accurate target_len for generation.
"""

import torch
import torchaudio
import re
from faster_whisper import WhisperModel


def calibrate_voice(ref_audio_path, whisper_model, language="en"):
    """Measure speaking rate from reference audio.

    Returns dict with:
        chars_per_sec: characters spoken per second (excluding silence)
        words_per_sec: words spoken per second
        total_speech_sec: total speaking time (excluding leading/trailing silence)
        transcript: full transcript
    """
    segments, info = whisper_model.transcribe(
        ref_audio_path, language=language, word_timestamps=True
    )

    words = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)

    if not words:
        # Fallback: use segment-level timing
        segments, info = whisper_model.transcribe(ref_audio_path, language=language)
        seg_list = list(segments)
        if not seg_list:
            return {"chars_per_sec": 15.0, "words_per_sec": 2.5,
                    "total_speech_sec": 0, "transcript": ""}
        transcript = " ".join(s.text.strip() for s in seg_list)
        total_dur = seg_list[-1].end - seg_list[0].start
        return {
            "chars_per_sec": len(transcript) / max(total_dur, 0.1),
            "words_per_sec": len(transcript.split()) / max(total_dur, 0.1),
            "total_speech_sec": total_dur,
            "transcript": transcript,
        }

    # Compute speaking rate from word timestamps
    transcript = " ".join(w.word.strip() for w in words)
    total_chars = sum(len(w.word.strip()) for w in words)
    total_words = len(words)

    # Speech duration: first word start to last word end
    speech_start = words[0].start
    speech_end = words[-1].end
    speech_dur = speech_end - speech_start

    # Also measure inter-word gaps to detect pauses
    gaps = []
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        gaps.append(gap)

    # Exclude long pauses (>0.5s) from speaking rate calculation
    active_dur = speech_dur
    for gap in gaps:
        if gap > 0.5:
            active_dur -= (gap - 0.15)  # keep 150ms as natural pause

    active_dur = max(active_dur, 0.5)

    return {
        "chars_per_sec": total_chars / active_dur,
        "words_per_sec": total_words / active_dur,
        "total_speech_sec": speech_dur,
        "active_speech_sec": active_dur,
        "transcript": transcript,
        "n_words": total_words,
        "avg_gap_ms": sum(gaps) / len(gaps) * 1000 if gaps else 0,
    }


def estimate_target_len(text, voice_cal, fps=25, padding=1.08):
    """Estimate target_len for a text chunk given voice calibration.

    Args:
        text: the text to synthesize
        voice_cal: dict from calibrate_voice()
        fps: audio frames per second (OmniVoice = 25)
        padding: multiplier for slight over-allocation (1.05-1.10)

    Returns:
        target_len in frames
    """
    n_chars = len(text.strip())
    duration_sec = n_chars / voice_cal["chars_per_sec"]
    target_len = int(duration_sec * fps * padding)
    return max(60, min(280, target_len))


if __name__ == "__main__":
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")

    voices = {
        "barth": "barth_ref.wav",
        "astarion": "astarion_ref.wav",
        "vesper": "vesper_ref.wav",
    }

    for name, path in voices.items():
        cal = calibrate_voice(path, whisper_model)
        print(f"{name}:")
        print(f"  chars/sec: {cal['chars_per_sec']:.1f}")
        print(f"  words/sec: {cal['words_per_sec']:.1f}")
        print(f"  speech: {cal['total_speech_sec']:.1f}s "
              f"(active: {cal.get('active_speech_sec', 0):.1f}s)")
        print(f"  avg gap: {cal.get('avg_gap_ms', 0):.0f}ms")
        print(f"  transcript: {cal['transcript'][:80]}")

        # Test: estimate frames for sample text
        test = "The old lighthouse keeper had not spoken to another human being in three years."
        tf = estimate_target_len(test, cal)
        print(f"  test '{test[:40]}...' -> {tf} frames ({tf/25:.1f}s)")
        print()
