"""
Synthesizes a small labeled ASR test set: (audio, reference_text, intent_label)
triples, using pyttsx3 (the same offline TTS engine the assistant already
uses for spoken confirmations) to generate audio for a sample of command
utterances.

IMPORTANT CAVEAT for the report: TTS-synthesized audio is cleaner and more
consistent than real human speech (no accent variation, background noise,
mic quality issues, mumbling, filler words, etc.), so WER measured on it is
a *best-case* / sanity-check number, not a realistic estimate of real-world
ASR accuracy. Say so explicitly in the report. For stronger results,
supplement or replace this with real recordings — evaluate_asr.py works
with any manifest CSV of (audio_path, reference_text, intent_label), so
recordings you make yourself (or collect from a few classmates, for speaker
variety) can be evaluated with exactly the same script.

Usage:
    python generate_asr_testset.py [--n-samples 80] [--seed 42]

Writes: asr_audio/*.wav
        data/asr_test_manifest.csv   (audio_path, reference_text, intent_label)
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import _GENERATORS, LABELS  # reuse the same templates/labels

AUDIO_DIR = Path(__file__).parent / "asr_audio"
MANIFEST_PATH = Path(__file__).parent / "data" / "asr_test_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples", type=int, default=80,
        help="Total number of audio samples to synthesize (spread evenly across intents)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pyttsx3
    except ImportError:
        raise SystemExit("pyttsx3 not installed. Install with: pip install pyttsx3")

    random.seed(args.seed)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    per_class = max(1, args.n_samples // len(LABELS))
    rows: list[tuple[str, str]] = []
    for label in LABELS:
        gen_fn = _GENERATORS[label]
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_class and attempts < per_class * 20:
            text = gen_fn()
            attempts += 1
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, label))
    random.shuffle(rows)

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    manifest_rows = []
    print(f"Synthesizing {len(rows)} audio samples via pyttsx3...")
    for i, (text, label) in enumerate(rows):
        wav_path = AUDIO_DIR / f"sample_{i:04d}.wav"
        engine.save_to_file(text, str(wav_path))
        manifest_rows.append((str(wav_path), text, label))
        if (i + 1) % 10 == 0:
            print(f"  queued {i + 1}/{len(rows)}")
    engine.runAndWait()  # actually writes all queued files to disk

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "reference_text", "intent_label"])
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} audio files -> {AUDIO_DIR}")
    print(f"Manifest -> {MANIFEST_PATH}")
    print(
        "\nNOTE: this is synthetic TTS audio — clean and consistent, unlike "
        "real speech. Treat resulting WER as a best-case sanity check, and "
        "say so explicitly in your report. For stronger results, record "
        "real speech samples for the same utterances (or additional ones) "
        "and point evaluate_asr.py at a manifest CSV with the same three "
        "columns (audio_path, reference_text, intent_label)."
    )


if __name__ == "__main__":
    main()