"""
Evaluates the assistant's Whisper transcription stage against a labeled ASR
test manifest, computing Word Error Rate (WER), Character Error Rate (CER)
via jiwer — and, as an end-to-end metric, how much intent-classification
accuracy drops when real ASR output (not clean ground-truth text) is fed
into the intent parser. That last comparison is arguably the most
reportable number here: it isolates how much of the pipeline's real-world
error comes from ASR mistakes versus parser mistakes.

Usage:
    python evaluate_asr.py [--manifest data/asr_test_manifest.csv] [--whisper-model base]

Reads:  a manifest CSV with columns: audio_path, reference_text, intent_label
        (generate_asr_testset.py produces one; a manifest of real recordings
        with the same three columns works identically — see its docstring)
Writes: results/asr_per_sample.csv
        results/asr_summary.json
        results/wer_distribution.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MANIFEST = Path(__file__).parent / "data" / "asr_test_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--whisper-model", default="base",
        help="Whisper model size — match your voice_assistant config.json "
             "whisper_model setting for a fair comparison to real app behavior",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Run generate_asr_testset.py first, or point --manifest at your "
            "own CSV with columns: audio_path, reference_text, intent_label"
        )

    try:
        import jiwer
        import whisper as whisper_pkg
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}\nInstall with: pip install jiwer openai-whisper")

    from voice_assistant.speech.transcriber import SpeechTranscriber
    from voice_assistant.intent.parser import IntentParser

    manifest_df = pd.read_csv(args.manifest)
    print(f"Evaluating ASR on {len(manifest_df)} samples using Whisper '{args.whisper_model}'...")

    transcriber = SpeechTranscriber(args.whisper_model)
    intent_parser = IntentParser()

    rows = []
    for _, row in manifest_df.iterrows():
        audio_path = row["audio_path"]
        reference = str(row["reference_text"])
        true_intent = str(row["intent_label"])

        # load_audio resamples to 16kHz mono float32 — the same format
        # AudioRecorder captures at, so this mirrors real pipeline input.
        audio = whisper_pkg.load_audio(audio_path)

        start = time.time()
        hypothesis = transcriber.transcribe(audio) or ""
        elapsed = time.time() - start

        sample_wer = jiwer.wer(reference, hypothesis) if hypothesis else 1.0
        sample_cer = jiwer.cer(reference, hypothesis) if hypothesis else 1.0

        # End-to-end: parse intent from what Whisper actually produced (not
        # the clean reference text) and compare to the ground-truth label —
        # this reflects real pipeline behavior, ASR errors included.
        predicted_intent = (
            intent_parser.parse(hypothesis).intent.name if hypothesis else "UNKNOWN"
        )
        intent_correct = predicted_intent == true_intent

        # For comparison: what the same parser produces on the clean
        # reference text, isolating "parser errors" from "ASR errors."
        clean_intent = intent_parser.parse(reference).intent.name
        clean_intent_correct = clean_intent == true_intent

        rows.append({
            "audio_path": audio_path,
            "reference_text": reference,
            "hypothesis_text": hypothesis,
            "true_intent": true_intent,
            "predicted_intent_from_asr": predicted_intent,
            "predicted_intent_from_clean_text": clean_intent,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
            "intent_correct_from_asr": intent_correct,
            "intent_correct_from_clean_text": clean_intent_correct,
            "transcribe_time_sec": round(elapsed, 3),
        })

    results_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_sample_path = RESULTS_DIR / "asr_per_sample.csv"
    results_df.to_csv(per_sample_path, index=False)

    # Corpus-level WER/CER (computed over the whole reference/hypothesis
    # corpus at once) is the standard way to report WER — averaging
    # per-sample WER weights every sentence equally regardless of length,
    # which can be misleading with short utterances like these commands.
    all_refs = results_df["reference_text"].tolist()
    all_hyps = results_df["hypothesis_text"].tolist()
    corpus_wer = jiwer.wer(all_refs, all_hyps)
    corpus_cer = jiwer.cer(all_refs, all_hyps)

    mean_sample_wer = results_df["wer"].mean()
    mean_sample_cer = results_df["cer"].mean()

    asr_intent_acc = results_df["intent_correct_from_asr"].mean()
    clean_intent_acc = results_df["intent_correct_from_clean_text"].mean()

    avg_latency = results_df["transcribe_time_sec"].mean()

    summary = {
        "n_samples": len(results_df),
        "whisper_model": args.whisper_model,
        "corpus_wer": round(corpus_wer, 4),
        "corpus_cer": round(corpus_cer, 4),
        "mean_per_sample_wer": round(mean_sample_wer, 4),
        "mean_per_sample_cer": round(mean_sample_cer, 4),
        "intent_accuracy_from_asr_output": round(asr_intent_acc, 4),
        "intent_accuracy_from_clean_text": round(clean_intent_acc, 4),
        "intent_accuracy_drop_due_to_asr": round(clean_intent_acc - asr_intent_acc, 4),
        "avg_transcription_latency_sec": round(avg_latency, 4),
    }

    summary_path = RESULTS_DIR / "asr_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ASR Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nPer-sample results -> {per_sample_path}")
    print(f"Summary -> {summary_path}")

    # Histogram of per-sample WER — shows error spread, not just the mean.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(results_df["wer"], bins=20, color="#4c72b0", edgecolor="black")
    ax.set_xlabel("Word Error Rate (per sample)")
    ax.set_ylabel("Count")
    ax.set_title(f"WER Distribution — Whisper '{args.whisper_model}' (n={len(results_df)})")
    plt.tight_layout()
    hist_path = RESULTS_DIR / "wer_distribution.png"
    plt.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"WER distribution plot -> {hist_path}")

    worst = results_df.sort_values("wer", ascending=False).head(5)
    print("\nWorst 5 samples by WER (useful for qualitative error analysis in the report):")
    for _, r in worst.iterrows():
        print(f"  WER={r['wer']:.2f}  ref='{r['reference_text']}'  hyp='{r['hypothesis_text']}'")


if __name__ == "__main__":
    main()