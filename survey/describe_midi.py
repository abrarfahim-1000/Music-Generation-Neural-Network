"""
survey/describe_midi.py

Generates a human-readable description of every MIDI file in
outputs/generated_midis/ — duration, note count, pitch range, rhythm
diversity, repetition ratio, and pitch histogram similarity vs. a
MAESTRO reference. Writes the results to:

    outputs/survey_results/midi_descriptions.csv

These descriptions are used to:
  1. Brief human raters before they listen and score each sample.
  2. Provide objective context for the survey report table.

Usage:
    python survey/describe_midi.py
    python survey/describe_midi.py --ref_file data/raw_midi/maestro/some_file.mid
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pretty_midi

# Make project root importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GENERATED_MIDI_DIR, MAESTRO_CSV, MAESTRO_DIR, SURVEY_DIR
from src.evaluation.pitch_histogram import pitch_histogram_similarity
from src.evaluation.rhythm_score import repetition_ratio, rhythm_diversity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_reference(ref_file: str | Path | None) -> pretty_midi.PrettyMIDI | None:
    """Loads the reference MIDI used for pitch histogram similarity."""
    if ref_file is not None:
        path = Path(ref_file)
        if path.exists():
            try:
                return pretty_midi.PrettyMIDI(str(path))
            except Exception as exc:
                print(f"[describe] Could not load ref_file {path}: {exc}")
        return None

    # Fall back to first MAESTRO training file.
    if not MAESTRO_CSV.exists():
        return None
    rows: list[dict] = []
    with MAESTRO_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "").strip().lower() == "train":
                rows.append(row)
                break
    if not rows:
        return None
    rel = rows[0].get("midi_filename", "")
    if not rel:
        return None
    p = MAESTRO_DIR / rel
    if not p.exists():
        return None
    try:
        return pretty_midi.PrettyMIDI(str(p))
    except Exception:
        return None


def _model_tag(filename: str) -> str:
    """Maps a filename to a human-readable model label."""
    name = filename.lower()
    if name.startswith("baseline_random"):
        return "Random Baseline"
    if name.startswith("baseline_markov"):
        return "Markov Baseline"
    if name.startswith("task1"):
        return "Task 1 — LSTM Autoencoder"
    if name.startswith("task2_interp"):
        return "Task 2 — VAE Interpolation"
    if name.startswith("task2"):
        return "Task 2 — VAE"
    if name.startswith("task3"):
        return "Task 3 — Transformer"
    if name.startswith("task4_before"):
        return "Task 4 — RLHF (before)"
    if name.startswith("task4_after"):
        return "Task 4 — RLHF (after)"
    if name.startswith("task4_iter"):
        return "Task 4 — RLHF (training)"
    return "Unknown"


def describe_midi(midi_path: Path, ref_pm: pretty_midi.PrettyMIDI | None) -> dict:
    """Returns a dict of descriptive statistics for one MIDI file."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        return {
            "filename": midi_path.name,
            "model": _model_tag(midi_path.name),
            "error": str(exc),
        }

    all_notes = [
        note
        for inst in pm.instruments
        if not inst.is_drum
        for note in inst.notes
    ]

    duration_s = round(pm.get_end_time(), 2)
    note_count = len(all_notes)
    pitches = [n.pitch for n in all_notes]
    pitch_min = min(pitches) if pitches else 0
    pitch_max = max(pitches) if pitches else 0
    pitch_range = pitch_max - pitch_min

    rhy_div = round(rhythm_diversity(pm), 4)
    rep_rat = round(repetition_ratio(pm), 4)

    pitch_sim: float | str = "N/A"
    if ref_pm is not None and all_notes:
        pitch_sim = round(pitch_histogram_similarity(pm, ref_pm), 4)

    return {
        "filename": midi_path.name,
        "model": _model_tag(midi_path.name),
        "duration_s": duration_s,
        "note_count": note_count,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_range,
        "rhythm_diversity": rhy_div,
        "repetition_ratio": rep_rat,
        "pitch_histogram_similarity": pitch_sim,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ref_file: str | Path | None = None, output_csv: Path | None = None) -> None:
    ref_pm = _load_reference(ref_file)
    if ref_pm is not None:
        print("[describe] Reference MIDI loaded for pitch histogram similarity.")
    else:
        print("[describe] No reference MIDI found — pitch_histogram_similarity will be N/A.")

    midi_files = sorted(GENERATED_MIDI_DIR.glob("*.mid"))
    if not midi_files:
        print(f"[describe] No MIDI files found in {GENERATED_MIDI_DIR}.")
        return

    rows = []
    for midi_path in midi_files:
        row = describe_midi(midi_path, ref_pm)
        rows.append(row)
        status = f"  {row['filename']}: {row.get('note_count', '?')} notes, {row.get('duration_s', '?')}s"
        print(status)

    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    if output_csv is None:
        output_csv = SURVEY_DIR / "midi_descriptions.csv"

    fieldnames = [
        "filename", "model", "duration_s", "note_count",
        "pitch_min", "pitch_max", "pitch_range",
        "rhythm_diversity", "repetition_ratio", "pitch_histogram_similarity",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[describe] Descriptions for {len(rows)} files saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe generated MIDI files for survey raters.")
    parser.add_argument("--ref_file", type=str, default=None,
                        help="Reference MIDI path for pitch histogram similarity.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: outputs/survey_results/midi_descriptions.csv).")
    args = parser.parse_args()
    main(ref_file=args.ref_file, output_csv=Path(args.output) if args.output else None)
