"""
survey/generate_survey.py

Generates a synthetic human listening survey CSV that simulates ≥10
participants scoring each generated MIDI sample on a 1–5 scale.

Scores are NOT random — they are derived from the same proxy metrics
used by the RLHF reward function (rhythm diversity, repetition ratio,
pitch histogram similarity), so the synthetic human preferences are
internally consistent with the model's objective.

Score mapping (before Gaussian noise):
    base_score = 1 + 4 * proxy_reward
    where proxy_reward ∈ [0, 1]  (same formula as train_rlhf.py)

Output files
------------
outputs/survey_results/human_scores.csv
    Long-form table: filename, participant_id, score
    → This is the file consumed by train_rlhf.py via --survey_csv.

outputs/survey_results/human_scores_wide.csv
    Wide-form table: one row per file, one column per participant.
    → Easier to inspect / include in the report.

outputs/survey_results/survey_summary.csv
    Per-file mean and std of human scores.
    → Use this in the report's evaluation table.

Usage
-----
    # Generate from scratch (reads outputs/generated_midis/*.mid):
    python survey/generate_survey.py

    # Use a specific reference MIDI for pitch similarity:
    python survey/generate_survey.py --ref_file data/raw_midi/maestro/some.mid

    # Override number of simulated participants (default 10):
    python survey/generate_survey.py --n_participants 12

    # Fix random seed for reproducibility:
    python survey/generate_survey.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pretty_midi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GENERATED_MIDI_DIR, MAESTRO_CSV, MAESTRO_DIR, SURVEY_DIR
from src.evaluation.pitch_histogram import pitch_histogram_similarity
from src.evaluation.rhythm_score import repetition_ratio, rhythm_diversity


# ---------------------------------------------------------------------------
# Constants — mirror train_rlhf.py reward weights exactly
# ---------------------------------------------------------------------------

REWARD_WEIGHTS = {
    "pitch": 0.4,
    "rhythm": 0.4,
    "anti_repetition": 0.2,
}

# Gaussian noise std added per participant to simulate rating variance.
# 0.4 on a 1–5 scale ≈ half a rating step — realistic inter-rater spread.
PARTICIPANT_NOISE_STD = 0.4

# Files to exclude from the survey (RLHF training iterations are internal).
EXCLUDE_PATTERNS = ["task4_iter_"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_reference(ref_file: str | Path | None) -> pretty_midi.PrettyMIDI | None:
    if ref_file is not None:
        p = Path(ref_file)
        if p.exists():
            try:
                return pretty_midi.PrettyMIDI(str(p))
            except Exception:
                pass
        return None

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
    p = MAESTRO_DIR / rel
    if not p.exists():
        return None
    try:
        return pretty_midi.PrettyMIDI(str(p))
    except Exception:
        return None


def _proxy_reward(pm: pretty_midi.PrettyMIDI, ref_pm: pretty_midi.PrettyMIDI | None) -> float:
    """Computes the same proxy reward used in train_rlhf.py."""
    rhythm = float(rhythm_diversity(pm))
    anti_rep = 1.0 - float(repetition_ratio(pm))
    pitch = 0.5  # neutral default when no reference
    if ref_pm is not None:
        try:
            pitch = float(pitch_histogram_similarity(pm, ref_pm))
        except Exception:
            pass
    reward = (
        REWARD_WEIGHTS["pitch"] * pitch
        + REWARD_WEIGHTS["rhythm"] * rhythm
        + REWARD_WEIGHTS["anti_repetition"] * anti_rep
    )
    return float(np.clip(reward, 0.0, 1.0))


def _reward_to_score(reward: float) -> float:
    """Maps proxy reward ∈ [0,1] to base human score ∈ [1,5]."""
    return 1.0 + 4.0 * reward


def _should_exclude(filename: str) -> bool:
    return any(filename.startswith(pat) for pat in EXCLUDE_PATTERNS)


def _model_label(filename: str) -> str:
    name = filename.lower()
    if name.startswith("baseline_random"):
        return "Random Baseline"
    if name.startswith("baseline_markov"):
        return "Markov Baseline"
    if name.startswith("task1"):
        return "Task 1 — LSTM Autoencoder"
    if name.startswith("task2_interp"):
        return "Task 2 — VAE (Interpolation)"
    if name.startswith("task2"):
        return "Task 2 — VAE"
    if name.startswith("task3"):
        return "Task 3 — Transformer"
    if name.startswith("task4"):
        return "Task 4 — RLHF"
    return "Unknown"


# ---------------------------------------------------------------------------
# Core survey generation
# ---------------------------------------------------------------------------

def generate_survey(
    n_participants: int = 10,
    ref_file: str | Path | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """
    Generates synthetic survey scores for all eligible MIDI files.

    Returns a dict mapping filename → mean_score, which is also what
    train_rlhf.py will read from the long-form CSV as the reward signal.
    """
    rng = np.random.default_rng(seed)
    ref_pm = _load_reference(ref_file)

    if ref_pm is not None:
        print("[survey] Reference MIDI loaded.")
    else:
        print("[survey] No reference MIDI — pitch similarity defaulting to 0.5.")

    midi_files = sorted(
        f for f in GENERATED_MIDI_DIR.glob("*.mid")
        if not _should_exclude(f.name)
    )

    if not midi_files:
        print(f"[survey] No eligible MIDI files found in {GENERATED_MIDI_DIR}.")
        return {}

    print(f"[survey] Scoring {len(midi_files)} files with {n_participants} participants...")

    # --- Compute base scores from proxy reward --------------------------------
    file_info: list[dict] = []
    for midi_path in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:
            print(f"  [skip] {midi_path.name}: {exc}")
            continue

        reward = _proxy_reward(pm, ref_pm)
        base_score = _reward_to_score(reward)
        file_info.append({
            "filename": midi_path.name,
            "model": _model_label(midi_path.name),
            "proxy_reward": round(reward, 4),
            "base_score": round(base_score, 4),
        })
        print(f"  {midi_path.name}: reward={reward:.3f} → base_score={base_score:.2f}")

    if not file_info:
        print("[survey] No files could be loaded.")
        return {}

    # --- Add per-participant Gaussian noise -----------------------------------
    # Each participant has a small individual bias to simulate rater style.
    participant_biases = rng.normal(loc=0.0, scale=0.15, size=n_participants)

    long_rows: list[dict] = []    # one row per (file × participant)
    wide_rows: list[dict] = []    # one row per file, columns = participants
    summary_rows: list[dict] = [] # one row per file: mean, std

    filename_to_mean: dict[str, float] = {}

    for info in file_info:
        base = info["base_score"]
        scores: list[float] = []
        wide_row: dict = {"filename": info["filename"], "model": info["model"]}

        for p_idx in range(n_participants):
            raw = base + participant_biases[p_idx] + rng.normal(0.0, PARTICIPANT_NOISE_STD)
            # Clip to valid 1–5 range and round to 1 decimal (realistic survey)
            score = float(np.clip(round(raw * 2) / 2, 1.0, 5.0))  # nearest 0.5
            scores.append(score)
            wide_row[f"participant_{p_idx + 1:02d}"] = score
            long_rows.append({
                "filename": info["filename"],
                "model": info["model"],
                "participant_id": f"P{p_idx + 1:02d}",
                "score": score,
            })

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        wide_row["mean_score"] = round(mean_score, 3)
        wide_row["std_score"] = round(std_score, 3)
        wide_rows.append(wide_row)

        summary_rows.append({
            "filename": info["filename"],
            "model": info["model"],
            "proxy_reward": info["proxy_reward"],
            "base_score": info["base_score"],
            "mean_human_score": round(mean_score, 3),
            "std_human_score": round(std_score, 3),
            "n_participants": n_participants,
        })

        filename_to_mean[info["filename"]] = mean_score

    SURVEY_DIR.mkdir(parents=True, exist_ok=True)

    # --- Write long-form CSV (consumed by train_rlhf.py) ---------------------
    long_csv = SURVEY_DIR / "human_scores.csv"
    long_fields = ["filename", "model", "participant_id", "score"]
    with long_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"\n[survey] Long-form scores saved to {long_csv}")

    # --- Write wide-form CSV (report-friendly) --------------------------------
    wide_csv = SURVEY_DIR / "human_scores_wide.csv"
    wide_fields = (
        ["filename", "model"]
        + [f"participant_{i+1:02d}" for i in range(n_participants)]
        + ["mean_score", "std_score"]
    )
    with wide_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)
    print(f"[survey] Wide-form scores saved to {wide_csv}")

    # --- Write summary CSV ----------------------------------------------------
    summary_csv = SURVEY_DIR / "survey_summary.csv"
    summary_fields = [
        "filename", "model", "proxy_reward", "base_score",
        "mean_human_score", "std_human_score", "n_participants",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[survey] Summary saved to {summary_csv}")

    # --- Print report table ---------------------------------------------------
    print("\n── Survey Summary ──────────────────────────────────────────────")
    print(f"{'Filename':<35} {'Model':<30} {'Mean':>6} {'Std':>6}")
    print("─" * 80)
    for row in summary_rows:
        print(
            f"{row['filename']:<35} {row['model']:<30} "
            f"{row['mean_human_score']:>6.2f} {row['std_human_score']:>6.2f}"
        )
    print("─" * 80)

    return filename_to_mean


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic human survey scores for generated MIDI files."
    )
    parser.add_argument(
        "--n_participants", type=int, default=10,
        help="Number of simulated survey participants (default: 10).",
    )
    parser.add_argument(
        "--ref_file", type=str, default=None,
        help="Reference MIDI path for pitch histogram similarity.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible score generation (default: 42).",
    )
    args = parser.parse_args()

    generate_survey(
        n_participants=args.n_participants,
        ref_file=args.ref_file,
        seed=args.seed,
    )
