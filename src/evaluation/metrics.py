import pretty_midi
import argparse
from pathlib import Path
import fnmatch
import sys

import pandas as pd

# Add src to path when run as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import GENERATED_MIDI_DIR, PLOTS_DIR, MAESTRO_CSV, MAESTRO_DIR
from src.evaluation.pitch_histogram import pitch_histogram_similarity
from src.evaluation.rhythm_score import rhythm_diversity, repetition_ratio


MODEL_PATTERNS = {
    "random_baseline": "baseline_random_*.mid",
    "markov_baseline": "baseline_markov_*.mid",
    "task1_ae": "task1_sample_*.mid",
    "task2_vae": "task2_sample_*.mid",
    "task3_transformer": "task3_sample_*.mid",
    "task4_rlhf_before": "task3_sample_*.mid",
    "task4_rlhf_after": "task4_after_*.mid",
}


def _write_csv_with_legacy_mirror(df: pd.DataFrame, path: Path, legacy_filename: str) -> None:
    """Write CSV to canonical path and mirror to legacy generated_midis path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    legacy_path = GENERATED_MIDI_DIR / legacy_filename
    if legacy_path != path:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(legacy_path, index=False)
        print(f"[metrics] Mirrored CSV to legacy path: {legacy_path}")


def default_reference_file() -> Path | None:
    if not MAESTRO_CSV.exists():
        return None
    df = pd.read_csv(MAESTRO_CSV)
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        return None
    return MAESTRO_DIR / train_df.iloc[0]["midi_filename"]

def compute_all_metrics(gen_midi_path, ref_midi_path=None):
    gen_pm = pretty_midi.PrettyMIDI(gen_midi_path)
    
    metrics = {
        "rhythm_diversity": rhythm_diversity(gen_pm),
        "repetition_ratio": repetition_ratio(gen_pm)
    }
    
    if ref_midi_path:
        ref_pm = pretty_midi.PrettyMIDI(ref_midi_path)
        metrics["pitch_histogram_similarity"] = pitch_histogram_similarity(gen_pm, ref_pm)
        
    return metrics


def evaluate_generated_midis(output_csv: Path | None = None, ref_file: str | Path | None = None):
    """
    Evaluates all .mid files in outputs/generated_midis and writes a CSV report.

    Args:
        output_csv: Destination CSV path. Defaults to outputs/plots and eval metrics/evaluation_results.csv.
        ref_file:   Reference MIDI for pitch histogram similarity. If None, falls back
                    to the first MAESTRO training file when the CSV is available.
                    Pass an explicit path for Task 2 / Task 3 evaluations that use
                    non-MAESTRO reference material.
    """

    if ref_file is None:
        ref_file = default_reference_file()
        if ref_file is not None:
            print(f"Using reference file: {ref_file}")
        else:
            print("[metrics] No ref_file provided and MAESTRO CSV not found. Skipping pitch similarity.")

    results = []
    for midi_path in sorted(GENERATED_MIDI_DIR.glob("*.mid")):
        filename = midi_path.name
        print(f"Evaluating {filename}...")
        metrics = compute_all_metrics(str(midi_path), str(ref_file) if ref_file else None)
        metrics["filename"] = filename
        results.append(metrics)

    if not results:
        print("No MIDI files found for evaluation.")
        return None

    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df.to_string())

    # Tag each row with model name based on filename pattern.
    results_df["model"] = results_df["filename"].apply(_infer_model_name)

    if output_csv is None:
        output_csv = PLOTS_DIR / "evaluation_results.csv"
    _write_csv_with_legacy_mirror(results_df, output_csv, "evaluation_results.csv")
    print(f"\nSaved results to {output_csv}")

    # Save grouped cross-model summary for rubric comparison.
    numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
    grouped = (
        results_df.groupby("model", dropna=False)[numeric_cols]
        .mean()
        .reset_index()
        .sort_values("model")
    )
    grouped_csv = PLOTS_DIR / "all_models_comparison.csv"
    _write_csv_with_legacy_mirror(grouped, grouped_csv, "all_models_comparison.csv")
    print(f"Saved grouped model comparison to {grouped_csv}")

    # Save Task 4 before-vs-after summary.
    task4_subset = results_df[results_df["model"].isin(["task3_transformer", "task4_rlhf_after"])].copy()
    if not task4_subset.empty:
        task4_numeric = task4_subset.select_dtypes(include="number").columns.tolist()
        task4_comparison = (
            task4_subset.groupby("model", dropna=False)[task4_numeric]
            .mean()
            .reset_index()
            .sort_values("model")
        )
        task4_csv = PLOTS_DIR / "task4_comparison.csv"
        _write_csv_with_legacy_mirror(task4_comparison, task4_csv, "task4_comparison.csv")
        print("\nTask 4 before-vs-after comparison:")
        print(task4_comparison.to_string(index=False))
        print(f"Saved Task 4 comparison to {task4_csv}")
    else:
        print("No Task 4 before/after files found; skipped task4_comparison.csv")

    return results_df


def _infer_model_name(filename: str) -> str:
    for model_name, pattern in MODEL_PATTERNS.items():
        if fnmatch.fnmatch(filename, pattern):
            return model_name
    return "unknown"


def collect_metrics_for_pattern(pattern: str, ref_file: Path | None) -> list[dict]:
    rows: list[dict] = []
    for midi_path in sorted(GENERATED_MIDI_DIR.glob(pattern)):
        metrics = compute_all_metrics(str(midi_path), str(ref_file) if ref_file else None)
        rows.append(
            {
                "filename": midi_path.name,
                "rhythm_diversity": float(metrics.get("rhythm_diversity", float("nan"))),
                "repetition_ratio": float(metrics.get("repetition_ratio", float("nan"))),
                "pitch_histogram_similarity": float(metrics.get("pitch_histogram_similarity", float("nan"))),
            }
        )
    return rows


def aggregate_all_models(output_csv: Path | None = None, ref_file: str | Path | None = None) -> pd.DataFrame:
    if ref_file is None:
        resolved_ref = default_reference_file()
    else:
        resolved_ref = Path(ref_file)

    per_file_rows = []
    for model_name, pattern in MODEL_PATTERNS.items():
        model_rows = collect_metrics_for_pattern(pattern, resolved_ref)
        for row in model_rows:
            row["model"] = model_name
        per_file_rows.extend(model_rows)

    if not per_file_rows:
        raise RuntimeError("No matching MIDI files found in outputs/generated_midis.")

    df = pd.DataFrame(per_file_rows)
    summary = (
        df.groupby("model", dropna=False)
        .agg(
            num_files=("filename", "count"),
            rhythm_diversity=("rhythm_diversity", "mean"),
            repetition_ratio=("repetition_ratio", "mean"),
            pitch_histogram_similarity=("pitch_histogram_similarity", "mean"),
        )
        .reset_index()
        .sort_values("model")
    )

    if output_csv is None:
        output_csv = PLOTS_DIR / "all_models_comparison.csv"
    _write_csv_with_legacy_mirror(summary, output_csv, "all_models_comparison.csv")
    return summary


def evaluate_group(midi_paths: list[Path], group_name: str, ref_file: Path | None) -> pd.DataFrame:
    rows = []
    for midi_path in midi_paths:
        metrics = compute_all_metrics(str(midi_path), str(ref_file) if ref_file else None)
        rows.append(
            {
                "group": group_name,
                "filename": midi_path.name,
                "rhythm_diversity": float(metrics.get("rhythm_diversity", float("nan"))),
                "repetition_ratio": float(metrics.get("repetition_ratio", float("nan"))),
                "pitch_histogram_similarity": float(metrics.get("pitch_histogram_similarity", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def compare_rlhf(output_csv: Path | None = None, ref_file: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    before_files = sorted(GENERATED_MIDI_DIR.glob("task3_sample_*.mid"))
    after_files = sorted(GENERATED_MIDI_DIR.glob("task4_after_*.mid"))

    if not before_files:
        raise RuntimeError("No files found for pattern task3_sample_*.mid")
    if not after_files:
        raise RuntimeError("No files found for pattern task4_after_*.mid")

    if ref_file is None:
        resolved_ref = default_reference_file()
    else:
        resolved_ref = Path(ref_file)

    before_df = evaluate_group(before_files, "before", resolved_ref)
    after_df = evaluate_group(after_files, "after", resolved_ref)

    all_df = pd.concat([before_df, after_df], ignore_index=True)
    summary = (
        all_df.groupby("group", dropna=False)
        .agg(
            num_files=("filename", "count"),
            rhythm_diversity=("rhythm_diversity", "mean"),
            repetition_ratio=("repetition_ratio", "mean"),
            pitch_histogram_similarity=("pitch_histogram_similarity", "mean"),
        )
        .reset_index()
        .sort_values("group")
    )

    if output_csv is None:
        output_csv = PLOTS_DIR / "task4_comparison.csv"
    _write_csv_with_legacy_mirror(summary, output_csv, "task4_comparison.csv")

    return all_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Aggregate metrics by model and save all-model comparison CSV.",
    )
    parser.add_argument(
        "--compare_rlhf",
        action="store_true",
        help="Compare Task 4 before-vs-after RLHF groups and save summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--ref_file",
        type=str,
        default=None,
        help="Optional reference MIDI path for pitch histogram similarity.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    if args.all and args.compare_rlhf:
        raise ValueError("Use either --all or --compare_rlhf, not both.")

    if args.all:
        table = aggregate_all_models(output_csv=output_path, ref_file=args.ref_file)
        print(table.to_string(index=False))
        print(f"\nSaved all-model comparison to {output_path or (PLOTS_DIR / 'all_models_comparison.csv')}")
    elif args.compare_rlhf:
        _, summary_df = compare_rlhf(output_csv=output_path, ref_file=args.ref_file)
        print(summary_df.to_string(index=False))
        print(f"\nSaved RLHF before/after comparison to {output_path or (PLOTS_DIR / 'task4_comparison.csv')}")
    else:
        evaluate_generated_midis(output_csv=output_path, ref_file=args.ref_file)
