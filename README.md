# Unsupervised Neural Network for Multi-Genre Music Generation

CSE425 / EEE474 — Spring 2026

This repository implements four required project tasks for symbolic MIDI generation:

1. LSTM Autoencoder (Task 1)
2. Variational Autoencoder (Task 2)
3. Transformer language model on event tokens (Task 3)
4. RLHF fine-tuning from survey-derived rewards (Task 4)

## What You Get Immediately

This repo already contains trained checkpoints, generated MIDI samples, plots, and survey outputs.

- Checkpoints: `checkpoints/`
- Generated samples: `outputs/generated_midis/`
- Survey and RLHF logs: `outputs/survey_results/`
- Existing plots + metric snapshots from prior runs: `outputs/plots and eval metrics/`
- Final report sources: `report/final_report.tex`, `report/references.bib`
- Final report PDF: `report/Unsupervised Neural Network for Multi Genre Music Generation.pdf`

If you only need to inspect results, you can skip training and go directly to these folders.

## Requirements

Python dependencies are listed in `requirements.txt`.

Install:

```bash
pip install -r requirements.txt
```

Main packages include:

- torch, torchvision
- numpy, pandas, scikit-learn
- pretty_midi, music21, mido
- matplotlib, seaborn, librosa
- tqdm, pyyaml, tensorboard, jupyterlab

Hardware/device selection is automatic in `src/config.py`:

- `xpu` if Intel XPU is available
- else `cuda` if NVIDIA CUDA is available
- else `cpu`

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Git Bash (Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

You need **at least one** of the following datasets:

| Dataset | Best for | Download Link |
|---------|----------|---------------|
| **MAESTRO** | Task 1 (Classical Piano, single genre) | https://magenta.tensorflow.org/datasets/maestro |
| **Lakh MIDI** | Task 2+ (multi-genre) | https://colinraffel.com/projects/lmd/ |
| **Groove MIDI** | Jazz/Drums rhythm diversity | https://magenta.tensorflow.org/datasets/groove |

> **Recommended:** Start with MAESTRO for Task 1 (clean, single-genre). Add Lakh MIDI for Task 2.

After downloading, place the datasets in the following directories:

- MAESTRO: `data/raw_midi/maestro/` (should contain `maestro-v3.0.0.csv` and MIDI files)
- Groove: `data/raw_midi/groove/`
- Lakh cleaned MIDI: `data/raw_midi/clean_midi/` (note: Lakh is expected under `clean_midi` not `lakh`)

### 4. Run preprocessing

```bash
# MAESTRO
python src/preprocessing/midi_parser.py --dataset maestro

# Groove
python src/preprocessing/midi_parser.py --dataset groove

# Lakh (normal mode)
python src/preprocessing/midi_parser.py --dataset lakh

# Lakh (streaming mode, lower RAM)
python src/preprocessing/midi_parser.py --dataset lakh --streaming

# Process all three supported datasets (maestro, groove, lakh)
python src/preprocessing/midi_parser.py --dataset all
```

This generates split arrays such as:

- `data/processed/maestro_train.npy`
- `data/processed/lakh_train.npy`
- `data/processed/*_validation.npy`
- `data/processed/*_test.npy`

Note: the `maestro` branch requires `data/raw_midi/maestro/maestro-v3.0.0.csv`.

## Startup Guides

Use these documents for guided execution and verification:

- Startup/smoke-test guide: `startup_test_guide.md`
- Full project workflow and planning: `workflow.md`
- Verification trace for report values: `VALUE_VERIFICATION_GUIDE.md`
- Known issues and debugging notes: `BUG_REPORT.md`, `ISSUES.md`, `tips.md`

## Quick Start Paths

### A) Quick validation (smoke tests)

```bash
# Task 1
python src/training/train_ae.py --train_max_batches 30 --val_max_batches 10

# Task 2
python src/training/train_vae.py --train_max_batches 30 --val_max_batches 10

# Task 3
python src/training/train_transformer.py --train_max_batches 30 --val_max_batches 10
```

### B) Full run from scratch

1) Preprocess data

```bash
python src/preprocessing/midi_parser.py --dataset all
```

2) Train Task 1/2/3

```bash
python src/training/train_ae.py
python src/training/train_vae.py
python src/training/train_transformer.py
```

3) Generate baseline and model samples

```bash
python src/generation/generate_music.py --model baseline_random --num_samples 5
python src/generation/generate_music.py --model baseline_markov --dataset maestro --num_samples 5

python src/generation/sample_latent.py --model ae --num_samples 5
python src/generation/sample_latent.py --model vae --num_samples 8
python src/generation/sample_latent.py --model vae --interpolate --interp_steps 8

python src/generation/generate_music.py --model transformer --num_samples 10 --max_new_tokens 512 --genre lakh
```

4) Build survey files, then run RLHF

```bash
# Optional: create per-file descriptive metadata for raters/reporting
python survey/describe_midi.py

# Required for RLHF reward input (writes outputs/survey_results/human_scores.csv)
python survey/generate_survey.py --n_participants 10 --seed 42

python src/training/train_rlhf.py --rl_steps 200 --episodes_per_step 16 --genre lakh
```

## Training Scripts and Outputs

### Task 1: Autoencoder

Command:

```bash
python src/training/train_ae.py
```

Outputs:

- `checkpoints/latest_ae.pt`
- `outputs/plots and eval metrics/task1_loss.png` (new runs)

### Task 2: VAE

Command:

```bash
python src/training/train_vae.py
```

Notes:

- Current script trains on Lakh split files by default.
- Hyperparameters come from `src/config.py` and optional CLI flags (`--beta`, `--batch_size`, `--lr`).

Outputs:

- `checkpoints/latest_vae.pt`
- `outputs/plots and eval metrics/task2_vae_loss.png` (new runs)

### Task 3: Transformer

Command:

```bash
python src/training/train_transformer.py
```

Notes:

- Current script trains on Lakh by default.
- Event-token pipeline is used internally.

Outputs:

- `checkpoints/latest_transformer.pt`
- `outputs/plots and eval metrics/task3_transformer_curves.png` (new runs)
- `outputs/plots and eval metrics/task3_perplexity_summary.json` (new runs)

### Task 4: RLHF

Command:

```bash
python src/training/train_rlhf.py --rl_steps 200 --episodes_per_step 16 --genre lakh
```

Default survey source:

- `outputs/survey_results/human_scores.csv`

Outputs:

- `checkpoints/latest_rlhf.pt`
- `checkpoints/best_rlhf.pt`
- `outputs/survey_results/task4_rlhf_training_log.csv`
- `outputs/survey_results/task4_rlhf_summary.json`
- `outputs/plots and eval metrics/task4_rlhf_curves.png` (new runs)

## Generation Commands

```bash
# AE samples
python src/generation/sample_latent.py --model ae --num_samples 5

# VAE samples
python src/generation/sample_latent.py --model vae --num_samples 8

# VAE interpolation
python src/generation/sample_latent.py --model vae --interpolate --interp_steps 8

# Transformer samples
python src/generation/generate_music.py --model transformer --num_samples 10 --max_new_tokens 512 --genre lakh

# Baselines
python src/generation/generate_music.py --model baseline_random --num_samples 5
python src/generation/generate_music.py --model baseline_markov --dataset maestro --num_samples 5
```

Generated files are written to `outputs/generated_midis/`.

## Evaluation Commands

```bash
# Evaluate every MIDI file in outputs/generated_midis/
python src/evaluation/metrics.py

# Aggregate by model
python src/evaluation/metrics.py --all

# Task 4 before vs after comparison
python src/evaluation/metrics.py --compare_rlhf
```

Default code outputs:

- `outputs/plots and eval metrics/evaluation_results.csv`
- `outputs/plots and eval metrics/all_models_comparison.csv`
- `outputs/plots and eval metrics/task4_comparison.csv` (created when both `task3_sample_*.mid` and `task4_after_*.mid` exist)

Compatibility note:

- `src/evaluation/metrics.py` also mirrors these CSVs to legacy paths in `outputs/generated_midis/`.

Existing snapshot copies from previous runs are currently available in:

- `outputs/plots and eval metrics/`

## Where Existing Results Are Right Now

### Pretrained checkpoints

- `checkpoints/latest_ae.pt`
- `checkpoints/latest_vae.pt`
- `checkpoints/latest_transformer.pt`
- `checkpoints/latest_rlhf.pt`
- `checkpoints/best_rlhf.pt`

### Generated MIDI examples

- Baselines: `outputs/generated_midis/baseline_random_*.mid`, `outputs/generated_midis/baseline_markov_*.mid`
- Task 1: `outputs/generated_midis/task1_sample_*.mid`
- Task 2: `outputs/generated_midis/task2_sample_*.mid`, `outputs/generated_midis/task2_interp_*.mid`
- Task 3: `outputs/generated_midis/task3_sample_*.mid`
- Task 4: `outputs/generated_midis/task4_after_*.mid`

### Plot and metric artifacts

- `outputs/plots and eval metrics/task1_loss.png`
- `outputs/plots and eval metrics/task2_vae_loss.png`
- `outputs/plots and eval metrics/task3_transformer_curves.png`
- `outputs/plots and eval metrics/task4_rlhf_curves.png`
- `outputs/plots and eval metrics/qualitative_pianoroll_comparison.png`
- `outputs/plots and eval metrics/task3_perplexity_summary.json`
- `outputs/plots and eval metrics/evaluation_results.csv`
- `outputs/plots and eval metrics/all_models_comparison.csv`
- `outputs/plots and eval metrics/task4_comparison.csv`

### Survey / RLHF metadata

- `outputs/survey_results/human_scores.csv`
- `outputs/survey_results/human_scores_wide.csv`
- `outputs/survey_results/survey_summary.csv`
- `outputs/survey_results/midi_descriptions.csv`
- `outputs/survey_results/task4_rlhf_training_log.csv`
- `outputs/survey_results/task4_rlhf_summary.json`

## Project Structure (Current Workspace)

```text
.
├── checkpoints/
├── data/
│   ├── raw_midi/
│   │   ├── maestro/
│   │   ├── groove/
│   │   └── clean_midi/                # Lakh expected here
│   ├── processed/
│   └── train_test_split/
├── notebooks/
├── outputs/
│   ├── generated_midis/
│   ├── plots and eval metrics/
│   └── survey_results/
├── report/
├── src/
│   ├── config.py
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── generation/
│   └── evaluation/
├── survey/
├── startup_test_guide.md
├── workflow.md
└── VALUE_VERIFICATION_GUIDE.md
```

## Notes

- `src/models/diffusion.py` is intentionally a placeholder and not part of the four required tasks.
- Some older docs refer to `outputs/plots/`; current canonical plot/metric output folder is `outputs/plots and eval metrics/`.
