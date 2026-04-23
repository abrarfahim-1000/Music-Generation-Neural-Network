import os
from pathlib import Path
import torch

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_midi"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "train_test_split"

# Dataset-Specific Paths
MAESTRO_DIR = RAW_DATA_DIR / "maestro"
LAKH_DIR = RAW_DATA_DIR / "clean_midi"  # Lakh data extracted as clean_midi
GROOVE_DIR = RAW_DATA_DIR / "groove"

MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"

# Output Paths
OUTPUT_DIR = BASE_DIR / "outputs"
GENERATED_MIDI_DIR = OUTPUT_DIR / "generated_midis"
PLOTS_DIR = OUTPUT_DIR / "plots"
SURVEY_DIR = OUTPUT_DIR / "survey_results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Create directories if they don't exist
for path in [PROCESSED_DATA_DIR, SPLIT_DATA_DIR, GENERATED_MIDI_DIR, PLOTS_DIR, SURVEY_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Hardware Configuration
# Native Intel support in PyTorch 2.9 (use 'xpu')
DEVICE = (
    "xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# Preprocessing Hyperparameters
FS = 16            # Sampling frequency (steps per bar)
SEQ_LEN = 64       # Number of steps per sequence segment
PIANO_ROLL_FS = 16 # Sampling rate for piano roll conversion

# Task 1: LSTM Autoencoder Hyperparameters
AE_CONFIG = {
    "hidden_size": 256,
    "latent_dim": 128,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "seq_len": SEQ_LEN
}

# Task 2: VAE Hyperparameters
VAE_CONFIG = {
    "hidden_size": 256,
    "latent_dim": 128,
    "beta": 1.0,  # KL Divergence weight
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "seq_len": SEQ_LEN
}

# Task 3: Transformer Hyperparameters (Conservative for CPU/XPU)
TRANSFORMER_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "max_seq_len": 256,
    "batch_size": 64,
    "epochs": 50,
    "lr": 1e-4
}

# Task 4: RLHF Hyperparameters
RLHF_CONFIG = {
    "rl_steps": 200,
    "episodes_per_step": 16,
    "max_new_tokens": 256,
    "lr": 5e-6,
    "temperature": 1.0,
    "top_k": 32,
    "num_eval_samples": 10,
    "reward_weights": {
        "pitch": 0.4,
        "rhythm": 0.4,
        "anti_repetition": 0.2,
    },
}

# General Training
SEED = 42
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)
