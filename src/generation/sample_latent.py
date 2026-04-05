import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import CHECKPOINT_DIR, GENERATED_MIDI_DIR, PROCESSED_DATA_DIR, DEVICE, AE_CONFIG, VAE_CONFIG, FS
from src.models.autoencoder import MusicAutoencoder
from src.models.vae import MusicVAE
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi


def _compute_ae_latent_stats(model: MusicAutoencoder, batch_size: int = 256):
    """
    Runs all training segments through the encoder to compute empirical
    per-dimension mean and std of the latent space.

    Returns (z_mean, z_std) tensors on CPU, or (None, None) if data is missing.
    """
    train_path = PROCESSED_DATA_DIR / "maestro_train.npy"
    if not train_path.exists():
        print(f"[AE] Training data not found at {train_path}; falling back to N(0,1) sampling.")
        return None, None

    x_train = np.load(train_path)
    ds = TensorDataset(torch.from_numpy(x_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    z_chunks = []
    print("[AE] Computing empirical latent distribution from training data...")
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch.to(DEVICE, dtype=torch.float32))
            z_chunks.append(z.cpu())

    z_all = torch.cat(z_chunks, dim=0)
    z_mean = z_all.mean(dim=0)
    z_std = z_all.std(dim=0).clamp(min=1e-6)
    print(f"[AE] Empirical latent stats: mean={z_mean.mean():.4f}, std={z_std.mean():.4f}")
    return z_mean, z_std


def sample_ae(num_samples=5):
    print(f"Using device: {DEVICE}")

    model = MusicAutoencoder(
        input_size=128,
        hidden_size=AE_CONFIG['hidden_size'],
        latent_dim=AE_CONFIG['latent_dim'],
        seq_len=AE_CONFIG['seq_len']
    ).to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / "latest_ae.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    else:
        model.load_state_dict(payload)
    model.eval()

    # Sample from the empirical latent distribution (not N(0,1)).
    # A non-regularized AE does not learn a unit-Gaussian latent space.
    z_mean, z_std = _compute_ae_latent_stats(model)

    print(f"Generating {num_samples} samples from latent space...")

    with torch.no_grad():
        for i in range(num_samples):
            if z_mean is not None:
                z = (z_mean + z_std * torch.randn(AE_CONFIG['latent_dim'])).unsqueeze(0).to(DEVICE)
            else:
                z = torch.randn(1, AE_CONFIG['latent_dim']).to(DEVICE)

            x_hat = model.decode(z)  # (1, seq_len, 128)
            roll = x_hat.squeeze(0).cpu().numpy().T  # (128, seq_len)

            pm = piano_roll_to_pretty_midi(roll, fs=FS)
            output_path = GENERATED_MIDI_DIR / f"task1_sample_{i}.mid"
            pm.write(str(output_path))
            print(f"Saved {output_path}")

def sample_vae(num_samples: int = 8):
    print(f"Using device: {DEVICE}")

    model = MusicVAE(
        input_size=128,
        hidden_size=VAE_CONFIG["hidden_size"],
        latent_dim=VAE_CONFIG["latent_dim"],
        seq_len=VAE_CONFIG["seq_len"],
    ).to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / "latest_vae.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    else:
        model.load_state_dict(payload)
    model.eval()

    print(f"Generating {num_samples} samples from VAE latent space...")
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, VAE_CONFIG["latent_dim"]).to(DEVICE)
            x_hat = model.decode(z)  # (1, seq_len, 128)
            roll = x_hat.squeeze(0).cpu().numpy().T  # (128, seq_len)
            pm = piano_roll_to_pretty_midi(roll, fs=FS)
            output_path = GENERATED_MIDI_DIR / f"task2_sample_{i}.mid"
            pm.write(str(output_path))
            print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ae", "vae"], default="ae")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    num_samples = args.num_samples
    if num_samples is None:
        num_samples = 8 if args.model == "vae" else 5

    if args.model == "ae":
        sample_ae(num_samples=num_samples)
    else:
        sample_vae(num_samples=num_samples)
