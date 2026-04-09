import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    PLOTS_DIR,
    GENERATED_MIDI_DIR,
    DEVICE,
    VAE_CONFIG,
    FS,
    SEED,
)
from src.models.vae import MusicVAE
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi

torch.manual_seed(SEED)
def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mu, sigma^2) and N(0, 1).

    Returns a scalar mean over batch.
    """

    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_per_sample = -0.5 * torch.sum(
        1.0 + log_var - mu.pow(2) - log_var.exp(),
        dim=1,
    )
    return kl_per_sample.mean()


def load_genre_splits(genres: list[str], split: str) -> dict[str, np.ndarray]:
    """
    Loads {genre}_{split}.npy if present from processed data.
    Returns dict: genre -> array of shape (N, seq_len, 128).
    """

    out: dict[str, np.ndarray] = {}
    for g in genres:
        path = PROCESSED_DATA_DIR / f"{g}_{split}.npy"
        if not path.exists():
            print(f"[VAE] Missing {path}, skipping genre '{g}'.")
            continue
        # Keep stored dtype (typically uint8) to avoid large RAM spikes.
        # We'll cast to float32 per-batch during training.
        arr = np.load(path)
        if arr.size == 0:
            print(f"[VAE] Empty array in {path}, skipping genre '{g}'.")
            continue
        out[g] = arr
        print(f"[VAE] Loaded {g}_{split}.npy with shape {arr.shape}")
    return out


def concat_splits(split_by_genre: dict[str, np.ndarray]) -> np.ndarray:
    if not split_by_genre:
        raise RuntimeError("No genre splits were loaded. Check processed data files.")
    arrays = [split_by_genre[g] for g in sorted(split_by_genre.keys())]
    if len(arrays) == 1:
        # Avoid copying the whole array for the common single-genre smoke test.
        return arrays[0]
    return np.concatenate(arrays, axis=0)



def train_vae(
    beta: float,
    batch_size: int,
    lr: float,
    train_max_batches: int | None = None,
    val_max_batches: int | None = None,
):
    epochs = VAE_CONFIG["epochs"]
    genres = ["lakh"]  # Task 2: multi-genre VAE trains on Lakh MIDI only
    print(f"Using device: {DEVICE}")

    train_by_genre = load_genre_splits(genres, split="train")
    val_by_genre = load_genre_splits(genres, split="validation")

    x_train = concat_splits(train_by_genre)
    x_val = concat_splits(val_by_genre)

    train_ds = TensorDataset(torch.from_numpy(x_train))
    val_ds = TensorDataset(torch.from_numpy(x_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MusicVAE(
        input_size=128,
        hidden_size=VAE_CONFIG["hidden_size"],
        latent_dim=VAE_CONFIG["latent_dim"],
        seq_len=VAE_CONFIG["seq_len"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()

    train_total_losses: list[float] = []
    val_total_losses: list[float] = []
    train_recon_losses: list[float] = []
    val_recon_losses: list[float] = []
    train_kl_losses: list[float] = []
    val_kl_losses: list[float] = []

    best_val_total = float("inf")
    start_epoch = 0
    resume_path = CHECKPOINT_DIR / "latest_vae.pt"

    if resume_path.exists():
        payload = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        saved_genres = sorted(payload.get("genres", []))
        current_genres = sorted(genres)
        if saved_genres and saved_genres != current_genres:
            raise RuntimeError(
                "Resume checkpoint genres do not match current --genres. "
                "Delete latest_vae.pt to start fresh with a different genre set."
            )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        best_val_total = float(payload.get("best_val_total", best_val_total))
        train_total_losses = list(payload.get("train_total_losses", []))
        val_total_losses = list(payload.get("val_total_losses", []))
        train_recon_losses = list(payload.get("train_recon_losses", []))
        val_recon_losses = list(payload.get("val_recon_losses", []))
        train_kl_losses = list(payload.get("train_kl_losses", []))
        val_kl_losses = list(payload.get("val_kl_losses", []))
        print(
            f"[VAE] Resumed from {resume_path} at epoch {start_epoch + 1} "
            f"(best_val_total={best_val_total:.6f})"
        )

    end_epoch = start_epoch + epochs

    print(f"[VAE] Starting training from epoch {start_epoch + 1} to {end_epoch} on genres={genres}")
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_train = 0.0
        total_train_recon = 0.0
        total_train_kl = 0.0
        train_batches_used = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Train]")):
            x = batch[0].to(DEVICE, dtype=torch.float32)

            if train_max_batches is not None and batch_idx >= train_max_batches:
                break

            optimizer.zero_grad()

            # Algorithm 2 (Task 2): (mu, log_var) -> z -> X_hat
            mu, log_var = model.encode(x)
            z = model.reparameterize(mu, log_var)
            x_hat = model.decode(z)

            recon_loss = recon_criterion(x_hat, x)
            kl_loss = kl_divergence(mu, log_var)
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            total_train += loss.item()
            total_train_recon += recon_loss.item()
            total_train_kl += kl_loss.item()
            train_batches_used += 1

        avg_train_total = total_train / max(1, train_batches_used)
        avg_train_recon = total_train_recon / max(1, train_batches_used)
        avg_train_kl = total_train_kl / max(1, train_batches_used)

        model.eval()
        total_val = 0.0
        total_val_recon = 0.0
        total_val_kl = 0.0
        val_batches_used = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Val]")):
                x = batch[0].to(DEVICE, dtype=torch.float32)
                if val_max_batches is not None and batch_idx >= val_max_batches:
                    break

                mu, log_var = model.encode(x)
                z = model.reparameterize(mu, log_var)
                x_hat = model.decode(z)

                recon_loss = recon_criterion(x_hat, x)
                kl_loss = kl_divergence(mu, log_var)
                loss = recon_loss + beta * kl_loss

                total_val += loss.item()
                total_val_recon += recon_loss.item()
                total_val_kl += kl_loss.item()
                val_batches_used += 1

            avg_val_total = total_val / max(1, val_batches_used)
            avg_val_recon = total_val_recon / max(1, val_batches_used)
            avg_val_kl = total_val_kl / max(1, val_batches_used)

        print(
            f"[VAE] Epoch {epoch+1}: "
            f"train_total={avg_train_total:.6f}, train_recon={avg_train_recon:.6f}, train_kl={avg_train_kl:.6f} | "
            f"val_total={avg_val_total:.6f}, val_recon={avg_val_recon:.6f}, val_kl={avg_val_kl:.6f}"
        )

        train_total_losses.append(avg_train_total)
        val_total_losses.append(avg_val_total)
        train_recon_losses.append(avg_train_recon)
        val_recon_losses.append(avg_val_recon)
        train_kl_losses.append(avg_train_kl)
        val_kl_losses.append(avg_val_kl)

        if avg_val_total < best_val_total:
            best_val_total = avg_val_total

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_completed": epoch,
                "best_val_total": best_val_total,
                "train_total_losses": train_total_losses,
                "val_total_losses": val_total_losses,
                "train_recon_losses": train_recon_losses,
                "val_recon_losses": val_recon_losses,
                "train_kl_losses": train_kl_losses,
                "val_kl_losses": val_kl_losses,
                "genres": sorted(genres),
            },
            resume_path,
        )

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_total_losses, label="Train Total")
    plt.plot(val_total_losses, label="Val Total")
    plt.plot(train_recon_losses, label="Train Recon", alpha=0.7)
    plt.plot(val_recon_losses, label="Val Recon", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Task 2: VAE Loss Curves")
    plot_path = PLOTS_DIR / "task2_vae_loss.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[VAE] Loss curve saved to {plot_path}")
    print(f"[VAE] Training complete. Use 'python src/generation/sample_latent.py --model vae' to generate samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=VAE_CONFIG["beta"])
    parser.add_argument("--batch_size", type=int, default=VAE_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=VAE_CONFIG["lr"])
    parser.add_argument("--train_max_batches", type=int, default=None, help="Smoke test: cap number of train batches.")
    parser.add_argument("--val_max_batches", type=int, default=None, help="Smoke test: cap number of val batches.")
    args = parser.parse_args()

    train_vae(
        beta=args.beta,
        batch_size=args.batch_size,
        lr=args.lr,
        train_max_batches=args.train_max_batches,
        val_max_batches=args.val_max_batches,
    )

