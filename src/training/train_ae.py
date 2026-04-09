import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, CHECKPOINT_DIR, PLOTS_DIR, DEVICE, AE_CONFIG, SEQ_LEN, SEED
from src.models.autoencoder import MusicAutoencoder


def train_ae(
    train_max_batches: int | None = None,
    val_max_batches: int | None = None,
):
    epochs = AE_CONFIG['epochs']
    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")

    train_path = PROCESSED_DATA_DIR / "maestro_train.npy"
    val_path = PROCESSED_DATA_DIR / "maestro_validation.npy"

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        return
    if not val_path.exists():
        print(f"Validation data not found at {val_path}")
        return

    print("Loading data...")
    x_train = np.load(train_path)
    x_val = np.load(val_path)

    if x_train.size == 0 or x_val.size == 0:
        print(
            "Processed data is empty. Re-run preprocessing and verify "
            f"{train_path} and {val_path} contain segments."
        )
        return

    train_ds = TensorDataset(torch.from_numpy(x_train))
    val_ds = TensorDataset(torch.from_numpy(x_val))

    train_loader = DataLoader(train_ds, batch_size=AE_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=AE_CONFIG['batch_size'])

    if len(train_loader) == 0 or len(val_loader) == 0:
        print(
            "No training/validation batches available. Check processed split sizes "
            "and batch size configuration."
        )
        return

    model = MusicAutoencoder(
        input_size=128,
        hidden_size=AE_CONFIG['hidden_size'],
        latent_dim=AE_CONFIG['latent_dim'],
        seq_len=AE_CONFIG['seq_len']
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=AE_CONFIG['lr'])
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0
    resume_path = CHECKPOINT_DIR / "latest_ae.pt"

    if resume_path.exists():
        payload = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = AE_CONFIG["lr"]
        start_epoch = int(payload.get("epoch_completed", -1)) + 1
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        train_losses = list(payload.get("train_losses", []))
        val_losses = list(payload.get("val_losses", []))
        print(
            f"Resumed AE from {resume_path} at epoch {start_epoch + 1} "
            f"(best_val_loss={best_val_loss:.6f})"
        )

    end_epoch = start_epoch + epochs
    print(f"Starting AE training from epoch {start_epoch + 1} to {end_epoch}...")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_train_loss = 0.0
        train_batches_used = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Train]")):
            if train_max_batches is not None and batch_idx >= train_max_batches:
                break

            x = batch[0].to(DEVICE, dtype=torch.float32)
            optimizer.zero_grad()

            # Algorithm 1 (Task 1): z = f_phi(X), X_hat = g_theta(z)
            z = model.encode(x)
            x_hat = model.decode(z)

            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches_used += 1

        avg_train_loss = total_train_loss / max(1, train_batches_used)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batches_used = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Val]")):
                if val_max_batches is not None and batch_idx >= val_max_batches:
                    break

                x = batch[0].to(DEVICE, dtype=torch.float32)
                z = model.encode(x)
                x_hat = model.decode(z)
                loss = criterion(x_hat, x)
                total_val_loss += loss.item()
                val_batches_used += 1

        avg_val_loss = total_val_loss / max(1, val_batches_used)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_completed": epoch,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            resume_path,
        )

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"ae_epoch_{epoch+1}.pt")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Task 1: LSTM Autoencoder Training Loss')
    plt.savefig(PLOTS_DIR / "task1_loss.png")
    plt.close()
    print(f"Loss curve saved to {PLOTS_DIR / 'task1_loss.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_max_batches", type=int, default=None, help="Smoke test: cap number of train batches")
    parser.add_argument("--val_max_batches", type=int, default=None, help="Smoke test: cap number of val batches")
    args = parser.parse_args()

    train_ae(
        train_max_batches=args.train_max_batches,
        val_max_batches=args.val_max_batches,
    )
