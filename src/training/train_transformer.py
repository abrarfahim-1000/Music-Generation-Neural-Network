from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR,
    DEVICE,
    FS,
    GENERATED_MIDI_DIR,
    PLOTS_DIR,
    TRANSFORMER_CONFIG,
    SEED,
    PROCESSED_DATA_DIR,
)
from src.models.transformer import MusicTransformer
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi
from src.preprocessing.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    piano_roll_batch_to_event_tokens,
    tokens_to_piano_roll,
)


torch.manual_seed(SEED)
def load_genre_tokens(genres: list[str], split: str, max_seq_len: int):
    """
    Loads processed piano-roll arrays and converts to fixed-length event token sequences.

    Returns:
      x_tokens: (N, max_seq_len)
      genre_ids: (N,)
      genre_to_id: mapping dict
    """

    token_batches = []
    genre_batches = []
    
    valid_genres = []
    for genre in genres:
        path = PROCESSED_DATA_DIR / f"{genre}_{split}.npy"
        if not path.exists():
            print(f"[TR] Missing {path}, skipping genre '{genre}'.")
            continue

        arr = np.load(path)
        if arr.size == 0:
            print(f"[TR] Empty data in {path}, skipping genre '{genre}'.")
            continue

        if arr.ndim != 3 or arr.shape[2] != 128:
            print(f"[TR] Unexpected shape {arr.shape} in {path}, skipping genre '{genre}'.")
            continue

        # Task 3 checklist asks for event-based tokenization.
        tokens = piano_roll_batch_to_event_tokens(arr, max_seq_len=max_seq_len)
        token_batches.append(tokens)
        valid_genres.append(genre)
        print(f"[TR] Loaded {path.name}: roll={arr.shape} -> event_tokens={tokens.shape}")

    if not token_batches:
        raise RuntimeError("No valid genre data loaded. Check processed .npy files.")

    genre_to_id = {g: i for i, g in enumerate(sorted(valid_genres))}

    for tokens, genre in zip(token_batches, valid_genres):
        gid = genre_to_id[genre]
        genre_vec = np.full((tokens.shape[0],), gid, dtype=np.int64)
        genre_batches.append(genre_vec)

    x_tokens = np.concatenate(token_batches, axis=0)
    genre_ids = np.concatenate(genre_batches, axis=0)
    return x_tokens, genre_ids, genre_to_id


def create_autoregressive_dataset(tokens: np.ndarray, genre_ids: np.ndarray):
    """
    Shift tokens by one for teacher forcing.

    input_ids: (N, T-1)
    target_ids:(N, T-1)
    """

    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]
    return input_ids, target_ids, genre_ids


def evaluate(
    model: MusicTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    max_batches: int | None = None,
):
    model.eval()
    total_loss = 0.0
    used = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y, g = batch
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            g = g.to(DEVICE, non_blocking=True)

            logits = model(x, genre_ids=g)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            total_loss += loss.item()
            used += 1

    avg_loss = total_loss / max(1, used)
    perplexity = float(np.exp(avg_loss))
    return avg_loss, perplexity


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, k=k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        scaled = torch.where(scaled < threshold, torch.full_like(scaled, -1e9), scaled)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def train_transformer(
    batch_size: int,
    lr: float,
    train_max_batches: int | None,
    val_max_batches: int | None,
):
    epochs = TRANSFORMER_CONFIG["epochs"]
    genres = ["lakh"]  # Task 3: multi-genre Transformer trains on Lakh MIDI
    print(f"Using device: {DEVICE}")

    x_train, g_train, train_genre_to_id = load_genre_tokens(
        genres=genres,
        split="train",
        max_seq_len=TRANSFORMER_CONFIG["max_seq_len"],
    )
    x_val, g_val, val_genre_to_id = load_genre_tokens(
        genres=genres,
        split="validation",
        max_seq_len=TRANSFORMER_CONFIG["max_seq_len"],
    )

    all_genres = sorted(set(train_genre_to_id.keys()) | set(val_genre_to_id.keys()))
    genre_to_id = {g: i for i, g in enumerate(all_genres)}

    def remap(ids: np.ndarray, local_map: dict[str, int]) -> np.ndarray:
        inv = {v: genre_to_id[k] for k, v in local_map.items()}
        return np.vectorize(inv.__getitem__)(ids)

    g_train = remap(g_train, train_genre_to_id)
    g_val = remap(g_val, val_genre_to_id)

    x_train_in, y_train, g_train = create_autoregressive_dataset(x_train, g_train)
    x_val_in, y_val, g_val = create_autoregressive_dataset(x_val, g_val)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_in).long(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(g_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_in).long(),
        torch.from_numpy(y_val).long(),
        torch.from_numpy(g_val).long(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MusicTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=TRANSFORMER_CONFIG["d_model"],
        nhead=TRANSFORMER_CONFIG["nhead"],
        num_layers=TRANSFORMER_CONFIG["num_layers"],
        max_seq_len=TRANSFORMER_CONFIG["max_seq_len"],
        num_genres=max(1, len(genre_to_id)),
        dropout=0.1,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    best_val_loss = float("inf")
    start_epoch = 0
    resume_path = CHECKPOINT_DIR / "latest_transformer.pt"

    if resume_path.exists():
        payload = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        saved_genre_to_id = payload.get("genre_to_id", {})
        if saved_genre_to_id and saved_genre_to_id != genre_to_id:
            raise RuntimeError(
                "Resume checkpoint genre mapping does not match current data mapping. "
                "Delete latest_transformer.pt to start fresh with the new genre setup."
            )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        start_epoch = int(payload.get("epoch_completed", -1)) + 1
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        train_losses = list(payload.get("train_losses", []))
        val_losses = list(payload.get("val_losses", []))
        train_ppls = list(payload.get("train_ppls", []))
        val_ppls = list(payload.get("val_ppls", []))
        print(
            f"[TR] Resumed from {resume_path} at epoch {start_epoch + 1} "
            f"(best_val_loss={best_val_loss:.6f})"
        )

    end_epoch = start_epoch + epochs

    print(
        f"[TR] Starting Task 3 training from epoch {start_epoch + 1} to {end_epoch} "
        f"(genres={all_genres}, train_batches={len(train_loader)}, val_batches={len(val_loader)})"
    )

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_train = 0.0
        used_train = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Train]")):
            if train_max_batches is not None and batch_idx >= train_max_batches:
                break

            x, y, g = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            g = g.to(DEVICE)

            optimizer.zero_grad()

            # Algorithm 3: p_theta(x_t | x_<t) under causal mask and autoregressive CE.
            logits = model(x, genre_ids=g)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            total_train += loss.item()
            used_train += 1

        avg_train_loss = total_train / max(1, used_train)
        train_ppl = float(np.exp(avg_train_loss))

        avg_val_loss, val_ppl = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            max_batches=val_max_batches,
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        print(
            f"[TR] Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.6f}, train_ppl={train_ppl:.3f} | "
            f"val_loss={avg_val_loss:.6f}, val_ppl={val_ppl:.3f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Single checkpoint save with all training history
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_completed": epoch,
                "best_val_loss": best_val_loss,
                "genre_to_id": genre_to_id,
                "vocab_size": VOCAB_SIZE,
                "tokenization": "event",
                "config": {
                    "d_model": TRANSFORMER_CONFIG["d_model"],
                    "nhead": TRANSFORMER_CONFIG["nhead"],
                    "num_layers": TRANSFORMER_CONFIG["num_layers"],
                    "max_seq_len": TRANSFORMER_CONFIG["max_seq_len"],
                },
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_ppls": train_ppls,
                "val_ppls": val_ppls,
            },
            resume_path,
        )

    # Plot cumulative losses/perplexities across all epochs trained so far
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    total_epochs_trained = len(train_losses)
    epoch_range = list(range(total_epochs_trained))

    axes[0].plot(epoch_range, train_losses, label="Train CE")
    axes[0].plot(epoch_range, val_losses, label="Val CE")
    axes[0].set_title(f"Task 3: Transformer Cross-Entropy (Cumulative: {total_epochs_trained} epochs)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch_range, train_ppls, label="Train PPL")
    axes[1].plot(epoch_range, val_ppls, label="Val PPL")
    axes[1].set_title(f"Task 3: Perplexity (Cumulative: {total_epochs_trained} epochs)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "task3_transformer_curves.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[TR] Saved cumulative plot ({total_epochs_trained} epochs) to {plot_path}")

    # Cumulative summary
    summary_path = PLOTS_DIR / "task3_perplexity_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_epochs_trained": total_epochs_trained,
                "best_val_loss": best_val_loss,
                "best_val_perplexity": float(np.exp(best_val_loss)),
                "last_val_loss": val_losses[-1] if val_losses else None,
                "last_val_perplexity": val_ppls[-1] if val_ppls else None,
                "epoch_range": epoch_range,
            },
            f,
            indent=2,
        )
    print(f"[TR] Saved cumulative summary ({total_epochs_trained} epochs total) to {summary_path}")
    print(f"[TR] Training complete. Use 'python src/generation/generate_music.py --model transformer' to generate samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=TRANSFORMER_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRANSFORMER_CONFIG["lr"])
    parser.add_argument("--train_max_batches", type=int, default=None, help="Smoke test: cap train batches")
    parser.add_argument("--val_max_batches", type=int, default=None, help="Smoke test: cap val batches")
    args = parser.parse_args()

    train_transformer(
        batch_size=args.batch_size,
        lr=args.lr,
        train_max_batches=args.train_max_batches,
        val_max_batches=args.val_max_batches,
    )
