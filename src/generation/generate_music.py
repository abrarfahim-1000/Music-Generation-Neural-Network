import numpy as np
from collections import defaultdict
from pathlib import Path
import sys
import argparse
import torch

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import CHECKPOINT_DIR, GENERATED_MIDI_DIR, PROCESSED_DATA_DIR, FS, DEVICE
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi
from src.preprocessing.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, VOCAB_SIZE, tokens_to_piano_roll
from src.models.transformer import MusicTransformer

def generate_random_music(num_steps=64, num_samples=5):
    print("Generating Random Baseline samples...")
    for i in range(num_samples):
        # Random binary matrix (128, num_steps)
        roll = (np.random.rand(128, num_steps) > 0.98).astype(np.float32)
        pm = piano_roll_to_pretty_midi(roll, fs=FS)
        output_path = GENERATED_MIDI_DIR / f"baseline_random_{i}.mid"
        pm.write(str(output_path))
        print(f"Saved {output_path}")

def _row_to_state(row: np.ndarray) -> tuple:
    """Active pitch indices only — keeps state space tractable (avoids O(2^128))."""
    return tuple(np.where(row > 0)[0])


def _state_to_row(state: tuple, num_pitches: int = 128) -> np.ndarray:
    row = np.zeros(num_pitches, dtype=np.float32)
    if state:
        row[list(state)] = 1.0
    return row


def train_markov_chain(dataset: str = "maestro"):
    train_data_path = PROCESSED_DATA_DIR / f"{dataset}_train.npy"
    if not train_data_path.exists():
        print(f"Training data not found at {train_data_path}")
        return None

    train_data = np.load(train_data_path)  # (N, seq_len, 128)
    counts: defaultdict = defaultdict(lambda: defaultdict(int))

    print("Training Markov Chain...")
    for seq in train_data[:2000]:
        for t in range(len(seq) - 1):
            counts[_row_to_state(seq[t])][_row_to_state(seq[t + 1])] += 1

    transitions = {
        curr: {nxt: c / sum(d.values()) for nxt, c in d.items()}
        for curr, d in counts.items()
    }
    return transitions


def generate_markov_music(transitions, num_steps=64, num_samples=5):
    if transitions is None:
        return

    print("Generating Markov Baseline samples...")
    all_keys = list(transitions.keys())

    for i in range(num_samples):
        curr = all_keys[np.random.randint(len(all_keys))]
        roll_seq = [_state_to_row(curr)]

        for _ in range(num_steps - 1):
            if curr in transitions:
                options = list(transitions[curr].keys())
                probs = list(transitions[curr].values())
                curr = options[np.random.choice(len(options), p=probs)]
            else:
                curr = all_keys[np.random.randint(len(all_keys))]
            roll_seq.append(_state_to_row(curr))

        roll = np.stack(roll_seq).T  # (128, num_steps)
        pm = piano_roll_to_pretty_midi(roll, fs=FS)
        output_path = GENERATED_MIDI_DIR / f"baseline_markov_{i}.mid"
        pm.write(str(output_path))
        print(f"Saved {output_path}")


def generate_transformer_music(num_samples=10, max_new_tokens=512, genre="maestro"):
    checkpoint_path = CHECKPOINT_DIR / "latest_transformer.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = payload.get("config", {})
    genre_to_id = payload.get("genre_to_id", {"maestro": 0})
    vocab_size = int(payload.get("vocab_size", VOCAB_SIZE))

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 3)),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
        num_genres=max(1, len(genre_to_id)),
        dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    genre_id = genre_to_id.get(genre, 0)
    print(f"Using device: {DEVICE}")
    print(f"Generating {num_samples} Task 3 transformer samples for genre='{genre}'...")

    with torch.no_grad():
        for i in range(num_samples):
            tokens = torch.full((1, 1), BOS_TOKEN_ID, dtype=torch.long, device=DEVICE)
            genre_tensor = torch.tensor([genre_id], dtype=torch.long, device=DEVICE)

            for _ in range(max_new_tokens):
                context = tokens[:, -model.max_seq_len :]
                logits = model(context, genre_ids=genre_tensor)
                next_logits = logits[:, -1, :]
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)

                if int(next_token.item()) == EOS_TOKEN_ID:
                    break

            token_seq = tokens.squeeze(0).cpu().numpy()
            roll = tokens_to_piano_roll(token_seq, num_pitches=128)
            pm = piano_roll_to_pretty_midi(roll, fs=FS)
            output_path = GENERATED_MIDI_DIR / f"task3_sample_{i}.mid"
            pm.write(str(output_path))
            print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline_random", "baseline_markov", "ae", "vae", "transformer"], default="baseline_random")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--genre", type=str, default="maestro")
    parser.add_argument("--dataset", type=str, default="maestro", help="Dataset for Markov baseline (e.g. maestro, lakh, groove)")
    args = parser.parse_args()

    if args.model == "baseline_random":
        generate_random_music(num_samples=args.num_samples)
    elif args.model == "baseline_markov":
        transitions = train_markov_chain(dataset=args.dataset)
        generate_markov_music(transitions, num_samples=args.num_samples)
    elif args.model == "ae":
        from src.generation.sample_latent import sample_ae
        sample_ae(num_samples=args.num_samples)
    elif args.model == "vae":
        from src.generation.sample_latent import sample_vae
        sample_vae(num_samples=args.num_samples)
    elif args.model == "transformer":
        generate_transformer_music(
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            genre=args.genre,
        )