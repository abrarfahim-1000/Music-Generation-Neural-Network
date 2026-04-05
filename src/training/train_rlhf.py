from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR,
    DEVICE,
    FS,
    GENERATED_MIDI_DIR,
    MAESTRO_CSV,
    MAESTRO_DIR,
    PLOTS_DIR,
    RLHF_CONFIG,
    SEED,
    SURVEY_DIR,
)
from src.evaluation.pitch_histogram import pitch_histogram_similarity
from src.evaluation.rhythm_score import repetition_ratio, rhythm_diversity
from src.models.transformer import MusicTransformer
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi
from src.preprocessing.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, VOCAB_SIZE, tokens_to_piano_roll


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed(seed)


def sample_next_token_with_log_prob(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, k=k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        scaled = torch.where(scaled < threshold, torch.full_like(scaled, -1e9), scaled)

    dist = Categorical(logits=scaled)
    token = dist.sample()
    log_prob = dist.log_prob(token)
    return token.unsqueeze(-1), log_prob


def load_transformer_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Task 3 checkpoint missing: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = payload.get("config", {})
    genre_to_id = payload.get("genre_to_id", {"maestro": 0})

    model = MusicTransformer(
        vocab_size=int(payload.get("vocab_size", VOCAB_SIZE)),
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 3)),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
        num_genres=max(1, len(genre_to_id)),
        dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"])

    return model, payload, genre_to_id


def load_reference_pretty_midi() -> pretty_midi.PrettyMIDI | None:
    if not MAESTRO_CSV.exists():
        return None

    rows = []
    with MAESTRO_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "").strip().lower() == "train":
                rows.append(row)
                break

    if not rows:
        return None

    rel_path = rows[0].get("midi_filename", "")
    if not rel_path:
        return None

    ref_path = MAESTRO_DIR / rel_path
    if not ref_path.exists():
        return None

    try:
        return pretty_midi.PrettyMIDI(str(ref_path))
    except Exception:
        return None


def tokens_to_pretty_midi(tokens: np.ndarray) -> pretty_midi.PrettyMIDI:
    roll = tokens_to_piano_roll(tokens, num_pitches=128)
    return piano_roll_to_pretty_midi(roll, fs=FS)


def generate_sequence_with_log_prob(
    model: MusicTransformer,
    genre_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[np.ndarray, torch.Tensor]:
    tokens = torch.full((1, 1), BOS_TOKEN_ID, dtype=torch.long, device=DEVICE)
    genre_tensor = torch.tensor([genre_id], dtype=torch.long, device=DEVICE)
    log_probs = []

    for _ in range(max_new_tokens):
        context = tokens[:, -model.max_seq_len :]
        logits = model(context, genre_ids=genre_tensor)
        next_logits = logits[:, -1, :]

        next_token, log_prob = sample_next_token_with_log_prob(
            next_logits,
            temperature=temperature,
            top_k=top_k,
        )
        tokens = torch.cat([tokens, next_token], dim=1)
        log_probs.append(log_prob)

        if int(next_token.item()) == EOS_TOKEN_ID:
            break

    seq = tokens.squeeze(0).detach().cpu().numpy()
    if log_probs:
        log_prob_sum = torch.stack(log_probs).sum()
    else:
        log_prob_sum = torch.tensor(0.0, device=DEVICE)

    return seq, log_prob_sum


def generate_sequence_no_grad(
    model: MusicTransformer,
    genre_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> np.ndarray:
    with torch.no_grad():
        seq, _ = generate_sequence_with_log_prob(
            model=model,
            genre_id=genre_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return seq


def compute_proxy_reward(
    pm: pretty_midi.PrettyMIDI,
    ref_pm: pretty_midi.PrettyMIDI | None,
    reward_weights: dict[str, float],
) -> float:
    rhythm = float(rhythm_diversity(pm))
    repetition = float(repetition_ratio(pm))
    anti_repetition = 1.0 - repetition

    pitch = 0.5
    if ref_pm is not None:
        pitch = float(pitch_histogram_similarity(pm, ref_pm))

    reward = (
        reward_weights["pitch"] * pitch
        + reward_weights["rhythm"] * rhythm
        + reward_weights["anti_repetition"] * anti_repetition
    )
    return float(reward)


def load_survey_rewards(survey_csv: Path | None) -> dict[str, float]:
    if survey_csv is None or not survey_csv.exists():
        return {}

    filename_keys = ["filename", "file", "sample", "sample_name"]
    score_keys = ["score", "human_score", "rating", "preference_score", "reward"]

    reward_map: dict[str, float] = {}
    with survey_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        filename_col = next((k for k in filename_keys if k in fields), None)
        score_col = next((k for k in score_keys if k in fields), None)
        if filename_col is None or score_col is None:
            print(
                "[RLHF] Survey CSV found but missing required columns. "
                "Expected one filename-like column and one score-like column."
            )
            return {}

        for row in reader:
            name = str(row.get(filename_col, "")).strip()
            if not name:
                continue
            try:
                score = float(row.get(score_col, ""))
            except (TypeError, ValueError):
                continue
            reward_map[name] = score

    print(f"[RLHF] Loaded {len(reward_map)} survey rewards from {survey_csv}")
    return reward_map


def save_midi_samples(
    model: MusicTransformer,
    genre_id: int,
    output_prefix: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    model.eval()
    for i in range(num_samples):
        seq = generate_sequence_no_grad(
            model=model,
            genre_id=genre_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        pm = tokens_to_pretty_midi(seq)
        out_path = GENERATED_MIDI_DIR / f"{output_prefix}_{i}.mid"
        pm.write(str(out_path))
        print(f"[RLHF] Saved {out_path}")


def train_rlhf(
    rl_steps: int,
    episodes_per_step: int,
    max_new_tokens: int,
    lr: float,
    temperature: float,
    top_k: int,
    genre: str,
    survey_csv: Path | None,
    num_eval_samples: int,
):
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    base_ckpt = CHECKPOINT_DIR / "latest_transformer.pt"
    resume_path = CHECKPOINT_DIR / "latest_rlhf.pt"

    start_step = 0
    step_rewards: list[float] = []
    step_losses: list[float] = []
    best_mean_reward = float("-inf")

    if resume_path.exists():
        resume_payload = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model, payload, base_genre_to_id = load_transformer_checkpoint(base_ckpt)
        saved_genre_to_id = resume_payload.get("genre_to_id", {})
        if saved_genre_to_id and base_genre_to_id and saved_genre_to_id != base_genre_to_id:
            raise RuntimeError(
                "RLHF resume checkpoint genre mapping does not match current latest_transformer.pt mapping. "
                "Delete latest_rlhf.pt to start from the current base model."
            )
        genre_to_id = saved_genre_to_id if saved_genre_to_id else base_genre_to_id
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        start_step = int(resume_payload.get("step_completed", -1)) + 1
        best_mean_reward = float(resume_payload.get("best_mean_reward", best_mean_reward))
        step_rewards = list(resume_payload.get("step_rewards", []))
        step_losses = list(resume_payload.get("step_losses", []))
        print(
            f"[RLHF] Resumed from {resume_path} at step {start_step + 1} "
            f"(best_mean_reward={best_mean_reward:.6f})"
        )
    else:
        model, payload, genre_to_id = load_transformer_checkpoint(base_ckpt)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    if genre in genre_to_id:
        genre_id = int(genre_to_id[genre])
    elif genre_to_id:
        fallback = sorted(genre_to_id.keys())[0]
        genre_id = int(genre_to_id[fallback])
        print(f"[RLHF] Genre '{genre}' not found. Falling back to '{fallback}'.")
    else:
        genre_id = 0

    ref_pm = load_reference_pretty_midi()
    reward_weights = RLHF_CONFIG["reward_weights"]

    survey_rewards = load_survey_rewards(survey_csv)
    use_survey = len(survey_rewards) > 0

    end_step = start_step + rl_steps

    # Save before-RLHF reference samples from the pretrained Task 3 policy.
    if start_step == 0:
        frozen_before_model, _, _ = load_transformer_checkpoint(base_ckpt)
        frozen_before_model.eval()
        save_midi_samples(
            model=frozen_before_model,
            genre_id=genre_id,
            output_prefix="task4_before",
            num_samples=num_eval_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    print(
        f"[RLHF] Starting RLHF from step {start_step + 1} to {end_step} "
        f"with {episodes_per_step} episodes/iteration."
    )

    for step_idx in range(start_step, end_step):
        model.train()

        episode_log_probs = []
        episode_rewards = []

        for ep in tqdm(range(episodes_per_step), desc=f"RL step {step_idx+1}/{end_step}"):
            seq, log_prob_sum = generate_sequence_with_log_prob(
                model=model,
                genre_id=genre_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            pm = tokens_to_pretty_midi(seq)

            sample_name = f"task4_iter_{step_idx}_sample_{ep}.mid"
            sample_path = GENERATED_MIDI_DIR / sample_name
            pm.write(str(sample_path))

            if use_survey and sample_name in survey_rewards:
                reward = float(survey_rewards[sample_name])
            else:
                reward = compute_proxy_reward(
                    pm=pm,
                    ref_pm=ref_pm,
                    reward_weights=reward_weights,
                )

            episode_log_probs.append(log_prob_sum)
            episode_rewards.append(reward)

        log_probs_tensor = torch.stack(episode_log_probs)
        rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32, device=DEVICE)

        # Variance-reduction baseline for REINFORCE.
        advantages = rewards_tensor - rewards_tensor.mean()
        advantages = advantages / (rewards_tensor.std(unbiased=False) + 1e-8)

        # Gradient ascent on E[r * log p_theta(X)] implemented via minimizing negative objective.
        loss = -(advantages.detach() * log_probs_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_reward = float(rewards_tensor.mean().item())
        step_rewards.append(mean_reward)
        step_losses.append(float(loss.item()))

        print(
            f"[RLHF] Step {step_idx+1}: mean_reward={mean_reward:.6f}, "
            f"loss={loss.item():.6f}"
        )

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            tuned_payload = dict(payload)
            tuned_payload["model_state_dict"] = model.state_dict()
            tuned_payload["rlhf"] = {
                "best_mean_reward": best_mean_reward,
                "step": step_idx + 1,
                "reward_source": "survey" if use_survey else "proxy",
            }
            torch.save(tuned_payload, CHECKPOINT_DIR / "latest_rlhf.pt")
            print("[RLHF] Saved latest_rlhf.pt")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step_completed": step_idx,
                "best_mean_reward": best_mean_reward,
                "step_rewards": step_rewards,
                "step_losses": step_losses,
                "genre_to_id": genre_to_id,
            },
            resume_path,
        )

    # Load latest tuned model for after-RLHF generation.
    tuned_ckpt_path = CHECKPOINT_DIR / "latest_rlhf.pt"
    if tuned_ckpt_path.exists():
        tuned = torch.load(tuned_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(tuned["model_state_dict"])

    model.eval()
    save_midi_samples(
        model=model,
        genre_id=genre_id,
        output_prefix="task4_after",
        num_samples=num_eval_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # Save RLHF training log for report/survey linkage.
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    log_csv = SURVEY_DIR / "task4_rlhf_training_log.csv"
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_reward", "loss"])
        for i, (r, l) in enumerate(zip(step_rewards, step_losses), start=1):
            writer.writerow([i, r, l])
    print(f"[RLHF] Saved training log to {log_csv}")

    summary_json = SURVEY_DIR / "task4_rlhf_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "reward_source": "survey" if use_survey else "proxy",
                "rl_steps": rl_steps,
                "episodes_per_step": episodes_per_step,
                "max_new_tokens": max_new_tokens,
                "best_mean_reward": best_mean_reward,
                "final_mean_reward": step_rewards[-1] if step_rewards else None,
                "num_before_samples": num_eval_samples,
                "num_after_samples": num_eval_samples,
            },
            f,
            indent=2,
        )
    print(f"[RLHF] Saved summary to {summary_json}")

    plt.figure(figsize=(10, 5))
    plt.plot(step_rewards, label="Mean Reward")
    plt.xlabel("RL Iteration")
    plt.ylabel("Reward")
    plt.title("Task 4: RLHF Reward Curve")
    plt.legend()
    plot_path = PLOTS_DIR / "task4_rlhf_reward_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[RLHF] Saved reward curve to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl_steps",
        type=int,
        default=RLHF_CONFIG["rl_steps"],
        help="Number of additional RLHF steps to run per invocation",
    )
    parser.add_argument("--episodes_per_step", type=int, default=RLHF_CONFIG["episodes_per_step"])
    parser.add_argument("--max_new_tokens", type=int, default=RLHF_CONFIG["max_new_tokens"])
    parser.add_argument("--lr", type=float, default=RLHF_CONFIG["lr"])
    parser.add_argument("--temperature", type=float, default=RLHF_CONFIG["temperature"])
    parser.add_argument("--top_k", type=int, default=RLHF_CONFIG["top_k"])
    parser.add_argument("--genre", type=str, default="maestro")
    parser.add_argument(
        "--survey_csv",
        type=str,
        default=None,
        help="Optional CSV with filename+score columns for human rewards.",
    )
    parser.add_argument("--num_eval_samples", type=int, default=RLHF_CONFIG["num_eval_samples"])
    args = parser.parse_args()

    survey_path = Path(args.survey_csv) if args.survey_csv else None

    train_rlhf(
        rl_steps=args.rl_steps,
        episodes_per_step=args.episodes_per_step,
        max_new_tokens=args.max_new_tokens,
        lr=args.lr,
        temperature=args.temperature,
        top_k=args.top_k,
        genre=args.genre,
        survey_csv=survey_path,
        num_eval_samples=args.num_eval_samples,
    )
