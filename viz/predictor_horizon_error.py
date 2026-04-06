"""
Evaluates predictor error across rollout horizons.

For each sequence of length max_horizon, we:
  1. Encode board_t → z_t (hard categorical indices via bottleneck)
  2. Roll the predictor forward k steps using the sequence of actions
  3. At each step k, compare predicted z vs. actual encoded z_t+k
     using token accuracy (fraction of categorical codes predicted correctly)

Usage:
    python viz/predictor_horizon_error.py \
        --checkpoint checkpoints/checkpoint_v3_epoch1.pt \
        --dataset data/dataset.pt \
        --max_horizon 8 \
        --n_seqs 512 \
        --batch_size 64
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from jepa.jepa import ChessJEPA
from util.chessdataset import ChessDataset


def load_model(checkpoint_path: str, device: torch.device) -> ChessJEPA:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ChessJEPA()
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_hard(model: ChessJEPA, board: torch.Tensor, level: int) -> torch.Tensor:
    """Encode board → hard categorical indices (B, N, n_cats)."""
    taps = model.encoder(board)
    z, _ = model.bottleneck(taps[level], tau=0.1)
    return z.argmax(dim=-1)  # (B, N, n_cats)


@torch.no_grad()
def encode_soft(model: ChessJEPA, board: torch.Tensor, level: int) -> torch.Tensor:
    """Encode board → soft categorical embedding (B, N, n_cats, n_codes)."""
    taps = model.encoder(board)
    z, _ = model.bottleneck(taps[level], tau=0.1)
    return z  # (B, N, n_cats, n_codes)


@torch.no_grad()
def predict_step(model: ChessJEPA, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Run predictor one step.
    z: (B, N, n_cats, n_codes)  — soft embeddings
    action: (B,) int64

    Returns: z_hat (B, N, n_cats, n_codes) — predicted soft embeddings (after softmax)
    """
    logits = model.predictor(z, action)  # (B, N, n_cats, n_codes)
    return torch.softmax(logits, dim=-1)


def build_sequences(dataset: ChessDataset, max_horizon: int):
    """
    Returns all valid starting indices for rollouts of length max_horizon.
    A starting index i is valid if transitions i, i+1, ..., i+max_horizon-1
    are all within the same game.
    """
    states      = dataset.data["states"]
    next_states = dataset.data["next_states"]
    N = len(states)

    same_game = (next_states[:-1] == states[1:]).all(dim=(1, 2, 3))  # (N-1,) bool

    # For index i to be a valid start, same_game[i], same_game[i+1], ...,
    # same_game[i+max_horizon-2] must all be True (max_horizon-1 consecutive links).
    # Use a sliding window: valid[i] = all(same_game[i : i+max_horizon-1])
    same_game_int = same_game.long()  # easier to cumsum
    cumsum = torch.cat([torch.zeros(1, dtype=torch.long), same_game_int.cumsum(0)])
    # window sum from i to i+max_horizon-2 (inclusive) = cumsum[i+max_horizon-1] - cumsum[i]
    window = max_horizon - 1
    valid_starts = []
    for i in range(N - window):
        if (cumsum[i + window] - cumsum[i]).item() == window:
            valid_starts.append(i)

    return valid_starts


def evaluate_horizons(
    model: ChessJEPA,
    dataset: ChessDataset,
    max_horizon: int,
    n_seqs: int,
    batch_size: int,
    device: torch.device,
    level: int,
):
    valid_starts = build_sequences(dataset, max_horizon)
    print(f"Found {len(valid_starts)} valid starting positions for horizon {max_horizon}")

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(valid_starts), size=min(n_seqs, len(valid_starts)), replace=False)

    starts = [valid_starts[i] for i in chosen]
    # Each sequence is just max_horizon+1 consecutive indices from its start
    seq_idx_lists = [list(range(s, s + max_horizon + 1)) for s in starts]

    states      = dataset.data["states"]
    next_states = dataset.data["next_states"]
    actions     = dataset.data["actions"]

    # acc_per_horizon[k] = list of per-sample token accuracies at horizon k
    acc_per_horizon = {k: [] for k in range(1, max_horizon + 1)}

    for b_start in range(0, len(starts), batch_size):
        batch_seqs = seq_idx_lists[b_start : b_start + batch_size]
        B = len(batch_seqs)

        # board at t=0
        board0 = torch.stack([states[s[0]] for s in batch_seqs]).to(device)
        z = encode_soft(model, board0, level)  # (B, N, n_cats, n_codes)

        for k in range(1, max_horizon + 1):
            # action at step k-1
            act = torch.tensor([actions[s[k - 1]] for s in batch_seqs], device=device)

            z = predict_step(model, z, act)  # predicted z at step k

            # actual encoding at step k
            board_k = torch.stack([states[s[k]] for s in batch_seqs]).to(device)
            z_real_idx = encode_hard(model, board_k, level)  # (B, N, n_cats)
            z_pred_idx = z.argmax(dim=-1)                    # (B, N, n_cats)

            acc = (z_pred_idx == z_real_idx).float().mean(dim=(1, 2))  # (B,)
            acc_per_horizon[k].extend(acc.cpu().tolist())

        if (b_start // batch_size) % 5 == 0:
            print(f"  processed {min(b_start + batch_size, len(starts))}/{len(starts)} sequences")

    return acc_per_horizon


def plot_results(acc_per_horizon: dict, out_path: str):
    horizons = sorted(acc_per_horizon.keys())
    means = [np.mean(acc_per_horizon[k]) for k in horizons]
    stds  = [np.std(acc_per_horizon[k])  for k in horizons]
    errors = [1 - m for m in means]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Token accuracy
    ax = axes[0]
    ax.errorbar(horizons, means, yerr=stds, marker='o', capsize=4, linewidth=2)
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Token Accuracy")
    ax.set_title("Predictor Token Accuracy vs. Horizon")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Token error
    ax = axes[1]
    ax.errorbar(horizons, errors, yerr=stds, marker='o', capsize=4, linewidth=2, color='tab:red')
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Token Error Rate")
    ax.set_title("Predictor Token Error vs. Horizon")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_v3_epoch1.pt")
    parser.add_argument("--dataset",    default="data/dataset.pt")
    parser.add_argument("--max_horizon", type=int, default=8)
    parser.add_argument("--n_seqs",      type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--level",       type=int, default=6,
                        help="Which tap layer to evaluate (default: last = 6)")
    parser.add_argument("--out", default="viz/predictor_horizon_error.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}…")
    model = load_model(args.checkpoint, device)

    print(f"Loading dataset from {args.dataset}…")
    dataset = ChessDataset(args.dataset)

    acc_per_horizon = evaluate_horizons(
        model, dataset,
        max_horizon=args.max_horizon,
        n_seqs=args.n_seqs,
        batch_size=args.batch_size,
        device=device,
        level=args.level,
    )

    print("\nResults:")
    print(f"{'Horizon':>8} {'Accuracy':>10} {'Error':>10} {'Std':>8}")
    for k in sorted(acc_per_horizon.keys()):
        m = np.mean(acc_per_horizon[k])
        s = np.std(acc_per_horizon[k])
        print(f"{k:>8} {m:>10.4f} {1-m:>10.4f} {s:>8.4f}")

    plot_results(acc_per_horizon, args.out)


if __name__ == "__main__":
    main()
