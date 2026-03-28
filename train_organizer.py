"""
Train the EvalOrganizer on top of frozen JEPA encoder taps (mean-pooled).
Uses raw encoder output (256-dim) instead of bottleneck codes, which preserves
material and positional information that the discrete bottleneck loses.

Usage:
    python train_organizer.py \
        --dataset  data/dataset.pt \
        --jepa_ckpt checkpoints/checkpoint_epoch5.pt \
        --out checkpoints/organizer.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from jepa.jepa import ChessJEPA
from jepa.eval_organizer import EvalOrganizer


EVAL_CLIP      = 1_500
BOTTLENECK_TAU = 1e-5


def encode_dataset(jepa: ChessJEPA, states: torch.Tensor, device, batch_size: int = 256):
    """
    Returns integer category indices: (N, 512)
    Each position → 64 patches × 8 categories, each index in [0, 15].
    Far more compact than one-hot (8192) — the EvalOrganizer embeds these.
    """
    jepa.eval()
    all_indices = []
    for i in range(0, len(states), batch_size):
        batch = states[i : i + batch_size].to(device)
        with torch.no_grad():
            tap_dict = jepa.encoder(batch)
            last_tap = tap_dict[max(tap_dict.keys())]
            z, _     = jepa.bottleneck(last_tap, tau=BOTTLENECK_TAU)
            # z: (B, 64, 8, 16) → argmax over codes → (B, 64, 8) → flatten → (B, 512)
            indices = z.argmax(dim=-1).flatten(start_dim=1)
            all_indices.append(indices.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(states))}/{len(states)}")
    return torch.cat(all_indices)  # (N, 512) int64


def ranking_loss(eval_pred: torch.Tensor, evals: torch.Tensor, margin: float = 0.5):
    """Margin ranking loss over random pairs in the batch."""
    B = eval_pred.size(0)
    i = torch.randint(0, B, (B,), device=eval_pred.device)
    j = torch.randint(0, B, (B,), device=eval_pred.device)
    target = torch.sign(evals[i] - evals[j])
    return nn.functional.margin_ranking_loss(
        eval_pred[i], eval_pred[j], target, margin=margin, reduction="mean"
    )


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # --- load dataset ---
    print(f"Loading dataset from {args.dataset}...")
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)

    if "evals" not in data:
        raise ValueError("dataset.pt has no 'evals' key — rebuild with --stockfish flag")

    states = data["states"].float()   # (N, 17, 8, 8)
    evals  = data["evals"].float()    # (N,) int16 -> float

    if args.num_samples:
        idx = torch.randperm(len(states))[:args.num_samples]
        states, evals = states[idx], evals[idx]

    print(f"Dataset: {len(states)} positions")

    # --- load JEPA (frozen) ---
    jepa = ChessJEPA().to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    print(f"Loaded JEPA from {args.jepa_ckpt}")

    # --- encode all positions ---
    print("Encoding positions through JEPA bottleneck...")
    X = encode_dataset(jepa, states, device, batch_size=args.encode_batch)
    print(f"Encoded: {X.shape}")

    evals_norm = (evals / EVAL_CLIP).clamp(-1, 1)

    dataset = TensorDataset(X, evals_norm, evals)
    n_val   = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    # --- organizer ---
    organizer = EvalOrganizer(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    opt = torch.optim.AdamW(organizer.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        organizer.train()
        total_loss = 0.0
        for X_b, e_norm, e_raw in train_dl:
            X_b    = X_b.to(device)
            e_norm = e_norm.to(device)
            e_raw  = e_raw.to(device)

            z, pred = organizer(X_b)
            loss = mse(pred, e_norm) + args.rank_weight * ranking_loss(pred, e_raw)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        organizer.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, e_norm, _ in val_dl:
                _, pred = organizer(X_b.to(device))
                val_preds.append(pred.cpu())
                val_targets.append(e_norm)

        val_preds   = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_mse     = np.mean((val_preds - val_targets) ** 2)
        val_r2      = 1 - val_mse / (np.var(val_targets) + 1e-8)

        print(f"Epoch {epoch:3d} | train_loss={total_loss/len(train_dl):.4f} "
              f"| val_MSE={val_mse:.4f} | val_R²={val_r2:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": organizer.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
            }, args.out)
            print(f"  → saved to {args.out}")

    print(f"\nDone. Best val MSE: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="data/dataset.pt")
    parser.add_argument("--jepa_ckpt",   default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--num_samples",  default=None, type=int,
                        help="subsample N positions (default: use all)")
    parser.add_argument("--latent_dim",   default=128,  type=int)
    parser.add_argument("--hidden_dim",   default=512,  type=int)
    parser.add_argument("--epochs",       default=50,   type=int)
    parser.add_argument("--batch_size",   default=256,  type=int)
    parser.add_argument("--encode_batch", default=256,  type=int,
                        help="batch size for encoding (tune to fit VRAM)")
    parser.add_argument("--lr",           default=1e-3, type=float)
    parser.add_argument("--rank_weight",  default=0.1,  type=float)
    parser.add_argument("--out",          default="checkpoints/organizer.pt")
    args = parser.parse_args()

    train(args)
