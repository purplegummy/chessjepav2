"""
Train the EvalOrganizer on top of frozen JEPA bottleneck codes.

Usage:
    python train_organizer.py \
        --fens data/fen_analysis.csv \
        --jepa_ckpt checkpoints/checkpoint_epoch5.pt \
        --num_samples 2000 \
        --stockfish /opt/homebrew/bin/stockfish \
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
from viz.evaluate_embeddings import load_checkpoint, load_and_encode


EVAL_CLIP = 1_500


def ranking_loss(z: torch.Tensor, evals: torch.Tensor, margin: float = 0.5):
    """
    Margin ranking loss: positions with higher eval should project to higher
    score along the eval direction (first dim of z as proxy, or use eval_head output).
    Uses random pairs within the batch.
    """
    scores = z[:, 0]  # proxy: first latent dim acts as eval axis before training settles
    B = z.size(0)
    i = torch.randint(0, B, (B,), device=z.device)
    j = torch.randint(0, B, (B,), device=z.device)
    target = torch.sign(evals[i] - evals[j])
    loss = nn.functional.margin_ranking_loss(
        scores[i], scores[j], target, margin=margin, reduction="mean"
    )
    return loss


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # --- load JEPA (frozen) and encode positions ---
    jepa = ChessJEPA().to(device)
    jepa = load_checkpoint(jepa, args.jepa_ckpt, device)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    print(f"Encoding {args.num_samples} positions...")
    embeddings, metadata = load_and_encode(
        args.fens, jepa, device, args.num_samples, args.stockfish, args.depth
    )

    X      = torch.stack(embeddings)                                          # (N, 8192)
    evals  = torch.tensor([m["eval_cp"] for m in metadata], dtype=torch.float32)
    evals_norm = evals / EVAL_CLIP                                            # normalise to [-1, 1]

    print(f"Dataset: {len(X)} positions, input_dim={X.shape[1]}")

    dataset = TensorDataset(X, evals_norm, evals)
    n_val   = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    # --- organizer ---
    organizer = EvalOrganizer(
        input_dim=X.shape[1],
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
            X_b, e_norm, e_raw = X_b.to(device), e_norm.to(device), e_raw.to(device)
            z, pred = organizer(X_b)

            loss_reg  = mse(pred, e_norm)
            loss_rank = ranking_loss(z, e_raw)
            loss = loss_reg + args.rank_weight * loss_rank

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        organizer.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, e_norm, _ in val_dl:
                X_b, e_norm = X_b.to(device), e_norm.to(device)
                _, pred = organizer(X_b)
                val_preds.append(pred.cpu())
                val_targets.append(e_norm.cpu())

        val_preds   = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_mse     = np.mean((val_preds - val_targets) ** 2)
        val_r2      = 1 - val_mse / np.var(val_targets)

        print(f"Epoch {epoch:3d} | train_loss={total_loss/len(train_dl):.4f} "
              f"| val_MSE={val_mse:.4f} | val_R²={val_r2:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model_state_dict": organizer.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "input_dim":  X.shape[1],
            }, args.out)
            print(f"  → saved to {args.out}")

    print(f"\nDone. Best val MSE: {best_val:.4f}")
    print("To use: load EvalOrganizer, pass discrete JEPA bottleneck codes,")
    print("        the latent z encodes eval — arithmetic in that space is meaningful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens",        default="data/fen_analysis.csv")
    parser.add_argument("--jepa_ckpt",   default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--num_samples", default=2000, type=int)
    parser.add_argument("--stockfish",   default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth",       default=12,   type=int)
    parser.add_argument("--latent_dim",  default=128,  type=int)
    parser.add_argument("--hidden_dim",  default=512,  type=int)
    parser.add_argument("--epochs",      default=50,   type=int)
    parser.add_argument("--batch_size",  default=64,   type=int)
    parser.add_argument("--lr",          default=1e-3, type=float)
    parser.add_argument("--rank_weight", default=0.1,  type=float,
                        help="weight for ranking loss vs regression loss")
    parser.add_argument("--out",         default="checkpoints/organizer.pt")
    args = parser.parse_args()

    train(args)
