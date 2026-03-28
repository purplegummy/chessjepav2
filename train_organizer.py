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
    Returns encoder taps: (N, 64, 256) float
    Pre-bottleneck activations preserve all information the encoder learned.
    """
    jepa.eval()
    all_taps = []
    for i in range(0, len(states), batch_size):
        batch = states[i : i + batch_size].to(device)
        with torch.no_grad():
            tap_dict = jepa.encoder(batch)
            last_tap = tap_dict[max(tap_dict.keys())]  # (B, 64, 256)
            all_taps.append(last_tap.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(states))}/{len(states)}")
    return torch.cat(all_taps)  # (N, 64, 256) float32


def ranking_loss(eval_pred: torch.Tensor, evals: torch.Tensor, margin: float = 0.5):
    """Margin ranking loss over random pairs in the batch."""
    B = eval_pred.size(0)
    i = torch.randint(0, B, (B,), device=eval_pred.device)
    j = torch.randint(0, B, (B,), device=eval_pred.device)
    target = torch.sign(evals[i] - evals[j])
    return nn.functional.margin_ranking_loss(
        eval_pred[i], eval_pred[j], target, margin=margin, reduction="mean"
    )


def contrastive_loss(z: torch.Tensor, evals: torch.Tensor,
                     win_thresh: float = 150, lose_thresh: float = -150,
                     margin: float = 2.0):
    """
    Organizes latent space geometry by eval label.
    Works on L2-normalized embeddings (cosine space) so margin is meaningful.
      - Pull intra-class pairs together (centroid pull)
      - Push inter-class pairs apart
    """
    win_mask  = evals >  win_thresh
    lose_mask = evals < lose_thresh

    if win_mask.sum() < 2 or lose_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # Normalize to unit sphere so distances are in [0, 2]
    z_n = nn.functional.normalize(z, dim=-1)
    z_win  = z_n[win_mask]
    z_lose = z_n[lose_mask]

    # Pull: minimize variance around each centroid
    pull_win  = (z_win  - z_win.mean(0)).pow(2).sum(1).mean()
    pull_lose = (z_lose - z_lose.mean(0)).pow(2).sum(1).mean()
    pull = pull_win + pull_lose

    # Push: inter-class pairs should be > margin apart
    n = min(len(z_win), len(z_lose), 64)
    idx_w = torch.randperm(len(z_win),  device=z.device)[:n]
    idx_l = torch.randperm(len(z_lose), device=z.device)[:n]
    dists = torch.norm(z_win[idx_w] - z_lose[idx_l], dim=1)
    push = torch.clamp(margin - dists, min=0).mean()

    return pull + push


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
        tap_dim=256,
        n_patches=64,
    ).to(device)

    opt = torch.optim.AdamW(organizer.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        organizer.train()
        total_loss = 0.0
        for X_b, e_norm, e_raw in train_dl:
            X_b    = X_b.to(device)
            e_norm = e_norm.to(device)
            e_raw  = e_raw.to(device)

            z, pred, struct = organizer(X_b)
            loss = (mse(pred, e_norm)
                    + args.contrastive_lambda * contrastive_loss(z, e_raw, margin=args.margin)
                    + args.orth_lambda * organizer.orthogonal_penalty())

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation: measure cluster separation (higher = better geometry)
        organizer.eval()
        val_z, val_e = [], []
        with torch.no_grad():
            for X_b, _, e_raw in val_dl:
                z, _, _ = organizer(X_b.to(device))
                val_z.append(z.cpu())
                val_e.append(e_raw)

        val_z = torch.cat(val_z)
        val_e = torch.cat(val_e)

        z_win  = val_z[val_e >  150]
        z_lose = val_z[val_e < -150]
        if len(z_win) > 0 and len(z_lose) > 0:
            win_centroid  = z_win.mean(0)
            lose_centroid = z_lose.mean(0)
            separation = torch.norm(win_centroid - lose_centroid).item()
            intra_win  = torch.pdist(z_win[:256]).mean().item()  if len(z_win)  > 1 else 0
            intra_lose = torch.pdist(z_lose[:256]).mean().item() if len(z_lose) > 1 else 0
        else:
            separation, intra_win, intra_lose = 0.0, 0.0, 0.0

        orth = organizer.orthogonal_penalty().item()
        val_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch:3d} | loss={val_loss:.4f} | orth={orth:.4f} "
              f"| sep={separation:.3f} | intra_win={intra_win:.3f} | intra_lose={intra_lose:.3f}")

        # save when separation improves
        if separation > best_val:
            best_val = separation
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": organizer.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "tap_dim": 256,
                "n_patches": 64,
            }, args.out)
            print(f"  → saved (sep={separation:.3f})")

    print(f"\nDone. Best separation: {best_val:.4f}")


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
    parser.add_argument("--orth_lambda", default=0.1, type=float,
                        help="weight for orthogonal penalty between value and structure branches")
    parser.add_argument("--contrastive_lambda", default=1.0, type=float,
                        help="weight for contrastive loss term")
    parser.add_argument("--margin", default=1.5, type=float,
                        help="contrastive margin (on normalized embeddings, max=2.0)")
    parser.add_argument("--out",                default="checkpoints/organizer.pt")
    args = parser.parse_args()

    train(args)
