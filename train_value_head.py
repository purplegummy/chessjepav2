"""
Train a ValueHead on top of frozen JEPA encoder taps.

Usage:
    python train_value_head.py \
        --dataset   data/dataset.pt \
        --jepa_ckpt checkpoints/checkpoint_epoch5.pt \
        --out       checkpoints/value_head.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from jepa.jepa import ChessJEPA
from jepa.value_head import ValueHead

EVAL_CLIP = 1_500


def encode_dataset(jepa: ChessJEPA, states: torch.Tensor, device, batch_size: int = 256):
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
    return torch.cat(all_taps)  # (N, 64, 256)


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    data = torch.load(args.dataset, map_location="cpu", weights_only=True)
    if "evals" not in data:
        raise ValueError("dataset.pt has no 'evals' key — rebuild with --stockfish flag")

    states = data["states"].float()
    evals  = data["evals"].float()

    if args.num_samples:
        idx = torch.randperm(len(states))[:args.num_samples]
        states, evals = states[idx], evals[idx]

    print(f"Dataset: {len(states)} positions")

    jepa = ChessJEPA().to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    print(f"Loaded JEPA from {args.jepa_ckpt}")

    print("Encoding positions...")
    X = encode_dataset(jepa, states, device, batch_size=args.encode_batch)
    print(f"Encoded: {X.shape}")

    evals_norm = (evals / EVAL_CLIP).clamp(-1, 1)

    dataset = TensorDataset(X, evals_norm)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = ValueHead(
        tap_dim=256,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse     = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, e_b in train_dl:
            X_b = X_b.to(device)
            e_b = e_b.to(device)
            _, pred = model(X_b)
            loss = mse(pred, e_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, e_b in val_dl:
                _, pred = model(X_b.to(device))
                val_loss += mse(pred, e_b.to(device)).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "tap_dim":    256,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
            }, args.out)
            print(f"  → saved (val={val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="data/dataset.pt")
    parser.add_argument("--jepa_ckpt",    default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--num_samples",  default=None, type=int)
    parser.add_argument("--latent_dim",   default=128,  type=int)
    parser.add_argument("--hidden_dim",   default=512,  type=int)
    parser.add_argument("--epochs",       default=50,   type=int)
    parser.add_argument("--batch_size",   default=256,  type=int)
    parser.add_argument("--encode_batch", default=256,  type=int)
    parser.add_argument("--lr",           default=1e-3, type=float)
    parser.add_argument("--out",          default="checkpoints/value_head.pt")
    args = parser.parse_args()
    train(args)
