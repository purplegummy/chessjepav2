"""
train_value.py — Train the ValueHead on frozen encoder + bottleneck weights.

The JEPA encoder and bottleneck are loaded from a checkpoint and kept frozen.
Only the ValueHead parameters are updated.

Usage
-----
python train_value.py \
    --jepa_ckpt checkpoints/checkpoint_epoch1.pt \
    --out       checkpoints/value_head.pt \
    --data      data/dataset.pt \
    --epochs    5 \
    --batch     512 \
    --lr        3e-4
"""

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from jepa.jepa import ChessJEPA
from jepa.head import ValueHead
from util.chessdataset import ChessDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

BOTTLENECK_TAU = 1e-5   # near-deterministic during inference / value training
N_CATS  = 8
N_CODES = 16


def encode_batch(encoder, bottleneck, board, device):
    """Encode a batch of boards to hard latent codes. No gradients."""
    with torch.no_grad():
        taps = encoder(board.to(device))
        final_layer = max(taps.keys())
        h = taps[final_layer]                           # (B, 64, 256)
        z, _ = bottleneck(h, tau=BOTTLENECK_TAU)        # (B, 64, 32, 64)
    return z


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # ------------------------------------------------------------------ #
    # Load frozen JEPA encoder + bottleneck
    # ------------------------------------------------------------------ #
    jepa = ChessJEPA(n_cats=N_CATS, n_codes=N_CODES).to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False

    encoder    = jepa.encoder
    bottleneck = jepa.bottleneck

    # ------------------------------------------------------------------ #
    # Value head (trainable)
    # ------------------------------------------------------------------ #
    value_head = ValueHead(n_cats=N_CATS, n_codes=N_CODES).to(device)
    total = sum(p.numel() for p in value_head.parameters())
    logging.info(f"ValueHead params: {total:,}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset = ChessDataset(args.data)
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(value_head.parameters(), lr=args.lr)
    # MSE against float target in (-1, 1) — simple and effective
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(args.epochs):
        # ---- train ----
        value_head.train()
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            board  = data["state"]               # (B, 17, 8, 8)
            result = data["result"].float().to(device)  # (B,) in {-1, 0, 1}

            z = encode_batch(encoder, bottleneck, board, device)

            optimizer.zero_grad()
            pred = value_head(z)                 # (B,)
            loss = criterion(pred, result)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                logging.info(
                    f"epoch {epoch+1:>2} | batch {batch_idx:>5} | loss {loss.item():.4f}"
                )

        # ---- validate ----
        value_head.eval()
        val_loss = 0.0
        correct  = 0
        total_samples = 0
        with torch.no_grad():
            for data in val_loader:
                board  = data["state"]
                result = data["result"].float().to(device)
                z      = encode_batch(encoder, bottleneck, board, device)
                pred   = value_head(z)
                val_loss += criterion(pred, result).item()

                # sign accuracy: did we predict the right winner?
                pred_sign   = pred.sign()
                result_sign = result.sign()
                correct += (pred_sign == result_sign).sum().item()
                total_samples += result.shape[0]

        val_loss /= len(val_loader)
        acc = correct / total_samples
        logging.info(
            f"epoch {epoch+1:>2} | val_loss {val_loss:.4f} | sign_acc {acc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"value_head_state_dict": value_head.state_dict(), "epoch": epoch + 1},
                args.out,
            )
            logging.info(f"Saved best value head to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True,
                        help="Path to trained ChessJEPA checkpoint")
    parser.add_argument("--data",      default="data/dataset.pt")
    parser.add_argument("--out",       default="checkpoints/value_head.pt")
    parser.add_argument("--epochs",    default=5,    type=int)
    parser.add_argument("--batch",     default=512,  type=int)
    parser.add_argument("--lr",        default=3e-4, type=float)
    args = parser.parse_args()
    main(args)
