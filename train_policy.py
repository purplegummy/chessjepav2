"""
train_policy.py — Train the PolicyHead on frozen encoder + bottleneck weights.

Usage
-----
python train_policy.py \
    --jepa_ckpt checkpoints/checkpoint_epoch4.pt \
    --out       checkpoints/policy_head.pt \
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
from jepa.head import PolicyHead
from util.chessdataset import ChessDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

BOTTLENECK_TAU = 1e-5
N_CATS  = 8
N_CODES = 16


def encode_batch(encoder, bottleneck, board, device):
    with torch.no_grad():
        taps = encoder(board.to(device))
        final_layer = max(taps.keys())
        h = taps[final_layer]
        z, _ = bottleneck(h, tau=BOTTLENECK_TAU)
    return z


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"device: {device}")

    jepa = ChessJEPA(n_cats=N_CATS, n_codes=N_CODES).to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False

    encoder    = jepa.encoder
    bottleneck = jepa.bottleneck

    policy_head = PolicyHead(n_cats=N_CATS, n_codes=N_CODES).to(device)
    total = sum(p.numel() for p in policy_head.parameters())
    logging.info(f"PolicyHead params: {total:,}")

    start_epoch = 0
    best_val    = float("inf")
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        policy_head.load_state_dict(resume_ckpt["policy_head_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0)
        best_val    = resume_ckpt.get("best_val", float("inf"))
        logging.info(f"Resumed from {args.resume} (epoch {start_epoch}, best_val {best_val:.4f})")

    dataset = ChessDataset(args.data)
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,   num_workers=0)

    optimizer = torch.optim.Adam(policy_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=start_epoch + args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        policy_head.train()
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            board  = data["state"]               # (B, 17, 8, 8)
            action = data["action"].to(device)   # (B,) int64 move indices

            z = encode_batch(encoder, bottleneck, board, device)

            optimizer.zero_grad()
            logits = policy_head(z)              # (B, 4672)
            loss = criterion(logits, action)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    top5 = logits.topk(5, dim=-1).indices
                    top1_acc = (top5[:, 0] == action).float().mean().item()
                    top5_acc = (top5 == action.unsqueeze(1)).any(dim=1).float().mean().item()
                logging.info(
                    f"epoch {epoch+1:>2} | batch {batch_idx:>5} | loss {loss.item():.4f} | top1 {top1_acc:.3f} | top5 {top5_acc:.3f}"
                )

        policy_head.eval()
        val_loss = 0.0
        correct_top1  = 0
        correct_top5  = 0
        total_samples = 0
        with torch.no_grad():
            for data in val_loader:
                board  = data["state"]
                action = data["action"].to(device)
                z      = encode_batch(encoder, bottleneck, board, device)
                logits = policy_head(z)

                val_loss += criterion(logits, action).item()

                top5 = logits.topk(5, dim=-1).indices
                correct_top1 += (top5[:, 0] == action).sum().item()
                correct_top5 += (top5 == action.unsqueeze(1)).any(dim=1).sum().item()
                total_samples += action.shape[0]

        scheduler.step()

        val_loss /= len(val_loader)
        top1_acc = correct_top1 / total_samples
        top5_acc = correct_top5 / total_samples
        logging.info(
            f"epoch {epoch+1:>2} | val_loss {val_loss:.4f} | top1 {top1_acc:.3f} | top5 {top5_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"policy_head_state_dict": policy_head.state_dict(), "epoch": epoch + 1, "best_val": best_val},
                args.out,
            )
            logging.info(f"Saved best policy head to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--data",      default="data/dataset.pt")
    parser.add_argument("--out",       default="checkpoints/policy/policy_head.pt")
    parser.add_argument("--epochs",    default=5,    type=int)
    parser.add_argument("--batch",     default=2048, type=int)
    parser.add_argument("--lr",        default=3e-4, type=float)
    parser.add_argument("--resume",    default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)
