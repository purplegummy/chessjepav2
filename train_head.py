"""
Train a PolicyHead on top of a frozen JEPA encoder.

Usage:
    python train_head.py \
        --jepa_ckpt checkpoints/checkpoint_v3_epoch0.pt \
        --data      data/dataset.pt \
        --out       checkpoints/policy_head.pt
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import logging

from jepa.jepa import ChessJEPA
from jepa.head import PolicyHead
from util.chessdataset import ChessDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def save_checkpoint(head, optimizer, epoch, out_path):
    torch.save({
        "model_state_dict":     head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
    }, out_path)
    logging.info(f"Saved checkpoint → {out_path}")


def validate(head, encoder, dataloader, criterion, device):
    head.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for data in dataloader:
            states  = data["state"].to(device)
            actions = data["action"].to(device)

            taps   = encoder(states)
            logits = head(taps[max(taps.keys())])

            total_loss    += criterion(logits, actions).item()
            total_correct += (logits.argmax(dim=-1) == actions).sum().item()
            total         += len(actions)

    return total_loss / len(dataloader), total_correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen encoder
    logging.info(f"Loading JEPA from {args.jepa_ckpt}…")
    jepa = ChessJEPA().to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    encoder = jepa.encoder

    head = PolicyHead().to(device)
    total = sum(p.numel() for p in head.parameters())
    logging.info(f"PolicyHead params: {total:,}")

    dataset = ChessDataset(args.data)
    if args.data_frac < 1.0:
        keep = int(args.data_frac * len(dataset))
        dataset, _ = random_split(dataset, [keep, len(dataset) - keep])
        logging.info(f"Using {args.data_frac:.0%} of data → {keep:,} samples")

    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    optimizer   = torch.optim.Adam(head.parameters(), lr=args.lr)
    criterion   = torch.nn.CrossEntropyLoss()
    total_steps = args.epochs * (train_size // args.batch)
    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        head.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.lr)
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * (train_size // args.batch)
        logging.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup,
        last_epoch=min(global_step, args.warmup) - 1,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup, eta_min=1e-6,
        last_epoch=max(-1, global_step - args.warmup - 1),
    )

    for epoch in range(start_epoch, args.epochs):
        head.train()
        for batch_idx, data in enumerate(train_loader):
            states  = data["state"].to(device)
            actions = data["action"].to(device)

            with torch.no_grad():
                taps = encoder(states)

            optimizer.zero_grad()
            logits = head(taps[max(taps.keys())])
            loss   = criterion(logits, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            if global_step < args.warmup:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            global_step += 1

            if batch_idx % 100 == 0:
                acc = (logits.argmax(dim=-1) == actions).float().mean().item()
                logging.info(
                    f"epoch {epoch:>3} | batch {batch_idx:>6} | "
                    f"loss {loss.item():.4f} | acc {acc:.4f}"
                )

        val_loss, val_acc = validate(head, encoder, val_loader, criterion, device)
        logging.info(f"epoch {epoch:>3} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
        save_checkpoint(head, optimizer, epoch, args.out.replace(".pt", f"_epoch{epoch}.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt",  required=True)
    parser.add_argument("--data",       default="data/dataset.pt")
    parser.add_argument("--out",        default="checkpoints/policy_head.pt")
    parser.add_argument("--batch",      default=512,   type=int)
    parser.add_argument("--lr",         default=3e-4,  type=float)
    parser.add_argument("--warmup",     default=1000,  type=int)
    parser.add_argument("--epochs",     default=10,    type=int)
    parser.add_argument("--resume",     default=None,  type=str)
    parser.add_argument("--data-frac",  default=1.0,   type=float)
    args = parser.parse_args()
    main(args)
