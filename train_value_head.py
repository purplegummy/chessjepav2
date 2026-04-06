"""
Train a ValueHead on top of a frozen JEPA encoder.

Targets are centipawn evals clipped to [-1000, 1000] and normalized to [-1, 1].

Usage:
    python train_value_head.py \
        --jepa_ckpt checkpoints/checkpoint_epoch6.pt \
        --data      data/dataset.pt \
        --out       checkpoints/value_head.pt
"""

import torch
from torch.utils.data import DataLoader, random_split
import logging

from jepa.jepa import ChessJEPA
from jepa.head import ValueHead
from util.chessdataset import ChessDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

EVAL_CLIP = 3000.0  # centipawns


def normalize_eval(evals: torch.Tensor) -> torch.Tensor:
    return evals.float().clamp(-EVAL_CLIP, EVAL_CLIP) / EVAL_CLIP


def save_checkpoint(head, optimizer, epoch, out_path):
    torch.save({
        "model_state_dict":     head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
    }, out_path)
    logging.info(f"Saved checkpoint → {out_path}")


def validate(head, encoder, dataloader, device):
    head.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for data in dataloader:
            states = data["state"].to(device)
            evals  = normalize_eval(data["eval"]).to(device)

            taps  = encoder(states)
            preds = head(taps[max(taps.keys())])
            total_loss += criterion(preds, evals).item()

    return total_loss / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading JEPA from {args.jepa_ckpt}…")
    jepa = ChessJEPA().to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    encoder = jepa.encoder

    head = ValueHead().to(device)
    logging.info(f"ValueHead params: {sum(p.numel() for p in head.parameters()):,}")

    dataset = ChessDataset(args.data)
    if not dataset.has_evals:
        raise ValueError("Dataset has no 'evals' key — run util/add_next_evals.py first")

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
    criterion   = torch.nn.MSELoss()
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
            states = data["state"].to(device)
            evals  = normalize_eval(data["eval"]).to(device)

            with torch.no_grad():
                taps = encoder(states)

            optimizer.zero_grad()
            preds = head(taps[max(taps.keys())])
            loss  = criterion(preds, evals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            if global_step < args.warmup:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            global_step += 1

            if batch_idx % 100 == 0:
                logging.info(
                    f"epoch {epoch:>3} | batch {batch_idx:>6} | loss {loss.item():.4f}"
                )

        val_loss = validate(head, encoder, val_loader, device)
        logging.info(f"epoch {epoch:>3} | val_loss {val_loss:.4f}")
        save_checkpoint(head, optimizer, epoch, args.out.replace(".pt", f"_epoch{epoch}.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--data",      default="data/dataset.pt")
    parser.add_argument("--out",       default="checkpoints/value_head.pt")
    parser.add_argument("--batch",     default=512,  type=int)
    parser.add_argument("--lr",        default=3e-4, type=float)
    parser.add_argument("--warmup",    default=1000, type=int)
    parser.add_argument("--epochs",    default=10,   type=int)
    parser.add_argument("--resume",    default=None, type=str)
    parser.add_argument("--data-frac", default=1.0,  type=float)
    args = parser.parse_args()
    main(args)
