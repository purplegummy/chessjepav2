from jepa.jepa import ChessJEPA
from jepa.delta_predictor import DeltaPredictor
import torch
import torch.nn.functional as F
from util.chessdataset import ChessDataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

LAMBDA_DELTA = 1.0


def log_step(epoch, batch_idx, loss, l_delta):
    logging.info(
        f"epoch {epoch:>3} | batch {batch_idx:>6} | "
        f"loss {loss:.4f} | l_delta {l_delta:.4f}"
    )


def save_checkpoint(jepa, delta_pred, optimizer, epoch, out_path):
    torch.save({
        "jepa_state_dict":         jepa.state_dict(),
        "delta_pred_state_dict":   delta_pred.state_dict(),
        "optimizer_state_dict":    optimizer.state_dict(),
        "epoch":                   epoch,
    }, out_path)
    logging.info(f"Saved checkpoint → {out_path}")


def calc_loss(z_t1_pred, z_t1_target):
    """MSE between predicted and target bottleneck soft codes."""
    return F.mse_loss(z_t1_pred, z_t1_target)


def validate(jepa, delta_pred, dataloader, device):
    jepa.eval()
    delta_pred.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            board_t     = data["state"].to(device)
            board_t1    = data["next_state"].to(device)
            delta_evals = data["delta_eval"].float().to(device)

            taps_t  = jepa.encoder(board_t)
            taps_t1 = jepa.encoder(board_t1)
            z_t,  _ = jepa.bottleneck(taps_t[max(taps_t.keys())],   tau=0.1)
            z_t1, _ = jepa.bottleneck(taps_t1[max(taps_t1.keys())], tau=0.1)

            z_t1_pred = delta_pred(z_t, delta_evals)
            loss = calc_loss(z_t1_pred, z_t1.detach())
            total_loss += loss.item()
    return total_loss / len(dataloader)


def log_model_info(delta_pred, device):
    total = sum(p.numel() for p in delta_pred.parameters())
    logging.info(f"device:               {device}")
    logging.info(f"delta_pred params:    {total:,}")
    logging.info(f"GPUs available:       {torch.cuda.device_count()}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen JEPA encoder + bottleneck
    print(f"Loading JEPA from {args.jepa_ckpt}…")
    jepa = ChessJEPA(dropout=0.0).to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    delta_pred = DeltaPredictor(dropout=args.dropout).to(device)
    log_model_info(delta_pred, device)

    dataset = ChessDataset(args.data)
    if "delta_evals" not in torch.load(args.data, map_location="cpu", weights_only=True):
        raise ValueError("dataset has no 'delta_evals' — run util/add_next_evals.py first")

    if args.data_frac < 1.0:
        keep = int(args.data_frac * len(dataset))
        dataset, _ = torch.utils.data.random_split(dataset, [keep, len(dataset) - keep])
        logging.info(f"Using {args.data_frac:.0%} of data → {keep:,} samples")

    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    optimizer   = torch.optim.Adam(delta_pred.parameters(), lr=args.lr)
    total_steps = (args.total_epochs or args.epochs) * (train_size // args.batch)
    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        delta_pred.load_state_dict(ckpt["delta_pred_state_dict"])
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
        delta_pred.train()
        for batch_idx, data in enumerate(train_loader):
            board_t     = data["state"].to(device)
            board_t1    = data["next_state"].to(device)
            delta_evals = data["delta_eval"].float().to(device)

            with torch.no_grad():
                taps_t  = jepa.encoder(board_t)
                taps_t1 = jepa.encoder(board_t1)
                z_t,  _ = jepa.bottleneck(taps_t[max(taps_t.keys())],   tau=0.1)
                z_t1, _ = jepa.bottleneck(taps_t1[max(taps_t1.keys())], tau=0.1)

            optimizer.zero_grad()
            z_t1_pred = delta_pred(z_t, delta_evals)
            loss = calc_loss(z_t1_pred, z_t1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(delta_pred.parameters(), 1.0)
            optimizer.step()

            if global_step < args.warmup:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            global_step += 1

            if batch_idx % 100 == 0:
                log_step(epoch, batch_idx, loss.item(), loss.item())

        val_loss = validate(jepa, delta_pred, val_loader, device)
        logging.info(f"epoch {epoch:>3} | val_loss {val_loss:.4f}")
        save_checkpoint(jepa, delta_pred, optimizer, epoch,
                        args.out.replace(".pt", f"_epoch{epoch}.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt",    required=True)
    parser.add_argument("--batch",        default=512,   type=int)
    parser.add_argument("--lr",           default=3e-4,  type=float)
    parser.add_argument("--warmup",       default=1000,  type=int)
    parser.add_argument("--dropout",      default=0.1,   type=float)
    parser.add_argument("--data",         default="data/dataset.pt")
    parser.add_argument("--out",          default="checkpoints/delta_pred.pt")
    parser.add_argument("--epochs",       default=10,    type=int)
    parser.add_argument("--total-epochs", default=None,  type=int)
    parser.add_argument("--resume",       default=None,  type=str)
    parser.add_argument("--data-frac",    default=1.0,   type=float)
    args = parser.parse_args()
    main(args)
