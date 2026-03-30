from jepa.jepa import ChessJEPA
import torch

from util.chessdataset import ChessDataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

LAMBDA_ENTROPY = 0.05
LAMBDA_INV     = 0.5


def log_step(epoch, batch_idx, loss, l_pred, l_entropy, l_inv):
    logging.info(
        f"epoch {epoch:>3} | batch {batch_idx:>6} | "
        f"loss {loss:.4f} | l_pred {l_pred:.4f} | l_entropy {l_entropy:.4f} | "
        f"l_inv {l_inv:.4f}"
    )


def save_checkpoint(model, optimizer, epoch, out_path):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
    }, out_path)
    logging.info(f"Saved checkpoint → {out_path}")


def calc_loss(pred_logits, target_indices, bottleneck_logits,
              inv_logits, actions, criterion,
              lambda_entropy=LAMBDA_ENTROPY, lambda_inv=LAMBDA_INV):

    l_pred = 0.0
    for level in pred_logits:
        B, N, n_cats, n_codes = pred_logits[level].shape
        logits  = pred_logits[level].view(B * N * n_cats, n_codes)
        targets = target_indices[level].view(B * N * n_cats)
        l_pred += criterion(logits, targets)
    l_pred /= len(pred_logits)

    l_entropy = 0.0
    eps = 1e-8
    for level in bottleneck_logits:
        probs = torch.softmax(bottleneck_logits[level], dim=-1)
        avg_dist = probs.mean(dim=(0, 1))
        entropy_per_cat = -torch.sum(avg_dist * torch.log(avg_dist + eps), dim=-1)
        l_entropy -= entropy_per_cat.mean()
    l_entropy /= len(bottleneck_logits)

    l_inv = criterion(inv_logits, actions)

    total = l_pred + lambda_entropy * l_entropy + lambda_inv * l_inv

    return total, l_pred, l_entropy, l_inv


def validate(model, dataloader, criterion, device, lambdas):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            board_t  = data["state"].to(device)
            board_t1 = data["next_state"].to(device)
            actions  = data["action"].to(device)

            pred_logits, target_indices, bottleneck_logits, inv_logits = \
                model(board_t, board_t1, actions, tau=1.0)

            loss, *_ = calc_loss(
                pred_logits, target_indices, bottleneck_logits,
                inv_logits, actions, criterion, **lambdas,
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def log_model_info(model, device):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"device:         {device}")
    logging.info(f"total params:   {total:,}")
    logging.info(f"trainable:      {trainable:,}")
    logging.info(f"GPUs available: {torch.cuda.device_count()}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessJEPA(dropout=args.dropout).to(device)
    log_model_info(model, device)

    dataset = ChessDataset(args.data)
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

    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion   = torch.nn.CrossEntropyLoss()
    lambdas     = dict(lambda_entropy=args.lambda_entropy, lambda_inv=args.lambda_inv)
    logging.info(f"lambdas: {lambdas}")
    total_steps = (args.total_epochs or args.epochs) * (train_size // args.batch)
    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.lr)
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * (train_size // args.batch)
        logging.info(f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup,
        last_epoch=min(global_step, args.warmup) - 1,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup, eta_min=1e-6,
        last_epoch=max(-1, global_step - args.warmup - 1),
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            board_t  = data["state"].to(device)
            board_t1 = data["next_state"].to(device)
            actions  = data["action"].to(device)

            tau = max(0.1, 1.0 - 0.9 * (global_step / total_steps))

            optimizer.zero_grad()
            pred_logits, target_indices, bottleneck_logits, inv_logits = \
                model(board_t, board_t1, actions, tau=tau)

            loss, l_pred, l_entropy, l_inv = calc_loss(
                pred_logits, target_indices, bottleneck_logits,
                inv_logits, actions, criterion, **lambdas,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step < args.warmup:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            global_step += 1

            if batch_idx % 100 == 0:
                log_step(epoch, batch_idx, loss.item(), l_pred.item(),
                         l_entropy.item(), l_inv.item())

        val_loss = validate(model, val_loader, criterion, device, lambdas)
        logging.info(f"epoch {epoch:>3} | val_loss {val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, args.out.replace(".pt", f"_epoch{epoch}.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",          default=512,   type=int)
    parser.add_argument("--lr",             default=3e-4,  type=float)
    parser.add_argument("--warmup",         default=1000,  type=int)
    parser.add_argument("--dropout",        default=0.1,   type=float)
    parser.add_argument("--data",           default="data/dataset.pt")
    parser.add_argument("--out",            default="checkpoints/checkpoint.pt")
    parser.add_argument("--epochs",         default=10,    type=int)
    parser.add_argument("--total-epochs",   default=None,  type=int)
    parser.add_argument("--resume",         default=None,  type=str)
    parser.add_argument("--data-frac",      default=1.0,   type=float)
    parser.add_argument("--lambda-entropy", default=0.05,  type=float)
    parser.add_argument("--lambda-inv",     default=0.5,   type=float)
    args = parser.parse_args()
    main(args)
