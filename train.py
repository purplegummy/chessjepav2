from jepa.jepa import ChessJEPA
import torch
from util.chessdataset import ChessDataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

def log_step(epoch: int, batch_idx: int, loss: float, l_pred: float, l_entropy: float):
    logging.info(
        f"epoch {epoch:>3} | batch {batch_idx:>6} | "
        f"loss {loss:.4f} | l_pred {l_pred:.4f} | l_entropy {l_entropy:.4f}"
    )


def save_checkpoint(model: ChessJEPA, optimizer: torch.optim.Optimizer, epoch: int, out_path: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, out_path)
    logging.info(f"Saved checkpoint to {out_path}")


LAMBDA = 0.1
def calc_loss(pred_logits: dict[int, torch.Tensor], target_indices: dict[int, torch.Tensor], criterion: torch.nn.CrossEntropyLoss) -> torch.Tensor:
    l_pred = 0.0
    for level in pred_logits:
        # pred_logits[level]: (B, 16, n_cats, n_codes)
        # target_indices[level]: (B, 16, n_cats)
        B, N, n_cats, n_codes = pred_logits[level].shape
        # reshape to (B*N*n_cats, n_codes) and (B*N*n_cats) for cross-entropy loss
        logits = pred_logits[level].view(B * N * n_cats, n_codes)
        targets = target_indices[level].view(B * N * n_cats)
        l_pred += criterion(logits, targets)
    
    l_pred /= len(pred_logits)  # average over the different tap levels

    l_entropy = 0.0  # Initialize entropy loss

    for level in pred_logits:
        probs = torch.softmax(pred_logits[level], dim=-1)  # (B, 16, n_cats, n_codes)
        avg_dist = probs.mean(dim=(0, 1))                  # (n_cats, n_codes)
        # Calculate the entropy of the average distribution
        # we want to maximize entropy so we use negative entropy as a loss (since we minimize the loss, minimizing negative entropy is equivalent to maximizing entropy)
        entropy = -torch.sum(avg_dist * torch.log(avg_dist + 1e-8))  # Add small value to prevent log(0)
        l_entropy -= entropy

    return l_pred + LAMBDA * l_entropy, l_pred, l_entropy

def validate(model: ChessJEPA, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            board_t  = data["state"].to(device)
            board_t1 = data["next_state"].to(device)
            actions  = data["action"].to(device)
            pred_logits, target_indices = model(board_t, board_t1, actions, tau=1.0)
            loss, _, _ = calc_loss(pred_logits, target_indices, criterion)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessJEPA().to(device)

    dataset = ChessDataset(args.data)
    val_size  = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            board_t  = data["state"].to(device)
            board_t1 = data["next_state"].to(device)
            actions  = data["action"].to(device)

            optimizer.zero_grad()
            pred_logits, target_indices = model(board_t, board_t1, actions, tau=1.0)
            loss, l_pred, l_entropy = calc_loss(pred_logits, target_indices, criterion)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                log_step(epoch, batch_idx, loss.item(), l_pred.item(), l_entropy.item())

        val_loss = validate(model, val_loader, criterion, device)
        logging.info(f"epoch {epoch:>3} | val_loss {val_loss:.4f}")

        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, args.out.replace(".pt", f"_epoch{epoch}.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",   default=64,                          type=int)
    parser.add_argument("--data",    default="data/dataset.pt")
    parser.add_argument("--out",     default="checkpoints/checkpoint.pt")
    parser.add_argument("--epochs",  default=10,                          type=int)
    args = parser.parse_args()

    main(args)
