"""
Latent rollout evaluation for ChessJEPA.

Encodes a sequence of real transitions (s_0, a_0, s_1, a_1, ..., s_H) through
the bottleneck to get ground-truth latent codes z_0..z_H, then rolls out from
z_0 using the predictor with real actions and compares ẑ_t against z_t.

Metrics reported per horizon step:
  - MSE between predicted and target one-hot latents (float)
  - Cosine similarity between predicted and target one-hot latents
  - Token accuracy: fraction of (patch, category) tokens where argmax(ẑ_t) == argmax(z_t)

Usage:
    python eval_rollout.py --checkpoint checkpoints/jepa.pt --data data/dataset_test.pt
    python eval_rollout.py --checkpoint checkpoints/jepa.pt --data data/dataset_test.pt \
        --horizon 10 --n-seqs 500 --level 6
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jepa.jepa import ChessJEPA
from util.chessdataset import ChessDataset


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def extract_sequences(dataset: ChessDataset, horizon: int, n_seqs: int):
    """
    Pull `n_seqs` contiguous windows of length `horizon+1` from the flat dataset.

    The dataset stores transitions in game order, so a window [i, i+horizon]
    is a real game sequence as long as it doesn't cross a game boundary.
    We detect boundaries by checking next_state[t] ≈ state[t+1]; if they differ
    we skip that window and try the next one.

    Returns:
        states  : (n_seqs, H+1, 17, 8, 8)  float
        actions : (n_seqs, H)               long   — a_0 .. a_{H-1}
    """
    data = dataset.data
    states_all  = data["states"]       # (N, 17, 8, 8)
    actions_all = data["actions"]      # (N,)
    N = len(actions_all)

    seq_states  = []
    seq_actions = []

    i = 0
    while len(seq_states) < n_seqs and i + horizon < N:
        # Check that this window doesn't cross a game boundary.
        # next_state[t] should equal state[t+1] within the same game.
        valid = True
        for t in range(horizon):
            if not torch.equal(data["next_states"][i + t], states_all[i + t + 1]):
                valid = False
                i += t + 1   # skip past the break point
                break

        if not valid:
            continue

        s_window = states_all[i : i + horizon + 1].float()   # (H+1, 17, 8, 8)
        a_window = actions_all[i : i + horizon].long()        # (H,)

        seq_states.append(s_window)
        seq_actions.append(a_window)
        i += 1

    if len(seq_states) == 0:
        raise RuntimeError(
            f"Could not find any valid sequences of horizon {horizon}. "
            "Try a smaller horizon or a larger dataset."
        )

    if len(seq_states) < n_seqs:
        print(f"Warning: only found {len(seq_states)} valid sequences (requested {n_seqs})")

    return torch.stack(seq_states), torch.stack(seq_actions)


# ---------------------------------------------------------------------------
# Rollout eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_rollout(
    model: ChessJEPA,
    states: torch.Tensor,   # (M, H+1, 17, 8, 8)
    actions: torch.Tensor,  # (M, H)
    level: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict[int, dict[str, float]]:
    """
    For each horizon step t in 1..H, compute:
        mse, cosine_sim, token_acc
    averaged over all sequences in the batch.

    Returns a dict: {t: {"mse": float, "cos": float, "acc": float}}
    """
    model.eval()
    M, Hp1, *board_shape = states.shape
    H = Hp1 - 1

    # Accumulators per horizon step
    sum_mse = torch.zeros(H, device=device)
    sum_cos = torch.zeros(H, device=device)
    sum_acc = torch.zeros(H, device=device)
    count   = 0

    for start in range(0, M, batch_size):
        s_batch = states[start : start + batch_size].to(device)   # (B, H+1, 17, 8, 8)
        a_batch = actions[start : start + batch_size].to(device)  # (B, H)
        B = s_batch.shape[0]

        # --- 1. Encode ALL time steps to get ground-truth latents ---
        # Reshape to (B*(H+1), 17, 8, 8) so we can run a single encoder pass.
        s_flat = s_batch.view(B * Hp1, *board_shape)
        taps   = model.encoder(s_flat)                       # {level: (B*(H+1), 16, 256)}
        feat   = taps[level]                                 # (B*(H+1), 16, 256)
        z_hard, _ = model.bottleneck(feat, tau=1.0)          # (B*(H+1), 16, n_cats, n_codes) one-hot
        z_hard = z_hard.view(B, Hp1, *z_hard.shape[1:])     # (B, H+1, 16, n_cats, n_codes)

        # Ground-truth latents at each step: z_gt[t] = z_hard[:, t]
        # z_0 is our starting point; z_1..z_H are targets.

        # --- 2. Roll out from z_0 using the predictor ---
        z_pred = z_hard[:, 0]   # (B, 16, n_cats, n_codes)  — start from z_0

        for t in range(H):
            a_t     = a_batch[:, t]                          # (B,)
            logits  = model.predictor(z_pred, a_t)           # (B, 16, n_cats, n_codes) logits
            z_gt_t1 = z_hard[:, t + 1]                       # (B, 16, n_cats, n_codes) one-hot

            # Next predicted latent: argmax → one-hot (hard, so rollout stays discrete)
            idx     = logits.argmax(dim=-1)                  # (B, 16, n_cats)
            n_codes = logits.shape[-1]
            z_pred  = F.one_hot(idx, num_classes=n_codes).float()  # (B, 16, n_cats, n_codes)

            # --- Metrics ---
            flat_pred = z_pred.view(B, -1)                   # (B, 16*n_cats*n_codes)
            flat_gt   = z_gt_t1.view(B, -1)

            # MSE
            mse = F.mse_loss(flat_pred, flat_gt, reduction="none").mean(dim=-1)  # (B,)
            sum_mse[t] += mse.sum()

            # Cosine similarity
            cos = F.cosine_similarity(flat_pred, flat_gt, dim=-1)  # (B,)
            sum_cos[t] += cos.sum()

            # Token accuracy: does argmax of predicted logits match argmax of ground-truth?
            # ground-truth is one-hot so argmax(z_gt) = the single hot index
            pred_idx = logits.argmax(dim=-1)                 # (B, 16, n_cats)
            gt_idx   = z_gt_t1.argmax(dim=-1)                # (B, 16, n_cats)
            acc      = (pred_idx == gt_idx).float().mean(dim=(1, 2))  # (B,)
            sum_acc[t] += acc.sum()

        count += B

    results = {}
    for t in range(H):
        results[t + 1] = {
            "mse": (sum_mse[t] / count).item(),
            "cos": (sum_cos[t] / count).item(),
            "acc": (sum_acc[t] / count).item(),
        }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChessJEPA latent rollout evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/jepa.pt",
                        help="Path to ChessJEPA checkpoint (.pt)")
    parser.add_argument("--data", default="data/dataset_test.pt",
                        help="Path to test dataset (.pt)")
    parser.add_argument("--horizon", type=int, default=8,
                        help="Number of prediction steps (H)")
    parser.add_argument("--n-seqs", type=int, default=500,
                        help="Number of sequences to evaluate over")
    parser.add_argument("--level", type=int, default=6,
                        help="Which encoder tap level to use (2, 4, or 6)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (auto-detected if omitted)")
    # ChessJEPA hparams (must match checkpoint)
    parser.add_argument("--n-cats",    type=int, default=8)
    parser.add_argument("--n-codes",   type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Model
    tap_layers = (2, 4, 6)
    model = ChessJEPA(
        n_cats=args.n_cats,
        n_codes=args.n_codes,
        embed_dim=args.embed_dim,
        tap_layers=tap_layers,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    assert args.level in tap_layers, f"--level must be one of {tap_layers}"

    # Data
    dataset = ChessDataset(args.data)
    print(f"Dataset size: {len(dataset)} transitions")

    print(f"Extracting {args.n_seqs} sequences of horizon {args.horizon}...")
    states, actions = extract_sequences(dataset, args.horizon, args.n_seqs)
    print(f"Sequences shape: states={tuple(states.shape)}, actions={tuple(actions.shape)}")

    # Eval
    print(f"\nRunning rollout eval at encoder level {args.level}...\n")
    results = eval_rollout(
        model, states, actions,
        level=args.level,
        device=device,
        batch_size=args.batch_size,
    )

    # Print table
    print(f"{'horizon':>8}  {'MSE':>10}  {'cosine_sim':>12}  {'token_acc':>10}")
    print("-" * 46)
    for t, metrics in sorted(results.items()):
        print(
            f"{t:>8d}  {metrics['mse']:>10.5f}  "
            f"{metrics['cos']:>12.5f}  {metrics['acc']:>10.4f}"
        )


if __name__ == "__main__":
    main()
