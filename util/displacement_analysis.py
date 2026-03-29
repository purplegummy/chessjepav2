"""
Displacement vector analysis in encoder space.

For a sample of transition pairs (s_t, s_{t+1}), computes:
    delta = mean_pool(encoder(s_{t+1})) - mean_pool(encoder(s_t))

Then checks whether these delta vectors cluster by move type:
  - Capture vs quiet
  - Pawn vs piece
  - Kingside vs queenside
  - Castle
  - Promotion
  - Specific frequent moves (e.g. e2e4)

Outputs:
  1. Cosine similarity heatmap between move-type centroids
  2. PCA scatter coloured by move type
  3. Within-type vs between-type cosine similarity statistics
  4. Alignment score: mean cosine similarity of each move instance to its type centroid

Usage:
    python util/displacement_analysis.py \
        --checkpoint checkpoints/checkpoint_epoch5.pt \
        --dataset data/dataset.pt \
        --n_samples 8000 \
        --out displacement_results/
"""

import argparse
import os
import sys
import torch
import numpy as np
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# ── repo root on path ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jepa.encoder import Encoder
from util.chessdataset import ChessDataset

# ── AlphaZero action → UCI move ──────────────────────────────────────────────
# Standard AZ encoding: 73 planes × 64 squares = 4672 actions
# planes 0–55: queen moves (8 dirs × 7 distances)
# planes 56–63: knight moves
# planes 64–66: underpromotions (knight, bishop, rook) — last rank pawn moves
# We only need rough move properties, so we'll decode via python-chess board diff.

def action_to_uci(action_idx: int) -> str:
    """Convert AZ action index to UCI string. Returns '' on failure."""
    # 73 move planes, 64 squares
    plane = action_idx // 64
    sq    = action_idx %  64
    from_rank, from_file = sq // 8, sq % 8

    # Queen-like moves: planes 0-55
    if plane < 56:
        direction = plane // 7   # 0-7
        distance  = plane %  7 + 1  # 1-7
        dr = [0, 1, 1, 1, 0, -1, -1, -1][direction]
        df = [1, 1, 0, -1, -1, -1, 0, 1][direction]
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            promo = ""
            if plane >= 48:  # last 8 planes are queen underpromos from rank 6 to 7
                promo = "q"  # treat as queen promo tentatively; board diff overrides
            from_sq = chess.square(from_file, from_rank)
            to_sq   = chess.square(to_file, to_rank)
            return chess.square_name(from_sq) + chess.square_name(to_sq) + promo
    # Knight moves: planes 56-63
    elif plane < 64:
        knight_deltas = [(2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2),(-2,1),(-2,-1)]
        dr, df = knight_deltas[plane - 56]
        to_rank = from_rank + dr
        to_file = from_file + df
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            from_sq = chess.square(from_file, from_rank)
            to_sq   = chess.square(to_file, to_rank)
            return chess.square_name(from_sq) + chess.square_name(to_sq)
    # Underpromotions: planes 64-72
    else:
        promo_pieces = ["n", "b", "r"]
        # Same logic as pawn push to last rank
        promo = promo_pieces[(plane - 64) % 3]
        direction = (plane - 64) // 3  # 0=forward, 1=capture-left, 2=capture-right
        df_map = [0, -1, 1]
        df = df_map[direction]
        dr = 1  # white pawn; we won't distinguish colour here
        to_rank = from_rank + dr
        to_file = from_file + df
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            from_sq = chess.square(from_file, from_rank)
            to_sq   = chess.square(to_file, to_rank)
            return chess.square_name(from_sq) + chess.square_name(to_sq) + promo
    return ""


def classify_move(uci: str, state: torch.Tensor) -> dict:
    """
    Return a dict of binary labels for the move.
    state: (17, 8, 8) float tensor  (channel layout: see dataset.py)
    """
    labels = {
        "is_capture":    False,
        "is_pawn":       False,
        "is_promotion":  False,
        "is_castle":     False,
        "is_kingside":   False,
        "is_queenside":  False,
        "piece_type":    "unknown",
    }
    if len(uci) < 4:
        return labels

    from_sq_name = uci[:2]
    to_sq_name   = uci[2:4]
    promo        = uci[4:] if len(uci) > 4 else ""

    try:
        from_sq = chess.parse_square(from_sq_name)
        to_sq   = chess.parse_square(to_sq_name)
    except Exception:
        return labels

    from_file = chess.square_file(from_sq)
    to_file   = chess.square_file(to_sq)
    from_rank = chess.square_rank(from_sq)
    to_rank   = chess.square_rank(to_sq)

    # Determine if capture: destination square occupied by opponent.
    # Channel layout (from dataset.py, standard JEPA chess encoding):
    #   ch 0-5:  white pieces (P,N,B,R,Q,K)
    #   ch 6-11: black pieces (P,N,B,R,Q,K)
    #   ch 12:   en-passant
    #   ch 13-14: castling rights
    #   ch 15:   side to move (all 1 = white)
    #   ch 16:   move count (normalized)
    white_occ = state[:6, :, :].sum(0)   # (8,8)
    black_occ = state[6:12, :, :].sum(0)
    side_to_move_white = (state[15, 0, 0].item() > 0.5)

    dest_r, dest_f = to_rank, to_file
    if side_to_move_white:
        labels["is_capture"] = (black_occ[dest_r, dest_f].item() > 0)
    else:
        labels["is_capture"] = (white_occ[dest_r, dest_f].item() > 0)

    # En-passant capture
    if state[12, dest_r, dest_f].item() > 0 and not labels["is_capture"]:
        labels["is_capture"] = True  # en passant

    # Piece type from source square
    piece_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    if side_to_move_white:
        piece_channels = state[:6, from_rank, from_file]
    else:
        piece_channels = state[6:12, from_rank, from_file]

    for i, name in enumerate(piece_names):
        if piece_channels[i].item() > 0:
            labels["piece_type"] = name
            break

    labels["is_pawn"]      = (labels["piece_type"] == "pawn")
    labels["is_promotion"] = (len(promo) > 0)

    # Castling: king moves 2 squares horizontally
    if labels["piece_type"] == "king" and abs(to_file - from_file) == 2:
        labels["is_castle"] = True

    # Kingside / queenside  (file of destination)
    labels["is_kingside"]  = (to_file >= 4)
    labels["is_queenside"] = (to_file < 4)

    return labels


@torch.no_grad()
def extract_deltas(encoder, dataset, indices, device, batch_size=256):
    """Returns delta vectors (N, 256) and action list."""
    deltas  = []
    actions = []
    states_list = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start: start + batch_size]
        s_t  = torch.stack([dataset[i]["state"]      for i in batch_idx]).to(device)
        s_t1 = torch.stack([dataset[i]["next_state"] for i in batch_idx]).to(device)
        acts = [dataset[i]["action"].item()           for i in batch_idx]

        taps_t  = encoder(s_t)   # {2: (B,64,256), 4:..., 6:...}
        taps_t1 = encoder(s_t1)

        # Use last tap (layer 6) — most processed, closest to bottleneck
        z_t  = taps_t[6].mean(dim=1)   # (B, 256)
        z_t1 = taps_t1[6].mean(dim=1)

        delta = z_t1 - z_t             # (B, 256)
        deltas.append(delta.cpu())
        actions.extend(acts)
        states_list.extend([dataset[i]["state"] for i in batch_idx])

    return torch.cat(deltas, dim=0).numpy(), actions, states_list


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def within_between_stats(deltas_norm, group_ids):
    """Compute mean within-group and between-group cosine similarities."""
    unique_groups = sorted(set(group_ids))
    within, between = [], []

    group_vecs = {g: deltas_norm[np.array(group_ids) == g] for g in unique_groups}

    for g in unique_groups:
        vecs = group_vecs[g]
        if len(vecs) < 2:
            continue
        # sample pairs
        n = min(len(vecs), 200)
        idx = np.random.choice(len(vecs), (n, 2), replace=True)
        for i, j in idx:
            if i != j:
                within.append(float(vecs[i] @ vecs[j]))

    # between: sample one vec from each of two different groups
    group_list = [g for g in unique_groups if len(group_vecs[g]) >= 1]
    if len(group_list) < 2:
        return np.mean(within) if within else 0.0, np.std(within) if within else 0.0, 0.0, 0.0
    for _ in range(2000):
        g1, g2 = np.random.choice(len(group_list), 2, replace=False)
        g1, g2 = group_list[g1], group_list[g2]
        v1 = group_vecs[g1][np.random.randint(len(group_vecs[g1]))]
        v2 = group_vecs[g2][np.random.randint(len(group_vecs[g2]))]
        between.append(float(v1 @ v2))

    return np.mean(within), np.std(within), np.mean(between), np.std(between)


def centroid_cosine_matrix(deltas_norm, group_ids):
    unique_groups = sorted(set(group_ids))
    centroids = {}
    for g in unique_groups:
        vecs = deltas_norm[np.array(group_ids) == g]
        c = vecs.mean(0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids[g] = c
    n = len(unique_groups)
    mat = np.zeros((n, n))
    for i, g1 in enumerate(unique_groups):
        for j, g2 in enumerate(unique_groups):
            mat[i, j] = float(centroids[g1] @ centroids[g2])
    return mat, unique_groups, centroids


def alignment_score(deltas_norm, group_ids, centroids):
    """Mean cosine similarity of each delta to its group centroid."""
    scores = []
    for vec, g in zip(deltas_norm, group_ids):
        if g in centroids:
            scores.append(float(vec @ centroids[g]))
    return np.mean(scores), np.std(scores)


def plot_pca(deltas_norm, group_ids, group_name, out_path, max_pts=3000):
    pca = PCA(n_components=2)
    if len(deltas_norm) > max_pts:
        sel = np.random.choice(len(deltas_norm), max_pts, replace=False)
        vecs = deltas_norm[sel]
        ids  = [group_ids[i] for i in sel]
    else:
        vecs, ids = deltas_norm, group_ids

    coords = pca.fit_transform(vecs)
    unique = sorted(set(ids))
    cmap   = plt.cm.get_cmap("tab10", len(unique))
    color_map = {g: cmap(i) for i, g in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for g in unique:
        mask = np.array(ids) == g
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[g]], s=6, alpha=0.5, label=str(g))
    ax.set_title(f"PCA of Δ vectors — coloured by {group_name}\n"
                 f"var explained: {pca.explained_variance_ratio_.sum():.1%}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved PCA plot → {out_path}")


def plot_heatmap(mat, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3.5, len(labels) * 0.8)))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(mat[i,j]) < 0.6 else "white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved heatmap → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--dataset",    default="data/dataset.pt")
    parser.add_argument("--n_samples",  type=int, default=8000)
    parser.add_argument("--out",        default="displacement_results")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading checkpoint…")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Support both raw state-dict and wrapped checkpoint
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # Extract only encoder weights
    encoder = Encoder(tap_layers=(2, 4, 6)).to(device).eval()
    enc_sd  = {k.replace("encoder.", "", 1): v
               for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd, strict=True)
    print(f"  Encoder loaded ({sum(p.numel() for p in encoder.parameters()):,} params)")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset…")
    dataset = ChessDataset(args.dataset)
    N = len(dataset)
    print(f"  Dataset size: {N:,}")

    n = min(args.n_samples, N)
    indices = rng.choice(N, n, replace=False).tolist()

    # ── Extract deltas ────────────────────────────────────────────────────────
    print(f"Extracting Δ vectors for {n} pairs…")
    deltas, actions, states = extract_deltas(encoder, dataset, indices, device)
    print(f"  Δ shape: {deltas.shape}")

    # Normalise for cosine computations
    norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
    deltas_norm = deltas / norms

    # ── Decode move labels ────────────────────────────────────────────────────
    print("Decoding move labels…")
    ucis   = [action_to_uci(a) for a in actions]
    labels_list = [classify_move(u, s) for u, s in zip(ucis, states)]

    # Build group arrays
    capture_ids   = ["capture" if l["is_capture"]   else "quiet"    for l in labels_list]
    piece_ids     = [l["piece_type"]                                 for l in labels_list]
    side_ids      = ["kingside" if l["is_kingside"] else "queenside" for l in labels_list]
    castle_ids    = ["castle"   if l["is_castle"]   else "non-castle" for l in labels_list]

    # Frequent specific moves
    from collections import Counter
    uci_counts = Counter(ucis)
    top_moves  = [m for m, _ in uci_counts.most_common(12) if m]
    freq_ids   = [u if u in top_moves else "other" for u in ucis]

    # ── Stats: within vs between ──────────────────────────────────────────────
    print("\n=== Within-type vs Between-type cosine similarity ===")
    results_text = []
    for group_name, group_ids in [
        ("capture/quiet", capture_ids),
        ("piece type",    piece_ids),
        ("side",          side_ids),
        ("castle",        castle_ids),
        ("top moves",     freq_ids),
    ]:
        w_mean, w_std, b_mean, b_std = within_between_stats(deltas_norm, group_ids)
        sep = w_mean - b_mean
        line = (f"  {group_name:20s}  within={w_mean:+.4f}±{w_std:.4f}  "
                f"between={b_mean:+.4f}±{b_std:.4f}  separation={sep:+.4f}")
        print(line)
        results_text.append(line)

    # Alignment scores per grouping
    print("\n=== Alignment score (mean cosine to type centroid) ===")
    for group_name, group_ids in [
        ("capture/quiet", capture_ids),
        ("piece type",    piece_ids),
        ("side",          side_ids),
        ("top moves",     freq_ids),
    ]:
        _, _, centroids = centroid_cosine_matrix(deltas_norm, group_ids)
        a_mean, a_std = alignment_score(deltas_norm, group_ids, centroids)
        line = f"  {group_name:20s}  alignment={a_mean:+.4f}±{a_std:.4f}"
        print(line)
        results_text.append(line)

    # ── Centroid heatmaps ──────────────────────────────────────────────────────
    print("\nGenerating centroid heatmaps…")
    for group_name, group_ids, fname in [
        ("capture/quiet", capture_ids, "heatmap_capture.png"),
        ("piece type",    piece_ids,   "heatmap_piece.png"),
        ("top moves",     freq_ids,    "heatmap_topmoves.png"),
    ]:
        mat, labels_ax, centroids = centroid_cosine_matrix(deltas_norm, group_ids)
        plot_heatmap(mat, labels_ax, f"Centroid cosine sim — {group_name}",
                     os.path.join(args.out, fname))

    # ── PCA plots ──────────────────────────────────────────────────────────────
    print("Generating PCA plots…")
    for group_name, group_ids, fname in [
        ("capture vs quiet", capture_ids, "pca_capture.png"),
        ("piece type",       piece_ids,   "pca_piece.png"),
        ("top moves",        freq_ids,    "pca_topmoves.png"),
        ("side",             side_ids,    "pca_side.png"),
    ]:
        plot_pca(deltas_norm, group_ids, group_name,
                 os.path.join(args.out, fname))

    # ── Magnitude analysis ────────────────────────────────────────────────────
    print("\n=== Δ-vector magnitude by move type ===")
    mags = np.linalg.norm(deltas, axis=1)
    for group_name, group_ids in [
        ("capture/quiet", capture_ids),
        ("piece type",    piece_ids),
        ("castle",        castle_ids),
    ]:
        unique = sorted(set(group_ids))
        parts = []
        for g in unique:
            mask = np.array(group_ids) == g
            m = mags[mask]
            parts.append(f"{g}: {m.mean():.3f}±{m.std():.3f} (n={mask.sum()})")
        line = f"  {group_name:20s}  " + "  |  ".join(parts)
        print(line)
        results_text.append(line)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.out, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(results_text))
    print(f"\nSummary saved → {summary_path}")
    print(f"\nAll outputs in: {args.out}/")


if __name__ == "__main__":
    main()
