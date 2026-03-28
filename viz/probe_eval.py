"""
Probe whether Stockfish eval is linearly decodable from JEPA embeddings.

Usage:
    python viz/probe_eval.py \
        --fens data/fen_analysis.csv \
        --jepa_ckpt checkpoints/checkpoint_epoch4.pt \
        --num_samples 500 \
        --stockfish /opt/homebrew/bin/stockfish
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report
from sklearn.preprocessing import StandardScaler

from viz.evaluate_embeddings import load_checkpoint, load_and_encode
from jepa.jepa import ChessJEPA


def probe_regression(X_train, X_val, y_train, y_val):
    """Ridge regression: can we predict the raw centipawn eval?"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    probe = Ridge(alpha=1.0)
    probe.fit(X_train_s, y_train)
    preds = probe.predict(X_val_s)

    r2  = r2_score(y_val, preds)
    mae = np.mean(np.abs(preds - y_val))
    print(f"\n=== Regression probe (raw centipawn eval) ===")
    print(f"  R²  : {r2:.4f}   (1.0 = perfect, 0.0 = no better than mean)")
    print(f"  MAE : {mae:.1f} cp")

    # Winning direction vector (unnormalised)
    win_vec = probe.coef_
    print(f"  Win-direction norm: {np.linalg.norm(win_vec):.4f}")
    return probe, scaler, win_vec


def probe_classification(X_train, X_val, y_train_label, y_val_label):
    """Logistic regression: can we classify winning / equal / losing?"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_s, y_train_label)
    preds = clf.predict(X_val_s)

    print(f"\n=== Classification probe (winning / equal / losing) ===")
    print(classification_report(y_val_label, preds, zero_division=0))
    return clf, scaler


def latent_arithmetic(embeddings: np.ndarray, evals: np.ndarray):
    """
    Compute a 'winning direction' as the mean difference between
    high-eval and low-eval embeddings, then score all positions.
    """
    winning = embeddings[evals >  150]
    losing  = embeddings[evals < -150]

    if len(winning) == 0 or len(losing) == 0:
        print("\n[skip] Not enough winning/losing samples for latent arithmetic.")
        return

    win_vec = winning.mean(0) - losing.mean(0)
    win_vec /= np.linalg.norm(win_vec) + 1e-8

    scores = embeddings @ win_vec
    corr   = np.corrcoef(scores, evals)[0, 1]
    print(f"\n=== Latent arithmetic ===")
    print(f"  Mean(winning) - Mean(losing) direction")
    print(f"  Pearson r with Stockfish eval: {corr:.4f}")
    print(f"  (|r| > 0.5 suggests the direction is meaningful)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens",        default="data/fen_analysis.csv")
    parser.add_argument("--jepa_ckpt",   default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--stockfish",   default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth",       default=12, type=int)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = ChessJEPA().to(device)
    model = load_checkpoint(model, args.jepa_ckpt, device)

    print(f"Encoding {args.num_samples} positions...")
    embeddings, metadata = load_and_encode(
        args.fens, model, device, args.num_samples, args.stockfish, args.depth
    )

    X      = np.stack([e.numpy() for e in embeddings])
    evals  = np.array([m["eval_cp"]    for m in metadata], dtype=float)
    labels = np.array([m["eval_label"] for m in metadata])

    print(f"\nDataset: {len(X)} positions, embedding dim={X.shape[1]}")
    print(f"  Winning: {(labels=='winning').sum()}  "
          f"Equal: {(labels=='equal').sum()}  "
          f"Losing: {(labels=='losing').sum()}")

    X_tr, X_val, e_tr, e_val, l_tr, l_val = train_test_split(
        X, evals, labels, test_size=0.25, random_state=42
    )

    probe_regression(X_tr, X_val, e_tr, e_val)
    probe_classification(X_tr, X_val, l_tr, l_val)
    latent_arithmetic(X, evals)

    print("\n--- Interpretation guide ---")
    print("  R² ≥ 0.5  → eval signal is strongly present in embeddings")
    print("  R² 0.2–0.5 → partial signal; latent arithmetic may help")
    print("  R² < 0.2  → eval is not linearly encoded; consider retraining with eval loss")


if __name__ == "__main__":
    main()
