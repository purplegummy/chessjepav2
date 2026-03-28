"""
server.py — Flask backend for ChessJEPA GUI.

Run with:
    python app/server.py \
        --jepa_ckpt checkpoints/checkpoint_epoch5.pt

Endpoints
---------
GET  /              — serves the chess UI
POST /api/best_move — uses latent arithmetic to rank moves by embedding delta
                      projected onto a learned "winning direction"
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess
import torch
import numpy as np
from flask import Flask, jsonify, render_template, request

from jepa.jepa import ChessJEPA
from util.parse import board_to_tensor

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--host",   default="127.0.0.1")
parser.add_argument("--port",   default=5000, type=int)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Model (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
N_CATS  = 8
N_CODES = 16
BOTTLENECK_TAU = 1e-5

device = torch.device(args.device)
print(f"Loading JEPA from {args.jepa_ckpt} on {device}…")
jepa = ChessJEPA(n_cats=N_CATS, n_codes=N_CODES).to(device)
ckpt = torch.load(args.jepa_ckpt, map_location=device)
jepa.load_state_dict(ckpt["model_state_dict"])
jepa.eval()
for p in jepa.parameters():
    p.requires_grad = False

encoder    = jepa.encoder
bottleneck = jepa.bottleneck
print("Model ready.")

# Winning direction: will be estimated lazily from the first batch of positions
# seen, or can be set externally. Stored as a unit vector in bottleneck space.
_win_direction: torch.Tensor | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)


def encode(board: chess.Board) -> torch.Tensor:
    """Return flattened discrete bottleneck codes for a board: (64*8*16,)."""
    tensor = board_to_tensor(board).float().unsqueeze(0).to(device)
    with torch.no_grad():
        taps = encoder(tensor)
        z, _ = bottleneck(taps[max(taps.keys())], tau=BOTTLENECK_TAU)
    return z.squeeze(0).flatten()  # (8192,)


def winning_direction(boards: list[chess.Board]) -> torch.Tensor:
    """
    Estimate the winning direction from a set of positions using material
    difference as a proxy: mean(material>0 embeddings) - mean(material<0).
    Returns a unit vector in embedding space.
    """
    pos, neg = [], []
    for b in boards:
        z = encode(b)
        mat = sum(
            len(b.pieces(pt, chess.WHITE)) * v - len(b.pieces(pt, chess.BLACK)) * v
            for pt, v in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),
                          (chess.ROOK,5),(chess.QUEEN,9)]
        )
        if mat > 0:
            pos.append(z)
        elif mat < 0:
            neg.append(z)

    if not pos or not neg:
        # fallback: random unit vector (no signal available)
        v = torch.randn(8192, device=device)
    else:
        v = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
    return v / (v.norm() + 1e-8)


def pick_move(board: chess.Board, top_n: int = 5):
    """
    Latent arithmetic move selection.
    Score each legal move by projecting the embedding delta
    (z_after - z_before) onto the winning direction.
    """
    global _win_direction

    legal = list(board.legal_moves)
    if not legal:
        return None, []

    z_before = encode(board)

    # Build the winning direction lazily from candidate positions
    if _win_direction is None:
        candidate_boards = []
        for m in legal:
            b2 = board.copy()
            b2.push(m)
            candidate_boards.append(b2)
        _win_direction = winning_direction(candidate_boards)

    # Score each move: delta projected onto winning direction
    # Flip sign if it's black's turn (black wants to minimise white's advantage)
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scored = []
    for m in legal:
        b2 = board.copy()
        b2.push(m)
        z_after = encode(b2)
        score = float(sign * (z_after - z_before) @ _win_direction)
        scored.append((m, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_move = scored[0][0]

    # Normalise scores to [0,1] range for display
    raw = [s for _, s in scored]
    lo, hi = min(raw), max(raw)
    span = (hi - lo) if hi != lo else 1.0
    top_moves = [
        {"san": board.san(m), "uci": m.uci(), "prob": round((s - lo) / span, 4)}
        for m, s in scored[:top_n]
    ]

    return best_move, top_moves


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/best_move", methods=["POST"])
def best_move():
    data  = request.get_json(force=True)
    fen   = data.get("fen", chess.STARTING_FEN)
    top_n = data.get("top_n", 5)

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    if board.is_game_over():
        return jsonify({"error": "Game over"}), 400

    move, top_moves = pick_move(board, top_n=top_n)

    if move is None:
        return jsonify({"error": "No legal moves"}), 400

    confidence = top_moves[0]["prob"] if top_moves else 0.0

    return jsonify({
        "move":       move.uci(),
        "san":        board.san(move),
        "confidence": confidence,
        "top_moves":  top_moves,
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
