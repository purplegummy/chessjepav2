"""
server.py — Flask backend for ChessJEPA GUI.

Run with:
    python app/server.py \
        --jepa_ckpt      checkpoints/checkpoint_epoch5.pt \
        --organizer_ckpt checkpoints/organizer.pt

Endpoints
---------
GET  /              — serves the chess UI
POST /api/best_move — predictor-based filtering pipeline:
                      1. Encode current position → z_t
                      2. Run all legal moves through Predictor → imagined z_{t+1}
                      3. Score via EvalOrganizer projected onto eval_head weight (win direction)
                      4. Prune moves worse than current position, keep top-k for lookahead
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess
import torch
from flask import Flask, jsonify, render_template, request

from jepa.jepa import ChessJEPA
from jepa.eval_organizer import EvalOrganizer
from util.parse import board_to_tensor

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt",      required=True)
parser.add_argument("--organizer_ckpt", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--host",   default="127.0.0.1")
parser.add_argument("--port",   default=5001, type=int)
parser.add_argument("--top_k",  default=3, type=int, help="moves kept per ply for lookahead")
parser.add_argument("--depth",  default=2, type=int, help="lookahead depth")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device(args.device)

print(f"Loading JEPA from {args.jepa_ckpt}…")
jepa = ChessJEPA().to(device)
ckpt = torch.load(args.jepa_ckpt, map_location=device)
jepa.load_state_dict(ckpt["model_state_dict"])
jepa.eval()
for p in jepa.parameters():
    p.requires_grad_(False)

encoder = jepa.encoder

print(f"Loading organizer from {args.organizer_ckpt}…")
org_ckpt  = torch.load(args.organizer_ckpt, map_location=device)
organizer = EvalOrganizer(
    latent_dim=org_ckpt["latent_dim"],
    hidden_dim=org_ckpt["hidden_dim"],
    tap_dim=org_ckpt.get("tap_dim", 256),
    n_patches=org_ckpt.get("n_patches", 64),
    val_bottleneck=org_ckpt.get("val_bottleneck", 32),
).to(device)
organizer.load_state_dict(org_ckpt["model_state_dict"])
organizer.eval()
for p in organizer.parameters():
    p.requires_grad_(False)

print("Models ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)


def encode_position(board: chess.Board) -> torch.Tensor:
    """Encode a single position to organizer latent z: (1, latent_dim)."""
    t = board_to_tensor(board).float().unsqueeze(0).to(device)
    with torch.no_grad():
        tap_dict = encoder(t)
        last_tap = tap_dict[max(tap_dict.keys())]
        z, _, _ = organizer(last_tap)
    return z


def encode_moves(board: chess.Board, moves: list[chess.Move]) -> torch.Tensor:
    """Encode resulting positions to organizer latent z: (N, latent_dim)."""
    tensors = []
    for m in moves:
        b2 = board.copy()
        b2.push(m)
        tensors.append(board_to_tensor(b2).float())
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        tap_dict = encoder(batch)
        last_tap = tap_dict[max(tap_dict.keys())]
        z, _, _ = organizer(last_tap)
    return z


def score_moves(board: chess.Board, moves: list[chess.Move]) -> list[float]:
    """Score moves by how much their latent delta aligns with the winning direction."""
    val_weight = organizer.val_head.weight  # (1, latent_dim)

    z_curr  = encode_position(board)     # (1, latent_dim)
    z_nexts = encode_moves(board, moves) # (N, latent_dim)

    delta = z_nexts - z_curr             # (N, latent_dim)

    delta_norm  = delta      / (delta.norm(dim=-1, keepdim=True)      + 1e-8)
    weight_norm = val_weight / (val_weight.norm(dim=-1, keepdim=True) + 1e-8)

    alignment = torch.matmul(delta_norm, weight_norm.T).squeeze(-1)  # (N,)
    return alignment.tolist()


def negamax(board: chess.Board, depth: int) -> float:
    """
    Negamax: always returns best score from the current player's POV.
    Uses organizer eval on resulting positions, keeps top_k survivors per ply.
    """
    legal = list(board.legal_moves)
    if not legal or depth == 0:
        if not legal:
            return 0.0
        scores = score_moves(board, legal)
        return max(scores)

    scores = score_moves(board, legal)
    ranked = sorted(zip(legal, scores), key=lambda x: x[1], reverse=True)
    survivors = ranked[:args.top_k]

    best = float("-inf")
    for m, _ in survivors:
        b2 = board.copy()
        b2.push(m)
        score = -negamax(b2, depth - 1)
        best = max(best, score)
    return best


def pick_move(board: chess.Board, top_n: int = 5):
    legal = list(board.legal_moves)
    if not legal:
        return None, []

    scores = score_moves(board, legal)
    ranked = sorted(zip(legal, scores), key=lambda x: x[1], reverse=True)

    print(f"\n[debug] turn={'white' if board.turn == chess.WHITE else 'black'}")
    for m, s in ranked[:5]:
        print(f"  {board.san(m):10s}  score={s:.4f}")

    if args.depth > 1:
        survivors = ranked[:args.top_k]
        final = []
        for m, s in survivors:
            b2 = board.copy()
            b2.push(m)
            lookahead = -negamax(b2, args.depth - 1)
            final.append((m, lookahead))
        # merge: survivors with lookahead score + rest with single-ply score
        survivor_uci = {m.uci() for m, _ in survivors}
        final += [(m, s) for m, s in ranked if m.uci() not in survivor_uci]
        ranked = sorted(final, key=lambda x: x[1], reverse=True)

    best_move = ranked[0][0]

    raw = [s for _, s in ranked]
    lo, hi = min(raw), max(raw)
    span = (hi - lo) if hi != lo else 1.0
    top_moves = [
        {"san": board.san(m), "uci": m.uci(), "prob": round((s - lo) / span, 4)}
        for m, s in ranked[:top_n]
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
