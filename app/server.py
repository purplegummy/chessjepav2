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
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
N_CATS         = 8
N_CODES        = 16
BOTTLENECK_TAU = 1e-5

device = torch.device(args.device)

print(f"Loading JEPA from {args.jepa_ckpt}…")
jepa = ChessJEPA(n_cats=N_CATS, n_codes=N_CODES).to(device)
ckpt = torch.load(args.jepa_ckpt, map_location=device)
jepa.load_state_dict(ckpt["model_state_dict"])
jepa.eval()
for p in jepa.parameters():
    p.requires_grad_(False)

encoder    = jepa.encoder
bottleneck = jepa.bottleneck
predictor  = jepa.predictor

print(f"Loading organizer from {args.organizer_ckpt}…")
org_ckpt  = torch.load(args.organizer_ckpt, map_location=device)
organizer = EvalOrganizer(
    input_dim=org_ckpt["input_dim"],
    latent_dim=org_ckpt["latent_dim"],
    hidden_dim=org_ckpt["hidden_dim"],
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


def board_to_z(board: chess.Board) -> torch.Tensor:
    """Encode a board → bottleneck codes z of shape (1, 64, 8, 16)."""
    tensor = board_to_tensor(board).float().unsqueeze(0).to(device)
    with torch.no_grad():
        taps = encoder(tensor)
        z, _ = bottleneck(taps[max(taps.keys())], tau=BOTTLENECK_TAU)
    return z  # (1, 64, 8, 16)


def encode_moves(board: chess.Board, moves: list[chess.Move]) -> torch.Tensor:
    """
    Actually play each move and encode the resulting board.
    Returns (N, 64, 8, 16) — ground truth encodings, no predictor.
    """
    tensors = []
    for m in moves:
        b2 = board.copy()
        b2.push(m)
        tensors.append(board_to_tensor(b2).float())
    batch = torch.stack(tensors).to(device)  # (N, 17, 8, 8)
    with torch.no_grad():
        taps = encoder(batch)
        z, _ = bottleneck(taps[max(taps.keys())], tau=BOTTLENECK_TAU)
    return z  # (N, 64, 8, 16)


def score_z(z: torch.Tensor) -> torch.Tensor:
    """
    Score positions using organizer's eval head directly.
    z:       (N, 64, 8, 16)
    Returns: (N,) predicted eval normalised to [-1, 1], current-player POV
    """
    z_flat = z.flatten(start_dim=1)           # (N, 8192)
    with torch.no_grad():
        _, eval_pred = organizer(z_flat)      # (N,)
    return eval_pred


PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

def material_score(board: chess.Board) -> float:
    """Material balance from white's POV, normalised to ~[-1, 1]."""
    score = sum(
        PIECE_VALUES[pt] * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        for pt in PIECE_VALUES
    )
    return score / 39.0  # max material = 39 pawns worth


def pick_move(board: chess.Board, top_n: int = 5):
    """
    Hybrid scoring: material (dominant) + organizer (positional tiebreak).
    The organizer bottleneck codes lose material info in complex positions,
    so material is weighted heavily to prevent obvious blunders.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None, []

    z_next_all = encode_moves(board, legal)   # (N, 64, 8, 16)
    org_scores = score_z(z_next_all)          # (N,) current-player POV
    org_scores = -org_scores                  # flip to current player's advantage

    move_scores = []
    for i, m in enumerate(legal):
        b2 = board.copy()
        b2.push(m)
        mat = material_score(b2)
        if board.turn == chess.BLACK:
            mat = -mat  # black wants negative material balance
        org = org_scores[i].item()
        combined = 0.8 * mat + 0.2 * org
        move_scores.append((m, combined, mat, org))

    ranked = sorted(move_scores, key=lambda x: x[1], reverse=True)

    print(f"\n[debug] turn={'white' if board.turn == chess.WHITE else 'black'}")
    for m, combined, mat, org in ranked[:5]:
        print(f"  {board.san(m):10s}  combined={combined:.3f}  mat={mat:.3f}  org={org:.3f}")

    best_move = ranked[0][0]

    raw = [s for _, s, _, _ in ranked]
    lo, hi = min(raw), max(raw)
    span = (hi - lo) if hi != lo else 1.0
    top_moves = [
        {"san": board.san(m), "uci": m.uci(), "prob": round((s - lo) / span, 4)}
        for m, s, _, _ in ranked[:top_n]
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
