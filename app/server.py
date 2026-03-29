"""
server.py — Flask backend for ChessJEPA GUI.

Run with:
    python app/server.py \
        --jepa_ckpt       checkpoints/checkpoint_epoch5.pt \
        --value_head_ckpt checkpoints/value_head.pt
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
from jepa.value_head import ValueHead
from util.parse import board_to_tensor

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt",       required=True)
parser.add_argument("--value_head_ckpt", required=True)
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

print(f"Loading value head from {args.value_head_ckpt}…")
vh_ckpt    = torch.load(args.value_head_ckpt, map_location=device)
value_head = ValueHead(
    tap_dim=vh_ckpt.get("tap_dim", 256),
    hidden_dim=vh_ckpt["hidden_dim"],
    latent_dim=vh_ckpt["latent_dim"],
).to(device)
value_head.load_state_dict(vh_ckpt["model_state_dict"])
value_head.eval()
for p in value_head.parameters():
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




def score_moves(board: chess.Board, moves: list[chess.Move]) -> list[float]:
    """Score each move by the value head prediction of the resulting position."""
    tensors = []
    for m in moves:
        b2 = board.copy()
        b2.push(m)
        tensors.append(board_to_tensor(b2).float())
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        tap_dict = encoder(batch)
        last_tap = tap_dict[max(tap_dict.keys())]
        _, pred  = value_head(last_tap)         # (N,) in [-1, 1]
    # negate: resulting position is opponent's turn
    return (-pred).tolist()


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

    # current position eval from side-to-move's POV, then convert to white's POV
    with torch.no_grad():
        taps = encoder(board_to_tensor(board).float().unsqueeze(0).to(device))
        _, raw_eval = value_head(taps[max(taps.keys())])
        eval_val = raw_eval.item()  # positive = good for side to move
    white_eval = eval_val if board.turn == chess.WHITE else -eval_val
    black_eval = -white_eval

    move, top_moves = pick_move(board, top_n=top_n)

    if move is None:
        return jsonify({"error": "No legal moves"}), 400

    confidence = top_moves[0]["prob"] if top_moves else 0.0

    return jsonify({
        "move":       move.uci(),
        "san":        board.san(move),
        "confidence": confidence,
        "top_moves":  top_moves,
        "eval": {
            "white": round(white_eval, 4),
            "black": round(black_eval, 4),
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
