"""
server.py — Flask backend for ChessJEPA GUI.

Move selection pipeline:
  1. Encode s_t → z_t (bottleneck)
  2. DeltaPredictor(z_t, delta=+1.0) → z_goal  (imagined next state)
  3. InversePredictor(z_t, z_goal) → action logits
  4. Pick highest logit legal move

Run with:
    python app/server.py \
        --jepa_ckpt       checkpoints/checkpoint_v2_epoch8.pt \
        --delta_pred_ckpt checkpoints/delta_pred_epoch0.pt
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

from jepa.jepa import ChessJEPA
from jepa.delta_predictor import DeltaPredictor
from util.parse import board_to_tensor, move_to_index

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt",       required=True)
parser.add_argument("--delta_pred_ckpt", required=True)
parser.add_argument("--delta",           default=1.0,          type=float,
                    help="Target eval delta passed to DeltaPredictor (raw centipawns)")
parser.add_argument("--device",          default="cpu")
parser.add_argument("--host",            default="127.0.0.1")
parser.add_argument("--port",            default=5001,         type=int)
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

print(f"Loading DeltaPredictor from {args.delta_pred_ckpt}…")
delta_pred = DeltaPredictor().to(device)
dp_ckpt = torch.load(args.delta_pred_ckpt, map_location=device)
delta_pred.load_state_dict(dp_ckpt["delta_pred_state_dict"])
delta_pred.eval()
for p in delta_pred.parameters():
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
    """
    1. Encode s_t → z_t
    2. DeltaPredictor(z_t, delta) → z_goal
    3. InversePredictor(z_t, z_goal) → logits over 4672 actions
    4. Return logit for each legal move's action index
    """
    s_t = board_to_tensor(board).float().unsqueeze(0).to(device)
    delta = torch.tensor([args.delta], dtype=torch.float32).to(device)

    with torch.no_grad():
        taps_t = jepa.encoder(s_t)
        z_t, _ = jepa.bottleneck(taps_t[max(taps_t.keys())], tau=0.1)  # (1, N, n_cats, n_codes)
        z_goal = delta_pred(z_t, delta)                                  # (1, N, n_cats, n_codes)
        logits = jepa.inv_predictor(z_t, z_goal)                        # (1, 4672)

    scores = []
    for m in moves:
        action_idx = move_to_index(m, board)
        scores.append(logits[0, action_idx].item())

    return scores


def pick_move(board: chess.Board, top_n: int = 5):
    legal = list(board.legal_moves)
    if not legal:
        return None, []

    scores = score_moves(board, legal)
    ranked = sorted(zip(legal, scores), key=lambda x: x[1], reverse=True)

    print(f"\n[debug] turn={'white' if board.turn == chess.WHITE else 'black'}")
    for m, s in ranked[:5]:
        print(f"  {board.san(m):10s}  score={s:.4f}")

    all_scores = torch.tensor([s for _, s in ranked], dtype=torch.float32)
    probs = F.softmax(all_scores, dim=0).tolist()
    top_moves = [
        {"san": board.san(m), "uci": m.uci(), "prob": round(probs[i], 4)}
        for i, (m, _) in enumerate(ranked[:top_n])
    ]

    return ranked[0][0], top_moves


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

    return jsonify({
        "move":       move.uci(),
        "san":        board.san(move),
        "confidence": top_moves[0]["prob"] if top_moves else 0.0,
        "top_moves":  top_moves,
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
