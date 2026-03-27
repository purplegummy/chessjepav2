"""
server.py — Flask backend for ChessJEPA GUI.

Run with:
    python app/server.py \
        --jepa_ckpt    checkpoints/checkpoint_epoch4.pt \
        --policy_ckpt  checkpoints/policy/policy_head.pt

Endpoints
---------
GET  /              — serves the chess UI
POST /api/best_move — runs policy head, returns best move + top moves
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
from jepa.head import PolicyHead
from util.parse import board_to_tensor, index_to_move, move_to_index

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt",   required=True)
parser.add_argument("--policy_ckpt", required=True)
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

print(f"Loading policy head from {args.policy_ckpt}…")
policy_head = PolicyHead(n_cats=N_CATS, n_codes=N_CODES).to(device)
ph_ckpt = torch.load(args.policy_ckpt, map_location=device)
policy_head.load_state_dict(ph_ckpt["policy_head_state_dict"])
policy_head.eval()
print("Models ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)


def pick_move(board: chess.Board, top_n: int = 5):
    """Encode the board, run the policy head, return best legal move + top moves."""
    tensor = board_to_tensor(board).float().unsqueeze(0).to(device)

    with torch.no_grad():
        taps = encoder(tensor)
        h = taps[max(taps.keys())]
        z, _ = bottleneck(h, tau=BOTTLENECK_TAU)
        logits = policy_head(z)          # (1, 4672)
        logits = logits.squeeze(0)       # (4672,)

    # Build legal (move, index) pairs
    move_index_pairs = []
    for m in board.legal_moves:
        try:
            move_index_pairs.append((m, move_to_index(m, board)))
        except Exception:
            continue

    if not move_index_pairs:
        return None, []

    legal_moves   = [m   for m, _ in move_index_pairs]
    legal_indices = [idx for _, idx in move_index_pairs]

    # Mask to legal moves only
    legal_logits = logits[legal_indices]
    probs = F.softmax(legal_logits, dim=0).cpu().tolist()

    # Best move
    best_local = max(range(len(probs)), key=lambda i: probs[i])
    best_move  = legal_moves[best_local]

    # Top-N by probability
    ranked = sorted(zip(legal_moves, probs), key=lambda x: x[1], reverse=True)
    top_moves = [
        {"san": board.san(m), "uci": m.uci(), "prob": round(p, 4)}
        for m, p in ranked[:top_n]
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
