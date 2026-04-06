"""
server.py — Flask backend for ChessJEPA GUI.

Move selection pipeline:
  1. Encode s_t → z_t (bottleneck)
  2. delta_predictor(z_t, delta) → z_goal  (imagined next state)
  3. inv_predictor(z_t, z_goal)  → action logits
  4. Pick highest logit legal move

Run with:
    python app/server.py --jepa_ckpt checkpoints/checkpoint_v3_epoch0.pt \
                         --value_ckpt checkpoints/value_head_epoch0.pt
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess
import chess.engine
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

from jepa.jepa import ChessJEPA
from jepa.head import ValueHead
from util.parse import board_to_tensor, move_to_index

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt",  required=True)
parser.add_argument("--value_ckpt", default=None,
                    help="Path to value head checkpoint (optional)")
parser.add_argument("--stockfish",  default="/opt/homebrew/bin/stockfish",
                    help="Path to stockfish binary")
parser.add_argument("--sf_depth",   default=12, type=int,
                    help="Stockfish search depth")
parser.add_argument("--device",     default="cpu")
parser.add_argument("--host",       default="127.0.0.1")
parser.add_argument("--port",       default=5001, type=int)
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

print("JEPA ready.")

# Value head (optional)
value_head = None
if args.value_ckpt:
    print(f"Loading ValueHead from {args.value_ckpt}…")
    value_head = ValueHead().to(device)
    vckpt = torch.load(args.value_ckpt, map_location=device)
    value_head.load_state_dict(vckpt["model_state_dict"])
    value_head.eval()
    for p in value_head.parameters():
        p.requires_grad_(False)
    print("ValueHead ready.")

# Stockfish engine (persistent process)
sf_engine = None
if os.path.exists(args.stockfish):
    try:
        sf_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        print(f"Stockfish ready ({args.stockfish}).")
    except Exception as e:
        print(f"Stockfish failed to start: {e}")
else:
    print(f"Stockfish not found at {args.stockfish}, skipping.")

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
    For each legal move:
      1. Encode s_t → z_t
      2. predictor(z_t, action) → z_t1_hat
      3. inv_predictor(z_t, z_t1_hat) → logits over 4672 actions
      4. Score = logit at that move's action index
    """
    s_t = board_to_tensor(board).float().unsqueeze(0).to(device)
    action_indices = torch.tensor(
        [move_to_index(m, board) for m in moves], dtype=torch.long, device=device
    )  # (M,)
    M = len(moves)

    with torch.no_grad():
        taps_t = jepa.encoder(s_t)
        z_t, _ = jepa.bottleneck(taps_t[max(taps_t.keys())], tau=0.1)  # (1, N, n_cats, n_codes)

        # Expand z_t to batch over all moves
        z_t_exp = z_t.expand(M, -1, -1, -1)  # (M, N, n_cats, n_codes)

        # Predict next state for each action
        pred_logits = jepa.predictor(z_t_exp, action_indices)  # (M, N, n_cats, n_codes)
        z_t1_hat = torch.softmax(pred_logits, dim=-1)

        # Score each (z_t, z_t1_hat) pair with the inverse predictor
        inv_logits = jepa.inv_predictor(z_t_exp, z_t1_hat)  # (M, 4672)

    # Score for move i = logit at its own action index
    scores = inv_logits[torch.arange(M), action_indices].tolist()
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


@app.route("/api/eval", methods=["POST"])
def eval_position():
    data = request.get_json(force=True)
    fen  = data.get("fen", chess.STARTING_FEN)

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    result = {}

    # ── JEPA value head ───────────────────────────────────────────────────────
    if value_head is not None:
        s_t = board_to_tensor(board).float().unsqueeze(0).to(device)
        with torch.no_grad():
            taps = jepa.encoder(s_t)
            jepa_val = value_head(taps[max(taps.keys())]).item()  # [-1, 1]
        # jepa_val is from white's perspective (normalized centipawns / 1000)
        result["jepa_cp"] = round(jepa_val * 3000, 1)   # back to centipawns (EVAL_CLIP=3000)
    else:
        result["jepa_cp"] = None

    # ── Stockfish ─────────────────────────────────────────────────────────────
    if sf_engine is not None:
        try:
            info = sf_engine.analyse(board, chess.engine.Limit(depth=args.sf_depth))
            score = info["score"].white()
            if score.is_mate():
                m = score.mate()
                result["sf_cp"]   = None
                result["sf_mate"] = m
            else:
                result["sf_cp"]   = score.score()
                result["sf_mate"] = None
        except Exception as e:
            result["sf_cp"]   = None
            result["sf_mate"] = None
            result["sf_error"] = str(e)
    else:
        result["sf_cp"]   = None
        result["sf_mate"] = None

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
