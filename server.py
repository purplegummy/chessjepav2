"""
server.py — Flask backend for ChessJEPA GUI.

Run with:
    python server.py --checkpoint checkpoints/checkpoint.pt

Endpoints
---------
GET  /              — serves the chess UI
POST /api/best_move — runs CEM planner, returns best move + analysis
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import torch
from flask import Flask, jsonify, render_template, request

from util.parse import board_to_tensor, index_to_move, move_to_index
from util.planner import NUM_MOVES, load_models, encode_obs, cem_planner

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
parser.add_argument("--device",     default="cpu")
parser.add_argument("--host",       default="127.0.0.1")
parser.add_argument("--port",       default=5000, type=int)
# CEM defaults (can be overridden per-request in future)
parser.add_argument("--horizon",    default=3,  type=int)
parser.add_argument("--n_samples",  default=64, type=int)
parser.add_argument("--n_elites",   default=8,  type=int)
parser.add_argument("--n_iters",    default=10, type=int)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Model (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device(args.device)
print(f"Loading model from {args.checkpoint} on {device}…")
encoder, bottleneck, predictor, _ = load_models(args.checkpoint, device)
print("Model ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


PIECE_MAP = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP,
             'n': chess.KNIGHT, 'p': chess.PAWN, 'k': chess.KING}


def make_goal_board(board: chess.Board, data: dict) -> chess.Board:
    """
    Build the goal board according to the selected goal type.

    king_capture  — remove opponent's king. Drives planner toward checkmate-like states.
    piece_capture — remove a specific opponent piece type. E.g. "capture their queen."
    target_fen    — use a user-supplied FEN as the goal position directly.
    preset        — same as target_fen but from the preset dropdown.

    In all cases the goal board is encoded by the same encoder → z_goal.
    CEM minimises MSE(z_predicted, z_goal) in latent space.
    """
    goal_type    = data.get("goal_type", "king_capture")
    planner_color = board.turn                              # planner moves next
    human_color   = not planner_color                      # human's pieces are the target

    if goal_type == "king_capture":
        goal = board.copy()
        for sq in list(goal.pieces(chess.KING, human_color)):
            goal.remove_piece_at(sq)
        return goal

    elif goal_type == "piece_capture":
        piece_char = data.get("capture_piece", "q")
        piece_type = PIECE_MAP.get(piece_char, chess.QUEEN)
        goal = board.copy()
        for sq in list(goal.pieces(piece_type, human_color)):
            goal.remove_piece_at(sq)
        return goal

    elif goal_type in ("target_fen", "preset"):
        goal_fen = data.get("goal_fen", "").strip()
        try:
            return chess.Board(goal_fen)
        except Exception:
            # Fall back to king capture if FEN is invalid
            goal = board.copy()
            for sq in list(goal.pieces(chess.KING, human_color)):
                goal.remove_piece_at(sq)
            return goal

    # Default fallback
    goal = board.copy()
    for sq in list(goal.pieces(chess.KING, human_color)):
        goal.remove_piece_at(sq)
    return goal


def run_planner(board: chess.Board, data: dict, top_n: int = 5):
    """
    Score every legal move by a 1-step rollout to build a display ranking,
    then run categorical CEM over legal moves to pick the best first action.
    """
    from util.planner import encode_obs as _enc, rollout as _roll, cem_planner as _cem

    obs_init = board_to_tensor(board).float()

    # Build (move, index) pairs for every legal move, skipping any that
    # can't be encoded (shouldn't happen, but be safe).
    move_index_pairs = []
    for m in board.legal_moves:
        try:
            move_index_pairs.append((m, move_to_index(m, board)))
        except Exception:
            continue

    if not move_index_pairs:
        return None, {}

    legal_moves_list   = [m   for m, _ in move_index_pairs]
    legal_indices_list = [idx for _, idx in move_index_pairs]

    z_init = _enc(encoder, bottleneck, obs_init, device)

    # ── Categorical CEM to pick the best move ─────────────────────────────
    config = {
        "horizon":   args.horizon,
        "n_samples": args.n_samples,
        "n_elites":  args.n_elites,
        "n_iters":   args.n_iters,
        "device":    device,
    }

    best_idx = _cem(
        encoder, bottleneck, predictor,
        obs_init,
        legal_indices_list,
        config,
        value_head=None,
    )

    # Map the returned move index back to a chess.Move
    try:
        best_move = index_to_move(best_idx, board)
        if best_move not in board.legal_moves:
            best_move = legal_moves_list[0]
    except Exception:
        best_move = legal_moves_list[0]

    # ── Build top-moves display with CEM choice always first ──────────────
    others = [m for m in legal_moves_list if m != best_move]
    ordered = [best_move] + others
    prob = round(1.0 / len(legal_moves_list), 4)
    top_moves = [{"san": board.san(m), "uci": m.uci(), "prob": prob}
                 for m in ordered[:top_n]]

    return best_move, {
        "confidence": prob,
        "top_moves":  top_moves,
        "value":      0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/best_move", methods=["POST"])
def best_move():
    data  = request.get_json(force=True)
    fen   = data.get("fen",   chess.STARTING_FEN)
    top_n = data.get("top_n", 5)

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    if board.is_game_over():
        return jsonify({"error": "Game over"}), 400

    move, analysis = run_planner(board, data, top_n=top_n)

    if move is None:
        return jsonify({"error": "No legal moves"}), 400

    return jsonify({
        "move":       move.uci(),
        "san":        board.san(move),
        "confidence": analysis.get("confidence", 0.0),
        "top_moves":  analysis.get("top_moves",  []),
        "value":      analysis.get("value",       0.0),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
