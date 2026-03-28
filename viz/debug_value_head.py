"""
Test whether the value head can distinguish material and tactical differences.

python viz/debug_value_head.py \
    --jepa_ckpt       checkpoints/checkpoint_epoch5.pt \
    --value_head_ckpt checkpoints/value_head.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import chess
from jepa.jepa import ChessJEPA
from jepa.value_head import ValueHead
from util.parse import board_to_tensor


def load_models(jepa_ckpt, vh_ckpt_path, device):
    jepa = ChessJEPA().to(device)
    jepa.load_state_dict(torch.load(jepa_ckpt, map_location=device)["model_state_dict"])
    jepa.eval()

    vh_ckpt    = torch.load(vh_ckpt_path, map_location=device)
    value_head = ValueHead(
        tap_dim=vh_ckpt.get("tap_dim", 256),
        hidden_dim=vh_ckpt["hidden_dim"],
        latent_dim=vh_ckpt["latent_dim"],
    ).to(device)
    value_head.load_state_dict(vh_ckpt["model_state_dict"])
    value_head.eval()
    return jepa, value_head


def score(fen, jepa, value_head, device):
    board  = chess.Board(fen)
    tensor = board_to_tensor(board).float().unsqueeze(0).to(device)
    with torch.no_grad():
        taps     = jepa.encoder(tensor)
        last_tap = taps[max(taps.keys())]
        _, pred  = value_head(last_tap)
    return pred.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt",       default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--value_head_ckpt", default="checkpoints/value_head.pt")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    jepa, value_head = load_models(args.jepa_ckpt, args.value_head_ckpt, device)

    print("=== Test 1: material difference ===")
    print(f"  Start:               val={score(chess.STARTING_FEN, jepa, value_head, device):+.3f}")
    print(f"  Black missing queen: val={score('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', jepa, value_head, device):+.3f}  (should be strongly positive)")
    print(f"  White missing queen: val={score('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1', jepa, value_head, device):+.3f}  (should be strongly negative)")
    print(f"  Black missing pawn:  val={score('rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', jepa, value_head, device):+.3f}  (should be mildly positive)")

    print("\n=== Test 2: free capture (white to move, can take free rook) ===")
    fen_before = "4k3/8/8/4r3/4R3/8/8/4K3 w - - 0 1"
    fen_after  = "4k3/8/8/4R3/8/8/8/4K3 b - - 0 1"
    s_b = score(fen_before, jepa, value_head, device)
    s_a = score(fen_after,  jepa, value_head, device)
    print(f"  Before capture: val={s_b:+.3f}")
    print(f"  After capture:  val={s_a:+.3f}  (after flip: {-s_a:+.3f})")
    print(f"  → capture ranks {'HIGHER ✓' if -s_a > s_b else 'LOWER ✗ (bug)'}")

    print("\n=== Test 3: rank all moves in free-capture position ===")
    board = chess.Board(fen_before)
    scored = []
    for m in board.legal_moves:
        b2 = board.copy()
        b2.push(m)
        s = -score(b2.fen(), jepa, value_head, device)
        scored.append((board.san(m), s))
    scored.sort(key=lambda x: x[1], reverse=True)
    for san, s in scored[:8]:
        print(f"  {san:12s}  val={s:+.3f}")


if __name__ == "__main__":
    main()
