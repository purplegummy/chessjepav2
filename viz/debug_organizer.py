"""
Test whether the organizer can distinguish material differences.
python viz/debug_organizer.py --jepa_ckpt checkpoints/checkpoint_epoch5.pt --organizer_ckpt checkpoints/organizer.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import chess
from jepa.jepa import ChessJEPA
from jepa.eval_organizer import EvalOrganizer
from util.parse import board_to_tensor

BOTTLENECK_TAU = 1e-5

def load_models(jepa_ckpt, org_ckpt, device):
    jepa = ChessJEPA().to(device)
    jepa.load_state_dict(torch.load(jepa_ckpt, map_location=device)["model_state_dict"])
    jepa.eval()

    org_data  = torch.load(org_ckpt, map_location=device)
    organizer = EvalOrganizer(
        latent_dim=org_data["latent_dim"],
        hidden_dim=org_data["hidden_dim"],
    ).to(device)
    organizer.load_state_dict(org_data["model_state_dict"])
    organizer.eval()
    return jepa, organizer

def score_fen(fen, jepa, organizer, device):
    board  = chess.Board(fen)
    tensor = board_to_tensor(board).float().unsqueeze(0).to(device)
    with torch.no_grad():
        taps = jepa.encoder(tensor)
        z, _ = jepa.bottleneck(taps[max(taps.keys())], tau=BOTTLENECK_TAU)
        z_flat = z.flatten(start_dim=1)
        _, eval_pred, _ = organizer(z_flat)
    return eval_pred.item(), board

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt",      default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--organizer_ckpt", default="checkpoints/organizer.pt")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    jepa, organizer = load_models(args.jepa_ckpt, args.organizer_ckpt, device)

    win_dir = organizer.eval_head.weight.squeeze(0)
    win_dir = win_dir / (win_dir.norm() + 1e-8)

    def score(fen):
        board  = chess.Board(fen)
        tensor = board_to_tensor(board).float().unsqueeze(0).to(device)
        with torch.no_grad():
            tap_dict = jepa.encoder(tensor)
            taps    = tap_dict[max(tap_dict.keys())]
            z, _    = jepa.bottleneck(taps, tau=BOTTLENECK_TAU)
            indices = z.argmax(dim=-1).flatten(start_dim=1)  # (1, 512)
            latent, eval_pred, _ = organizer(indices)
            proj = (latent @ win_dir).item()
        return eval_pred.item(), proj

    print("=== Test 1: material difference ===")
    # Starting position
    s, p = score(chess.STARTING_FEN)
    print(f"  Start:              eval={s:+.3f}  proj={p:+.3f}")

    # White up a queen
    s, p = score("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"  Black missing queen: eval={s:+.3f}  proj={p:+.3f}  (should be strongly positive)")

    # Black up a queen
    s, p = score("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    print(f"  White missing queen: eval={s:+.3f}  proj={p:+.3f}  (should be strongly negative)")

    # White up a pawn
    s, p = score("rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"  Black missing pawn:  eval={s:+.3f}  proj={p:+.3f}  (should be mildly positive)")

    print("\n=== Test 2: free capture (white to move, can take free rook) ===")
    # White rook on e4, black rook on e5 hanging (no defenders)
    fen_before = "4k3/8/8/4r3/4R3/8/8/4K3 w - - 0 1"
    fen_after  = "4k3/8/8/4R3/8/8/8/4K3 b - - 0 1"
    s_b, p_b = score(fen_before)
    s_a, p_a = score(fen_after)
    print(f"  Before capture: eval={s_b:+.3f}  proj={p_b:+.3f}")
    print(f"  After capture:  eval={s_a:+.3f}  proj={p_a:+.3f}  (after flip: {-s_a:+.3f} / {-p_a:+.3f})")
    print(f"  → capture ranks {'HIGHER ✓' if -s_a > s_b else 'LOWER ✗ (bug)'} than not capturing")

    print("\n=== Test 3: rank all moves in free-capture position ===")
    board = chess.Board(fen_before)
    fens_after = []
    moves = list(board.legal_moves)
    for m in moves:
        b2 = board.copy(); b2.push(m)
        fens_after.append((board.san(m), b2.fen()))

    scored = []
    for san, fen in fens_after:
        s, p = score(fen)
        scored.append((san, -s, -p))  # negate: white moved, resulting pos is black's turn

    scored.sort(key=lambda x: x[1], reverse=True)
    for san, s, p in scored[:8]:
        print(f"  {san:12s}  eval={s:+.3f}  proj={p:+.3f}")

if __name__ == "__main__":
    main()
