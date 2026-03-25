# Usage:
#   python3 -m util.dataset --pgn data/games.pgn --out data/dataset.pt
#   python3 -m util.dataset --pgn data/games.pgn --out data/dataset.pt --max-games 10000

import chess
import chess.pgn
import torch
from pathlib import Path
from util.parse import board_to_tensor, move_to_index


def pgn_to_dataset(pgn_path: str, out_path: str, max_games: int | None = None):
    """
    Parse a PGN file and write a .pt file containing:
        states:       (N, 17, 8, 8)  uint8   — board at time t (current player POV)
        next_states:  (N, 17, 8, 8)  uint8   — board at time t+1 (current player POV)
        actions:      (N,)           int32   — AlphaZero action index
        results:      (N,)           int8    — outcome from current player's POV
                                               1 = win, -1 = loss, 0 = draw
    """
    states, next_states, actions, results = [], [], [], []
    n_games = 0

    _result_map = {"1-0": (1, -1), "0-1": (-1, 1), "1/2-1/2": (0, 0)}

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result_str = game.headers.get("Result", "*")
            white_result, black_result = _result_map.get(result_str, (0, 0))

            board = game.board()
            for move in game.mainline_moves():
                s_t  = board_to_tensor(board)
                a_t  = move_to_index(move, board)
                r_t  = white_result if board.turn == chess.WHITE else black_result
                board.push(move)
                s_t1 = board_to_tensor(board)

                states.append(s_t)
                next_states.append(s_t1)
                actions.append(a_t)
                results.append(r_t)

            n_games += 1
            if n_games % 1000 == 0:
                print(f"  {n_games} games, {len(actions)} positions")
            if max_games is not None and n_games >= max_games:
                break

    dataset = {
        "states":      torch.stack(states).to(torch.uint8),
        "next_states": torch.stack(next_states).to(torch.uint8),
        "actions":     torch.tensor(actions, dtype=torch.int32),
        "results":     torch.tensor(results, dtype=torch.int8),
    }
    torch.save(dataset, out_path)
    print(f"Saved {len(actions)} positions from {n_games} games → {out_path}")
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn",      default="data/games.pgn")
    parser.add_argument("--out",      default="data/dataset.pt")
    parser.add_argument("--max-games", type=int, default=None,
                        help="cap number of games (useful for testing)")
    args = parser.parse_args()

    pgn_to_dataset(args.pgn, args.out, args.max_games)
