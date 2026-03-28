# Usage:
#   python3 -m util.dataset --pgn data/games.pgn --out data/dataset.pt
#   python3 -m util.dataset --pgn data/games.pgn --out data/dataset.pt --max-games 10000
#   python3 -m util.dataset --pgn data/games.pgn --out data/dataset.pt --stockfish /usr/bin/stockfish --depth 12 --workers 8

import chess
import chess.pgn
import chess.engine
import torch
import multiprocessing as mp
from util.parse import board_to_tensor, move_to_index

MATE_SCORE = 10_000   # centipawns used to represent mate
EVAL_CLIP  = 1_500    # clamp evals to ±this value


def _eval_worker(args):
    """
    Runs in a subprocess. Evaluates a chunk of FENs with its own Stockfish instance.
    Returns a list of int16 centipawn evals (current-player POV, clipped to ±EVAL_CLIP).
    """
    fens, stockfish_path, depth, worker_id = args
    results = []
    n = len(fens)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            info  = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].relative  # current player's POV
            if score.is_mate():
                cp = MATE_SCORE if score.mate() > 0 else -MATE_SCORE
            else:
                cp = score.score()
            results.append(max(-EVAL_CLIP, min(EVAL_CLIP, cp)))
            if (i + 1) % 1000 == 0:
                print(f"  worker {worker_id}: {i + 1}/{n} ({100*(i+1)/n:.0f}%)", flush=True)
    print(f"  worker {worker_id}: done ({n} positions)", flush=True)
    return results


def _stockfish_evals(fens: list[str], stockfish_path: str, depth: int, n_workers: int) -> list[int]:
    """Parallel Stockfish eval over all FENs using n_workers subprocesses."""
    chunk_size = max(1, len(fens) // n_workers)
    chunks = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]
    tasks  = [(chunk, stockfish_path, depth, i) for i, chunk in enumerate(chunks)]

    with mp.Pool(processes=len(chunks)) as pool:
        results = pool.map(_eval_worker, tasks)

    return [cp for chunk in results for cp in chunk]


def pgn_to_dataset(
    pgn_path: str,
    out_path: str,
    max_games: int | None = None,
    stockfish_path: str | None = None,
    depth: int = 12,
    workers: int = 8,
):
    """
    Parse a PGN file and write a .pt file containing:
        states:       (N, 17, 8, 8)  uint8   — board at time t (current player POV)
        next_states:  (N, 17, 8, 8)  uint8   — board at time t+1 (current player POV)
        actions:      (N,)           int32   — AlphaZero action index
        results:      (N,)           int8    — outcome from current player's POV
                                               1 = win, -1 = loss, 0 = draw
        evals:        (N,)           int16   — Stockfish centipawn eval (current player POV,
                                               clipped to ±1500, ±10000 for mate)
                                               only present if --stockfish is given
    """
    states, next_states, actions, results, fens = [], [], [], [], []
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
                if stockfish_path:
                    fens.append(board.fen())
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

    if stockfish_path:
        print(f"Running Stockfish (depth={depth}) on {len(fens)} positions with {workers} workers...")
        print(f"  chunk size: ~{max(1, len(fens) // workers)} positions per worker")
        print("  spawning workers...", flush=True)
        evals = _stockfish_evals(fens, stockfish_path, depth, workers)
        dataset["evals"] = torch.tensor(evals, dtype=torch.int16)
        print(f"Stockfish evals done. ({len(evals)} positions evaluated)")

    torch.save(dataset, out_path)
    print(f"Saved {len(actions)} positions from {n_games} games → {out_path}")
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn",        default="data/games.pgn")
    parser.add_argument("--out",        default="data/dataset.pt")
    parser.add_argument("--max-games",  type=int,  default=None)
    parser.add_argument("--stockfish",  type=str,  default=None,
                        help="path to stockfish binary (e.g. /usr/bin/stockfish)")
    parser.add_argument("--depth",      type=int,  default=12,
                        help="stockfish search depth per position")
    parser.add_argument("--workers",    type=int,  default=8,
                        help="number of parallel stockfish processes")
    args = parser.parse_args()

    pgn_to_dataset(args.pgn, args.out, args.max_games, args.stockfish, args.depth, args.workers)
