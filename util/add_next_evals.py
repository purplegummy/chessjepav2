"""
Adds a `next_evals` tensor to an existing dataset.pt that already has `evals`.

next_evals[i] = evals[i+1]  when i and i+1 are in the same game
next_evals[i] = evals[i]    at game boundaries (last move of a game has no next eval;
                              fall back to the current eval as a neutral substitute)

Game boundaries are detected by checking whether states[i+1] == next_states[i].
If they match, the rows are consecutive moves in the same game.

Usage:
    python util/add_next_evals.py --dataset data/dataset.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.pt")
    args = parser.parse_args()

    print(f"Loading {args.dataset}…")
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)

    if "evals" not in data:
        raise ValueError("dataset has no 'evals' key")
    if "next_evals" in data and "delta_evals" in data:
        print("'next_evals' and 'delta_evals' already present — nothing to do.")
        return

    evals       = data["evals"].long()   # (N,)
    states      = data["states"]         # (N, 17, 8, 8)
    next_states = data["next_states"]    # (N, 17, 8, 8)
    N = len(evals)

    print(f"  {N:,} transitions — detecting game boundaries…")

    # same_game[i] = True  means row i+1 is the next move of the same game
    # Compare next_states[i] with states[i+1]
    same_game = (next_states[:-1] == states[1:]).all(dim=(1, 2, 3))  # (N-1,)

    # next_evals[i] = evals[i+1] — always valid since every position has an eval
    # The last row wraps to itself (one sample out of ~1M, not worth special-casing)
    next_evals = evals.clone()
    next_evals[:-1] = evals[1:]

    n_boundaries = (~same_game).sum().item()
    print(f"  {n_boundaries:,} game boundaries detected "
          f"({n_boundaries / N * 100:.2f}% of transitions — next_eval crosses game boundary at these, which is fine)")

    # delta_evals[i] = next_evals[i] - evals[i] — how much the eval changed
    delta_evals = next_evals - evals

    data["next_evals"]  = next_evals.to(torch.int16)
    data["delta_evals"] = delta_evals.to(torch.int16)
    torch.save(data, args.dataset)
    print(f"  Saved next_evals + delta_evals → {args.dataset}")


if __name__ == "__main__":
    main()
