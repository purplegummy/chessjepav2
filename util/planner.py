"""
planner.py — Latent planning for ChessJEPA via the Cross-Entropy Method (CEM).

Pipeline recap
--------------
Raw board (B, 17, 8, 8)
    ↓  Encoder           → taps: dict[layer_idx → (B, 64, 256)]
    ↓  CategoricalBottleneck → z: (B, 64, 32, 64)  ← one-hot categorical codes
    ↓  Predictor(z, a)   → next-z logits: (B, 64, 32, 64)

With patch_size=1 the board is split into 64 individual-cell patches (8×8=64).
The bottleneck projects each 256-dim patch embedding into 32 categorical variables,
each with 64 possible codes — giving a discrete, compact state representation.

CEM overview
------------
CEM is a gradient-free optimisation method that maintains a distribution over
action sequences and iteratively refines it:
  1. Sample N action sequences from the current distribution.
  2. Evaluate each by rolling out the world model and measuring latent cost.
  3. Keep the top-K ("elite") sequences.
  4. Fit a new distribution to the elites.
  5. Repeat for T iterations; return the mean as the best plan.

Used as MPC: only the *first* action of the plan is executed, then re-plan.
"""

import torch
import torch.nn.functional as F

from jepa.jepa import ChessJEPA
from jepa.head import ValueHead
from util.parse import index_to_move, move_to_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The encoder is trained with these defaults; changing them requires a new ckpt.
NUM_PATCHES: int = 64   # 8×8 board with 1×1 patch_size
N_CATS:      int = 8    # categorical variables per patch
N_CODES:     int = 16   # possible values per categorical variable
EMBED_DIM:   int = 256
NUM_MOVES:   int = 4672  # total legal-ish UCI move indices in the dataset

# Gumbel-softmax temperature used at inference.
# tau→0 makes the softmax collapse to a hard argmax (fully discrete).
# We want deterministic hard codes during planning, not stochastic samples.
BOTTLENECK_TAU: float = 1e-5


# ---------------------------------------------------------------------------
# 1. load_models
# ---------------------------------------------------------------------------

def load_models(
    checkpoint_path: str,
    device: torch.device,
    value_checkpoint_path: str | None = None,
):
    """
    Instantiate the ChessJEPA model, load weights from a checkpoint, and
    return the sub-modules used during planning.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by train.py.
    device : torch.device
    value_checkpoint_path : str or None
        Optional path to a value-head checkpoint saved by train_value.py.
        If provided, the ValueHead is returned; otherwise None is returned.

    Returns
    -------
    encoder    : jepa.encoder.Encoder
    bottleneck : jepa.categoricalbottleneck.CategoricalBottleneck
    predictor  : jepa.predictor.Predictor
    value_head : jepa.head.ValueHead or None
    """
    model = ChessJEPA(
        n_cats=N_CATS,
        n_codes=N_CODES,
        embed_dim=EMBED_DIM,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    value_head = None
    if value_checkpoint_path is not None:
        value_head = ValueHead(n_cats=N_CATS, n_codes=N_CODES).to(device)
        vh_ckpt = torch.load(value_checkpoint_path, map_location=device)
        value_head.load_state_dict(vh_ckpt["value_head_state_dict"])
        value_head.eval()

    return model.encoder, model.bottleneck, model.predictor, value_head


# ---------------------------------------------------------------------------
# 2. encode_obs
# ---------------------------------------------------------------------------

def encode_obs(
    encoder,
    bottleneck,
    obs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a raw board tensor into the discrete latent code used by the
    predictor.

    Parameters
    ----------
    encoder    : jepa.encoder.Encoder
    bottleneck : jepa.categoricalbottleneck.CategoricalBottleneck
    obs        : torch.Tensor, shape (17, 8, 8) or (1, 17, 8, 8)
                 17-channel board representation (piece planes + meta planes).
    device     : torch.device

    Returns
    -------
    z : torch.Tensor, shape (1, 64, 32, 64)
        Hard one-hot latent code for the board position.
    """
    # Ensure batch dimension exists — the model expects (B, C, H, W).
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)          # (1, 17, 8, 8)
    obs = obs.to(device)

    # torch.no_grad() disables autograd tracking. During planning we never
    # call .backward(), so tracking gradients would waste memory and time.
    with torch.no_grad():
        # The encoder returns a dict of "taps" — intermediate representations
        # extracted after each specified transformer layer (default: 2, 4, 6).
        taps = encoder(obs)             # {2: (1,64,256), 4: ..., 6: ...}

        # We use the *final* tap (highest layer index) because it has processed
        # the input through the most attention layers and therefore captures the
        # richest, most globally-integrated board representation.
        # Earlier taps are used during training as auxiliary supervision targets;
        # here we only need the best single representation for planning.
        final_layer = max(taps.keys())
        h = taps[final_layer]           # (1, 64, 256) — one 256-d vector per cell

        # The bottleneck projects 256 → 32×64 and applies straight-through
        # Gumbel-softmax to obtain a discrete one-hot code per categorical var.
        # tau≈0 makes gumbel_softmax deterministic (≈ argmax), which is what we
        # want: a single, stable latent point to plan from rather than a noisy
        # stochastic sample.
        z, _ = bottleneck(h, tau=BOTTLENECK_TAU)   # (1, 64, 32, 64)

    return z


# ---------------------------------------------------------------------------
# 3. rollout
# ---------------------------------------------------------------------------

def rollout_batch(
    predictor,
    z_init: torch.Tensor,
    action_sequences: torch.Tensor,
) -> torch.Tensor:
    """
    Batched rollout: predict latent states for N action sequences in parallel.

    Parameters
    ----------
    predictor        : jepa.predictor.Predictor
    z_init           : (1, P, n_cats, n_codes)  — current latent state
    action_sequences : (N, H)                   — N sequences of H move indices

    Returns
    -------
    z_all : (N, H, P, n_cats, n_codes)  — predicted latent state at every step
    """
    N, H = action_sequences.shape
    z = z_init.expand(N, -1, -1, -1).clone()
    steps = []

    with torch.no_grad():
        for t in range(H):
            a_t    = action_sequences[:, t]
            logits = predictor(z, a_t)
            z      = F.one_hot(logits.argmax(dim=-1), num_classes=N_CODES).float()
            steps.append(z)

    return torch.stack(steps, dim=1)  # (N, H, P, n_cats, n_codes)


# ---------------------------------------------------------------------------
# 4. cem_planner
# ---------------------------------------------------------------------------

def cem_planner(
    encoder,
    bottleneck,
    predictor,
    obs_init: torch.Tensor,
    legal_move_indices: list[int],
    config: dict,
    value_head=None,
    obs_goal: torch.Tensor | None = None,
) -> int:
    """
    Find the best first move using CEM with a categorical distribution over
    the legal moves at the current position.

    Unlike Gaussian CEM, this samples directly from a distribution over the
    actual legal move indices, so every sampled action is valid and the
    distribution update is meaningful (move index 100 and 101 are unrelated
    integers — averaging them is meaningless; counting elite occurrences is not).

    Parameters
    ----------
    encoder, bottleneck, predictor : sub-modules from load_models()
    obs_init           : (17, 8, 8) current board tensor
    legal_move_indices : list of int — move indices legal at obs_init
    config             : dict with keys:
        horizon    (int) — rollout depth
        n_samples  (int) — sequences sampled per CEM iteration
        n_elites   (int) — top-K sequences kept
        n_iters    (int) — CEM refinement iterations
        device     (torch.device)
    value_head : ValueHead or None
    obs_goal   : (17, 8, 8) or None — used only when value_head is None

    Returns
    -------
    best_move : int  — the move index to execute (MPC: one step at a time)
    """
    horizon   = config["horizon"]
    n_samples = config["n_samples"]
    n_elites  = config["n_elites"]
    n_iters   = config["n_iters"]
    device    = config["device"]
    gamma     = config.get("gamma", 1.5)  # weight multiplier per step; >1 = later steps matter more

    n_legal = len(legal_move_indices)
    # Tensor of legal move indices so we can index into it easily.
    legal_moves = torch.tensor(legal_move_indices, dtype=torch.long, device=device)  # (L,)

    # ------------------------------------------------------------------
    # a. Encode current (and optional goal) board.
    # ------------------------------------------------------------------
    z_init = encode_obs(encoder, bottleneck, obs_init, device)   # (1, 64, 32, 64)
    z_goal = encode_obs(encoder, bottleneck, obs_goal, device) if obs_goal is not None else None


    # ------------------------------------------------------------------
    # b. Categorical distribution: logits over legal moves, one per step.
    #    Shape: (H, n_legal) — start uniform (all zeros before softmax).
    # ------------------------------------------------------------------
    logits = torch.zeros(horizon, n_legal, device=device)

    with torch.no_grad():
        for _ in range(n_iters):
            # ----------------------------------------------------------
            # c-i. Sample (n_samples, H) sequences from the categorical.
            # ----------------------------------------------------------
            probs = torch.softmax(logits, dim=-1)  # (H, n_legal)

            probs_tiled = probs.unsqueeze(0).expand(n_samples, -1, -1)      # (N, H, L)
            probs_flat  = probs_tiled.reshape(n_samples * horizon, n_legal)  # (N*H, L)
            local_idx   = torch.multinomial(probs_flat, num_samples=1)       # (N*H, 1)
            local_idx   = local_idx.reshape(n_samples, horizon)              # (N, H)

            # Convert local indices (into legal_moves) → actual move indices.
            samples_int = legal_moves[local_idx]   # (N, H)

            # ----------------------------------------------------------
            # c-ii. Score all steps, accumulate with increasing weights.
            # ----------------------------------------------------------
            z_all = rollout_batch(predictor, z_init, samples_int)  # (N, H, P, n_cats, n_codes)

            # weights: gamma^0, gamma^1, ..., gamma^(H-1) — later steps weighted more
            weights = torch.tensor([gamma ** t for t in range(horizon)], device=device)
            weights = weights / weights.sum()  # normalise so scale is independent of H

            costs = torch.zeros(n_samples, device=device)
            for t in range(horizon):
                z_t = z_all[:, t]   # (N, P, n_cats, n_codes)
                if value_head is not None:
                    step_cost = -value_head(z_t).squeeze(-1)                                    # (N,)
                else:
                    z_goal_exp = z_goal.expand(n_samples, -1, -1, -1)
                    step_cost = F.mse_loss(z_t, z_goal_exp, reduction='none').mean(dim=(1, 2, 3))  # (N,)
                costs += weights[t] * step_cost

            # ----------------------------------------------------------
            # c-iii. Keep top-K elites (lowest cost).
            # ----------------------------------------------------------
            elite_idx        = costs.argsort()[:n_elites]   # (K,)
            elite_local_idx  = local_idx[elite_idx]         # (K, H)

            # ----------------------------------------------------------
            # c-iv. Update logits: count how often each legal move appeared
            #        in the elite set at each timestep.  This is the
            #        maximum-likelihood update for a categorical distribution.
            # ----------------------------------------------------------
            new_logits = torch.zeros_like(logits)
            # elite_local_idx: (K, H) — scatter counts into (H, n_legal)
            new_logits.scatter_add_(1, elite_local_idx.t(), torch.ones_like(elite_local_idx.t(), dtype=torch.float))
            logits = new_logits + 0.1

    # ------------------------------------------------------------------
    # d. Return the best move and the softmax probabilities over legal moves.
    # ------------------------------------------------------------------
    probs_final = torch.softmax(logits[0], dim=-1)  # (L,)
    best_local  = probs_final.argmax().item()
    return legal_moves[best_local].item(), probs_final.cpu().tolist()


# ---------------------------------------------------------------------------
# Quick smoke-test (not a real unit test — use pytest for that)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CEM planner smoke-test")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading models...")
    encoder, bottleneck, predictor = load_models(args.checkpoint, device)

    # Random boards — just checking shapes and no runtime errors.
    obs_init = torch.rand(17, 8, 8)
    obs_goal = torch.rand(17, 8, 8)

    config = {
        "horizon":    5,
        "n_samples":  64,
        "n_elites":   8,
        "n_iters":    10,
        "action_low":  0,
        "action_high": NUM_MOVES - 1,
        "device":      device,
    }

    print("Running CEM planner...")
    best_actions = cem_planner(encoder, bottleneck, predictor, obs_init, obs_goal, config)
    print(f"Best action sequence (H={config['horizon']}): {best_actions.tolist()}")
    print(f"First action to execute (MPC step): {best_actions[0].item()}")
