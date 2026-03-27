"""
ValueHead  — predicts win/loss probability from a discrete latent state z.
PolicyHead — predicts a distribution over the 4672-move action space.

Both take z of shape (B, N, n_cats, n_codes) — hard one-hot bottleneck output.

ValueHead output:  scalar in (-1, 1)  (+1 = win, 0 = draw, -1 = loss)
PolicyHead output: (B, ACTION_SIZE) logits — apply softmax / mask illegal moves
                   before use.
"""

import torch
import torch.nn as nn

ACTION_SIZE = 73 * 64  # 4672 — AlphaZero-style encoding (see util/parse.py)


class ValueHead(nn.Module):
    def __init__(self, n_cats: int = 8, n_codes: int = 16, hidden_dim: int = 512):
        super().__init__()
        in_dim = n_cats * n_codes  # 2048

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, N, n_cats, n_codes)  hard one-hot latent codes

        Returns
        -------
        value : (B,) in (-1, 1)
        """
        B, N, n_cats, n_codes = z.shape
        z = z.view(B, N, n_cats * n_codes)   # (B, N, 2048)
        z = z.mean(dim=1)                     # (B, 2048)  — pool over patches
        return self.net(z).squeeze(-1).tanh() # (B,)


class PolicyHead(nn.Module):
    """
    Predicts logits over the full 4672-move action space from a latent state.

    Usage
    -----
    logits = policy_head(z)                        # (B, 4672)  raw logits
    # mask illegal moves before sampling:
    logits[illegal_mask] = -1e9
    probs = torch.softmax(logits, dim=-1)          # (B, 4672)

    Training
    --------
    loss = F.cross_entropy(logits, target_move_index)
    where target_move_index is the move played (or the stockfish best move),
    encoded with util.parse.move_to_index().
    """

    def __init__(self, n_cats: int = 8, n_codes: int = 16, n_patches: int = 64, hidden_dim: int = 512):
        super().__init__()
        in_dim = n_patches * n_cats * n_codes  # 64 * 128 = 8192 — full spatial layout

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ACTION_SIZE),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, N, n_cats, n_codes)  hard one-hot latent codes

        Returns
        -------
        logits : (B, ACTION_SIZE)  unnormalised log-probabilities
        """
        B, N, n_cats, n_codes = z.shape
        z = z.view(B, N * n_cats * n_codes)  # (B, 8192) — flatten all patches
        return self.net(z)                   # (B, 4672)
