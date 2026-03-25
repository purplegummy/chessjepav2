"""
ValueHead — predicts win/loss probability from a discrete latent state z.

Input:  z of shape (B, N, n_cats, n_codes) — hard one-hot bottleneck output.
Output: scalar in (-1, 1) representing expected game result from current
        player's perspective (+1 = win, 0 = draw, -1 = loss).

Architecture
------------
1. Flatten z to (B, N, n_cats*n_codes).
2. Mean-pool over the N patch dimension  → (B, n_cats*n_codes).
3. Two-layer MLP with LayerNorm + GELU   → scalar logit.
4. tanh to squash output to (-1, 1).
"""

import torch
import torch.nn as nn


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
