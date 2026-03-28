"""
EvalOrganizer: a small network that projects discrete JEPA bottleneck codes
into an eval-organized latent space, enabling latent arithmetic for move quality.

Input:  flattened bottleneck one-hots  (B, 64 * 8 * 16) = (B, 8192)
Output: latent vector                  (B, latent_dim)

Training losses:
  - MSE regression loss against normalized Stockfish eval
  - Ranking (margin) loss: winning positions should score higher than losing ones
"""

import torch
import torch.nn as nn


class EvalOrganizer(nn.Module):
    def __init__(self, input_dim: int = 8192, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        # scalar eval head for regression loss
        self.eval_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, 8192) one-hot flattened bottleneck codes."""
        z = self.encoder(x)
        eval_pred = self.eval_head(z).squeeze(-1)  # (B,)
        return z, eval_pred
