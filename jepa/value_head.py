import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    def __init__(self, tap_dim: int = 256, hidden_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tap_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.val_head = nn.Linear(latent_dim, 1, bias=False)

        self.tap_dim    = tap_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        """
        x: (B, 64, 256) — JEPA encoder taps, all patches.
        Returns:
            z:    (B, latent_dim) — L2-normalized latent embedding
            pred: (B,)            — scalar value in [-1, 1]
        """
        z    = F.normalize(self.mlp(x.mean(dim=1)), dim=-1)
        pred = self.val_head(z).squeeze(-1)
        return z, pred
