"""
EvalOrganizer: projects JEPA representations into an eval-organized latent space.

Input:
  codes: (B, n_patches * n_cats * n_codes) = (B, 8192)  — discrete bottleneck codes
  taps:  (B, n_patches, tap_dim)           = (B, 64, 256) — raw encoder patch tokens

Architecture:
  1. Self-attention over the 64 patch tokens so the model sees inter-square relationships
     (e.g. queen on d4 + rook on d1 = battery along the d-file).
  2. Mean-pool the attended tokens → (B, tap_dim)
  3. Concatenate with bottleneck codes → (B, 8192 + 256)
  4. MLP → latent vector → scalar eval prediction

Training losses:
  - MSE regression against normalised Stockfish eval
  - Margin ranking loss: winning positions should score higher than losing ones
"""

import torch
import torch.nn as nn


class EvalOrganizer(nn.Module):
    def __init__(
        self,
        tap_dim: int   = 256,
        n_patches: int = 64,
        n_cats: int    = 8,
        n_codes: int   = 16,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        attn_heads: int = 8,
    ):
        super().__init__()
        bottleneck_dim = n_patches * n_cats * n_codes  # 8192

        # Spatial attention over patch tokens
        self.spatial_attn = nn.MultiheadAttention(tap_dim, attn_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(tap_dim)

        # MLP over concat(bottleneck, attended_pool)
        input_dim = bottleneck_dim + tap_dim  # 8448
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.eval_head = nn.Linear(latent_dim, 1)

        # store for checkpoint
        self.tap_dim    = tap_dim
        self.n_patches  = n_patches
        self.n_cats     = n_cats
        self.n_codes    = n_codes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads

    def forward(self, codes: torch.Tensor, taps: torch.Tensor):
        """
        codes: (B, 8192)   — flattened bottleneck one-hots
        taps:  (B, 64, 256) — raw encoder patch tokens
        """
        # Spatial self-attention: let squares attend to each other
        attn_out, _ = self.spatial_attn(taps, taps, taps, need_weights=False)
        taps = self.spatial_norm(taps + attn_out)   # residual + norm
        pooled = taps.mean(dim=1)                   # (B, 256) — spatially aware

        x = torch.cat([codes, pooled], dim=1)       # (B, 8448)
        z = self.mlp(x)                             # (B, latent_dim)
        eval_pred = self.eval_head(z).squeeze(-1)   # (B,)
        return z, eval_pred
