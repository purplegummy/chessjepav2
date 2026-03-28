import torch
import torch.nn as nn
import torch.nn.functional as F


class EvalOrganizer(nn.Module):
    def __init__(
        self,
        tap_dim: int     = 256,   # encoder output dim per patch
        n_patches: int   = 64,    # number of patches (8×8 board)
        latent_dim: int  = 128,
        hidden_dim: int  = 512,
        val_bottleneck: int = 32, # compress before val_head to tighten eval concept
    ):
        super().__init__()
        # Pre-alignment block: global avg pool input → latent sphere
        self.pre_align = nn.Sequential(
            nn.Linear(tap_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Value bottleneck: compress latent → tight eval subspace
        self.val_bottleneck = nn.Linear(latent_dim, val_bottleneck, bias=False)

        # Fork: value head reads from bottleneck, structure from full latent
        # bias=False so orthogonality penalty is well-defined
        self.val_head    = nn.Linear(val_bottleneck, 1,              bias=False)
        self.struct_head = nn.Linear(latent_dim,     latent_dim - 1, bias=False)

        self.tap_dim        = tap_dim
        self.n_patches      = n_patches
        self.latent_dim     = latent_dim
        self.hidden_dim     = hidden_dim
        self.val_bottleneck_dim = val_bottleneck

    def forward(self, x: torch.Tensor):
        """x: (B, 64, 256) float — encoder taps, all patches."""
        z_raw = self.pre_align(x.mean(dim=1))       # (B, latent_dim)
        z     = F.normalize(z_raw, dim=-1)           # L2 normalize onto unit sphere

        v     = self.val_bottleneck(z)               # (B, val_bottleneck)
        eval_pred = self.val_head(v).squeeze(-1)     # (B,)
        struct    = self.struct_head(z)              # (B, latent_dim-1)

        return z, eval_pred, struct

    def orthogonal_penalty(self) -> torch.Tensor:
        """
        Penalise alignment between val_bottleneck directions and struct directions.
        Projects val_bottleneck into latent space via val_head for the penalty.
        """
        # effective val direction in latent space: val_head.weight @ val_bottleneck.weight
        # shape: (1, val_bottleneck) @ (val_bottleneck, latent_dim) → (1, latent_dim)
        val_dir = self.val_head.weight @ self.val_bottleneck.weight  # (1, latent_dim)
        dot = self.struct_head.weight @ val_dir.T                    # (latent_dim-1, 1)
        return dot.pow(2).sum()
