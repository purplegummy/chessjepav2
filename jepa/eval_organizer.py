import torch
import torch.nn as nn


class EvalOrganizer(nn.Module):
    def __init__(
        self,
        tap_dim: int    = 256,   # encoder output dim per patch
        n_patches: int  = 64,    # number of patches (8×8 board)
        latent_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        input_dim = n_patches * tap_dim  # 64 * 256 = 16384

        # Pre-alignment block: prepares JEPA features for the eval task
        self.pre_align = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Fork: value branch (scalar eval) and structure branch (residual features)
        # bias=False so orthogonality penalty is well-defined on weight vectors
        self.val_head    = nn.Linear(latent_dim, 1,              bias=False)
        self.struct_head = nn.Linear(latent_dim, latent_dim - 1, bias=False)

        self.tap_dim    = tap_dim
        self.n_patches  = n_patches
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        """x: (B, 64, 256) float — encoder taps, all patches."""
        z = self.pre_align(x.flatten(start_dim=1))  # (B, latent_dim)

        eval_pred = self.val_head(z).squeeze(-1)  # (B,)
        struct    = self.struct_head(z)            # (B, latent_dim-1)

        return z, eval_pred, struct

    def orthogonal_penalty(self) -> torch.Tensor:
        """
        Penalise alignment between the value direction and all structure directions.
        val_head.weight:    (1, latent_dim)
        struct_head.weight: (latent_dim-1, latent_dim)
        """
        dot = self.struct_head.weight @ self.val_head.weight.T  # (latent_dim-1, 1)
        return dot.pow(2).sum()
