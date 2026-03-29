import torch
import torch.nn as nn
from jepa.encoder import TransformerBlock


class DeltaPredictor(nn.Module):
    """
    Predicts z_{t+1} given z_t and a scalar delta value (v_{t+1} - v_t).

    Inputs:
        z_t   : (B, N, n_cats, n_codes)  — bottleneck encoding of s_t
        delta : (B,)                     — scalar eval delta (raw centipawns)

    Output:
        z_t1_pred : (B, N, n_cats, n_codes)  — predicted bottleneck encoding of s_{t+1}
    """

    def __init__(
        self,
        n_cats: int = 8,
        n_codes: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_cats  = n_cats
        self.n_codes = n_codes

        self.z_proj     = nn.Linear(n_cats * n_codes, embed_dim)
        self.delta_proj = nn.Linear(1, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm       = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, n_cats * n_codes)

    def forward(self, z_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        B, N, n_cats, n_codes = z_t.shape

        z = self.z_proj(z_t.view(B, N, n_cats * n_codes))              # (B, N, embed_dim)
        d = self.delta_proj(torch.tanh(delta.float() / 400.0).view(B, 1, 1))  # (B, 1, embed_dim)

        x = torch.cat([z, d], dim=1)                                    # (B, N+1, embed_dim)

        for block in self.layers:
            x = block(x)

        x = self.norm(x)[:, :N, :]                                      # (B, N, embed_dim)
        x = self.output_proj(x)                                         # (B, N, n_cats*n_codes)

        return x.view(B, N, n_cats, n_codes)                            # (B, N, n_cats, n_codes)
