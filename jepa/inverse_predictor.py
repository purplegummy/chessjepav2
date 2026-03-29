import torch
import torch.nn as nn
from jepa.encoder import TransformerBlock


class InversePredictor(nn.Module):
    """
    Predicts the action taken between two encoded states.

    Inputs:
        z_t  : (B, N, n_cats, n_codes)  — bottleneck encoding of s_t
        z_t1 : (B, N, n_cats, n_codes)  — bottleneck encoding of s_{t+1}

    Output:
        logits : (B, num_moves)  — unnormalised scores over the action space
    """

    def __init__(
        self,
        n_cats: int = 8,
        n_codes: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depth: int = 4,
        num_moves: int = 4672,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project each token's flat categorical representation into embed_dim
        self.input_proj = nn.Linear(n_cats * n_codes * 2, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Pool across tokens then classify
        self.head = nn.Linear(embed_dim, num_moves)

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        B, N, n_cats, n_codes = z_t.shape

        # Flatten categorical dims: (B, N, n_cats*n_codes)
        z_t  = z_t.view(B, N, n_cats * n_codes)
        z_t1 = z_t1.view(B, N, n_cats * n_codes)

        # Concatenate along feature dim so each token sees both states
        x = torch.cat([z_t, z_t1], dim=-1)   # (B, N, n_cats*n_codes*2)
        x = self.input_proj(x)                # (B, N, embed_dim)

        for block in self.layers:
            x = block(x)

        x = self.norm(x)                      # (B, N, embed_dim)
        x = x.mean(dim=1)                     # (B, embed_dim)  — pool over tokens

        return self.head(x)                   # (B, num_moves)
