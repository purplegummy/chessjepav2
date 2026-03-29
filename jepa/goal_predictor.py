import torch
import torch.nn as nn
from jepa.encoder import TransformerBlock


class GoalConditionedPredictor(nn.Module):
    """
    Predicts action a_t given the current bottleneck state z_t and
    a scalar goal value v_{t+1} (e.g. board evaluation after the move).

    Inputs:
        z_t  : (B, N, n_cats, n_codes)  — bottleneck encoding of s_t
        v_t1 : (B,)                     — scalar board value of s_{t+1}

    Output:
        logits : (B, num_moves)
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

        self.z_proj = nn.Linear(n_cats * n_codes, embed_dim)

        # Scalar value → single token added to the sequence
        self.v_proj = nn.Linear(1, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_moves)

    def forward(self, z_t: torch.Tensor, v_t1: torch.Tensor) -> torch.Tensor:
        B, N, n_cats, n_codes = z_t.shape

        z = self.z_proj(z_t.view(B, N, n_cats * n_codes))  # (B, N, embed_dim)

        # Squash raw centipawns to [-1, 1] then embed as a single conditioning token
        v = self.v_proj(torch.tanh(v_t1.float() / 400.0).view(B, 1, 1))  # (B, 1, embed_dim)

        x = torch.cat([z, v], dim=1)                        # (B, N+1, embed_dim)

        for block in self.layers:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)                                   # (B, embed_dim)

        return self.head(x)                                  # (B, num_moves)
