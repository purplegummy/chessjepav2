import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """
    Simple policy head on top of frozen JEPA encoder taps.

    Input:  last encoder tap (B, N, embed_dim)
    Output: action logits    (B, num_moves)
    """

    def __init__(self, embed_dim: int = 256, num_moves: int = 4672):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_moves),
        )

    def forward(self, tap: torch.Tensor) -> torch.Tensor:
        x = tap.mean(dim=1)   # (B, embed_dim) — pool over tokens
        return self.net(x)    # (B, num_moves)
