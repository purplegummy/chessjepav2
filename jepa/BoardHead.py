import torch.nn as nn
import torch 
class BoardHead(nn.Module):
    def __init__(self,  embed_dim: int = 256, n_channels: int = 17, board_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_channels * board_size * board_size),
        )

    def forward(self, taps: torch.Tensor) -> torch.Tensor:
        B, N, embed_dim = taps.shape
        x = taps.mean(dim=1)   # (B, embed_dim) — pool over tokens