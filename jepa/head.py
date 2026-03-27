"""
ValueHead  — predicts win/loss probability from a discrete latent state z.
PolicyHead — predicts a distribution over the 4672-move action space.

Both take z of shape (B, N, n_cats, n_codes) — hard one-hot bottleneck output.

ValueHead output:  scalar in (-1, 1)  (+1 = win, 0 = draw, -1 = loss)
PolicyHead output: (B, ACTION_SIZE) logits — apply softmax / mask illegal moves
                   before use.
"""

import torch
import torch.nn as nn
from encoder import TransformerBlock
ACTION_SIZE = 73 * 64  # 4672 — AlphaZero-style encoding (see util/parse.py)


class ValueHead(nn.Module):
    def __init__(self, n_cats: int = 8, n_codes: int = 16, hidden_dim: int = 512):
        super().__init__()
        in_dim = n_cats * n_codes  # 2048

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, N, n_cats, n_codes)  hard one-hot latent codes

        Returns
        -------
        value : (B,) in (-1, 1)
        """
        B, N, n_cats, n_codes = z.shape
        z = z.view(B, N, n_cats * n_codes)   # (B, N, 2048)
        z = z.mean(dim=1)                     # (B, 2048)  — pool over patches
        return self.net(z).squeeze(-1).tanh() # (B,)


class PolicyHead(nn.Module):
    def __init__(
        self,
        n_cats: int = 8,
        n_codes: int = 16,
        n_patches: int = 64,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depth: int = 3,
    ):
        super().__init__()
 
        # --- 1. Per-patch projection: one-hot codes → dense embedding ---
        # Each patch has n_cats * n_codes = 128 dims (sparse, one-hot per cat).
        # Project to embed_dim so attention has something dense to work with.
        self.input_proj = nn.Linear(n_cats * n_codes, embed_dim)
 
        # --- 2. Positional embeddings ---
        # 64 patches = 64 squares on the board. The model needs to know
        # *where* each patch is — "knight on g1" vs "knight on f3" matters.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
 
        # --- 3. Transformer layers ---
        # Self-attention lets the model attend across squares:
        # "there's a bishop on c4 AND the king is on e8 → Bxf7 is interesting"
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
 
        self.norm = nn.LayerNorm(embed_dim)
 
        # --- 4. Move prediction head ---
        # After pooling 64 tokens → 1 vector, predict over all 4672 moves.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, ACTION_SIZE),
        )
 
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, 64, n_cats, n_codes)  — hard one-hot latent codes
 
        Returns
        -------
        logits : (B, 4672)  — unnormalised log-probabilities over moves
        """
        B, N, n_cats, n_codes = z.shape
 
        # Flatten cats: (B, 64, 128) — still sparse but now 2D per patch
        x = z.view(B, N, n_cats * n_codes)
 
        # Dense projection: (B, 64, 256)
        x = self.input_proj(x)
 
        # Add positional info so the model knows which square is which
        x = x + self.pos_embed
 
        # Self-attention across all 64 squares
        for layer in self.layers:
            x = layer(x)
 
        x = self.norm(x)
 
        # Mean pool over patches: (B, 256)
        x = x.mean(dim=1)
 
        # Predict move logits: (B, 4672)
        return self.head(x)
 