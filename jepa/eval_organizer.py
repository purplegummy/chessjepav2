import torch
import torch.nn as nn


class EvalOrganizer(nn.Module):
    def __init__(
        self,
        n_tokens: int   = 512,   # 64 patches × 8 categories
        n_codes: int    = 16,    # codebook size per category
        embed_dim: int  = 16,    # embedding dim per token
        latent_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        # Embed each discrete code index into a dense vector
        self.embedding = nn.Embedding(n_codes, embed_dim)

        input_dim = n_tokens * embed_dim  # 512 * 16 = 8192 → but dense, not sparse
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

        self.n_tokens   = n_tokens
        self.n_codes    = n_codes
        self.embed_dim  = embed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        """x: (B, 512) int64 — category indices."""
        e = self.embedding(x)              # (B, 512, embed_dim)
        e = e.flatten(start_dim=1)         # (B, 512 * embed_dim)
        z = self.mlp(e)                    # (B, latent_dim)
        eval_pred = self.eval_head(z).squeeze(-1)  # (B,)
        return z, eval_pred
