import torch
import torch.nn as nn

class BoardTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 17,
        board_size: int = 8,
        patch_size: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size # 1x1 size patches
        self.num_patches = (board_size // patch_size) ** 2 # 64 patches if we use 1x1 patches
        patch_dim = in_channels * patch_size * patch_size # 17 channels * 1 * 1 = 17, each patch is a 17-dim vector, so we have 64x17 input to the linear layer
        self.embedding = torch.nn.Linear(patch_dim, embed_dim) # linear layer to embed each patch into a 256-dim vector  -> output is 64x256
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size # 1, so we will reshape into 1x1 patches
        
        # 1. Reshape into (B, C, H_patches, p, W_patches, p)
        # For an 8x8 board with 1x1 patches: (B, 17, 8, 1, 8, 1) -> (B, 17, 8, 8, 1, 1) 
        x = x.view(B, C, H // p, p, W // p, p)
        
        # 2. Permute to get patches together: (B, H_p, W_p, C, p, p)
        # for instance, if we want the patch in row 0, col 0, we would index H_p = 0, W_p = 0, and we would get the 17 channels and the 2x2 patch for that location, and we could do this for all 16 patches
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        
        # Now we have (B, 4, 4, 17, 2, 2), we can view the last three dimensions as a single dimension of 68 (17 channels * 2 * 2), so we get (B, 16, 68)
        # 3. Flatten into (B, 16, 68)
        x = x.view(B, self.num_patches, -1)
        
        # Then we project 68 -> 256
        return self.embedding(x) # Result: (B, 16, 256)
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim) # pass in 16x256 embeddings
        # multi-head attention layer, with 8 heads, each head has 256/8 = 32 dimensions, so we have 8x32 for each patch, 
        # and we set batch_first=True to make the input and output shapes more intuitive (B, N, D)
        # returns 16x256 output, same shape as input, but now each patch has attended to all other patches
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # feed-forward network, with an intermediate dimension of 512 (4x the embed_dim), and then back down to 256, with a GELU activation in between
        # we get back a 16x256 output, but as a unified representation of the concatenated head results
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, D) — N tokens, D embed_dim."""
        # Self-attention with pre-norm
        normed = self.norm1(x)

        # attn returns a 16x256 output and we add it back to the original x (residual connection), so we get a new 16x256 output that has attended to all other patches, and then we pass that through the feed-forward network with another residual connection, so we get a final 16x256 output that has been processed by both the attention and the feed-forward network
        # prevents original input from being lost 
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]

        normed2 = self.norm2(x) # normalize again as values from attention can be overly large, so we normalize again before passing through the feed-forward network to prevent instability
        # Feed-forward with pre-norm
        # another residual connection
        x = x + self.ffn(normed2)
        return x

# Here we use patch size = 1 but in the comments we us e patch size = 2 for easier understanding, but the code works for any patch size as long as it divides the board size evenly
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 17,
        board_size: int = 8,
        patch_size: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        tap_layers: tuple = (2, 4, 6),  # which layers to extract
    ):
        super().__init__()
        self.tap_layers = set(tap_layers)

        self.tokenizer = BoardTokenizer(
            in_channels=in_channels,
            board_size=board_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # learned positional embedding over the patch grid
        num_patches = (board_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # ensures initial values of the positional embedding are small
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)  # applied to all taps before returning

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = self.tokenizer(x)       # (B, 16, 256)
        x = x + self.pos_embed      # add positional embedding
        
        # we want to store the intermediate representations (taps) from the specified layers
        taps = {}
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)            # (B, 16, 256)
            if i in self.tap_layers:
                taps[i] = x         # store intermediate rep

        # normalise all taps so they're on the same scale before the bottleneck
        for k in taps:
            taps[k] = self.norm(taps[k])

        return taps                 #