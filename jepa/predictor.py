import torch
import torch.nn as nn
from jepa.encoder import TransformerBlock

class Predictor(nn.Module):
    def __init__(
            
            self, n_cats: int = 32, 
            n_codes: int = 64, 
            embed_dim: int = 256, num_heads: int = 8, 
            mlp_ratio: float = 4.0, 
            depth: int = 6, num_moves: int = 4672
            
            ):
        
        super().__init__()
        self.action_encoder = nn.Embedding(num_moves, embed_dim)

        self.input_proj = nn.Linear(n_cats * n_codes, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.output_proj = nn.Linear(embed_dim, n_cats * n_codes)

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        B, N, n_cats, n_codes = z.shape 

        #action embedding
        a = self.action_encoder(a)  # (B, 256)
        a = a.unsqueeze(1) 


        # flatten the categorical variables into a single dimension of 2048 (32*64)
        z = z.view(B, N, n_cats * n_codes) 

        # project to embed_dim
        z = self.input_proj(z)

       

        # concatenate along the sequence dimension, so we have 17 tokens (16 from z and 1 from a), each with an embedding dimension of 256
        x = torch.cat([z, a], dim=1)       # (B, 17, 256)


        for block in self.layers:
            x = block(x)
    
        #remove action token
        z_hat = x[:, :N, :]          # (B, 16, 256)

        # we want to project back to the original categorical space of (B, 16, 32*64), and then reshape to (B, 16, 32, 64) to get the logits for each categorical variable
        logits = self.output_proj(z_hat)                 # (B, 16, 2048)
        logits = logits.view(B, N, n_cats, n_codes)      # (B, 16, 32, 64)
        return logits
