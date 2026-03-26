import torch.nn as nn
import torch
class CategoricalBottleneck(nn.Module):
    def __init__(self, n_cats: int = 8, n_codes: int = 16, embed_dim: int = 256, dead_code_threshold: float = 0.01):
        super().__init__()
        self.n_cats = n_cats
        self.n_codes = n_codes
        self.dead_code_threshold = dead_code_threshold
        # we will project the input embedding into a space of n_cats * n_codes dimensions
        self.proj = nn.Linear(embed_dim, n_cats * n_codes)
        # EMA usage tracker: counts average usage per code across categories (n_cats, n_codes)
        self.register_buffer("ema_usage", torch.ones(n_cats, n_codes) / n_codes)

    @torch.no_grad()
    def _reset_dead_codes(self, logits: torch.Tensor):
        # logits: (B, N, n_cats, n_codes)
        # compute per-code usage frequency averaged over batch and positions
        probs = torch.softmax(logits.detach(), dim=-1)  # (B, N, n_cats, n_codes)
        avg = probs.mean(dim=(0, 1))                    # (n_cats, n_codes)
        self.ema_usage.mul_(0.99).add_(avg.mul(0.01))

        dead = self.ema_usage < self.dead_code_threshold  # (n_cats, n_codes)
        n_dead = dead.sum().item()
        if n_dead > 0:
            # reinitialize dead code weights with gaussian noise around current mean weight
            weight = self.proj.weight  # (n_cats*n_codes, embed_dim)
            weight_2d = weight.view(self.n_cats, self.n_codes, -1)
            for cat_idx in range(self.n_cats):
                dead_codes = dead[cat_idx].nonzero(as_tuple=True)[0]
                if len(dead_codes) == 0:
                    continue
                alive_codes = (~dead[cat_idx]).nonzero(as_tuple=True)[0]
                if len(alive_codes) == 0:
                    continue
                # sample alive codes to copy from (with noise)
                src = alive_codes[torch.randint(len(alive_codes), (len(dead_codes),))]
                noise = torch.randn_like(weight_2d[cat_idx, src]) * 0.01
                weight_2d[cat_idx, dead_codes] = weight_2d[cat_idx, src] + noise

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        # x: (B, N, embed_dim)
        B, N, _ = x.shape
        logits = self.proj(x).view(B, N, self.n_cats, self.n_codes)

        if self.training:
            self._reset_dead_codes(logits)

        soft = nn.functional.gumbel_softmax(logits, tau=tau, hard=False)
        hard = nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        z = hard - soft.detach() + soft  # straight-through estimator
        return z, logits  # z: (B, N, n_cats, n_codes)