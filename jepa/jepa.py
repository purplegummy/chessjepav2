import torch
import torch.nn as nn
from jepa.encoder import Encoder
from jepa.predictor import Predictor
from jepa.categoricalbottleneck import CategoricalBottleneck


class ChessJEPA(nn.Module):
    def __init__(self, n_cats=8, n_codes=16, embed_dim=256, tap_layers=(2, 4, 6), dropout=0.0):
        super().__init__()
        self.tap_layers = tap_layers
        self.encoder    = Encoder(tap_layers=tap_layers, dropout=dropout)
        self.bottleneck = CategoricalBottleneck(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim)
        self.predictor  = Predictor(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim, dropout=dropout)

    def forward(self, board_t, board_t1, a, tau=1.0):
        """
        Returns:
            pred_logits:       dict[level -> (B, 16, n_cats, n_codes)]
            target_indices:    dict[level -> (B, 16, n_cats)]
            bottleneck_logits: dict[level -> (B, 16, n_cats, n_codes)]  -- from t and t1 stacked
        """
        taps_t  = self.encoder(board_t)   # {2: (B,16,256), 4: ..., 6: ...}
        taps_t1 = self.encoder(board_t1)

        pred_logits       = {}
        target_indices    = {}
        bottleneck_logits = {}

        for level in self.tap_layers:
            z_t,  bn_logits_t  = self.bottleneck(taps_t[level],  tau=tau)
            z_t1, bn_logits_t1 = self.bottleneck(taps_t1[level], tau=tau)

            target_indices[level]    = z_t1.argmax(dim=-1)       # (B, 16, n_cats)
            pred_logits[level]       = self.predictor(z_t, a)    # (B, 16, n_cats, n_codes)
            # stack t and t1 so entropy sees both timesteps
            bottleneck_logits[level] = torch.cat([bn_logits_t, bn_logits_t1], dim=0)

        return pred_logits, target_indices, bottleneck_logits
