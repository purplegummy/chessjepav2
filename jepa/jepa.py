import torch.nn as nn
from jepa.encoder import Encoder
from jepa.predictor import Predictor
from jepa.categoricalbottleneck import CategoricalBottleneck


class ChessJEPA(nn.Module):
    def __init__(self, n_cats=32, n_codes=64, embed_dim=256, tap_layers=(2, 4, 6)):
        super().__init__()
        self.tap_layers = tap_layers
        self.encoder    = Encoder(tap_layers=tap_layers)
        self.bottleneck = CategoricalBottleneck(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim)
        self.predictor  = Predictor(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim)

    def forward(self, board_t, board_t1, a, tau=1.0):
        """
        Returns:
            pred_logits:    dict[level -> (B, 16, n_cats, n_codes)]
            target_indices: dict[level -> (B, 16, n_cats)]
        """
        taps_t  = self.encoder(board_t)   # {2: (B,16,256), 4: ..., 6: ...}
        taps_t1 = self.encoder(board_t1)

        pred_logits    = {}
        target_indices = {}

        for level in self.tap_layers:
            # discretise both states through the shared bottleneck
            z_t,  _ = self.bottleneck(taps_t[level],  tau=tau)  # (B, 16, n_cats, n_codes) hard one-hot encoding of the categorical variables
            z_t1, _ = self.bottleneck(taps_t1[level], tau=tau)

            # hard argmax gives the target code index per categorical variable
            target_indices[level] = z_t1.argmax(dim=-1)          # (B, 16, n_cats) 32 numbers per patch representing the index of the active code in each category, which is the target for the predictor to learn to predict from z_t and a

            # predictor estimates the next latent from current latent + action
            pred_logits[level] = self.predictor(z_t, a)          # (B, 16, n_cats, n_codes)

        return pred_logits, target_indices
