import torch
import torch.nn as nn
from jepa.encoder import Encoder
from jepa.predictor import Predictor
from jepa.categoricalbottleneck import CategoricalBottleneck
from jepa.inverse_predictor import InversePredictor
from jepa.goal_predictor import GoalConditionedPredictor

class ChessJEPA(nn.Module):
    def __init__(self, n_cats=8, n_codes=16, embed_dim=256, tap_layers=(2, 4, 6), dropout=0.0):
        super().__init__()
        self.tap_layers = tap_layers
        self.encoder    = Encoder(tap_layers=tap_layers, dropout=dropout)
        self.bottleneck = CategoricalBottleneck(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim)
        self.predictor  = Predictor(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim, dropout=dropout)
        self.inv_predictor = InversePredictor(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim, dropout=dropout)
        self.goal_predictor = GoalConditionedPredictor(n_cats=n_cats, n_codes=n_codes, embed_dim=embed_dim, dropout=dropout)
    def forward(self, board_t, board_t1, a, tau=1.0, delta_evals=None):
        """
        Returns:
            pred_logits:       dict[level -> (B, N, n_cats, n_codes)]
            target_indices:    dict[level -> (B, N, n_cats)]
            bottleneck_logits: dict[level -> (B, N, n_cats, n_codes)]  -- from t and t1 stacked
            inv_logits:        (B, num_moves)  — inverse predictor on last tap
            goal_logits:       (B, num_moves) | None  — goal predictor (requires delta_evals)
        """
        taps_t  = self.encoder(board_t)
        taps_t1 = self.encoder(board_t1)

        pred_logits       = {}
        target_indices    = {}
        bottleneck_logits = {}
        z_t_last = z_t1_last = None

        for level in self.tap_layers:
            z_t,  bn_logits_t  = self.bottleneck(taps_t[level],  tau=tau)
            z_t1, bn_logits_t1 = self.bottleneck(taps_t1[level], tau=tau)

            target_indices[level]    = z_t1.argmax(dim=-1)
            pred_logits[level]       = self.predictor(z_t, a)
            bottleneck_logits[level] = torch.cat([bn_logits_t, bn_logits_t1], dim=0)

            if level == max(self.tap_layers):
                z_t_last  = z_t
                z_t1_last = z_t1

        inv_logits  = self.inv_predictor(z_t_last, z_t1_last)

        goal_logits = None
        if delta_evals is not None:
            goal_logits = self.goal_predictor(z_t_last, delta_evals)  # delta_evals = delta_evals

        return pred_logits, target_indices, bottleneck_logits, inv_logits, goal_logits
