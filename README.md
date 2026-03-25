# ChessJEPA

A JEPA (Joint Embedding Predictive Architecture) for chess, using a transformer encoder with a categorical bottleneck (à la DreamerV2) to learn discrete world models from self-play data.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch chess
```

## Data

```bash
# Download PGN
gdown --id 1IbxOGHV31tz6jHNSa2Z--rqJDqugl68z -O data/

# Parse into tensors
python3 -m util.dataset --pgn data/games.pgn --out data/dataset.pt
```

## Training

```bash
python3 -m train --data data/dataset.pt --epochs 10 --batch 64
```

Checkpoints are saved to `checkpoints/` every 5 epochs.

## Architecture

- **Encoder** — Vision Transformer over 1×1 patches of the 17-channel board tensor, tapped at layers 2, 4, 6
- **Bottleneck** — Categorical (32 variables × 64 codes) with straight-through Gumbel-softmax
- **Predictor** — Transformer that takes the discrete latent + action and predicts the next latent
- **Loss** — Multi-level prediction CE loss + entropy regularisation to prevent codebook collapse
