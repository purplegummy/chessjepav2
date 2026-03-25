# ChessJEPA

A JEPA (Joint Embedding Predictive Architecture) for chess, using a transformer encoder with a categorical bottleneck (à la DreamerV2) to learn discrete world models from self-play data.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
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
mkdir checkpoints

# Train the JEPA world model
python train.py --data data/dataset.pt --epochs 10 --batch 512

# Train the value head on frozen encoder + bottleneck
python train_value.py \
    --jepa_ckpt checkpoints/checkpoint_epoch9.pt \
    --out checkpoints/value_head.pt
```

Checkpoints are saved to `checkpoints/` every 1000 steps and at the end of each epoch.

### train.py defaults

| argument | default |
|---|---|
| `--data` | `data/dataset.pt` |
| `--out` | `checkpoints/checkpoint.pt` |
| `--epochs` | `10` |
| `--batch` | `512` |
| `--lr` | `3e-4` |
| `--warmup` | `1000` steps |
| `--dropout` | `0.1` |

### train_value.py defaults

| argument | default |
|---|---|
| `--jepa_ckpt` | *(required)* |
| `--data` | `data/dataset.pt` |
| `--out` | `checkpoints/value_head.pt` |
| `--epochs` | `5` |
| `--batch` | `512` |
| `--lr` | `3e-4` |

## Inference

```bash
python server.py --checkpoint checkpoints/checkpoint_epoch9.pt --port 5001
```

### server.py defaults

| argument | default |
|---|---|
| `--checkpoint` | `checkpoints/checkpoint.pt` |
| `--device` | `cpu` |
| `--host` | `127.0.0.1` |
| `--port` | `5000` |
| `--horizon` | `3` |
| `--n_samples` | `64` |
| `--n_elites` | `8` |
| `--n_iters` | `10` |

Then open `http://127.0.0.1:5001` in your browser.

> **Note:** macOS reserves port 5000 for AirPlay Receiver. Use `--port 5001` or disable AirPlay in System Settings → General → AirDrop & Handoff.

## Architecture

- **Encoder** — Vision Transformer over 1×1 patches of the 17-channel board tensor, tapped at layers 2, 4, 6. Dropout applied to attention and FFN.
- **Bottleneck** — Categorical (8 variables × 16 codes per patch) with straight-through Gumbel-softmax. Temperature annealed from 1.0 → 0.1 over training.
- **Predictor** — Transformer that takes the discrete latent + action embedding and predicts the next latent.
- **Value Head** — MLP over the pooled bottleneck output predicting win/loss in (-1, 1).
- **Planner** — Cross-Entropy Method (CEM) with a categorical distribution over legal moves only. Scored by the value head when available.
- **Loss** — Multi-level prediction CE loss + entropy regularisation (λ=0.01) to prevent codebook collapse. Gradient clipping at 1.0. Cosine LR decay after linear warmup.
