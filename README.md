# ChessJEPA

Chess engine built on a Joint Embedding Predictive Architecture (JEPA). The encoder learns rich board representations in a discrete latent space; a policy head trained on top plays chess.

## Project structure

```
jepa/          model architecture (encoder, predictor, bottleneck, heads)
util/          dataset, parsing, CEM planner
app/           Flask server + frontend (static/, templates/)
viz/           UMAP embedding visualisation
data/          dataset.pt, games.pgn
checkpoints/   saved model weights
```

## Training

### 1. Build the dataset

If starting from a PGN:

```bash
python util/dataset.py --pgn data/games.pgn --out data/dataset.pt
```

### 2. Train the JEPA encoder + predictor

```bash
python train.py --data data/dataset.pt --out checkpoints/
```

Trains the encoder, categorical bottleneck, and predictor jointly via the JEPA objective. Checkpoints saved each epoch.

### 3. Train the policy head

```bash
python train_policy.py \
    --jepa_ckpt checkpoints/checkpoint_epoch4.pt \
    --data      data/dataset.pt \
    --out       checkpoints/policy/policy_head.pt
```

Encoder and bottleneck are frozen. Only the policy head is trained with cross-entropy against the computer moves in the dataset.

### 4. (Optional) Train the value head

```bash
python train_value.py \
    --jepa_ckpt checkpoints/checkpoint_epoch4.pt \
    --data      data/dataset.pt \
    --out       checkpoints/value_head.pt
```

### 5. Visualise embeddings

```bash
python viz/evaluate_embeddings.py \
    --jepa_ckpt   checkpoints/checkpoint_epoch4.pt \
    --fens        data/fen_analysis.csv \
    --num_samples 500 \
    --out         viz/umap.html
```

Open `viz/umap.html` in a browser. Click any point to open the position on Lichess.

### 6. Run the server

```bash
python app/server.py --checkpoint checkpoints/checkpoint_epoch4.pt
```

Opens at `http://127.0.0.1:5000`.
