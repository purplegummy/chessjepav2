import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import chess
import torch
import numpy as np

from jepa.jepa import ChessJEPA
from util.parse import board_to_tensor

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def load_checkpoint(model: ChessJEPA, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return model


def material_difference(board: chess.Board) -> int:
    """White material minus black material (pawns=1, N/B=3, R=5, Q=9)."""
    score = 0
    for piece_type, val in PIECE_VALUES.items():
        score += val * len(board.pieces(piece_type, chess.WHITE))
        score -= val * len(board.pieces(piece_type, chess.BLACK))
    return score


def load_and_encode(csv_path: str, model: ChessJEPA, device, num_samples: int):
    """
    Reads FENs from CSV, encodes each position, returns embeddings + metadata.
    """
    model.eval()
    embeddings = []
    metadata   = []  # per-point: fen, material_diff, side_to_move, lichess_url

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            fen   = row["FEN"]
            board = chess.Board(fen)
            tensor = board_to_tensor(board).float().unsqueeze(0).to(device)

            with torch.no_grad():
                taps = model.encoder(tensor)
                final_layer = max(taps.keys())
                h = taps[final_layer].mean(dim=1).squeeze(0).cpu()  # (256,)

            embeddings.append(h)
            metadata.append({
                "fen":           fen,
                "material_diff": material_difference(board),
                "side_to_move":  "white" if board.turn == chess.WHITE else "black",
                "url":           "https://lichess.org/analysis/" + fen.replace(" ", "_"),
            })

    return embeddings, metadata


def plot_umap_html(embeddings: list, metadata: list, out_path: str = "umap.html"):
    X = np.stack([e.numpy() for e in embeddings])
    import umap
    reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, random_state=42)
    X_2d = reducer.fit_transform(X)

    points = []
    for i, (x, y) in enumerate(X_2d):
        m = metadata[i]
        points.append({
            "x": float(x), "y": float(y),
            "fen":           m["fen"],
            "material_diff": m["material_diff"],
            "side_to_move":  m["side_to_move"],
            "url":           m["url"],
        })

    data_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Chess Embedding UMAP</title>
<style>
  body {{ font-family: sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  #controls {{ margin-bottom: 12px; display: flex; gap: 20px; align-items: center; }}
  canvas {{ border: 1px solid #ccc; cursor: pointer; }}
  label {{ font-size: 13px; }}
</style>
</head>
<body>
<h2>Chess Position Embeddings (UMAP)</h2>
<div id="controls">
  <label>Color by:
    <select id="colorBy">
      <option value="material_diff">Material difference</option>
      <option value="side_to_move">Side to move</option>
    </select>
  </label>
  <label>Material diff range:
    <input type="range" id="matMin" min="-20" max="20" value="-20" step="1"> <span id="matMinVal">-20</span>
    &nbsp;to&nbsp;
    <input type="range" id="matMax" min="-20" max="20" value="20"  step="1"> <span id="matMaxVal">20</span>
  </label>
  <label>Side:
    <select id="sideFilter">
      <option value="all">All</option>
      <option value="white">White to move</option>
      <option value="black">Black to move</option>
    </select>
  </label>
</div>
<canvas id="c" width="900" height="680"></canvas>

<script>
const points = {data_json};
const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");

const pad = 50;
const allX = points.map(p => p.x), allY = points.map(p => p.y);
const xMin = Math.min(...allX), xMax = Math.max(...allX);
const yMin = Math.min(...allY), yMax = Math.max(...allY);
const sx = x => pad + (x - xMin) / (xMax - xMin) * (canvas.width  - 2*pad);
const sy = y => pad + (y - yMin) / (yMax - yMin) * (canvas.height - 2*pad);

// diverging colormap for material diff: red=black winning, blue=white winning
function matColor(diff) {{
  const t = Math.max(-1, Math.min(1, diff / 15));
  if (t >= 0) return `rgb(${{Math.round(255*(1-t))}}, ${{Math.round(255*(1-t))}}, 255)`;
  return `rgb(255, ${{Math.round(255*(1+t))}}, ${{Math.round(255*(1+t))}})`;
}}
const sideColor = s => s === "white" ? "#f0c040" : "#555";

function getColor(p, colorBy) {{
  return colorBy === "material_diff" ? matColor(p.material_diff) : sideColor(p.side_to_move);
}}

function getFiltered() {{
  const colorBy    = document.getElementById("colorBy").value;
  const matMin     = parseInt(document.getElementById("matMin").value);
  const matMax     = parseInt(document.getElementById("matMax").value);
  const sideFilter = document.getElementById("sideFilter").value;
  return points.filter(p =>
    p.material_diff >= matMin && p.material_diff <= matMax &&
    (sideFilter === "all" || p.side_to_move === sideFilter)
  ).map(p => ({{ ...p, color: getColor(p, colorBy) }}));
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const visible = getFiltered();
  visible.forEach(p => {{
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 6, 0, 2*Math.PI);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
  }});
}}

["colorBy","matMin","matMax","sideFilter"].forEach(id => {{
  const el = document.getElementById(id);
  el.addEventListener("input", () => {{
    if (id === "matMin")  document.getElementById("matMinVal").textContent = el.value;
    if (id === "matMax")  document.getElementById("matMaxVal").textContent = el.value;
    draw();
  }});
}});

canvas.addEventListener("click", e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const visible = getFiltered();
  for (const p of visible) {{
    const dx = sx(p.x) - mx, dy = sy(p.y) - my;
    if (Math.sqrt(dx*dx + dy*dy) < 8) {{
      window.open(p.url, "_blank");
      return;
    }}
  }}
}});

draw();
</script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens",        default="data/fen_analysis.csv")
    parser.add_argument("--jepa_ckpt",   default="checkpoints/checkpoint_epoch4.pt")
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--out",         default="umap.html")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessJEPA().to(device)
    model = load_checkpoint(model, args.jepa_ckpt, device)
    embeddings, metadata = load_and_encode(args.fens, model, device, args.num_samples)
    plot_umap_html(embeddings, metadata, out_path=args.out)
