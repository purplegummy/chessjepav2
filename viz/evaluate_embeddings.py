import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import chess
import chess.engine
import torch
import numpy as np

from jepa.jepa import ChessJEPA
from jepa.eval_organizer import EvalOrganizer
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


MATE_CP   = 10_000
EVAL_CLIP = 1_500


def stockfish_eval_cp(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int) -> int:
    """Centipawn eval from current player's POV, clipped to ±EVAL_CLIP."""
    info  = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].relative
    if score.is_mate():
        return MATE_CP if score.mate() > 0 else -MATE_CP
    return max(-EVAL_CLIP, min(EVAL_CLIP, score.score()))


def eval_to_label(cp: int) -> str:
    if   cp >= 150:  return "winning"
    elif cp <= -150: return "losing"
    else:            return "equal"


def load_and_encode(
    csv_path: str,
    model: ChessJEPA,
    device,
    num_samples: int,
    stockfish_path: str | None = None,
    depth: int = 12,
    organizer: EvalOrganizer | None = None,
):
    """
    Reads FENs from CSV, encodes each position, returns embeddings + metadata.
    If organizer is provided, embeddings are the organizer's latent vectors (latent_dim,).
    Otherwise, embeddings are the JEPA bottleneck category indices (512,).
    """
    model.eval()
    embeddings = []
    metadata   = []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path) if stockfish_path else None

    try:
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
                    z, _ = model.bottleneck(taps[final_layer], tau=1.0)
                    indices = z.argmax(dim=-1).flatten(start_dim=1)  # (1, 512)

                    if organizer is not None:
                        h, _ = organizer(indices)   # (1, latent_dim)
                        h = h.squeeze(0).cpu()
                    else:
                        h = indices.squeeze(0).float().cpu()  # (512,)

                eval_cp = stockfish_eval_cp(board, engine, depth) if engine else 0

                embeddings.append(h)
                metadata.append({
                    "fen":           fen,
                    "material_diff": material_difference(board),
                    "side_to_move":  "white" if board.turn == chess.WHITE else "black",
                    "eval_cp":       eval_cp,
                    "eval_label":    eval_to_label(eval_cp) if engine else "unknown",
                    "url":           "https://lichess.org/analysis/" + fen.replace(" ", "_"),
                })
    finally:
        if engine:
            engine.quit()

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
            "eval_cp":       m["eval_cp"],
            "eval_label":    m["eval_label"],
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
      <option value="eval_cp">Stockfish eval</option>
      <option value="eval_label">Win / Equal / Lose</option>
    </select>
  </label>
  <label>Material diff range:
    <input type="range" id="matMin" min="-20" max="20" value="-20" step="1"> <span id="matMinVal">-20</span>
    &nbsp;to&nbsp;
    <input type="range" id="matMax" min="-20" max="20" value="20"  step="1"> <span id="matMaxVal">20</span>
  </label>
  <label>Stockfish eval (cp):
    <input type="range" id="evalMin" min="-1500" max="1500" value="-1500" step="50"> <span id="evalMinVal">-1500</span>
    &nbsp;to&nbsp;
    <input type="range" id="evalMax" min="-1500" max="1500" value="1500"  step="50"> <span id="evalMaxVal">1500</span>
  </label>
  <label>Position:
    <select id="evalLabel">
      <option value="all">All</option>
      <option value="winning">Winning (&gt;+150cp)</option>
      <option value="equal">Equal (±150cp)</option>
      <option value="losing">Losing (&lt;-150cp)</option>
    </select>
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

// diverging colormap: red=negative, blue=positive
function divergeColor(val, scale) {{
  const t = Math.max(-1, Math.min(1, val / scale));
  if (t >= 0) return `rgb(${{Math.round(255*(1-t))}}, ${{Math.round(255*(1-t))}}, 255)`;
  return `rgb(255, ${{Math.round(255*(1+t))}}, ${{Math.round(255*(1+t))}})`;
}}
const sideColor  = s => s === "white" ? "#f0c040" : "#555";
const labelColor = l => ({{ winning: "#4caf50", equal: "#aaa", losing: "#e53935", unknown: "#999" }})[l] || "#999";

function getColor(p, colorBy) {{
  if (colorBy === "material_diff") return divergeColor(p.material_diff, 15);
  if (colorBy === "eval_cp")       return divergeColor(p.eval_cp, 800);
  if (colorBy === "eval_label")    return labelColor(p.eval_label);
  return sideColor(p.side_to_move);
}}

function getFiltered() {{
  const colorBy    = document.getElementById("colorBy").value;
  const matMin     = parseInt(document.getElementById("matMin").value);
  const matMax     = parseInt(document.getElementById("matMax").value);
  const evalMin    = parseInt(document.getElementById("evalMin").value);
  const evalMax    = parseInt(document.getElementById("evalMax").value);
  const evalLabel  = document.getElementById("evalLabel").value;
  const sideFilter = document.getElementById("sideFilter").value;
  return points.filter(p =>
    p.material_diff >= matMin && p.material_diff <= matMax &&
    p.eval_cp >= evalMin && p.eval_cp <= evalMax &&
    (evalLabel  === "all" || p.eval_label  === evalLabel) &&
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

["colorBy","matMin","matMax","evalMin","evalMax","evalLabel","sideFilter"].forEach(id => {{
  const el = document.getElementById(id);
  el.addEventListener("input", () => {{
    if (id === "matMin")  document.getElementById("matMinVal").textContent  = el.value;
    if (id === "matMax")  document.getElementById("matMaxVal").textContent  = el.value;
    if (id === "evalMin") document.getElementById("evalMinVal").textContent = el.value;
    if (id === "evalMax") document.getElementById("evalMaxVal").textContent = el.value;
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
    parser.add_argument("--fens",           default="data/fen_analysis.csv")
    parser.add_argument("--jepa_ckpt",      default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--organizer_ckpt", default=None,
                        help="path to organizer checkpoint — plots organizer latent space instead of JEPA bottleneck")
    parser.add_argument("--num_samples",    default=200, type=int)
    parser.add_argument("--out",            default="umap.html")
    parser.add_argument("--stockfish",      default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth",          default=12, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessJEPA().to(device)
    model = load_checkpoint(model, args.jepa_ckpt, device)

    organizer = None
    if args.organizer_ckpt:
        org_ckpt  = torch.load(args.organizer_ckpt, map_location=device)
        organizer = EvalOrganizer(
            latent_dim=org_ckpt["latent_dim"],
            hidden_dim=org_ckpt["hidden_dim"],
        ).to(device)
        organizer.load_state_dict(org_ckpt["model_state_dict"])
        organizer.eval()
        print(f"Using organizer latent space (dim={org_ckpt['latent_dim']})")
    else:
        print("Using JEPA bottleneck indices (dim=512)")

    embeddings, metadata = load_and_encode(
        args.fens, model, device, args.num_samples, args.stockfish, args.depth,
        organizer=organizer,
    )
    plot_umap_html(embeddings, metadata, out_path=args.out)
