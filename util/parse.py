# to do, parse games in pgn format, and convert them into tensor format that can be fed into the model.
import chess
import chess.pgn
import torch
import numpy as np

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board: chess.Board, force_flip: bool | None = None) -> torch.Tensor:
    flip = board.turn == chess.BLACK if force_flip is None else force_flip
    t = torch.zeros(17, 8, 8, dtype=torch.uint8)
    
    us, them = (chess.BLACK, chess.WHITE) if flip else (chess.WHITE, chess.BLACK)

    def sq_to_rc(sq):
        r, c = sq // 8, sq % 8
        return (7 - r, 7 - c) if flip else (r, c)

    for i, piece in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece, us):
            r, c = sq_to_rc(sq)
            t[i, r, c] = 1
        for sq in board.pieces(piece, them):
            r, c = sq_to_rc(sq)
            t[i + 6, r, c] = 1

    # castling rights
    w_ks, w_qs = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE)
    b_ks, b_qs = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)
    t[12], t[13], t[14], t[15] = (b_ks, b_qs, w_ks, w_qs) if flip else (w_ks, w_qs, b_ks, b_qs)

    # en passant
    if board.ep_square is not None:
        r, c = sq_to_rc(board.ep_square)
        t[16, r, c] = 1

    return t


# ---------------------------------------------------------------------------
# Action encoding (AlphaZero-style, always from current player's POV)
#
# 73 planes × 64 squares = 4672 possible action indices
#   planes  0-55 : queen-style moves — 8 directions × 7 distances
#   planes 56-63 : knight moves (8 L-shapes)
#   planes 64-72 : underpromotions — 3 capture dirs × 3 pieces (N/B/R)
#                  queen promotions use the normal queen-move plane
#
# action_index = plane * 64 + from_r * 8 + from_c
# ---------------------------------------------------------------------------

# (dr, dc) in tensor space — r=0 is our back rank, r=7 is opponent's back rank
_QUEEN_DIRS    = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
_KNIGHT_DELTAS = [(2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2),(-2,1),(-2,-1)]
_PROMO_DIRS    = [(1,-1),(1,0),(1,1)]          # capture-left, straight, capture-right
_PROMO_PIECES  = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

ACTION_SIZE = 73 * 64


def _sq_to_rc(sq: int, flip: bool):
    r, c = sq // 8, sq % 8
    return (7 - r, 7 - c) if flip else (r, c)


def _rc_to_sq(r: int, c: int, flip: bool) -> int:
    return chess.square(7 - c, 7 - r) if flip else chess.square(c, r)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode a chess.Move as a scalar in [0, ACTION_SIZE)."""
    flip = board.turn == chess.BLACK
    fr, fc = _sq_to_rc(move.from_square, flip)
    tr, tc = _sq_to_rc(move.to_square,   flip)
    dr, dc = tr - fr, tc - fc

    # underpromotion (queen promotion falls through to queen-move plane)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        dir_idx   = _PROMO_DIRS.index((dr, dc))
        piece_idx = _PROMO_PIECES.index(move.promotion)
        plane = 64 + dir_idx * 3 + piece_idx
        return plane * 64 + fr * 8 + fc

    # knight move
    if (dr, dc) in _KNIGHT_DELTAS:
        plane = 56 + _KNIGHT_DELTAS.index((dr, dc))
        return plane * 64 + fr * 8 + fc

    # queen-style move (slides + queen promotions)
    dist  = max(abs(dr), abs(dc))
    unit  = (dr // dist, dc // dist)
    plane = _QUEEN_DIRS.index(unit) * 7 + (dist - 1)
    return plane * 64 + fr * 8 + fc


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode a scalar action index back to a chess.Move."""
    flip = board.turn == chess.BLACK
    plane, sq_idx = divmod(index, 64)
    fr, fc = divmod(sq_idx, 8)

    if plane >= 64:                                    # underpromotion
        dir_idx, piece_idx = divmod(plane - 64, 3)
        dr, dc    = _PROMO_DIRS[dir_idx]
        tr, tc    = fr + dr, fc + dc
        promotion = _PROMO_PIECES[piece_idx]
    elif plane >= 56:                                  # knight
        dr, dc    = _KNIGHT_DELTAS[plane - 56]
        tr, tc    = fr + dr, fc + dc
        promotion = None
    else:                                              # queen-style
        dir_idx, dist_m1 = divmod(plane, 7)
        dr, dc = _QUEEN_DIRS[dir_idx]
        dist   = dist_m1 + 1
        tr, tc = fr + dr * dist, fc + dc * dist
        # pawn reaching back rank → queen promotion
        from_sq   = _rc_to_sq(fr, fc, flip)
        promotion = (chess.QUEEN
                     if tr == 7 and board.piece_type_at(from_sq) == chess.PAWN
                     else None)

    return chess.Move(_rc_to_sq(fr, fc, flip), _rc_to_sq(tr, tc, flip),
                      promotion=promotion)
