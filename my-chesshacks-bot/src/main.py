from .utils import chess_manager, GameContext
from .model import ChessModel
import random
import torch
import chess
import numpy as np
import os
from huggingface_hub import hf_hub_download

MODEL = None
HF_REPO_ID = "IridescentBlue/Chess_01"

def board_to_array_with_metadata(board):
    """Convert board to 8x8x18 array with metadata channels."""
    array = np.zeros((8, 8, 18), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Channels 0-11: Piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            array[rank, file, channel] = 1.0
    
    # Channel 12: Turn to move (1.0 for white, 0.0 for black)
    array[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Channels 13-16: Castling rights
    array[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    array[:, :, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    array[:, :, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    array[:, :, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # Channel 17: En passant square
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        array[ep_rank, ep_file, 17] = 1.0
    
    return array

def load_model():
    global MODEL
    if MODEL is None:
        model_path = None
        
        try:
            print(f"Downloading model from HuggingFace: {HF_REPO_ID}")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename="chess_model_best.pth",
                cache_dir="./.model_cache"
            )
            print(f"Model cached at: {model_path}")
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            if os.path.exists('models/chess_model_best.pth'):
                model_path = 'models/chess_model_best.pth'
                print("Using local fallback model")
        
        if model_path:
            try:
                MODEL = ChessModel()
                checkpoint = torch.load(model_path, map_location='cpu')
                MODEL.load_state_dict(checkpoint["model_state_dict"])
                MODEL.eval()
                print("Model loaded successfully")
                return
            except Exception as e:
                print(f"Model load failed: {e}")
        
        print("No model available, using random moves")
        MODEL = False

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    load_model()
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    try:
        if MODEL and MODEL is not False:
            with torch.no_grad():
                # get numpy encoding (could be HWC, CHW, or other)
                board_np = board_to_array_with_metadata(ctx.board)  # user func

                # convert to torch tensor
                x = torch.from_numpy(board_np).float()

                # Ensure batch dim exists: accept (C,H,W) or (H,W,C) or (B,C,H,W) or (B,H,W,C)
                if x.ndim == 3:
                    # Could be (C,H,W) or (H,W,C) — detect by channels value
                    # If first dim is small (<= 18) and last dim is 8, assume (C,H,W)
                    if x.shape[0] in (12, 13, 18) and x.shape[1] == 8 and x.shape[2] == 8:
                        x = x.unsqueeze(0)  # (1,C,H,W) OK
                    else:
                        # assume (H,W,C)
                        x = x.permute(2, 0, 1).unsqueeze(0)  # -> (1,C,H,W)
                elif x.ndim == 4:
                    # Could be (B,H,W,C) or (B,C,H,W)
                    if x.shape[1] == 8 and x.shape[2] == 8 and x.shape[3] in (12,13,18):
                        # it's (B,H,W,C) -> move channels to axis=1
                        x = x.permute(0, 3, 1, 2)
                    # else assume already (B,C,H,W)
                else:
                    raise ValueError(f"Unexpected board tensor shape: {x.shape}")

                # Move to model device
                device = next(MODEL.parameters()).device
                x = x.to(device)

                # Sanity check: channels should match first conv weights
                expected_in_ch = MODEL.conv_input.weight.shape[1] if hasattr(MODEL, "conv_input") else None
                if expected_in_ch is None:
                    # try more common name
                    expected_in_ch = MODEL.conv1.weight.shape[1] if hasattr(MODEL, "conv1") else None

                if expected_in_ch is not None:
                    if x.shape[1] != expected_in_ch:
                        raise RuntimeError(f"Model expects {expected_in_ch} input channels but tensor has {x.shape[1]} channels. "
                                        "Check your board encoder or input ordering (should be channels-first).")

                # forward pass
                logits = MODEL(x)[0]          # shape: (num_moves,)
                probs = torch.softmax(logits, dim=0)

                move_probs = {}
                for move in legal_moves:
                    move_idx = move.from_square * 64 + move.to_square
                    move_probs[move] = probs[move_idx].item()

                if move_probs:
                    ctx.logProbabilities(move_probs)
                    return max(move_probs, key=move_probs.get)

    except Exception as e:
        print(f"Model inference failed: {e}")

    # fallback uniform
    move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    ctx.logProbabilities(move_probs)
    return random.choice(legal_moves)


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
