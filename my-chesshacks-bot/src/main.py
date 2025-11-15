from .utils import chess_manager, GameContext
from .model import ChessModel
import random
import torch
import chess
import numpy as np
import os

MODEL = None

def board_to_tensor(board):
    """Convert chess.Board to (1, 12, 8, 8) tensor"""
    array = np.zeros((8, 8, 12), dtype=np.float32)
    piece_to_channel = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            array[rank, file, channel] = 1.0
    
    return torch.FloatTensor(array).permute(2, 0, 1).unsqueeze(0)

def load_model():
    global MODEL
    if MODEL is None:
        try:
            MODEL = ChessModel()
            MODEL.load_state_dict(torch.load('models/chess_model.pth', map_location='cpu'))
            MODEL.eval()
            print("Model loaded successfully")
        except:
            print("Model not found, using random moves")
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
                board_tensor = board_to_tensor(ctx.board)
                logits = MODEL(board_tensor)[0]
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
    
    move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    ctx.logProbabilities(move_probs)
    return random.choice(legal_moves)


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
