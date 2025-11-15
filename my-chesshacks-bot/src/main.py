from .utils import chess_manager, GameContext
from .model import ChessModel
import random
import torch
import chess
import numpy as np
import os
from huggingface_hub import hf_hub_download

MODEL = None
HF_REPO_ID = "ricfinity242/chess"

def board_to_tensor(board):
    """convert chess.Board to (1, 12, 8, 8) tensor"""
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
        model_path = None
        
        try:
            print(f"Downloading model from HuggingFace: {HF_REPO_ID}")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename="chess_model.pth",
                cache_dir="./.model_cache"
            )
            print(f"Model cached at: {model_path}")
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            if os.path.exists('models/chess_model.pth'):
                model_path = 'models/chess_model.pth'
                print("Using local fallback model")
        
        if model_path:
            try:
                MODEL = ChessModel()
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Handle different save formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        MODEL.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        MODEL.load_state_dict(checkpoint['state_dict'])
                    else:
                        MODEL.load_state_dict(checkpoint)
                else:
                    MODEL.load_state_dict(checkpoint)
                
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
