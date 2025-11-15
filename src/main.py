import os
# Disable CUDA before importing torch to prevent GPU initialization
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from .utils import chess_manager, GameContext
from .model import ChessModel
import random
import torch
import chess
import numpy as np
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
        
        # try local model first (fastest)
        if os.path.exists('models/chess_model.pth'):
            model_path = 'models/chess_model.pth'
            print("Using local model from models/chess_model.pth")
        else:
            # Fallback to HuggingFace if local not available
            try:
                print(f"Local model not found, downloading from HuggingFace: {HF_REPO_ID}")
                model_path = hf_hub_download(
                    repo_id=HF_REPO_ID, 
                    filename="chess_model.pth",
                    cache_dir="./.model_cache",
                    resume_download=True
                )
                print(f"Model downloaded/cached at: {model_path}")
            except Exception as e:
                print(f"HuggingFace download failed: {e}")
                model_path = None
        
        if model_path:
            try:
                MODEL = ChessModel()
                # Force CPU to avoid CUDA initialization timeout during deployment
                # Small model doesn't benefit from GPU anyway
                device = 'cpu'
                checkpoint = torch.load(model_path, map_location=device)
                
                # handle different save formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        MODEL.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        MODEL.load_state_dict(checkpoint['state_dict'])
                    else:
                        MODEL.load_state_dict(checkpoint)
                else:
                    MODEL.load_state_dict(checkpoint)
                
                # move model to the appropriate device
                MODEL = MODEL.to(device)
                MODEL.eval()
                print(f"Model loaded successfully on {device}")
                return
            except Exception as e:
                print(f"Model load failed: {e}")
        
        print("No model available, using random moves")
        MODEL = False

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # Lazy load model on first call
    load_model()
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    try:
        if MODEL and MODEL is not False:
            with torch.no_grad():
                board_tensor = board_to_tensor(ctx.board)
                # Model is on CPU, tensor is already on CPU (no device transfer needed)
                logits = MODEL(board_tensor)[0]
                probs = torch.softmax(logits, dim=0)
                
                move_probs = {}
                for move in legal_moves:
                    move_idx = move.from_square * 64 + move.to_square
                    prob = probs[move_idx].item()

                    # prefer queen promotions by adding a small bonus
                    # knight promotions by a small bonus
                    # rook and bishop promotions get base probability
                    if move.promotion:
                        if move.promotion == chess.QUEEN:
                            prob *= 1.5
                        elif move.promotion == chess.KNIGHT:
                            prob *= 1.2
                    
                    move_probs[move] = prob
                
                if move_probs:
                    ctx.logProbabilities(move_probs)
                    best_move = max(move_probs, key=move_probs.get)
                    
                    # Debug: print selected move
                    promo_info = f" (promote to {chess.piece_name(best_move.promotion)})" if best_move.promotion else ""
                    print(f"Bot selected: {best_move.uci()}{promo_info} (prob: {move_probs[best_move]:.4f})")
                    
                    # verify the move is actually legal
                    if best_move in legal_moves:
                        return best_move
                    else:
                        print(f"Warning: Model selected illegal move {best_move}, falling back to random")
    except Exception as e:
        print(f"Model inference failed: {e}")
    
    # Fallback to random move
    move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    ctx.logProbabilities(move_probs)
    fallback_move = random.choice(legal_moves)
    promo_info = f" (promote to {chess.piece_name(fallback_move.promotion)})" if fallback_move.promotion else ""
    print(f"Fallback random move: {fallback_move.uci()}{promo_info}")
    return fallback_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
