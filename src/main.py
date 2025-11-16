import os
# Disable CUDA before importing torch to prevent GPU initialization
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from .utils import chess_manager, GameContext
from .model import ChessModelV1, ChessModelV2
import random
import torch
import chess
import numpy as np
from huggingface_hub import hf_hub_download

MODEL = None
HF_REPO_ID = "ricfinity242/chess"

# Standard piece values (centipawns) for material evaluation
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0  # King is invaluable
}

def board_to_tensor(board, include_piece_values=True):
    """
    Convert chess.Board to enhanced tensor representation.
    
    Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (P, N, B, R, Q, K)
    12: White kingside castling
    13: White queenside castling
    14: Black kingside castling
    15: Black queenside castling
    16: En passant target square
    17: Turn (1 for white, 0 for black)
    18: Material advantage (normalized) [optional]
    
    Returns:
        Tensor of shape (1, 19, 8, 8) or (1, 18, 8, 8)
    """
    num_channels = 19 if include_piece_values else 18
    array = np.zeros((8, 8, num_channels), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Piece positions (channels 0-11)
    white_material = 0.0
    black_material = 0.0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_channel[piece.piece_type]
            
            if piece.color == chess.WHITE:
                array[rank, file, channel] = 1.0
                white_material += PIECE_VALUES[piece.piece_type]
            else:
                array[rank, file, channel + 6] = 1.0
                black_material += PIECE_VALUES[piece.piece_type]
    
    # Castling rights (channels 12-15)
    if board.has_kingside_castling_rights(chess.WHITE):
        array[:, :, 12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        array[:, :, 13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        array[:, :, 14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        array[:, :, 15] = 1.0
    
    # En passant (channel 16)
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        array[ep_rank, ep_file, 16] = 1.0
    
    # Turn (channel 17)
    if board.turn == chess.WHITE:
        array[:, :, 17] = 1.0
    
    # Material advantage (channel 18) - normalized
    if include_piece_values:
        material_diff = white_material - black_material
        # Normalize to roughly [-1, 1] (queen advantage is ~9)
        normalized_material = np.tanh(material_diff / 10.0)
        array[:, :, 18] = normalized_material
    
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
                # Force CPU to avoid CUDA initialization timeout during deployment
                device = 'cpu'
                checkpoint = torch.load(model_path, map_location=device)
                
                # Extract state_dict from different save formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Auto-detect model version based on first conv layer input channels
                if 'conv1.weight' in state_dict:
                    # V1 model (simple CNN with 12 input channels)
                    input_channels = state_dict['conv1.weight'].shape[1]
                    print(f"Detected V1 model with {input_channels} input channels")
                    MODEL = ChessModelV1()
                elif 'conv_input.weight' in state_dict:
                    # V2 model (enhanced with 19 input channels)
                    input_channels = state_dict['conv_input.weight'].shape[1]
                    use_piece_values = (input_channels == 19)
                    print(f"Detected V2 model with {input_channels} input channels")
                    MODEL = ChessModelV2(use_piece_values=use_piece_values)
                else:
                    raise ValueError("Unknown model architecture")
                
                MODEL.load_state_dict(state_dict)
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
                # Use appropriate tensor format based on model version
                if isinstance(MODEL, ChessModelV1):
                    # V1 model expects 12 channels (pieces only)
                    board_tensor = board_to_tensor(ctx.board, include_piece_values=False)
                    # Extract only the piece channels (0-11)
                    board_tensor = board_tensor[:, :12, :, :]
                else:
                    # V2 model expects 19 channels (pieces + game state + material)
                    board_tensor = board_to_tensor(ctx.board, include_piece_values=True)
                
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
