from .utils import chess_manager, GameContext
from .model import ChessModel
import random
import torch
import chess
import chess.engine
import numpy as np
import os
from huggingface_hub import hf_hub_download

MODEL = None
STOCKFISH_ENGINE = None
HF_REPO_ID = "ricfinity242/chess"


def board_to_tensor(board):
    array = np.zeros((8, 8, 12), dtype=np.float32)
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

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
    if MODEL is None or MODEL is False:
        print("=" * 60)
        print("LOADING CHESS MODEL")
        print("=" * 60)
        model_path = None

        try:
            print("Attempting to download model from HuggingFace...")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="chess_model.pth",
                cache_dir="./.model_cache",
            )
            print(f"Model downloaded to: {model_path}")
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            if os.path.exists("models/chess_model.pth"):
                model_path = "models/chess_model.pth"
                print(f"Using local fallback model: {model_path}")

        if model_path:
            try:
                print("Loading model into memory...")
                MODEL = ChessModel()
                checkpoint = torch.load(model_path, map_location="cpu")

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    MODEL.load_state_dict(checkpoint["model_state_dict"])
                else:
                    MODEL.load_state_dict(checkpoint)

                MODEL.eval()
                print("✓ Model loaded successfully!")
                print("=" * 60)
                return
            except Exception as e:
                print(f"ERROR - Model load failed: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("ERROR - No model path found")

        print("Setting MODEL to False (will use random moves)")
        print("=" * 60)
        MODEL = False


def load_stockfish():
    global STOCKFISH_ENGINE
    if STOCKFISH_ENGINE is None:
        stockfish_paths = [
            r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
            r"C:\stockfish\stockfish.exe",
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "stockfish",
        ]

        for path in stockfish_paths:
            try:
                STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(path)
                STOCKFISH_ENGINE.configure({"Skill Level": 1})
                return
            except:
                continue

        print("WARNING: Stockfish not found, using random moves")
        STOCKFISH_ENGINE = False


def get_stockfish_move(board):
    """Get a move from Stockfish"""
    load_stockfish()

    if STOCKFISH_ENGINE and STOCKFISH_ENGINE is not False:
        try:
            # Very short time + depth 1 for weak play
            result = STOCKFISH_ENGINE.play(
                board, chess.engine.Limit(time=0.01, depth=1)
            )
            return result.move
        except Exception as e:
            print(f"Stockfish error: {e}")

    # Fallback to random
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Universal chess bot that alternates between Stockfish (white) and your model (black)
    """
    load_model()

    # Determine whose turn based on move count
    move_count = len(list(ctx.board.move_stack))
    is_white_turn = move_count % 2 == 0

    # White = Stockfish
    if is_white_turn:
        move = get_stockfish_move(ctx.board)
        ctx.logProbabilities({})
        print(f"[WHITE - STOCKFISH] Move {move_count + 1}: {move}")
        return move

    # Black = Your Model
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
                    prob = probs[move_idx].item()

                    # Promotion bonus
                    if move.promotion:
                        if move.promotion == chess.QUEEN:
                            prob *= 1.5
                        elif move.promotion == chess.KNIGHT:
                            prob *= 1.2

                    move_probs[move] = prob

                if move_probs:
                    ctx.logProbabilities(move_probs)
                    model_move = max(move_probs, key=move_probs.get)

                    if model_move in legal_moves:
                        print(
                            f"[BLACK - MODEL] Move {move_count + 1}: {model_move} (prob: {move_probs[model_move]:.4f})"
                        )
                        return model_move
                    else:
                        print(
                            f"[BLACK - MODEL] ERROR: Selected illegal move {model_move}"
                        )
        else:
            print("[BLACK - MODEL] ERROR: Model not loaded, using random move")
    except Exception as e:
        print(f"[BLACK - MODEL] ERROR: {e}")
        import traceback

        traceback.print_exc()

    # Fallback to random move
    move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    ctx.logProbabilities(move_probs)
    fallback = random.choice(legal_moves)
    print(f"[BLACK - MODEL] Fallback random: {fallback}")
    return fallback


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins"""
    pass
