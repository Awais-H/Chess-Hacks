"""
Enhanced data processing with piece values and additional game state features.
"""
import os
import subprocess
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
from typing import Tuple, Optional


# Standard piece values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0  # King is invaluable, but we use 0 for material count
}


def board_to_array_v2(board: chess.Board, include_piece_values=True) -> np.ndarray:
    """
    Convert chess board to enhanced representation with piece values.
    
    Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (P, N, B, R, Q, K)
    12: White kingside castling
    13: White queenside castling
    14: Black kingside castling
    15: Black queenside castling
    16: En passant target square
    17: Turn (1 for white, 0 for black)
    18: Material advantage (normalized piece values) [optional]
    
    Args:
        board: python-chess Board object
        include_piece_values: Whether to include material advantage channel
        
    Returns:
        8x8x19 or 8x8x18 numpy array
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
    
    return array


def calculate_position_value(board: chess.Board) -> float:
    """
    Calculate a simple position evaluation based on material.
    Returns value from white's perspective in range [-1, 1].
    
    This can be used as a training target for the value head.
    """
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    
    white_material = 0.0
    black_material = 0.0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    material_diff = white_material - black_material
    # Normalize using tanh
    return np.tanh(material_diff / 10.0)


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess move to index.
    Index = from_square * 64 + to_square
    
    Args:
        move: python-chess Move object
        
    Returns:
        Integer index (0-4095)
    """
    return move.from_square * 64 + move.to_square


def download_and_extract_pgn(url: str, output_dir: str = ".") -> Optional[str]:
    """
    Download and extract a Lichess PGN database.
    
    Args:
        url: URL to .pgn.zst file
        output_dir: Directory to save files
        
    Returns:
        Path to extracted .pgn file, or None if failed
    """
    filename = url.split("/")[-1]
    zst_path = os.path.join(output_dir, filename)
    pgn_path = zst_path.replace(".zst", "")
    
    # Skip if already extracted
    if os.path.exists(pgn_path):
        print(f"✓ PGN already exists: {pgn_path}")
        return pgn_path
    
    try:
        # Download
        if not os.path.exists(zst_path):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-q", url, "-O", zst_path], check=True)
            print(f"✓ Downloaded {filename}")
        
        # Extract
        print(f"Extracting {filename}...")
        subprocess.run(["unzstd", zst_path], check=True)
        print(f"✓ Extracted to {pgn_path}")
        
        # Clean up compressed file
        if os.path.exists(zst_path):
            os.remove(zst_path)
            print(f"✓ Cleaned up {filename}")
        
        return pgn_path
        
    except Exception as e:
        print(f"✗ Failed to download/extract: {e}")
        return None


def process_pgn_v2(
    pgn_path: str,
    max_positions: int = 500000,
    min_rating: int = 1800,
    skip_moves: int = 10,
    include_values: bool = True,
    include_position_eval: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process PGN file with enhanced features.
    
    Args:
        pgn_path: Path to .pgn file
        max_positions: Maximum number of positions to extract
        min_rating: Minimum ELO rating for both players
        skip_moves: Number of opening moves to skip
        include_values: Include piece value channel
        include_position_eval: Also return position evaluations for value head training
        
    Returns:
        Tuple of (positions, moves, values) as numpy arrays
        values is None if include_position_eval=False
    """
    if not os.path.exists(pgn_path):
        print(f"✗ PGN file not found: {pgn_path}")
        return None, None, None
    
    positions_list = []
    moves_list = []
    values_list = [] if include_position_eval else None
    
    print(f"\nProcessing {os.path.basename(pgn_path)}:")
    print(f"  Max positions: {max_positions}")
    print(f"  Min rating: {min_rating}")
    print(f"  Skip opening moves: {skip_moves}")
    print(f"  Include piece values: {include_values}")
    print(f"  Include position eval: {include_position_eval}\n")
    
    try:
        with open(pgn_path, 'r') as pgn_file:
            games_processed = 0
            games_skipped = 0
            pbar = tqdm(total=max_positions, desc="Extracting positions", unit="pos")
            
            while len(positions_list) < max_positions:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    
                    if game is None:
                        break
                    
                    # Filter by rating
                    try:
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))
                        if white_elo < min_rating or black_elo < min_rating:
                            games_skipped += 1
                            continue
                    except (ValueError, TypeError):
                        games_skipped += 1
                        continue
                    
                    # Process game moves
                    board = game.board()
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        move_count += 1
                        
                        # Skip opening moves
                        if move_count <= skip_moves:
                            board.push(move)
                            continue
                        
                        try:
                            positions_list.append(board_to_array_v2(board, include_values))
                            moves_list.append(move_to_index(move))
                            
                            if include_position_eval:
                                values_list.append(calculate_position_value(board))
                            
                            pbar.update(1)
                            
                            if len(positions_list) >= max_positions:
                                break
                        except Exception:
                            pass
                        
                        board.push(move)
                    
                    games_processed += 1
                    if games_processed % 100 == 0:
                        pbar.set_postfix({
                            'games': games_processed,
                            'skipped': games_skipped
                        })
                
                except Exception:
                    continue
            
            pbar.close()
            print(f"\n✓ Processed {games_processed} games")
            print(f"  Skipped: {games_skipped} games")
            print(f"  Extracted: {len(positions_list)} positions")
    
    except Exception as e:
        print(f"✗ Error processing PGN: {e}")
        return None, None, None
    
    if not positions_list:
        print("✗ No positions extracted")
        return None, None, None
    
    positions = np.array(positions_list, dtype=np.float32)
    moves = np.array(moves_list, dtype=np.int32)
    values = np.array(values_list, dtype=np.float32) if include_position_eval else None
    
    print(f"✓ Final shapes: positions={positions.shape}, moves={moves.shape}", end="")
    if values is not None:
        print(f", values={values.shape}")
    else:
        print()
    
    return positions, moves, values


def load_or_process_dataset_v2(
    dataset_url: str,
    cache_dir: str,
    max_positions: int = 500000,
    min_rating: int = 1800,
    skip_moves: int = 10,
    include_values: bool = True,
    include_position_eval: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cached dataset or download and process PGN with enhanced features.
    
    Args:
        dataset_url: URL to Lichess database
        cache_dir: Directory for caching processed data
        max_positions: Maximum positions to extract
        min_rating: Minimum ELO rating
        skip_moves: Opening moves to skip
        include_values: Include piece value channel
        include_position_eval: Include position evaluations
        
    Returns:
        Tuple of (positions, moves, values) numpy arrays
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filenames based on URL and settings
    dataset_name = dataset_url.split("/")[-1].replace(".pgn.zst", "")
    suffix = "_v2" if include_values else "_v2_novals"
    if include_position_eval:
        suffix += "_eval"
    
    positions_cache = os.path.join(cache_dir, f"{dataset_name}_positions{suffix}.npy")
    moves_cache = os.path.join(cache_dir, f"{dataset_name}_moves{suffix}.npy")
    values_cache = os.path.join(cache_dir, f"{dataset_name}_values{suffix}.npy")
    
    # Check cache
    cache_exists = os.path.exists(positions_cache) and os.path.exists(moves_cache)
    if include_position_eval:
        cache_exists = cache_exists and os.path.exists(values_cache)
    
    if cache_exists:
        print(f"✓ Loading cached dataset: {dataset_name}")
        positions = np.load(positions_cache)
        moves = np.load(moves_cache)
        values = np.load(values_cache) if include_position_eval else None
        print(f"  Loaded {len(positions)} positions")
        return positions, moves, values
    
    # Download and process
    print(f"Processing new dataset: {dataset_name}")
    pgn_path = download_and_extract_pgn(dataset_url, cache_dir)
    
    if not pgn_path:
        return None, None, None
    
    positions, moves, values = process_pgn_v2(
        pgn_path, max_positions, min_rating, skip_moves,
        include_values, include_position_eval
    )
    
    if positions is not None:
        # Cache the processed data
        np.save(positions_cache, positions)
        np.save(moves_cache, moves)
        if include_position_eval and values is not None:
            np.save(values_cache, values)
        print(f"✓ Cached dataset to {cache_dir}")
        
        # Clean up PGN file to save space
        if os.path.exists(pgn_path):
            os.remove(pgn_path)
            print(f"✓ Cleaned up {os.path.basename(pgn_path)}")
    
    return positions, moves, values

