# !pip install python-chess tqdm

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

def board_to_array(board):
    """Convert board to 8x8x12 array. Channels 0-5: white pieces, 6-11: black pieces."""
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
    
    return array

def move_to_index(move):
    """Convert move to index: from_square * 64 + to_square"""
    return move.from_square * 64 + move.to_square


def process_pgn(pgn_path, max_positions=500000, min_rating=1800, skip_moves=10):
    positions_list = []
    moves_list = []
    
    print(f"Processing {pgn_path}: {max_positions} positions, rating >{min_rating}, skip {skip_moves} moves\n")
    
    try:
        with open(pgn_path, 'r') as pgn_file:
            games_processed = 0
            games_skipped = 0
            pbar = tqdm(total=max_positions, desc="Collecting", unit="pos")
            
            while len(positions_list) < max_positions:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    
                    if game is None:
                        break
                    
                    try:
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))
                        if white_elo < min_rating or black_elo < min_rating:
                            games_skipped += 1
                            continue
                    except (ValueError, TypeError):
                        games_skipped += 1
                        continue
                    
                    board = game.board()
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        move_count += 1
                        if move_count <= skip_moves:
                            board.push(move)
                            continue
                        
                        try:
                            positions_list.append(board_to_array(board))
                            moves_list.append(move_to_index(move))
                            pbar.update(1)
                            if len(positions_list) >= max_positions:
                                break
                        except:
                            pass
                        board.push(move)
                    
                    games_processed += 1
                    if games_processed % 100 == 0:
                        pbar.set_postfix({'games': games_processed, 'skipped': games_skipped})
                
                except:
                    continue
            
            pbar.close()
            print(f"\nDone: {games_processed} games, {games_skipped} skipped, {len(positions_list)} positions")
    
    except FileNotFoundError:
        print(f"Error: {pgn_path} not found")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
    if not positions_list:
        return None, None
    
    positions = np.array(positions_list, dtype=np.float32)
    moves = np.array(moves_list, dtype=np.int32)
    print(f"Shapes: {positions.shape}, {moves.shape}")
    return positions, moves

# Download and extract
# !apt-get install -y zstd
# !wget https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst
# !unzstd lichess_db_standard_rated_2017-03.pgn.zst

# Process
positions, moves = process_pgn('lichess_db_standard_rated_2017-03.pgn', max_positions=500000)
if positions is not None:
    np.save('positions.npy', positions)
    np.save('moves.npy', moves)