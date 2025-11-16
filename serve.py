import os
import sys

# CRITICAL: Disable CUDA before any torch imports to prevent GPU initialization timeout
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Also disable other CUDA-related environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

# Debug: Print environment before imports
print(f"[STARTUP] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", file=sys.stderr, flush=True)
print(f"[STARTUP] Python version: {sys.version}", file=sys.stderr, flush=True)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess

print("[STARTUP] FastAPI and chess imported", file=sys.stderr, flush=True)

from src.utils import chess_manager
print("[STARTUP] chess_manager imported", file=sys.stderr, flush=True)

from src import main
print("[STARTUP] main module imported", file=sys.stderr, flush=True)

# Pre-load model at startup to avoid timeout on first request
print("[STARTUP] Pre-loading model...", file=sys.stderr, flush=True)
try:
    main.load_model()
    print("[STARTUP] Model pre-loaded successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[STARTUP] Model pre-load failed: {e}", file=sys.stderr, flush=True)

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn, file=sys.stderr, flush=True)
    print(f"[MOVE_REQUEST] timeleft={timeleft}ms", file=sys.stderr, flush=True)

    try:
        start_time = time.perf_counter()
        print(f"[MOVE_REQUEST] Calling get_model_move...", file=sys.stderr, flush=True)
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
        print(f"[MOVE_REQUEST] Move generated in {time_taken:.2f}ms", file=sys.stderr, flush=True)
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(content={m: None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
