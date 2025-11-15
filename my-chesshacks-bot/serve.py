from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os

from src.utils import chess_manager
from src import main

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(
            content={"error": "Missing pgn or timeleft"}, status_code=400
        )

    if "pgn" not in data or "timeleft" not in data:
        return JSONResponse(
            content={"error": "Missing pgn or timeleft"}, status_code=400
        )

    pgn = data["pgn"]
    timeleft = data["timeleft"]

    if not pgn or pgn.strip() == "":
        pgn = "*"

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
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

    stockfish_move = None
    if isinstance(move, tuple):
        move, stockfish_move = move

    if not isinstance(move_probs, dict):
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "error": "Failed to get move",
                "message": "Move probabilities is not a dictionary",
            },
            status_code=500,
        )

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(
                content={
                    m: None,
                    "move_probs": None,
                    "error": "Failed to get move",
                    "message": "Move probabilities is not a dictionary",
                },
                status_code=500,
            )

    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    response_data = {
        "move": move.uci(),
        "error": None,
        "time_taken": time_taken,
        "move_probs": move_probs_dict,
        "logs": logs,
    }

    if stockfish_move:
        response_data["stockfish_move"] = stockfish_move.uci()

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
