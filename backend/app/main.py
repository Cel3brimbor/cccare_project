from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[2]
trainer = Trainer(ROOT)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    trainer.start()
    yield
    trainer.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "metal": trainer.metal, "phase": trainer.mode}


@app.get("/api/rows")
def rows(split: str = "train", offset: int = 0, limit: int = 12) -> dict:
    if not trainer.data_ready:
        raise HTTPException(status_code=409, detail="The dataset is still being prepared.")
    return trainer.page(split, offset, limit)


@app.websocket("/ws")
async def studio_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    seen: dict[str, int] = {}
    cursor = trainer.bus.tip()
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
            except asyncio.TimeoutError:
                raw = None
            if raw:
                try:
                    trainer.submit(json.loads(raw))
                except json.JSONDecodeError:
                    pass
            messages, cursor = trainer.bus.take(seen, cursor)
            for message in messages:
                await websocket.send_text(json.dumps(message, separators=(",", ":")))
    except WebSocketDisconnect:
        return
