"""
WebSocket API
=============
Real-time pipeline updates via WebSocket.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.pipeline.graph import run_pipeline

router = APIRouter(tags=["websocket"])

# Active WebSocket connections
_connections: list[WebSocket] = []


async def broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    data = json.dumps(message, default=str)
    disconnected = []
    for ws in _connections:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _connections.remove(ws)


@router.websocket("/ws/pipeline")
async def pipeline_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline updates."""
    await websocket.accept()
    _connections.append(websocket)

    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "data": {"message": "Invalid JSON"}})
                continue

            if command.get("action") == "start":
                config = command.get("config", {})
                await websocket.send_json({
                    "type": "status",
                    "data": {"message": "Pipeline starting...", "status": "starting"},
                })

                # Run pipeline and stream real-time log updates
                try:
                    state = await run_pipeline(
                        max_posts=config.get("max_posts", 10),
                        max_age_days=config.get("max_age_days", 7),
                        limit_per_source=config.get("limit_per_source", 15),
                        max_retries=config.get("max_retries", 3),
                        sources=config.get("sources"),
                        broadcast_fn=broadcast,
                    )

                    # Send final results
                    await websocket.send_json({
                        "type": "result",
                        "data": {
                            "run_id": state.get("run_id"),
                            "accepted": len(state.get("accepted_posts", [])),
                            "rejected": len(state.get("rejected_items", [])),
                            "logs": state.get("logs", []),
                            "status": "completed",
                        },
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": str(e), "status": "failed"},
                    })

            elif command.get("action") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        _connections.remove(websocket)
