"""
WebSocket endpoint that streams inference results to the frontend.

Flow:
  1. Client connects to /ws/{job_id}
  2. Server subscribes to Redis channel job:{job_id}
  3. Each Redis pub/sub message is forwarded to the client as JSON
  4. Connection closes when job is completed or client disconnects
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.redis_client import redis_client

router = APIRouter()
logger = logging.getLogger(__name__)

PING_INTERVAL = 15  # seconds


@router.websocket("/ws/{job_id}")
async def job_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    logger.info("WebSocket client connected for job %s", job_id)

    # Immediately send current job state
    job = await redis_client.get_job(job_id)
    if job:
        await websocket.send_json({"type": "job_state", **job})
    else:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    # If job already done, send completion metadata; frontend fetches detections.
    if job.get("status") in ("completed", "failed"):
        detections = await redis_client.get_all_detections(job_id)
        await websocket.send_json({
            "type": "completed",
            "job_id": job_id,
            "total_detections": len(detections),
        })
        await websocket.close()
        return

    # On reconnect during a running job, send the detections accumulated so far.
    # Redis pub/sub only delivers future messages, so without this snapshot the
    # UI can show fresh progress while polygons remain stuck at the last chunk
    # it saw before the socket dropped.
    existing_detections = await redis_client.get_all_detections(job_id)
    if existing_detections:
        await websocket.send_json({
            "type": "detections_snapshot",
            "job_id": job_id,
            "count": len(existing_detections),
            "detections": existing_detections,
        })

    # Subscribe to Redis pub/sub for live updates
    pubsub = await redis_client.subscribe(f"job:{job_id}")

    async def ping_task():
        """Keep connection alive."""
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    ping = asyncio.create_task(ping_task())

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            data = json.loads(message["data"])
            await websocket.send_json(data)

            if data.get("type") in ("completed", "error"):
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from job %s", job_id)
    except Exception as e:
        logger.error("WebSocket error for job %s: %s", job_id, e)
    finally:
        ping.cancel()
        await pubsub.unsubscribe(f"job:{job_id}")
        logger.info("WebSocket closed for job %s", job_id)
