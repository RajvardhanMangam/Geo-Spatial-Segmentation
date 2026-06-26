"""
On-demand Road Enhancement API.

Exposes:
  POST /api/v1/jobs/{job_id}/enhance-roads  — start user-triggered enhancement
  GET  /ws/{job_id}/enhance                 — WebSocket for live step progress

The enhancement pipeline runs road_network_postprocess.enhance_road_network()
on the current merged detections and streams progress steps back to the client.
Buildings and water detections are not touched.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from app.core.redis_client import redis_client
from app.services.road_network_postprocess import ENHANCEMENT_STEPS, enhance_road_network

router    = APIRouter()    # mounted under /api/v1  (HTTP endpoints)
ws_router = APIRouter()   # mounted without prefix (WebSocket endpoint)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


# ── HTTP endpoint ─────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/enhance-roads")
async def start_road_enhancement(job_id: str, background_tasks: BackgroundTasks):
    """
    Trigger on-demand road enhancement for a completed job.

    Returns immediately; progress streams via /ws/{job_id}/enhance.
    """
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "completed":
        raise HTTPException(400, f"Job not completed (status: {job.get('status')})")
    if job.get("enhancement_status") == "running":
        raise HTTPException(409, "Enhancement already in progress")

    await redis_client.update_job(job_id, {
        "enhancement_status": "running",
        "enhancement_started_at": time.time(),
    })
    background_tasks.add_task(_run_road_enhancement, job_id)

    logger.info("Road enhancement queued for job %s", job_id)
    return {"status": "started", "job_id": job_id, "steps": ENHANCEMENT_STEPS}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@ws_router.websocket("/ws/{job_id}/enhance")
async def enhancement_stream(websocket: WebSocket, job_id: str):
    """Stream road enhancement progress steps to the client."""
    await websocket.accept()

    job = await redis_client.get_job(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    # If already completed, replay the result immediately
    if job.get("enhancement_status") == "completed":
        roads = await redis_client.get_enhanced_roads(job_id)
        await websocket.send_json({
            "type": "enhance_complete",
            "job_id": job_id,
            "roads": roads,
            "count": len(roads),
            "steps": ENHANCEMENT_STEPS,
        })
        await websocket.close()
        return

    channel = f"enhance:{job_id}"
    pubsub = await redis_client.subscribe(channel)

    async def _ping():
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    ping_task = asyncio.create_task(_ping())

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            await websocket.send_json(data)
            if data.get("type") in ("enhance_complete", "enhance_error"):
                break
    except WebSocketDisconnect:
        logger.info("Enhancement WebSocket disconnected for job %s", job_id)
    except Exception as e:
        logger.error("Enhancement WebSocket error for job %s: %s", job_id, e)
    finally:
        ping_task.cancel()
        await pubsub.unsubscribe(channel)


# ── Background task ───────────────────────────────────────────────────────────

async def _run_road_enhancement(job_id: str):
    """Run the full road enhancement pipeline and publish step-by-step progress."""
    job = await redis_client.get_job(job_id)
    if not job:
        return

    tif_path = job.get("tif_path", "")
    channel = f"enhance:{job_id}"
    loop = asyncio.get_event_loop()

    # Delay to let the WebSocket client connect before first message
    await asyncio.sleep(0.4)

    await redis_client.publish(channel, {
        "type": "enhance_started",
        "job_id": job_id,
        "total_steps": len(ENHANCEMENT_STEPS),
        "steps": ENHANCEMENT_STEPS,
    })

    try:
        detections = await redis_client.get_all_detections(job_id)

        def _progress(step_name: str, step_index: int, total_steps: int):
            """Thread-safe progress publisher — called from executor thread."""
            asyncio.run_coroutine_threadsafe(
                redis_client.publish(channel, {
                    "type": "enhance_step",
                    "job_id": job_id,
                    "step": step_name,
                    "step_index": step_index,
                    "total_steps": total_steps,
                }),
                loop,
            )

        enhanced_all = await loop.run_in_executor(
            _executor,
            lambda: enhance_road_network(detections, tif_path, progress_callback=_progress),
        )

        enhanced_roads = [d for d in enhanced_all if d.get("feature_type") == "road"]

        await redis_client.set_enhanced_roads(job_id, enhanced_roads)
        await redis_client.update_job(job_id, {
            "enhancement_status": "completed",
            "enhancement_road_count": len(enhanced_roads),
            "enhancement_completed_at": time.time(),
        })

        logger.info(
            "Road enhancement complete for job %s: %d road polygons",
            job_id, len(enhanced_roads),
        )

        await redis_client.publish(channel, {
            "type": "enhance_complete",
            "job_id": job_id,
            "roads": enhanced_roads,
            "count": len(enhanced_roads),
            "steps": ENHANCEMENT_STEPS,
        })

    except Exception as e:
        logger.exception("Road enhancement failed for job %s: %s", job_id, e)
        await redis_client.update_job(job_id, {
            "enhancement_status": "failed",
            "enhancement_error": str(e),
        })
        await redis_client.publish(channel, {
            "type": "enhance_error",
            "job_id": job_id,
            "message": str(e),
        })
