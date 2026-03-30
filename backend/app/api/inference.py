"""
Async inference pipeline.

Starts a job that:
1. Iterates GeoTIFF chunks (rasterio windowed read)
2. Runs Mask2Former on each chunk
3. Publishes results to Redis pub/sub channel (consumed by WebSocket handler)
4. Updates job progress in Redis

Uses FastAPI BackgroundTasks (no Celery needed for hackathon scale).
For production: swap with a Celery worker.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.redis_client import redis_client
from app.services.chunker import iter_chunks
from app.services.model_service import detection_model

router = APIRouter()
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound model inference
_executor = ThreadPoolExecutor(max_workers=2)


class InferenceRequest(BaseModel):
    upload_id: str
    confidence_threshold: float = 0.10


@router.post("/inference/start")
async def start_inference(req: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Kick off inference on an uploaded GeoTIFF.
    Returns a job_id immediately; results stream via WebSocket /ws/{job_id}.
    """
    upload = await redis_client.get_job(f"upload:{req.upload_id}")
    if not upload:
        raise HTTPException(404, "Upload not found")
    if upload["status"] != "ready":
        raise HTTPException(400, f"Upload not ready (status: {upload['status']})")

    job_id = str(uuid.uuid4())
    meta = upload["metadata"]

    job = {
        "job_id": job_id,
        "upload_id": req.upload_id,
        "tif_path": upload["dest_path"],
        "status": "queued",
        "total_chunks": meta["total_inference_chunks"],
        "chunks_done": 0,
        "detections_found": 0,
        "confidence_threshold": req.confidence_threshold,
        "metadata": meta,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    await redis_client.set_job(job_id, job)
    background_tasks.add_task(_run_inference, job_id)

    logger.info("Job %s queued for upload %s (%d chunks)", job_id, req.upload_id, meta["total_inference_chunks"])
    return {"job_id": job_id, "total_chunks": meta["total_inference_chunks"]}


async def _run_inference(job_id: str):
    """Background task: iterate chunks, infer, publish results."""
    job = await redis_client.get_job(job_id)
    if not job:
        return

    tif_path = job["tif_path"]
    threshold = job["confidence_threshold"]
    total = job["total_chunks"]
    loop = asyncio.get_event_loop()

    await redis_client.update_job(job_id, {"status": "running", "updated_at": time.time()})
    await redis_client.publish(f"job:{job_id}", {"type": "started", "job_id": job_id, "total_chunks": total})

    try:
        chunks_done = 0

        # Load model (no-op if already loaded)
        await loop.run_in_executor(_executor, detection_model.load)

        for chunk in iter_chunks(tif_path):
            # Run inference in thread pool (CPU/GPU bound)
            detections = await loop.run_in_executor(
                _executor, detection_model.infer_chunk, chunk
            )

            # Filter by threshold
            detections = [d for d in detections if d["confidence"] >= threshold]

            chunks_done += 1
            progress = round(chunks_done / total * 100, 2)

            # Persist detections
            for det in detections:
                await redis_client.append_detection(job_id, det)

            # Publish progress + new detections to WebSocket subscribers
            await redis_client.publish(f"job:{job_id}", {
                "type": "chunk_done",
                "job_id": job_id,
                "chunk_id": chunk.chunk_id,
                "chunks_done": chunks_done,
                "total_chunks": total,
                "progress": progress,
                "detections": detections,
            })

            await redis_client.update_job(job_id, {
                "chunks_done": chunks_done,
                "detections_found": (job.get("detections_found", 0) + len(detections)),
                "progress": progress,
                "updated_at": time.time(),
            })
            job = await redis_client.get_job(job_id)

            # Yield control back to event loop between chunks
            await asyncio.sleep(0)

        # Done
        all_detections = await redis_client.get_all_detections(job_id)
        await redis_client.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "detections_found": len(all_detections),
            "updated_at": time.time(),
        })
        await redis_client.publish(f"job:{job_id}", {
            "type": "completed",
            "job_id": job_id,
            "total_detections": len(all_detections),
        })
        logger.info("Job %s completed: %d detections", job_id, len(all_detections))

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        await redis_client.update_job(job_id, {
            "status": "failed",
            "error": str(e),
            "updated_at": time.time(),
        })
        await redis_client.publish(f"job:{job_id}", {
            "type": "error",
            "job_id": job_id,
            "message": str(e),
        })
