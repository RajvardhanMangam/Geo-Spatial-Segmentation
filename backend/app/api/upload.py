"""
Chunked upload endpoint for .tif files up to 6GB.

Strategy:
- Client sends file in chunks via multipart/form-data
- Each chunk is appended to a temp file
- On completion, metadata is extracted and job is created
"""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

import aiofiles
import rasterio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.redis_client import redis_client

router = APIRouter()
logger = logging.getLogger(__name__)

CHUNK_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB read buffer


@router.post("/upload/init")
async def init_upload(
    filename: str = Form(...),
    total_size: int = Form(...),
    total_chunks: int = Form(...),
):
    """Initialize a multipart upload session."""
    if total_size > settings.MAX_UPLOAD_SIZE_GB * 1024 ** 3:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_SIZE_GB}GB limit")

    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF (.tif/.tiff) files are supported")

    upload_id = str(uuid.uuid4())
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{upload_id}.tif")

    session = {
        "upload_id": upload_id,
        "filename": filename,
        "total_size": total_size,
        "total_chunks": total_chunks,
        "chunks_received": 0,
        "dest_path": dest_path,
        "status": "uploading",
    }
    await redis_client.set_job(f"upload:{upload_id}", session, ttl=3600)

    # Pre-create file
    async with aiofiles.open(dest_path, "wb") as f:
        pass

    logger.info("Upload session %s initialized for %s (%d bytes)", upload_id, filename, total_size)
    return {"upload_id": upload_id}


@router.post("/upload/chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
):
    """Receive one chunk and append it to the destination file."""
    session = await redis_client.get_job(f"upload:{upload_id}")
    if not session:
        raise HTTPException(404, "Upload session not found or expired")
    if session["status"] != "uploading":
        raise HTTPException(400, f"Upload already in state: {session['status']}")

    dest_path = session["dest_path"]
    data = await chunk.read()

    async with aiofiles.open(dest_path, "r+b") as f:
        # Each chunk is exactly CHUNK_SIZE_BYTES except possibly the last
        offset = chunk_index * CHUNK_SIZE_BYTES
        await f.seek(offset)
        await f.write(data)

    session["chunks_received"] += 1
    await redis_client.set_job(f"upload:{upload_id}", session)

    return {
        "upload_id": upload_id,
        "chunks_received": session["chunks_received"],
        "total_chunks": session["total_chunks"],
    }


@router.post("/upload/complete")
async def complete_upload(upload_id: str = Form(...)):
    """
    Finalize upload: validate GeoTIFF, extract metadata, return image info.
    """
    session = await redis_client.get_job(f"upload:{upload_id}")
    if not session:
        raise HTTPException(404, "Upload session not found")

    dest_path = session["dest_path"]

    if not os.path.exists(dest_path):
        raise HTTPException(500, "File not found after upload")

    # Validate and extract geospatial metadata
    try:
        with rasterio.open(dest_path) as src:
            bounds = src.bounds
            crs = src.crs.to_string() if src.crs else "Unknown"
            width = src.width
            height = src.height
            bands = src.count
            dtype = str(src.dtypes[0])
            transform = list(src.transform)[:6]

            # Calculate chunk grid
            chunk_cols = (width + settings.CHUNK_SIZE - 1) // settings.CHUNK_SIZE
            chunk_rows = (height + settings.CHUNK_SIZE - 1) // settings.CHUNK_SIZE
            total_chunks = chunk_cols * chunk_rows

            metadata = {
                "crs": crs,
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "width": width,
                "height": height,
                "bands": bands,
                "dtype": dtype,
                "transform": transform,
                "chunk_cols": chunk_cols,
                "chunk_rows": chunk_rows,
                "total_inference_chunks": total_chunks,
            }

    except Exception as e:
        raise HTTPException(422, f"Invalid GeoTIFF: {e}")

    session.update({"status": "ready", "metadata": metadata})
    await redis_client.set_job(f"upload:{upload_id}", session)

    logger.info(
        "Upload %s complete: %dx%d px, %d bands, CRS=%s, %d chunks to process",
        upload_id, width, height, bands, crs, total_chunks,
    )

    return {
        "upload_id": upload_id,
        "metadata": metadata,
        "message": "Upload complete. Ready for inference.",
    }


@router.get("/upload/{upload_id}/status")
async def upload_status(upload_id: str):
    session = await redis_client.get_job(f"upload:{upload_id}")
    if not session:
        raise HTTPException(404, "Upload session not found")
    return session
