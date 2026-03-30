"""Job status and GeoJSON export endpoints."""

import json
import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.redis_client import redis_client
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/detections")
async def get_detections(job_id: str, feature_type: str = None):
    """Return all detections (optionally filtered by feature type)."""
    detections = await redis_client.get_all_detections(job_id)
    if feature_type:
        detections = [d for d in detections if d.get("feature_type") == feature_type]
    return {"job_id": job_id, "count": len(detections), "detections": detections}


@router.get("/jobs/{job_id}/geojson")
async def export_geojson(job_id: str):
    """Export all detections as a downloadable GeoJSON FeatureCollection."""
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    detections = await redis_client.get_all_detections(job_id)

    features = []
    for det in detections:
        polygon = det.get("geo_polygon", [])
        if len(polygon) < 4:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon],
            },
            "properties": {
                "feature_type": det.get("feature_type"),
                "confidence": det.get("confidence"),
                "chunk_id": det.get("chunk_id"),
                "colour": det.get("colour"),
                "area_px": det.get("area_px"),
                "crs": det.get("crs"),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "name": f"mopr_detections_{job_id[:8]}",
        "crs": {
            "type": "name",
            "properties": {"name": job.get("metadata", {}).get("crs", "EPSG:4326")},
        },
        "features": features,
    }

    # Write to temp file
    output_path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_detections.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))

    return FileResponse(
        output_path,
        media_type="application/geo+json",
        filename=f"mopr_detections_{job_id[:8]}.geojson",
    )
