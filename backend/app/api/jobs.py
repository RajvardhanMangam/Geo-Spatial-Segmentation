"""Job status and GeoJSON export endpoints."""

import json
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.redis_client import redis_client

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Return full job state including progress and building/detection counts."""
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/detections")
async def get_detections(
    job_id: str,
    feature_type: str = Query(None, description="Filter by feature type, e.g. 'building'"),
):
    """Return all detections, optionally filtered by feature type."""
    detections = await redis_client.get_all_detections(job_id)
    if feature_type:
        detections = [d for d in detections if d.get("feature_type") == feature_type]
    return {"job_id": job_id, "count": len(detections), "detections": detections}


@router.get("/jobs/{job_id}/buildings")
async def get_buildings(job_id: str):
    """
    Return only building instance detections for a completed job.

    Each entry includes the standard detection fields plus:
      - ``building_id``    – globally unique ID (e.g. "B000042")
      - ``roof_features``  – [mean_R, mean_G, mean_B, texture, angle]

    Useful for downstream roof-type classification or inventory exports.
    """
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    all_detections = await redis_client.get_all_detections(job_id)
    buildings = [
        d for d in all_detections
        if d.get("base_feature_type", d.get("feature_type")) == "building"
    ]

    return {
        "job_id":          job_id,
        "buildings_found": len(buildings),
        "buildings":       buildings,
    }


@router.get("/jobs/{job_id}/geojson")
async def export_geojson(job_id: str):
    """
    Export all detections as a downloadable GeoJSON FeatureCollection.

    Building features include extended properties:
      ``building_id``, ``roof_features`` (array of 5 floats).
    """
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    detections = await redis_client.get_all_detections(job_id)

    features = []
    for det in detections:
        polygon = det.get("geo_polygon", [])
        if len(polygon) < 4:
            continue

        properties = {
            "feature_type": det.get("feature_type"),
            "display_label": det.get("display_label", det.get("feature_type")),
            "base_feature_type": det.get("base_feature_type", det.get("feature_type")),
            "subtype": det.get("subtype"),
            "confidence":   det.get("confidence"),
            "chunk_id":     det.get("chunk_id"),
            "colour":       det.get("colour"),
            "area_px":      det.get("area_px"),
            "crs":          det.get("crs"),
        }
        if det.get("classifier"):
            properties["classifier"] = det.get("classifier")
            properties["classifier_confidence"] = det.get("classifier_confidence")
        if det.get("source_feature_type"):
            properties["source_feature_type"] = det.get("source_feature_type")

        # Attach building-specific properties when present
        if det.get("base_feature_type", det.get("feature_type")) == "building":
            properties["building_id"]   = det.get("building_id")
            properties["roof_features"] = det.get("roof_features", [])

        features.append({
            "type":     "Feature",
            "geometry": {
                "type":        "Polygon",
                "coordinates": [polygon],
            },
            "properties": properties,
        })

    geojson = {
        "type": "FeatureCollection",
        "name": f"mopr_detections_{job_id[:8]}",
        "crs": {
            "type":       "name",
            "properties": {"name": job.get("metadata", {}).get("crs", "EPSG:4326")},
        },
        "features": features,
    }

    output_path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_detections.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))

    logger.info(
        "GeoJSON exported for job %s: %d features (%d buildings)",
        job_id,
        len(features),
        sum(
            1 for f in features
            if f["properties"].get("base_feature_type") == "building"
        ),
    )

    return FileResponse(
        output_path,
        media_type="application/geo+json",
        filename=f"mopr_detections_{job_id[:8]}.geojson",
    )


@router.get("/jobs/{job_id}/buildings/geojson")
async def export_buildings_geojson(job_id: str):
    """
    Export only building instances as a downloadable GeoJSON FeatureCollection.

    Produces a leaner file suitable for building inventory systems that do not
    need road or water geometries.
    """
    job = await redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    all_detections = await redis_client.get_all_detections(job_id)
    buildings = [
        d for d in all_detections
        if d.get("base_feature_type", d.get("feature_type")) == "building"
    ]

    features = []
    for det in buildings:
        polygon = det.get("geo_polygon", [])
        if len(polygon) < 4:
            continue
        features.append({
            "type":     "Feature",
            "geometry": {
                "type":        "Polygon",
                "coordinates": [polygon],
            },
            "properties": {
                "feature_type": det.get("feature_type"),
                "display_label": det.get("display_label", det.get("feature_type")),
                "base_feature_type": det.get("base_feature_type", det.get("feature_type")),
                "subtype": det.get("subtype"),
                "building_id":   det.get("building_id"),
                "confidence":    det.get("confidence"),
                "classifier":    det.get("classifier"),
                "classifier_confidence": det.get("classifier_confidence"),
                "chunk_id":      det.get("chunk_id"),
                "colour":        det.get("colour"),
                "area_px":       det.get("area_px"),
                "crs":           det.get("crs"),
                "roof_features": det.get("roof_features", []),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "name": f"mopr_buildings_{job_id[:8]}",
        "crs": {
            "type":       "name",
            "properties": {"name": job.get("metadata", {}).get("crs", "EPSG:4326")},
        },
        "features": features,
    }

    output_path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_buildings.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))

    return FileResponse(
        output_path,
        media_type="application/geo+json",
        filename=f"mopr_buildings_{job_id[:8]}.geojson",
    )
