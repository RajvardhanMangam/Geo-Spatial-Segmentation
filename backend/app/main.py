"""MoPR rural feature detection API."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import inference, jobs, upload, websocket_handler
from app.core.config import settings
from app.core.redis_client import redis_client
from app.services.model_service import classifier_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    await redis_client.connect()
    logger.info("MoPR Detection API started")

    try:
        yield
    finally:
        await redis_client.disconnect()
        logger.info("MoPR Detection API stopped")


app = FastAPI(
    title="MoPR Rural Feature Detector",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(websocket_handler.router)

app.mount(
    "/outputs",
    StaticFiles(directory=settings.OUTPUT_DIR, check_dir=False),
    name="outputs",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.MODEL_NAME,
        "runtime": "onnxruntime",
        "classifiers": classifier_status(),
    }
