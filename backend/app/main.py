"""
MoPR Hackathon - Rural Feature Detection API
Detects buildings, roads, and utilities from drone orthophotos
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import upload, inference, websocket_handler, jobs
from app.core.config import settings
from app.core.redis_client import redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting MoPR Detection Server...")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    await redis_client.connect()
    logger.info("Server ready. Upload dir: %s", settings.UPLOAD_DIR)
    yield
    logger.info("Shutting down...")
    await redis_client.disconnect()


app = FastAPI(
    title="MoPR Rural Feature Detector",
    description="AI-powered detection of buildings, roads, and utilities from drone orthophotos",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(websocket_handler.router, tags=["websocket"])

# Serve uploaded/output files
app.mount("/files", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
