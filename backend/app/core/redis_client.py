"""Async Redis client for job tracking and WebSocket pub/sub."""

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Connected to Redis")

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("Redis not connected")
        return self._client

    # ── Job state ──────────────────────────────────────────────────────────
    async def set_job(self, job_id: str, data: dict, ttl: int = 86400):
        await self.client.setex(f"job:{job_id}", ttl, json.dumps(data))

    async def get_job(self, job_id: str) -> Optional[dict]:
        raw = await self.client.get(f"job:{job_id}")
        return json.loads(raw) if raw else None

    async def update_job(self, job_id: str, updates: dict):
        job = await self.get_job(job_id) or {}
        job.update(updates)
        await self.set_job(job_id, job)

    # ── Pub/Sub for WebSocket streaming ───────────────────────────────────
    async def publish(self, channel: str, message: dict):
        await self.client.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str):
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub

    # ── Detection accumulation ────────────────────────────────────────────
    async def append_detection(self, job_id: str, feature: dict):
        await self.client.rpush(f"detections:{job_id}", json.dumps(feature))

    async def get_all_detections(self, job_id: str) -> list:
        raw_list = await self.client.lrange(f"detections:{job_id}", 0, -1)
        return [json.loads(r) for r in raw_list]


redis_client = RedisClient()
