# Geo Spatial Segmentation
**Hackathon: MoPR Problem Statement 1**  
AI-powered detection of buildings, roads, and utilities from high-resolution drone orthophotos.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (React + Leaflet)                                      │
│  ┌──────────────┐  ┌────────────────────────────────────────┐   │
│  │ Upload Panel │  │  DetectionMap (Leaflet + proj4)        │   │
│  │ 8MB chunks   │  │  Real-time GeoJSON polygon overlay     │   │
│  └──────┬───────┘  └────────────────────────────────────────┘   │
│         │ HTTP multipart              ▲ WebSocket JSON           │
└─────────┼─────────────────────────────┼───────────────────────── ┘
          │                             │
┌─────────▼─────────────────────────────┼─────────────────────────┐
│  FastAPI Backend                      │                          │
│  ┌──────────────┐  ┌──────────────┐  │                          │
│  │ /upload/*    │  │ /ws/{job_id} │──┘                          │
│  │ Chunked TIF  │  │ Redis pub/sub│                             │
│  └──────┬───────┘  └──────────────┘                             │
│         │ Background Task                                        │
│  ┌──────▼────────────────────────────────────────────────┐      │
│  │ Inference Pipeline                                    │      │
│  │  rasterio.Window → 1024px patches → Mask2Former      │      │
│  │  → Connected components → GeoJSON → Redis publish    │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
          │
    ┌─────▼──────┐
    │   Redis    │  Job state + pub/sub + detection list
    └────────────┘
```

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- (Optional) NVIDIA GPU + nvidia-container-toolkit for faster inference

### 1. Clone and run
```bash
git clone <repo>
cd Geo-Spatial-Segmentation
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Redis: localhost:6379

### 2. Without Docker (development)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
# Start Redis separately
redis-server

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install --legacy-peer-deps
REACT_APP_API_URL=http://localhost:8000 \
REACT_APP_WS_URL=ws://localhost:8000 \
npm start
```

---

## Configuration

Edit `backend/.env` (or set environment variables):

| Variable | Default | Description |
|---|---|---|
| `UPLOAD_DIR` | `/tmp/mopr_uploads` | Where .tif files are stored |
| `OUTPUT_DIR` | `/tmp/mopr_outputs` | GeoJSON exports |
| `CHUNK_SIZE` | `1024` | Pixels per inference chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `MODEL_NAME` | `facebook/mask2former-swin-large-ade-semantic` | HuggingFace model |
| `MODEL_DEVICE` | `cpu` | `cpu` or `cuda` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |

---

## Swapping the Model

The `MODEL_NAME` env var accepts any HuggingFace Mask2Former checkpoint.

For the MoPR hackathon, fine-tune on your labeled orthophoto dataset and upload to HuggingFace Hub, then:
```bash
MODEL_NAME=your-org/mopr-rural-detector-v1 docker compose up
```

The pipeline automatically:
- Patches the first conv layer to accept 4 bands (RGB + NIR)
- Maps ADE20K class IDs to `building / road / utility / vegetation / water`
- Adjust `FEATURE_CLASS_MAP` in `config.py` for your fine-tuned label set

---

## API Reference

### Upload
```
POST /api/v1/upload/init        — Start session
POST /api/v1/upload/chunk       — Send 8MB chunk
POST /api/v1/upload/complete    — Finalize + extract metadata
GET  /api/v1/upload/{id}/status — Check upload
```

### Inference
```
POST /api/v1/inference/start    — Start job, returns job_id
GET  /api/v1/jobs/{id}          — Job status + progress
GET  /api/v1/jobs/{id}/detections — All detections (JSON)
GET  /api/v1/jobs/{id}/geojson  — Export as GeoJSON file
```

### WebSocket
```
WS /ws/{job_id}
```
Messages:
- `job_state` — current state on connect
- `chunk_done` — per-chunk progress + new detections
- `completed` — inference finished
- `error` — failure details

---

## Memory Safety

The pipeline never loads the full 6GB TIF into RAM. Each `rasterio.Window` read loads exactly one 1024×1024 patch (≈ 4MB for 4-band float32), processes it, then discards it before the next chunk.

Peak RAM usage ≈ `BATCH_SIZE × 4 × 1024 × 1024 × 4 bytes` ≈ **64 MB per batch**.

---

## Coordinate Precision

Detections are re-projected using the source image's affine transform and CRS:
- `EPSG:32644` (UTM Zone 44N) — most drone flights over central India
- `EPSG:3857` (Web Mercator)
- `EPSG:4326` (WGS84 — pass-through)

The frontend uses `proj4` to convert all coordinates to WGS84 before rendering on Leaflet, ensuring pixel-perfect alignment with the satellite basemap.

---

## Hackathon Checklist

- [x] Handles GeoTIFF up to 6GB (chunked upload, no memory overload)
- [x] Windowed rasterio reads (1024×1024 patches)
- [x] 4-channel Mask2Former (RGB + NIR band)
- [x] Async inference with progress tracking
- [x] Real-time WebSocket streaming to frontend
- [x] Leaflet map with EPSG:32644/3857 support
- [x] Per-feature detection overlays (building/road/utility)
- [x] Progress bar "% of village processed"
- [x] GeoJSON export
- [x] Docker Compose for one-command deploy
