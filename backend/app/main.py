"""
MoPR Hackathon - Rural Feature Detection API
Handles TIFF upload → SegFormer inference → returns segmented output
"""

import os
import uuid
import shutil
import logging
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# =========================
# CONFIG
# =========================
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# LOAD MODEL (ON STARTUP)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting MoPR Detection Server...")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("📁 Upload dir: %s", UPLOAD_DIR)
    yield

    logger.info("🛑 Shutting down...")


# =========================
# FASTAPI INIT
# =========================
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

# Serve outputs
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# =========================
# HELPER FUNCTIONS
# =========================
def read_tiff(file_path: str):
    """Read TIFF image safely"""
    try:
        image = Image.open(file_path).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid TIFF file: {str(e)}")


def run_inference(image: Image.Image):
    """Run SegFormer model"""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # [1, num_classes, H, W]

    # Resize logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    return pred_mask


def save_mask(mask: np.ndarray, filename: str):
    """Save segmentation mask as colored image"""
    # Simple color map
    color_map = np.array([
        [0, 0, 0],        # background
        [255, 0, 0],      # class 1
        [0, 255, 0],      # class 2
        [0, 0, 255],      # class 3
        [255, 255, 0],    # class 4
    ])

    colored = color_map[mask % len(color_map)]

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, cv2.cvtColor(colored.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return output_path


# =========================
# ROUTES
# =========================

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/upload")
async def upload_tiff(file: UploadFile = File(...)):
    """Upload TIFF → Run segmentation → Return result"""

    if not file.filename.endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only TIFF files allowed")

    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.tif")

    # Save file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image
    image = read_tiff(input_path)

    # Run model
    mask = run_inference(image)

    # Save output
    output_filename = f"{file_id}.png"
    output_path = save_mask(mask, output_filename)

    return {
        "message": "Segmentation complete",
        "input_file": input_path,
        "output_url": f"/outputs/{output_filename}"
    }
