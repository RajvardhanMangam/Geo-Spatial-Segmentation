"""
Mask2Former inference service.

Loads a Mask2Former segmentation model (or a fine-tuned variant)
and runs inference on 1024×1024 4-channel patches.

For the hackathon, we default to the pretrained ADE20K weights.
Swap MODEL_NAME in config for your fine-tuned MoPR checkpoint.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

from app.core.config import settings
from app.services.chunker import ImageChunk

logger = logging.getLogger(__name__)

# Colour map for visualisation (BGR → RGB already handled by frontend)
FEATURE_COLOURS = {
    "building": "#FF4444",
    "road":     "#4488FF",
    "utility":  "#FFAA00",
    "vegetation": "#44BB44",
    "water":    "#00BBFF",
    "unknown":  "#888888",
}


class DetectionModel:
    """Singleton wrapper around Mask2Former."""

    _instance: Optional["DetectionModel"] = None

    def __init__(self):
        self.device = torch.device(settings.MODEL_DEVICE)
        self.processor: Optional[AutoImageProcessor] = None
        self.model: Optional[Mask2FormerForUniversalSegmentation] = None
        self._loaded = False

    @classmethod
    def get(cls) -> "DetectionModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Load weights — called once at startup or on first inference."""
        if self._loaded:
            return
        logger.info("Loading Mask2Former from %s …", settings.MODEL_NAME)
        cache = settings.MODEL_CACHE_DIR
        os.makedirs(cache, exist_ok=True)

        self.processor = AutoImageProcessor.from_pretrained(
            settings.MODEL_NAME, cache_dir=cache
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            settings.MODEL_NAME, cache_dir=cache
        )

        # Patch the first conv layer to accept 4 channels instead of 3
        self._patch_input_channels()

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded on %s", self.device)

    def _patch_input_channels(self):
        """
        Replace the first conv layer with a 4-channel version.
        Copies the pretrained RGB weights and initialises the 4th channel
        as the average of the first three (a reasonable NIR initialisation).
        """
        try:
            backbone = self.model.model.pixel_level_module.encoder
            first_conv = None
            first_conv_name = None

            # Find the first Conv2d
            for name, module in backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
                    first_conv = module
                    first_conv_name = name
                    break

            if first_conv is None:
                logger.warning("Could not find 3-channel conv — using 3-band input only")
                return

            new_conv = torch.nn.Conv2d(
                4,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = first_conv.weight.clone()
                new_conv.weight[:, 3:] = first_conv.weight[:, :1].clone()
                if first_conv.bias is not None:
                    new_conv.bias = first_conv.bias.clone()

            # Replace the layer
            parts = first_conv_name.split(".")
            parent = backbone
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            logger.info("Patched input conv to 4 channels")

        except Exception as e:
            logger.warning("Could not patch input channels: %s", e)

    @torch.no_grad()
    def infer_chunk(self, chunk: ImageChunk) -> List[dict]:
        """
        Run Mask2Former on one chunk.

        Returns a list of GeoJSON-compatible feature dicts, each with:
          - type: feature class name
          - confidence: float
          - mask_pixels: list of [col, row] pixel coords in the chunk space
          - geo_polygon: list of [lon, lat] for the bounding box (approx)
        """
        if not self._loaded:
            self.load()

        pixels = chunk.pixels  # (4, H, W) float32
        H, W = pixels.shape[1], pixels.shape[2]

        # Processor expects PIL (3-channel) or numpy. We'll pass numpy directly.
        # Use first 3 bands for the HuggingFace processor then inject 4th ourselves.
        rgb = (pixels[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # If model accepts 4 channels, inject the 4th band manually
        if "pixel_values" in inputs and inputs["pixel_values"].shape[1] == 3:
            nir_band = torch.from_numpy(pixels[3:]).unsqueeze(0).to(self.device)
            # Normalize NIR the same way HF processor normalises RGB
            nir_band = (nir_band - nir_band.mean()) / (nir_band.std() + 1e-6)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], nir_band], dim=1
            )

        outputs = self.model(**inputs)

        # Post-process to semantic segmentation
        pred = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(H, W)]
        )[0]  # (H, W) int64 label tensor

        pred_np = pred.cpu().numpy()

        detections = []
        for feature_name, class_ids in settings.FEATURE_CLASS_MAP.items():
            mask = np.isin(pred_np, class_ids).astype(np.uint8)
            if mask.sum() == 0:
                continue

            # Simple connected component bounding boxes
            import cv2
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 100:  # skip tiny noise blobs
                    continue

                x, y, w, h = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT],
                )

                # Convert pixel bbox corners to geo coordinates
                geo_poly = _bbox_to_geo_polygon(
                    x, y, x + w, y + h, chunk.geo_transform
                )

                confidence = float(area) / (W * H)  # crude area-based confidence

                detections.append({
                    "feature_type": feature_name,
                    "confidence": round(min(confidence * 20, 0.99), 4),
                    "chunk_id": chunk.chunk_id,
                    "pixel_bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "geo_polygon": geo_poly,
                    "crs": chunk.crs,
                    "area_px": int(area),
                    "colour": FEATURE_COLOURS.get(feature_name, "#888888"),
                })

        return detections


def _bbox_to_geo_polygon(
    px1: int, py1: int, px2: int, py2: int, geo_transform: list
) -> list:
    """
    Convert pixel bounding box to a geographic polygon using the affine transform.
    geo_transform = [x_origin, x_pixel_res, row_rot, y_origin, col_rot, y_pixel_res]
    """
    def px_to_geo(col, row):
        x = geo_transform[0] + col * geo_transform[1] + row * geo_transform[2]
        y = geo_transform[3] + col * geo_transform[4] + row * geo_transform[5]
        return [round(x, 8), round(y, 8)]

    return [
        px_to_geo(px1, py1),
        px_to_geo(px2, py1),
        px_to_geo(px2, py2),
        px_to_geo(px1, py2),
        px_to_geo(px1, py1),  # close ring
    ]


# Module-level singleton
detection_model = DetectionModel.get()
