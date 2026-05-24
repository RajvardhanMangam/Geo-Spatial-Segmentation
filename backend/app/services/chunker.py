"""
Geospatial windowed reading using rasterio.

Slices large GeoTIFFs into 1024x1024 patches while preserving
the affine transform for each window so detections can be
re-projected back to geographic coordinates.
"""

import logging
from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ImageChunk:
    """A single 1024x1024 (or smaller at edges) image patch."""
    chunk_id: str          # "{row}_{col}"
    row: int               # chunk row index
    col: int               # chunk column index
    window: Window         # rasterio Window for this chunk
    pixels: np.ndarray     # shape (C, H, W), float32 0-1
    geo_transform: list    # 6-element affine for this patch's origin
    crs: str               # source CRS string
    image_width: int       # full image width
    image_height: int      # full image height
    total_chunks: int      # total chunks for progress reporting


def pixel_to_geo(col_off: int, row_off: int, transform: list) -> Tuple[float, float]:
    """Convert pixel offset to geographic coordinate using affine transform."""
    # transform = [x_origin, x_res, 0, y_origin, 0, y_res]  (GDAL order)
    x = transform[0] + col_off * transform[1] + row_off * transform[2]
    y = transform[3] + col_off * transform[4] + row_off * transform[5]
    return x, y


def chunk_window_transform(src_transform, window: Window) -> list:
    """Compute the affine transform for a sub-window."""
    # Use rasterio's window_transform for accuracy
    win_transform = rasterio.transform.array_bounds(
        window.height, window.width,
        rasterio.windows.transform(window, src_transform)
    )
    t = rasterio.windows.transform(window, src_transform)
    return [t.c, t.a, t.b, t.f, t.d, t.e]


def iter_chunks(tif_path: str) -> Generator[ImageChunk, None, None]:
    """
    Yield ImageChunk objects for every 1024x1024 tile in the GeoTIFF.
    Handles:
    - Edge tiles smaller than 1024
    - Overlap padding to reduce edge artifacts
    - RGB preprocessing to float32 [0, 1]
    """
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        transform = src.transform

        # Number of tiles in each direction
        cols = (width + chunk_size - 1) // chunk_size
        rows = (height + chunk_size - 1) // chunk_size
        total = rows * cols

        logger.info(
            "Chunking %s: %dx%d pixels → %d cols × %d rows = %d chunks",
            tif_path, width, height, cols, rows, total
        )

        for row_idx in range(rows):
            for col_idx in range(cols):
                # Core window
                col_off = col_idx * chunk_size
                row_off = row_idx * chunk_size
                win_w = min(chunk_size, width - col_off)
                win_h = min(chunk_size, height - row_off)

                # Expand with overlap (clamped to image bounds)
                read_col = max(0, col_off - overlap)
                read_row = max(0, row_off - overlap)
                read_w = min(chunk_size + 2 * overlap, width - read_col)
                read_h = min(chunk_size + 2 * overlap, height - read_row)

                window = Window(read_col, read_row, read_w, read_h)

                try:
                    # Read all bands, then keep the RGB channels expected by SegFormer.
                    data = src.read(window=window)  # (C, H, W)
                    data = preprocess_tiff_window(data, src)

                    geo_transform = chunk_window_transform(transform, window)

                    yield ImageChunk(
                        chunk_id=f"{row_idx}_{col_idx}",
                        row=row_idx,
                        col=col_idx,
                        window=window,
                        pixels=data,
                        geo_transform=geo_transform,
                        crs=crs,
                        image_width=width,
                        image_height=height,
                        total_chunks=total,
                    )

                except Exception as e:
                    logger.error("Error reading chunk %d_%d: %s", row_idx, col_idx, e)
                    continue


def preprocess_tiff_window(data: np.ndarray, src) -> np.ndarray:
    """
    Convert a rasterio window read into model-ready RGB float32 data.

    The default percentile mode gives drone orthophotos a stable contrast stretch
    without letting a few very bright or dark pixels dominate the tile.
    """
    if data.shape[0] >= 3:
        data = data[:3]
    else:
        pad = np.zeros(
            (3 - data.shape[0], data.shape[1], data.shape[2]),
            dtype=data.dtype,
        )
        data = np.concatenate([data, pad], axis=0)

    dtypes = list(getattr(src, "dtypes", []))
    nodatavals = list(getattr(src, "nodatavals", []) or [])
    mode = settings.TIFF_PREPROCESS_MODE.lower()

    output = np.zeros(data.shape, dtype=np.float32)
    for band_idx in range(data.shape[0]):
        dtype = dtypes[band_idx] if band_idx < len(dtypes) else data.dtype
        nodata = nodatavals[band_idx] if band_idx < len(nodatavals) else None
        output[band_idx] = _normalise_band(data[band_idx], dtype, nodata, mode)

    return np.clip(output, 0.0, 1.0)


def _normalise_band(
    band: np.ndarray,
    dtype,
    nodata,
    mode: str,
) -> np.ndarray:
    band = band.astype(np.float32, copy=False)
    valid = np.isfinite(band)
    if nodata is not None:
        valid &= band != float(nodata)

    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)

    if mode == "dtype":
        return _scale_band_by_dtype(band, valid, dtype)

    values = band[valid]
    if mode == "minmax":
        low = float(values.min())
        high = float(values.max())
    else:
        low = float(np.percentile(values, settings.TIFF_PERCENTILE_LOW))
        high = float(np.percentile(values, settings.TIFF_PERCENTILE_HIGH))

    if high <= low:
        return _scale_band_by_dtype(band, valid, dtype)

    normalised = (band - low) / (high - low)
    normalised[~valid] = 0.0
    return normalised.astype(np.float32, copy=False)


def _scale_band_by_dtype(band: np.ndarray, valid: np.ndarray, dtype) -> np.ndarray:
    values = band[valid]
    valid_max = float(values.max()) if values.size else 0.0
    np_dtype = np.dtype(dtype)

    if np.issubdtype(np_dtype, np.integer):
        if valid_max <= 1.0:
            scale = 1.0
        elif valid_max <= 255.0:
            scale = 255.0
        else:
            scale = float(np.iinfo(np_dtype).max)
    else:
        scale = 255.0 if valid_max > 1.0 else 1.0

    normalised = band / max(scale, 1.0)
    normalised[~valid] = 0.0
    return normalised.astype(np.float32, copy=False)
