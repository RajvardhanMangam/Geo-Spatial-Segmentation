"""Run the fine-tuned SegFormer checkpoint on a GeoTIFF and save a mask TIFF."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/segformer_epoch_35.pth"),
        help="Path to the trained .pth checkpoint.",
    )
    parser.add_argument(
        "--input_tif",
        type=Path,
        required=True,
        help="Input GeoTIFF to segment.",
    )
    parser.add_argument(
        "--output_tif",
        type=Path,
        default=None,
        help="Output mask GeoTIFF. Defaults to <input>_segmentation.tif.",
    )
    parser.add_argument(
        "--model_name",
        default="nvidia/segformer-b1-finetuned-ade-512-512",
        help="Base SegFormer architecture used during training.",
    )
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--img_size", type=int, default=512)
    return parser.parse_args()


def load_model(model_name: str, model_path: Path, num_labels: int, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = normalise_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing checkpoint keys: {missing[:10]}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {unexpected[:10]}")

    model.to(device)
    model.eval()
    return model


def normalise_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    return state_dict


def normalise_tile(tile: np.ndarray) -> np.ndarray:
    if np.issubdtype(tile.dtype, np.integer):
        dtype_max = np.iinfo(tile.dtype).max
        scale = 255.0 if dtype_max <= 255 else float(dtype_max)
        tile = tile.astype(np.float32) / scale
    else:
        tile = tile.astype(np.float32)
        if tile.max(initial=0) > 1.0:
            tile = tile / 255.0

    return np.clip(tile, 0.0, 1.0)


def read_rgb(src, window: Window) -> np.ndarray:
    tile = src.read(window=window)
    if tile.shape[0] >= 3:
        tile = tile[:3]
    else:
        padding = np.zeros(
            (3 - tile.shape[0], tile.shape[1], tile.shape[2]),
            dtype=tile.dtype,
        )
        tile = np.concatenate([tile, padding], axis=0)

    return normalise_tile(tile)


@torch.no_grad()
def predict_tile(model, tile: np.ndarray, img_size: int, device) -> np.ndarray:
    height, width = tile.shape[1], tile.shape[2]
    tensor = torch.from_numpy(np.ascontiguousarray(tile)).float().unsqueeze(0)
    tensor = tensor.to(device)

    if height != img_size or width != img_size:
        tensor = F.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    outputs = model(pixel_values=tensor)
    logits = F.interpolate(
        outputs.logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    output_tif = args.output_tif
    if output_tif is None:
        output_tif = args.input_tif.with_name(
            f"{args.input_tif.stem}_segmentation.tif"
        )

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not args.input_tif.exists():
        raise FileNotFoundError(f"Input TIFF not found: {args.input_tif}")
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.model_path}")
    model = load_model(args.model_name, args.model_path, args.num_labels, device)

    with rasterio.open(args.input_tif) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

        tile_size = args.tile_size
        windows = []
        for row_off in range(0, src.height, tile_size):
            for col_off in range(0, src.width, tile_size):
                height = min(tile_size, src.height - row_off)
                width = min(tile_size, src.width - col_off)
                windows.append(Window(col_off, row_off, width, height))

        with rasterio.open(output_tif, "w", **profile) as dst:
            for window in tqdm(windows, desc="Segmenting TIFF"):
                tile = read_rgb(src, window)
                pred = predict_tile(model, tile, args.img_size, device)
                dst.write(pred, 1, window=window)

    print(f"Saved: {output_tif}")


if __name__ == "__main__":
    main()
