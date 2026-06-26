import cv2
import numpy as np
import onnxruntime as ort
import rasterio
from pathlib import Path
from rasterio.windows import Window
from tqdm import tqdm


# =====================================================
# CONFIG
# =====================================================

ONNX_MODEL = r"C:\Users\eswar\Downloads\Geo-Spatial-Segmentation\models\segformer_epoch_35.onnx"

INPUT_TIF = r"C:\Users\eswar\Downloads\dataset_train\train2.tif"
OUTPUT_DIR = "building_output"

TILE_SIZE = 1024
MODEL_SIZE = 512

BUILDING_CLASS = 1

MIN_BUILDING_AREA = 75
DIST_THRESHOLD = 0.35


# =====================================================
# LOAD ONNX
# =====================================================

session = ort.InferenceSession(
    ONNX_MODEL,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name


# =====================================================
# SEGFORMER INFERENCE
# =====================================================

def normalize_tile(tile):

    tile = tile.astype(np.float32)

    if tile.max() > 1:
        tile /= 255.0

    return tile


def predict_tile(tile):

    h, w = tile.shape[:2]

    resized = cv2.resize(
        tile,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = np.transpose(
        resized,
        (2, 0, 1)
    )

    tensor = np.expand_dims(
        tensor,
        axis=0
    ).astype(np.float32)

    logits = session.run(
        None,
        {input_name: tensor}
    )[0]

    pred = np.argmax(
        logits,
        axis=1
    )[0].astype(np.uint8)

    pred = cv2.resize(
        pred,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return pred


# =====================================================
# WATERSHED
# =====================================================

def separate_buildings(
    building_mask,
    min_area=75,
    dist_threshold=0.35
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    clean = cv2.morphologyEx(
        building_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        kernel
    )

    dist = cv2.distanceTransform(
        clean,
        cv2.DIST_L2,
        5
    )

    sure_fg = (
        dist >
        dist_threshold * dist.max()
    ).astype(np.uint8)

    sure_bg = cv2.dilate(
        clean,
        kernel,
        iterations=3
    )

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 1] = 0

    rgb = np.dstack([
        clean * 255,
        clean * 255,
        clean * 255
    ]).astype(np.uint8)

    markers = cv2.watershed(
        rgb,
        markers
    )

    instances = []

    for label in np.unique(markers):

        if label <= 1:
            continue

        mask = (
            markers == label
        ).astype(np.uint8)

        area = mask.sum()

        if area < min_area:
            continue

        instances.append(mask)

    return instances


# =====================================================
# MAIN
# =====================================================

def main():

    output_dir = Path(OUTPUT_DIR)

    crops_dir = output_dir / "crops"
    masks_dir = output_dir / "masks"

    crops_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    masks_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading GeoTIFF...")

    with rasterio.open(INPUT_TIF) as src:

        height = src.height
        width = src.width

        full_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        windows = []

        for row in range(
            0,
            height,
            TILE_SIZE
        ):
            for col in range(
                0,
                width,
                TILE_SIZE
            ):

                h = min(
                    TILE_SIZE,
                    height - row
                )

                w = min(
                    TILE_SIZE,
                    width - col
                )

                windows.append(
                    Window(
                        col,
                        row,
                        w,
                        h
                    )
                )

        print(
            f"Running SegFormer on {len(windows)} tiles..."
        )

        for window in tqdm(windows):

            tile = src.read(
                [1, 2, 3],
                window=window
            )

            tile = np.transpose(
                tile,
                (1, 2, 0)
            )

            tile = normalize_tile(tile)

            pred = predict_tile(tile)

            r = int(window.row_off)
            c = int(window.col_off)

            full_mask[
                r:r + pred.shape[0],
                c:c + pred.shape[1]
            ] = pred

        print("Extracting buildings...")

        building_mask = (
            full_mask ==
            BUILDING_CLASS
        ).astype(np.uint8)

        instances = separate_buildings(
            building_mask,
            MIN_BUILDING_AREA,
            DIST_THRESHOLD
        )

        print(
            f"Found {len(instances)} buildings"
        )

        rgb_full = src.read(
            [1, 2, 3]
        )

        rgb_full = np.transpose(
            rgb_full,
            (1, 2, 0)
        )

    vis = np.zeros(
        (*building_mask.shape, 3),
        dtype=np.uint8
    )

    rng = np.random.default_rng(42)

    for idx, mask in enumerate(instances):

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x1 = xs.min()
        x2 = xs.max()

        y1 = ys.min()
        y2 = ys.max()

        crop = rgb_full[
            y1:y2 + 1,
            x1:x2 + 1
        ]

        crop_mask = mask[
            y1:y2 + 1,
            x1:x2 + 1
        ]

        crop = crop.copy()

        crop[
            crop_mask == 0
        ] = 0

        cv2.imwrite(
            str(
                crops_dir /
                f"building_{idx:05d}.png"
            ),
            cv2.cvtColor(
                crop,
                cv2.COLOR_RGB2BGR
            )
        )

        cv2.imwrite(
            str(
                masks_dir /
                f"building_{idx:05d}.png"
            ),
            crop_mask * 255
        )

        color = rng.integers(
            0,
            255,
            size=3
        )

        vis[mask == 1] = color

    cv2.imwrite(
        str(
            output_dir /
            "building_instances.png"
        ),
        cv2.cvtColor(
            vis,
            cv2.COLOR_RGB2BGR
        )
    )

    print()
    print("Finished")
    print(
        f"Results saved to {output_dir}"
    )


if __name__ == "__main__":
    main()