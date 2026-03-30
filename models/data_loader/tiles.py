import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import rasterize

# =========================
# CONFIG
# =========================

ROOT = "/home/ssl30/Desktop/geospace/data"

TILE_SIZE = 1024
STRIDE = 512

OUT_IMG = os.path.join(ROOT, "processed/images")
OUT_MASK = os.path.join(ROOT, "processed/masks")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

# =========================
# AUTO FIND IMAGE FILES
# =========================

def get_all_tif_files(root):
    tif_files = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".tif"):
                tif_files.append(os.path.join(r, f))
    return tif_files


# =========================
# LOAD SHAPEFILES
# =========================

def load_shapefiles(base_path, is_pb=False):
    if is_pb:
        return [
            gpd.read_file(os.path.join(base_path, "Built_Up_Area_typ.shp")),
            gpd.read_file(os.path.join(base_path, "Road.shp")),
            gpd.read_file(os.path.join(base_path, "Water_Body.shp")),
            gpd.read_file(os.path.join(base_path, "Utility_Poly_.shp")),
        ]
    else:
        return [
            gpd.read_file(os.path.join(base_path, "Built_Up_Area_type.shp")),
            gpd.read_file(os.path.join(base_path, "Road.shp")),
            gpd.read_file(os.path.join(base_path, "Water_Body.shp")),
            gpd.read_file(os.path.join(base_path, "Utility_Poly.shp")),
        ]


# =========================
# MAIN TILE FUNCTION
# =========================

def process_image(img_path, shapefiles, tile_id):
    saved = 0

    with rasterio.open(img_path) as src:

        # Align CRS
        layers = [gdf.to_crs(src.crs) for gdf in shapefiles]

        for y in range(0, src.height - TILE_SIZE + 1, STRIDE):
            for x in range(0, src.width - TILE_SIZE + 1, STRIDE):

                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                transform = src.window_transform(window)

                img = src.read(window=window)

                mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

                for val, gdf in enumerate(layers, 1):

                    if len(gdf) == 0:
                        continue

                    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

                    raster = rasterize(
                        shapes,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        transform=transform
                    )

                    mask[raster == 1] = val

                # Skip empty
                if np.sum(mask > 0) < 0.01 * TILE_SIZE * TILE_SIZE:
                    continue

                # Save image
                img_out = os.path.join(OUT_IMG, f"tile_{tile_id}.tif")
                with rasterio.open(
                    img_out,
                    "w",
                    driver="GTiff",
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    count=src.count,
                    dtype=img.dtype,
                    crs=src.crs,
                    transform=transform
                ) as dst:
                    dst.write(img)

                # Save mask
                mask_out = os.path.join(OUT_MASK, f"tile_{tile_id}.tif")
                with rasterio.open(
                    mask_out,
                    "w",
                    driver="GTiff",
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    count=1,
                    dtype=np.uint8,
                    crs=src.crs,
                    transform=transform
                ) as dst:
                    dst.write(mask, 1)

                tile_id += 1
                saved += 1

    return tile_id, saved


# =========================
# MAIN PIPELINE
# =========================

def build_dataset():

    # CG shapefiles
    cg_shp = load_shapefiles(os.path.join(ROOT, "CG_shp-file/shp-file"), is_pb=False)

    # PB shapefiles
    pb_shp = load_shapefiles(
        os.path.join(ROOT, "PB_training_dataSet_shp_file/PB_training_dataSet_shp_file/shp-file"),
        is_pb=True
    )

    tif_files = get_all_tif_files(ROOT)

    print(f"Total images found: {len(tif_files)}")

    tile_id = 0

    for img_path in tif_files:

        # Skip unwanted formats
        if any(x in img_path.lower() for x in [".ecw", ".aux", ".pyrx"]):
            continue

        print(f"\nProcessing: {img_path}")

        # Choose shapefile set
        if "PB" in img_path:
            shapefiles = pb_shp
        else:
            shapefiles = cg_shp

        tile_id, saved = process_image(img_path, shapefiles, tile_id)

        print(f"Tiles saved from image: {saved}")

    print(f"\n✅ TOTAL TILES CREATED: {tile_id}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    build_dataset()