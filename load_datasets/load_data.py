import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import rasterize

# =========================
# CONFIG (UPDATED FOR YOUR DATA)
# =========================

IMAGE_DIRS = [
    "./data/CG_Training_dataSet_2/Training_dataSet_2",
    "./data/CG_Training_dataSet_3/Training_dataSet_3"
]

SHP_DIR = "./data/CG_shp-file/shp-file"

OUT_IMG_DIR = "./data/images"
OUT_MASK_DIR = "./data/masks"

TILE_SIZE = 1024
STRIDE = 512   # 50% overlap (change to 256 for more overlap)

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =========================
# LOAD SHAPEFILES
# =========================
print("Loading shapefiles...")

buildings = gpd.read_file(os.path.join(SHP_DIR, "Built_Up_Area_type.shp"))
roads = gpd.read_file(os.path.join(SHP_DIR, "Road.shp"))
water = gpd.read_file(os.path.join(SHP_DIR, "Water_Body.shp"))
utility = gpd.read_file(os.path.join(SHP_DIR, "Utility_Poly.shp"))

layers_all = [buildings, roads, water, utility]

# =========================
# MAIN FUNCTION
# =========================
def create_tiles():
    tile_id = 0

    for IMAGE_DIR in IMAGE_DIRS:
        print(f"\n📂 Processing folder: {IMAGE_DIR}")

        for img_name in os.listdir(IMAGE_DIR):

            # Ignore non-TIF (important for PB dataset)
            if not img_name.endswith(".tif"):
                continue

            img_path = os.path.join(IMAGE_DIR, img_name)
            print(f"Processing image: {img_name}")

            with rasterio.open(img_path) as src:

                # Convert shapefiles to match image CRS
                layers = [gdf.to_crs(src.crs) for gdf in layers_all]

                for y in range(0, src.height - TILE_SIZE + 1, STRIDE):
                    for x in range(0, src.width - TILE_SIZE + 1, STRIDE):

                        window = Window(x, y, TILE_SIZE, TILE_SIZE)
                        transform = src.window_transform(window)

                        img_tile = src.read(window=window)

                        # =========================
                        # CREATE MASK
                        # =========================
                        mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

                        for val, gdf in enumerate(layers, 1):

                            if len(gdf) == 0:
                                continue

                            shapes = [
                                (geom, 1)
                                for geom in gdf.geometry
                                if geom is not None
                            ]

                            raster = rasterize(
                                shapes,
                                out_shape=(TILE_SIZE, TILE_SIZE),
                                transform=transform
                            )

                            mask[raster == 1] = val

                        # =========================
                        # SKIP EMPTY TILES
                        # =========================
                        if np.sum(mask > 0) < 0.01 * TILE_SIZE * TILE_SIZE:
                            continue

                        # =========================
                        # SAVE IMAGE TILE
                        # =========================
                        img_out = os.path.join(OUT_IMG_DIR, f"tile_{tile_id}.tif")

                        with rasterio.open(
                            img_out,
                            "w",
                            driver="GTiff",
                            height=TILE_SIZE,
                            width=TILE_SIZE,
                            count=src.count,
                            dtype=img_tile.dtype,
                            crs=src.crs,
                            transform=transform
                        ) as dst:
                            dst.write(img_tile)

                        # =========================
                        # SAVE MASK TILE
                        # =========================
                        mask_out = os.path.join(OUT_MASK_DIR, f"tile_{tile_id}.tif")

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

    print(f"\n✅ Total tiles created: {tile_id}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    create_tiles()