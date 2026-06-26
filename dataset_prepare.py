import os
import json
import csv
import numpy as np
import rasterio
import geopandas as gpd

from pathlib import Path
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.features import rasterize


# ==========================================================
# CONFIG
# ==========================================================

TIFF_DIR = r"C:\Users\eswar\Downloads\dataset_train_2"
SHP_DIR = r"C:\Users\eswar\Downloads\Punjab_shp_file"

OUT_DIR = r"C:\Users\eswar\Downloads\segformer_geotiff_dataset2"

OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_MASK_DIR = os.path.join(OUT_DIR, "masks")

TILE_SIZE = 1024
STRIDE = 512

MIN_LABEL_PIXELS = 1000

# If your GeoTIFF is BGR, use [3, 2, 1]
# If it is RGB, use [1, 2, 3]
RGB_BANDS = [1, 2, 3]

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def read_shp_if_exists(path):
    if os.path.exists(path):
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]
        return gdf
    print("Missing shapefile:", path)
    return None


def find_column(gdf, name):
    if gdf is None:
        return None

    for col in gdf.columns:
        if col.lower() == name.lower():
            return col

    return None


def reproject_layer(gdf, target_crs):
    if gdf is None:
        return None

    if gdf.crs is None:
        print("Warning: layer has no CRS. Assigning raster CRS.")
        return gdf.set_crs(target_crs)

    return gdf.to_crs(target_crs)


def filter_tile(gdf, tile_box):
    if gdf is None or len(gdf) == 0:
        return gdf.iloc[0:0] if gdf is not None else None

    try:
        idx = list(gdf.sindex.intersection(tile_box.bounds))
        gdf_tile = gdf.iloc[idx]
    except Exception:
        gdf_tile = gdf

    gdf_tile = gdf_tile[gdf_tile.intersects(tile_box)]
    return gdf_tile


def burn_fixed_class(mask, gdf, tile_box, transform, class_id):
    if gdf is None:
        return mask

    gdf_tile = filter_tile(gdf, tile_box)

    if gdf_tile is None or len(gdf_tile) == 0:
        return mask

    shapes = [
        (geom, class_id)
        for geom in gdf_tile.geometry
        if geom is not None and not geom.is_empty
    ]

    if len(shapes) == 0:
        return mask

    raster = rasterize(
        shapes,
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )

    mask[raster > 0] = raster[raster > 0]
    return mask


def burn_attribute_class(mask, gdf, tile_box, transform, field_name, mapping):
    """
    Example:
    Road_type 3 -> class 5
    Road_type 5 -> class 6
    Road_type 6 -> class 7
    """
    if gdf is None:
        return mask

    col = find_column(gdf, field_name)

    if col is None:
        print(f"Field {field_name} not found. Skipping.")
        return mask

    gdf_tile = filter_tile(gdf, tile_box)

    if gdf_tile is None or len(gdf_tile) == 0:
        return mask

    shapes = []

    for _, row in gdf_tile.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        try:
            value = int(float(row[col]))
        except Exception:
            continue

        if value not in mapping:
            continue

        class_id = mapping[value]
        shapes.append((geom, class_id))

    if len(shapes) == 0:
        return mask

    raster = rasterize(
        shapes,
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )

    mask[raster > 0] = raster[raster > 0]
    return mask


# ==========================================================
# LOAD SHAPEFILES
# ==========================================================

print("Loading shapefiles...")

buildings = read_shp_if_exists(
    os.path.join(SHP_DIR, "Built_Up_Area_type.shp")
)

roads = read_shp_if_exists(
    os.path.join(SHP_DIR, "Road.shp")
)

water = read_shp_if_exists(
    os.path.join(SHP_DIR, "Water_Body.shp")
)

utility = read_shp_if_exists(
    os.path.join(SHP_DIR, "Utility_Poly.shp")
)

bridge = read_shp_if_exists(
    os.path.join(SHP_DIR, "Bridge.shp")
)


# ==========================================================
# CLASS MAPPING
# ==========================================================
# You can rename later after getting codebook.

id2label = {
    0: "background",

    1: "water",
    2: "utility",
    3: "bridge",

    4: "road_type_3",
    5: "road_type_5",
    6: "road_type_6",

    7: "roof_type_1",
    8: "roof_type_2",
    9: "roof_type_3",
    10: "roof_type_4"
}

road_type_to_class = {
    3: 4,
    5: 5,
    6: 6
}

roof_type_to_class = {
    1: 7,
    2: 8,
    3: 9,
    4: 10
}

with open(os.path.join(OUT_DIR, "id2label.json"), "w") as f:
    json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)


# ==========================================================
# PROCESS TIFFS
# ==========================================================

tile_id = 0
metadata_rows = []

tiff_files = [
    f for f in os.listdir(TIFF_DIR)
    if f.lower().endswith((".tif", ".tiff"))
]

print("Found TIFF files:", len(tiff_files))

for tif_name in tiff_files:

    tif_path = os.path.join(TIFF_DIR, tif_name)

    print("\n" + "=" * 80)
    print("Processing:", tif_name)
    print("=" * 80)

    try:
        with rasterio.open(tif_path) as src:

            print("CRS:", src.crs)
            print("Size:", src.width, "x", src.height)
            print("Bands:", src.count)
            print("Dtype:", src.dtypes)

            if src.crs is None:
                print("Skipping because raster has no CRS.")
                continue

            # Reproject shapefiles to this raster CRS
            buildings_r = reproject_layer(buildings, src.crs)
            roads_r = reproject_layer(roads, src.crs)
            water_r = reproject_layer(water, src.crs)
            utility_r = reproject_layer(utility, src.crs)
            bridge_r = reproject_layer(bridge, src.crs)

            # Valid RGB bands
            selected_bands = [
                b for b in RGB_BANDS
                if 1 <= b <= src.count
            ]

            if len(selected_bands) == 0:
                selected_bands = [1]

            for y in range(0, src.height - TILE_SIZE + 1, STRIDE):

                for x in range(0, src.width - TILE_SIZE + 1, STRIDE):

                    window = Window(
                        x,
                        y,
                        TILE_SIZE,
                        TILE_SIZE
                    )

                    transform = src.window_transform(window)

                    try:
                        img_tile = src.read(
                            selected_bands,
                            window=window
                        )
                    except Exception as e:
                        print("Image read failed:", e)
                        continue

                    # If only one band, repeat to 3 bands
                    if img_tile.shape[0] == 1:
                        img_tile = np.repeat(img_tile, 3, axis=0)

                    # If more than 3 selected somehow, keep only 3
                    if img_tile.shape[0] > 3:
                        img_tile = img_tile[:3]

                    # ======================================
                    # CREATE MASK
                    # ======================================

                    mask = np.zeros(
                        (TILE_SIZE, TILE_SIZE),
                        dtype=np.uint8
                    )

                    tile_bounds = rasterio.windows.bounds(
                        window,
                        src.transform
                    )

                    tile_box = box(*tile_bounds)

                    # Priority order:
                    # lower-priority classes first,
                    # high-priority classes later overwrite overlaps.

                    mask = burn_fixed_class(
                        mask,
                        water_r,
                        tile_box,
                        transform,
                        class_id=1
                    )

                    mask = burn_fixed_class(
                        mask,
                        utility_r,
                        tile_box,
                        transform,
                        class_id=2
                    )

                    mask = burn_fixed_class(
                        mask,
                        bridge_r,
                        tile_box,
                        transform,
                        class_id=3
                    )

                    mask = burn_attribute_class(
                        mask,
                        roads_r,
                        tile_box,
                        transform,
                        field_name="Road_type",
                        mapping=road_type_to_class
                    )

                    mask = burn_attribute_class(
                        mask,
                        buildings_r,
                        tile_box,
                        transform,
                        field_name="Roof_type",
                        mapping=roof_type_to_class
                    )

                    # ======================================
                    # SKIP EMPTY MASKS
                    # ======================================

                    non_zero = np.count_nonzero(mask)

                    if non_zero < MIN_LABEL_PIXELS:
                        continue

                    # print(
                    #     f"Tile {tile_id} | "
                    #     f"Labels={np.unique(mask)} | "
                    #     f"Pixels={non_zero}"
                    # )

                    # ======================================
                    # SAVE IMAGE TILE AS GEOTIFF
                    # ======================================

                    img_path = os.path.join(
                        OUT_IMG_DIR,
                        f"tile_{tile_id:07d}.tif"
                    )

                    img_profile = src.profile.copy()

                    img_profile.update({
                        "driver": "GTiff",
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "count": img_tile.shape[0],
                        "dtype": img_tile.dtype,
                        "crs": src.crs,
                        "transform": transform,
                        "compress": "lzw"
                    })

                    # Remove nodata if incompatible with dtype
                    if "nodata" in img_profile and img_profile["nodata"] is None:
                        img_profile.pop("nodata", None)

                    with rasterio.open(img_path, "w", **img_profile) as dst:
                        dst.write(img_tile)

                    # ======================================
                    # SAVE MASK TILE AS GEOTIFF
                    # ======================================

                    mask_path = os.path.join(
                        OUT_MASK_DIR,
                        f"tile_{tile_id:07d}.tif"
                    )

                    mask_profile = {
                        "driver": "GTiff",
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "count": 1,
                        "dtype": "uint8",
                        "crs": src.crs,
                        "transform": transform,
                        "compress": "lzw"
                    }

                    with rasterio.open(mask_path, "w", **mask_profile) as dst:
                        dst.write(mask, 1)

                    metadata_rows.append({
                        "tile_id": tile_id,
                        "image": img_path,
                        "mask": mask_path,
                        "source_tif": tif_path,
                        "x": x,
                        "y": y,
                        "tile_size": TILE_SIZE,
                        "stride": STRIDE,
                        "labels": ",".join(map(str, np.unique(mask).tolist())),
                        "label_pixels": int(non_zero),
                        "crs": str(src.crs),
                        "transform": str(transform)
                    })

                    tile_id += 1

    except Exception as e:
        print("Error:", tif_name)
        print(e)


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata_path = os.path.join(OUT_DIR, "metadata.csv")

with open(metadata_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "tile_id",
            "image",
            "mask",
            "source_tif",
            "x",
            "y",
            "tile_size",
            "stride",
            "labels",
            "label_pixels",
            "crs",
            "transform"
        ]
    )

    writer.writeheader()
    writer.writerows(metadata_rows)

print("\nDone!")
print("Total tiles:", tile_id)
print("Output:", OUT_DIR)
print("Images:", OUT_IMG_DIR)
print("Masks:", OUT_MASK_DIR)
print("Metadata:", metadata_path)