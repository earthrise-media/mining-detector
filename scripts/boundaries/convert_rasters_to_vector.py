"""
Convert mining difference rasters to vector format.

Run this BEFORE preprocess_mining_areas.py. Outputs one GeoJSON per
year/quarter into a `vectorized/` folder alongside the source rasters,
using the same base filename.

Existing outputs are skipped unless --overwrite is passed.
"""

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/convert_rasters_to_vector.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
#     "numpy",
#     "rasterio",
# ]
# ///

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from constants import (
    MINING_DIFFERENCES_RASTER_FILES,
    MINING_RASTER_YEARS_QUARTERS,
    generate_vectorized_raster_filename,
)
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_gdf(raster_path, value_filter=1):
    print(f"Converting {raster_path} to gdf...")
    try:
        with rasterio.open(raster_path) as src:
            print("Opened. Bands available:", src.count)
            print("Reported shape:", src.height, src.width)
            print("Reported dtype:", src.dtypes)
            print("Block shapes:", src.block_shapes)

            nodata = src.nodata
            geoms = []
            values = []

            for _, window in src.block_windows(1):
                block = src.read(1, window=window)

                # Skip blocks that are entirely nodata or have no matching pixels
                if nodata is not None and np.all(block == nodata):
                    continue
                mask = block == value_filter
                if not mask.any():
                    continue

                block_transform = src.window_transform(window)
                for geom, val in shapes(block, mask=mask, transform=block_transform):
                    if val == value_filter:
                        geoms.append(shape(geom))
                        values.append(val)

            gdf = gpd.GeoDataFrame({"value": values}, geometry=geoms, crs=src.crs)

        return gdf
    except Exception as e:
        print(f"ERROR in raster_to_gdf: {type(e).__name__}: {e}")
        raise  # re-raise so it still propagates


def ensure_output_path_exists(output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def process_raster(year, overwrite=False):
    output_file = generate_vectorized_raster_filename(year)

    if Path(output_file).exists() and not overwrite:
        print(f"Skipping {year}, {output_file} already exists (use --overwrite)")
        return output_file

    print(f"Processing raster for: {year}")
    gdf = raster_to_gdf(MINING_DIFFERENCES_RASTER_FILES[year], value_filter=1)
    gdf["year"] = year  # add year column

    ensure_output_path_exists(output_file)
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"Created: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert mining difference rasters to vector files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-vectorize and overwrite existing vector files.",
    )
    args = parser.parse_args()

    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda year: process_raster(year, overwrite=args.overwrite),
                MINING_RASTER_YEARS_QUARTERS,
            )
        )
    print(f"Raster conversion took {time.time() - start:.1f}s")
