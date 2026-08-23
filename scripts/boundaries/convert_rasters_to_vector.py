"""
Convert the mining single raster to vector format.

Run this BEFORE preprocess_mining_areas.py. Reads a single raster whose
pixel values encode the first-detection year/quarter (e.g. 201800, 202602)
and outputs one GeoJSON per period into a `vectorized/` folder alongside the
source raster.

Each output is a CUMULATIVE snapshot: the file for a given period contains
every pixel first detected at or before that period (so 2019 includes 2018).

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
    MINING_FIRST_YEAR_RASTER_FILE,
    MINING_RASTER_YEARS_QUARTERS,
    generate_vectorized_raster_filename,
)
from rasterio.features import shapes
from shapely.geometry import shape


def year_quarter_to_pixel_value(year_quarter):
    """Map a 6-digit YYYYQQ key to the raster's pixel encoding.

    Full years (quarter == 00) encode as YYYY (e.g. 202400 -> 2024).
    Quarters encode as YYYYQ (e.g. 202503 -> 20253, 202602 -> 20262).
    """
    year, quarter = divmod(year_quarter, 100)
    if quarter == 0:
        return year
    return year * 10 + quarter


def pixel_values_through(year_quarter):
    """Return the pixel values for every period at or before `year_quarter`.

    The raster's compact encoding is not monotonic across the year/quarter
    boundary -- a full year 2027 encodes as 2027, which is numerically *less*
    than the quarter 202602 encoded as 20262 -- so pixel values cannot be
    thresholded directly. The canonical 6-digit YYYYQQ keys do sort correctly,
    so we filter those and then map the survivors to their pixel encoding.
    """
    periods = [p for p in MINING_RASTER_YEARS_QUARTERS if p <= year_quarter]
    return sorted({year_quarter_to_pixel_value(p) for p in periods})


def raster_to_gdf(raster_path, include_values):
    print(
        f"Converting {raster_path} "
        f"({len(include_values)} pixel values, up to {include_values[-1]}) to gdf..."
    )
    try:
        with rasterio.open(raster_path) as src:
            print("Opened. Bands available:", src.count)
            print("Reported shape:", src.height, src.width)
            print("Reported dtype:", src.dtypes)
            print("Block shapes:", src.block_shapes)

            nodata = src.nodata
            geoms = []

            # np.isin is much faster against a sorted ndarray of candidates
            include_values = np.asarray(include_values, dtype=src.dtypes[0])

            for _, window in src.block_windows(1):
                block = src.read(1, window=window)

                # Skip blocks that are entirely nodata or have no matching pixels
                if nodata is not None and np.all(block == nodata):
                    continue
                mask = np.isin(block, include_values)
                if not mask.any():
                    continue

                # Vectorize a binary mask rather than the raw block: otherwise
                # shapes() would emit a separate polygon per year value and a
                # contiguous mining area would come back split along year seams.
                binary = mask.astype(np.uint8)

                block_transform = src.window_transform(window)
                for geom, val in shapes(binary, mask=mask, transform=block_transform):
                    if val == 1:
                        geoms.append(shape(geom))

            gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)

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

    print(f"Processing period: {year}")
    # Pixel values in the raster use a compact encoding (YYYY or YYYYQ),
    # not the 6-digit YYYYQQ keys, so map before filtering. Cumulative
    # snapshot: every period at or before `year` is included.
    pixel_values = pixel_values_through(year)

    if not pixel_values:
        print(f"No periods at or before {year}, skipping output.")
        return output_file

    gdf = raster_to_gdf(MINING_FIRST_YEAR_RASTER_FILE, include_values=pixel_values)

    if gdf.empty:
        print(f"No pixels found through {year} (pixel values {pixel_values}), skipping output.")
        return output_file

    gdf["value"] = year  # period this cumulative snapshot represents
    gdf["year"] = year  # add year column

    ensure_output_path_exists(output_file)
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"Created: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the mining single raster to cumulative vector files."
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
