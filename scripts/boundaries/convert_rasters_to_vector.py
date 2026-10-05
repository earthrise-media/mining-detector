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
#     "pyogrio",
#     "rasterio",
#     "shapely>=2",
# ]
# ///

import argparse
import os
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
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
from rasterio.windows import Window
from shapely.geometry import shape

# Rank value for pixels that are nodata or don't match any known period.
SENTINEL_RANK = 255


def year_quarter_to_pixel_value(year_quarter):
    """Map a 6-digit YYYYQQ key to the raster's pixel encoding.

    Full years (quarter == 00) encode as YYYY (e.g. 202400 -> 2024).
    Quarters encode as YYYYQ (e.g. 202503 -> 20253, 202602 -> 20262).
    """
    year, quarter = divmod(year_quarter, 100)
    if quarter == 0:
        return year
    return year * 10 + quarter


def build_rank_lut(periods):
    """Build a lookup table mapping raw pixel value -> period rank.

    The raster's compact encoding is not monotonic across the year/quarter
    boundary -- a full year 2027 encodes as 2027, which is numerically *less*
    than the quarter 202602 encoded as 20262 -- so pixel values cannot be
    thresholded directly. The canonical 6-digit YYYYQQ keys do sort correctly,
    so each pixel value is mapped to the index of its period in the sorted
    YYYYQQ list. "Cumulative through period i" then becomes `rank <= i`.

    The extra last slot holds SENTINEL_RANK and is used for out-of-range values.
    """
    if len(periods) >= SENTINEL_RANK:
        raise ValueError(f"Too many periods ({len(periods)}) for uint8 ranks")
    pixel_values = [year_quarter_to_pixel_value(p) for p in periods]
    lut = np.full(max(pixel_values) + 2, SENTINEL_RANK, dtype=np.uint8)
    for rank, value in enumerate(pixel_values):
        lut[value] = rank
    return lut


def pixel_ranks(block, lut):
    """Convert a block of raw pixel values to period ranks (uint8)."""
    if not np.issubdtype(block.dtype, np.integer):
        block = np.where(np.isfinite(block), block, -1).astype(np.int64)
    # Nodata and unknown values (negative, or larger than any period value)
    # point at the sentinel slot at the end of the LUT.
    oob = lut.size - 1
    idx = np.where((block >= 0) & (block < oob), block, oob)
    return lut[idx]


def iter_windows(width, height, size):
    for row_off in range(0, height, size):
        for col_off in range(0, width, size):
            yield Window(
                col_off,
                row_off,
                min(size, width - col_off),
                min(size, height - row_off),
            )


def vectorize_chunk(window, raster_path, lut, needed):
    """Vectorize one chunk for every needed period.

    Returns {period_index: [geometries]}. The chunk is only polygonized once per
    distinct rank actually present in it; periods with no new pixels in this
    chunk reuse the previous period's geometries.
    """
    try:
        with rasterio.open(raster_path) as src:
            block = src.read(1, window=window)
            block_transform = src.window_transform(window)

        ranks = pixel_ranks(block, lut)

        # Skip chunks that have no matching pixels at all
        present = np.unique(ranks)
        present = present[present != SENTINEL_RANK].tolist()
        if not present:
            return {}

        by_rank = {}
        result = {}
        for i in needed:
            k = bisect_right(present, i) - 1
            if k < 0:
                continue  # nothing detected in this chunk yet as of period i
            r = present[k]
            if r not in by_rank:
                mask = ranks <= r
                # Vectorize a binary mask rather than the raw block: otherwise
                # shapes() would emit a separate polygon per year value and a
                # contiguous mining area would come back split along year seams.
                binary = mask.astype(np.uint8)
                by_rank[r] = [
                    shape(geom)
                    for geom, val in shapes(binary, mask=mask, transform=block_transform)
                    if val == 1
                ]
            result[i] = by_rank[r]
        return result
    except Exception as e:
        print(f"ERROR in vectorize_chunk {window}: {type(e).__name__}: {e}")
        raise  # re-raise so it still propagates


def ensure_output_path_exists(output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def write_period(year, geoms, crs):
    output_file = generate_vectorized_raster_filename(year)

    if not geoms:
        print(f"No pixels found through {year}, skipping output.")
        return output_file

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["value"] = year  # period this cumulative snapshot represents
    gdf["year"] = year  # add year column

    ensure_output_path_exists(output_file)
    gdf.to_file(output_file, driver="GeoJSON", engine="pyogrio")
    print(f"Created: {output_file} ({len(gdf)} polygons)")
    return output_file


def main(overwrite=False, workers=None, chunk_size=4096):
    periods = sorted(set(MINING_RASTER_YEARS_QUARTERS))

    needed = []
    for i, year in enumerate(periods):
        output_file = generate_vectorized_raster_filename(year)
        if Path(output_file).exists() and not overwrite:
            print(f"Skipping {year}, {output_file} already exists (use --overwrite)")
        else:
            needed.append(i)

    if not needed:
        print("Nothing to do.")
        return

    lut = build_rank_lut(periods)

    with rasterio.open(MINING_FIRST_YEAR_RASTER_FILE) as src:
        print("Opened. Bands available:", src.count)
        print("Reported shape:", src.height, src.width)
        print("Reported dtype:", src.dtypes)
        print("Block shapes:", src.block_shapes)
        crs = src.crs
        windows = list(iter_windows(src.width, src.height, chunk_size))

    print(
        f"Vectorizing {len(needed)} period(s) across {len(windows)} chunk(s) "
        f"of up to {chunk_size}x{chunk_size} px..."
    )

    period_geoms = {i: [] for i in needed}
    worker = partial(
        vectorize_chunk,
        raster_path=MINING_FIRST_YEAR_RASTER_FILE,
        lut=lut,
        needed=needed,
    )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # map() keeps results in window order, so output is deterministic
        for n, chunk_result in enumerate(pool.map(worker, windows), 1):
            for i, geoms in chunk_result.items():
                period_geoms[i].extend(geoms)
            if n % 50 == 0 or n == len(windows):
                print(f"  {n}/{len(windows)} chunks done")

    # GeoJSON writing is I/O-heavy and pyogrio releases the GIL, so threads help
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: write_period(periods[i], period_geoms[i], crs),
                needed,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the mining single raster to cumulative vector files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-vectorize and overwrite existing vector files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (default: all CPUs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk edge length in pixels (default: 4096).",
    )
    args = parser.parse_args()

    start = time.time()
    main(overwrite=args.overwrite, workers=args.workers, chunk_size=args.chunk_size)
    print(f"Raster conversion took {time.time() - start:.1f}s")
