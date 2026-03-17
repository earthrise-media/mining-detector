#!/usr/bin/env python3
"""Extract embedding pixels from GeoTIFFs into a GeoDataFrame and write to parquet.

This is a utility function for uses outside the main ML workflow.

Each input GeoTIFF has shape (bands, H, W). Band count is inferred from the first
file and must match all others. Each pixel becomes one row: one column per band
(e0, e1, ...) plus point geometry at the pixel centroid.

Usage:
    python embeddings_rasters_to_parquet.py /path/to/folder/with/geotiffs
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point


def find_geotiffs(folder: Path):
    """Return sorted paths to .tif and .tiff files in folder."""
    paths = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    return sorted(paths)


def pixel_centroids(transform, height: int, width: int):
    """Return (x, y) arrays for each pixel centroid using the raster transform."""
    rows = np.arange(height, dtype=np.float64) + 0.5  # row index of pixel center
    cols = np.arange(width, dtype=np.float64) + 0.5   # col index of pixel center
    # Mesh grid: one (row, col) per pixel in row-major order
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()
    xs, ys = rasterio.transform.xy(transform, rr, cc)
    return np.array(xs), np.array(ys)


def tiff_to_rows(path: Path, num_bands: int):
    """Read one GeoTIFF and return (data array (n_pixels, num_bands), x, y, crs)."""
    with rasterio.open(path) as src:
        if src.count != num_bands:
            raise ValueError(
                f"{path.name}: expected {num_bands} bands (from first file), got {src.count}. "
                "All GeoTIFFs must have the same number of bands."
            )
        data = src.read()   # (num_bands, H, W)
        transform = src.transform
        crs = src.crs
        height, width = data.shape[1], data.shape[2]

    # (num_bands, H, W) -> (H*W, num_bands)
    n_pixels = height * width
    values = data.reshape(num_bands, -1).T   # (n_pixels, num_bands)

    xs, ys = pixel_centroids(transform, height, width)
    return values, xs, ys, crs


def main(args):
    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    paths = find_geotiffs(folder)
    if not paths:
        raise SystemExit(f"No .tif or .tiff files found in {folder}")

    # Infer band count from first GeoTIFF; tiff_to_rows will enforce consistency
    with rasterio.open(paths[0]) as src:
        num_bands = src.count

    out_path = Path(args.outpath).resolve() if args.outpath else folder / "embeddings.parquet"

    all_values = []
    all_xs = []
    all_ys = []
    crs = None

    for path in paths:
        values, xs, ys, file_crs = tiff_to_rows(path, num_bands)
        all_values.append(values)
        all_xs.append(xs)
        all_ys.append(ys)
        if crs is None:
            crs = file_crs
        elif file_crs != crs:
            raise ValueError(
                f"CRS mismatch: {paths[0].name} has {crs}, {path.name} has {file_crs}. "
                "All GeoTIFFs must share the same CRS."
            )

    values = np.vstack(all_values)
    xs = np.concatenate(all_xs)
    ys = np.concatenate(all_ys)

    columns = [f"e{i}" for i in range(num_bands)]
    gdf = gpd.GeoDataFrame(
        {c: values[:, i] for i, c in enumerate(columns)},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )

    gdf.to_parquet(out_path, index=False)
    print(f"Wrote {len(gdf):,} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract embedding pixels from GeoTIFFs into a GeoDataFrame and write embeddings.parquet."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder containing the GeoTIFFs",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Output parquet path (default: <folder>/embeddings.parquet)",
    )
    main(parser.parse_args())
