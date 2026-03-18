"""Tiles an AOI and exports data from GEE to GeoTiffs for supported collections.

This is a utility function for uses outside the main ML workflow.

Usage: 
python gee_data_pull.py my_aoi.geojson --start_date 2024-01-01 --end_date 2024-01-31 --collection AlphaEarth --tilesize 224

"""

import argparse
from dataclasses import fields
from pathlib import Path
import re
import sys

import geopandas as gpd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gee"))
import gee
from tile_utils import tiles_for_geometry

def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def main(args):
    """Pull raster data from Earth Engine."""
    tiles_written = []

    extractor = gee.GEE_Data_Extractor(
        args.start_date,
        args.end_date,
        args.config
    )

    gdf = gpd.read_file(args.geojson_path).to_crs("EPSG:4326")

    outdir = Path(args.geojson_path.split('.geojson')[0] + args.collection)
    outdir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Geometries"):
        geom = row.geometry
        if geom.is_empty or not geom.is_valid:
            continue
        tiles = tiles_for_geometry(
            geom,
            extractor.config.tilesize,
            extractor.config.pad
        )

        for tile in tiles:
            if not row.geometry.intersects(tile.geometry):
                continue
            try:
                pixels = extractor.get_tile_data(tile)
                extractor.save_tile(pixels, tile, outdir)
                tiles_written.append(tile)
            except Exception as e:
                print(f"Tile {tile.key} failed: {e}")

    print(f"{len(tiles_written)} tiles written from {len(gdf)} geometries.")
    return tiles_written

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Pull raster data from GEE."))

    # Required args
    parser.add_argument(
        "geojson_path", type=str,
        help="GeoJSON polygons for which to pull raster data.",
    )
    parser.add_argument(
        "--start_date", type=valid_date, required=True,
        help="Start date in YYYY-MM-DD format")
    parser.add_argument(
        "--end_date", type=valid_date, required=True,
        help="End date in YYYY-MM-DD format")

    # DataConfig args
    data_defaults = gee.DataConfig()

    parser.add_argument("--tilesize", type=int,
                        default=data_defaults.tilesize,
                        help="Tile width in pixels for requests to GEE")
    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Number of pixels to pad each tile")
    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite image collection")
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Clear sky (cloud absence) threshold")
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Maximum concurrent GEE requests")

    args = parser.parse_args()

    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }

    args.config = gee.DataConfig(**config_dict)
    main(args)
