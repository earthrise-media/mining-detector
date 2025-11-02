#!/usr/bin/env python3
"""
postprocess.py — buffered dissolve, NDVI masking, and confidence filtering
"""

import argparse
from dataclasses import fields
import warnings
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd

import gee

def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def dissolve_patches(
    gdf: gpd.GeoDataFrame, buffer_deg: float = 0.00001,
    conf_field: str = "confidence") -> gpd.GeoDataFrame:
    """Buffered dissolve of detection patches; aggregates confidence as mean."""
    gdf = gdf.copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dissolved_geom = gdf.buffer(buffer_deg, join_style=2).unary_union
        dissolved = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf.crs)
        dissolved = dissolved.explode(index_parts=False).reset_index(drop=True)
        dissolved.geometry = dissolved.buffer(-buffer_deg, join_style=2)

    joined = gpd.sjoin(gdf, dissolved, how="inner", predicate="intersects")
    mean_conf = joined.groupby("index_right")[conf_field].mean()
    dissolved[conf_field] = mean_conf.reindex(dissolved.index)

    dissolved.set_crs(gdf.crs, inplace=True)
    print(f'Dissolved {len(gdf)} to {len(dissolved)} polygons.')
    return dissolved

def filter_small_polygons(
    gdf, low_area_size=30.0, low_area_conf_threshold=0.975,
    conf_field="confidence", area_field="Polygon area (ha)"):
    """Filter polygons by dynamic confidence threshold depending on size."""
    mask = ((gdf[area_field] < low_area_size) &
                (gdf[conf_field] < low_area_conf_threshold))
    return gdf.loc[~mask].copy()

def main(args):
    "Postprocess polygons: dissolve, NDVI mask, and confidence filtering."
    gdfs = [gpd.read_file(p).to_crs("EPSG:4326") for p in args.geojson_paths]
    print(f"{sum(len(g) for g in gdfs)} polygons from {len(gdfs)} files.")
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    if args.threshold is not None:
        gdf = gdf[gdf["confidence"] >= args.threshold].copy()

    if args.dissolve:
        gdf = dissolve_patches(gdf)

    if args.ndvi_threshold is not None:
        extractor = gee.GEE_Data_Extractor(
            args.start_date, 
            args.end_date,
            args.config
        )
        masker = gee.Masker(extractor, ndvi_threshold=args.ndvi_threshold)
        gdf = masker.ndvi_mask_polygons(gdf)

    if (args.low_area_conf_threshold is not None and
            'Polygon area (ha)' in gdf.columns): 
        gdf = filter_small_polygons(
            gdf, low_area_size=args.low_area_size,
            low_area_conf_threshold=args.low_area_conf_threshold)
    
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(outpath, driver="GeoJSON", index=False)
    print(f"Wrote {len(gdf)} polygons to {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Postprocess polygons: dissolve, NDVI mask, and "
            "confidence filtering."))

    # Required args
    parser.add_argument(
        "geojson_paths", nargs="+",
        help="Input GeoJSON files to concatenate.",
    )
    parser.add_argument(
        "--outpath", required=True,
        help="Output GeoJSON file path.",
    )

    # Postprocessing options
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Overall confidence threshold applied before dissolving.",
    )
    parser.add_argument(
        "--dissolve", action="store_true",
        help="Run buffered dissolve (dissolve_patches). Default is False.",
    )
    parser.add_argument(
        "--ndvi_threshold", type=float, default=None,
        help="NDVI threshold for masking (omit to skip NDVI masking).",
    )
    parser.add_argument(
        "--start_date", type=valid_date,
        help="Start date in YYYY-MM-DD format")
    parser.add_argument(
        "--end_date", type=valid_date,
        help="End date in YYYY-MM-DD format")
    parser.add_argument("--low_area_size",
        type=float, default=30.0,
        help="Area (ha) below which to apply stricter confidence filtering.",
    )
    parser.add_argument(
        "--low_area_conf_threshold", type=float, default=None,
        help="Confidence threshold for small polygons (< low_area_size).",
    )

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
    parser.add_argument("--image_cache_dir", type=str,
                        default=data_defaults.image_cache_dir,
                        help="Optional directory to save/reload image rasters")

    args = parser.parse_args()

    if args.ndvi_threshold is not None:
        if args.start_date is None or args.end_date is None:
            parser.error("--start_date and --end_date are required "
                         "when --ndvi_threshold is provided.")
            
    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }
    config_dict.update({
        'bands': list(data_defaults._NDVI_BANDS.values())
    })
    
    args.config = gee.DataConfig(**config_dict)
    main(args)

