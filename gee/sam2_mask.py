
import argparse
from dataclasses import fields
from pathlib import Path
import re

import geopandas as gpd

import gee  
from sam2_build_cog import main as run_cogging 


def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def main(args):

    detections_path = Path(args.detections)
    start_date, end_date = args.start_date, args.end_date

    data_defaults = gee.DataConfig()

    data_config_dict = {
        f.name: getattr(args, f.name, getattr(data_defaults, f.name))
        for f in fields(gee.DataConfig)
    }
    data_config = gee.DataConfig(**data_config_dict)

    mask_defaults = gee.MaskConfig()

    mask_config_dict = {
        f.name: getattr(args, f.name, getattr(mask_defaults, f.name))
        for f in fields(gee.MaskConfig)
    }
    mask_config = gee.MaskConfig(**mask_config_dict)

    # Redirect mask_dir to subfolder per detections file
    detections_stem = detections_path.stem
    base_mask_dir = Path(mask_config.mask_dir)
    run_mask_dir = base_mask_dir / detections_stem
    run_mask_dir.mkdir(parents=True, exist_ok=True)
    mask_config.mask_dir = run_mask_dir

    dets = gpd.read_file(detections_path)
    data_extractor = gee.GEE_Data_Extractor(start_date, end_date, data_config)
    masker = gee.SAM2_Masker(data_extractor, mask_config)
    masker.bulk_mask_polygons(dets)

    if args.cog:
        input_dir = masker.config.mask_dir
        output_dir = Path(input_dir) / "cog_outputs"

        index_out = output_dir / "tile_index.parquet"
        stac_out = output_dir / "stac_catalog.json"

        run_cogging(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            index_out=str(index_out),
            stac_out=str(stac_out),
            max_workers=args.max_workers,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SAM2 segmentation on detection polygons"
    )

    # ----------------------------
    # Required detections file
    # ----------------------------
    parser.add_argument(
        "detections", type=str,
        help="Path to detection polygons GeoJSON (w/ date range in filename)"
    )
    parser.add_argument("--start_date", type=valid_date,
                        default="2023-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_date,
                        default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    
    # ----------------------------
    # MaskConfig args
    # ----------------------------
    parser.add_argument("--prior_sigma", type=float,
                        default=gee.MaskConfig().prior_sigma,
                        help="Spatial prior falloff (pixels)")

    parser.add_argument("--smoothing_sigma", type=float,
                        default=gee.MaskConfig().smoothing_sigma,
                        help="Gaussian smoothing sigma (pixels)")

    parser.add_argument("--sam2_checkpoint", type=str,
                        default=gee.MaskConfig().sam2_checkpoint,
                        help="Path to SAM2 checkpoint")

    parser.add_argument("--finetuned_weights", type=str,
                        default=gee.MaskConfig().finetuned_weights,
                        help="Path to fine-tuned SAM2 weights")

    parser.add_argument("--sam2_model_cfg", type=str,
                        default=gee.MaskConfig().sam2_model_cfg,
                        help="Path to SAM2 YAML config")

    parser.add_argument("--mask_dir", type=str,
                        default=gee.MaskConfig().mask_dir,
                        help="Directory to write SAM2 GeoTIFF masks")

    # ----------------------------
    # DataConfig args
    # ----------------------------
    data_defaults = gee.DataConfig()

    parser.add_argument("--tilesize", type=int,
                        default=256,
                        help="Tile width in pixels")

    parser.add_argument("--pad", type=int,
                        default=data_defaults.pad,
                        help="Padding pixels around tiles")

    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=gee.DataConfig.available_collections(),
                        help="Satellite collection")

    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Cloud-free threshold")

    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Parallel GEE requests")

    parser.add_argument("--image_cache_dir", type=str,
                        default=data_defaults.image_cache_dir,
                        help="Optional raster cache directory")

    # ----------------------------
    # Cogging flag
    # ----------------------------
    parser.add_argument("--cog", action="store_true",
                        help="Run COG mosaicking pipeline after masking")

    args = parser.parse_args()

    main(args)
