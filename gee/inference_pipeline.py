import argparse
from dataclasses import fields
from datetime import datetime
import logging
import logging.handlers
import re
from pathlib import Path

import geopandas as gpd

import gee
import tile_utils

def valid_date(s: str) -> str:
    """Validate date string in YYYY-MM-DD format and return it unchanged."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'.")

def get_logger(logpath: Path, maxBytes=1e6,
               backupCount=5, level=logging.INFO) -> logging.Logger:
    logpath.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        logpath, maxBytes=maxBytes, backupCount=backupCount
    )
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def main(data_config: gee.DataConfig,
         inference_config: gee.InferenceConfig,
         mask_config: gee.MaskConfig,
         cli_args: argparse.Namespace,
         logger: logging.Logger):

    region = gpd.read_file(cli_args.region_path).union_all()

    tiles = tile_utils.create_tiles(
        region, data_config.tilesize, data_config.pad)
    logger.info(f"Created {len(tiles)} tiles")

    engine = gee.InferenceEngine(
        start_date=cli_args.start_date,
        end_date=cli_args.end_date,
        data_config=data_config,
        config=inference_config,
        mask_config=mask_config,
        logger=logger,
    )
    preds = engine.bulk_predict(tiles, cli_args.region_path.stem)

    analyzed_area = len(tiles) * (data_config.tilesize / 100) ** 2
    logger.info(f"{analyzed_area} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above "
                f"{inference_config.pred_threshold}")


if __name__ == '__main__':
    data_defaults = gee.DataConfig()
    inference_defaults = gee.InferenceConfig()
    mask_defaults = gee.MaskConfig()

    parser = argparse.ArgumentParser(description="Run bulk ML inference.")

    # DataConfig args
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


    # InferenceConfig args
    parser.add_argument("--model_path", type=Path,
                        default=inference_defaults.model_path,
                        help="Path to saved Keras classifier (.h5)")
    parser.add_argument("--pred_threshold", type=float,
                        default=inference_defaults.pred_threshold,
                        help="Prediction threshold for positive chips")
    parser.add_argument("--stride_ratio", type=int,
                        default=inference_defaults.stride_ratio,
                        help="Stride is computed as chip_size//stride_ratio")
    parser.add_argument("--max_concurrent_tiles", type=int,
                        default=inference_defaults.max_concurrent_tiles,
                        help="Maximum number of tiles to process concurrently")
    parser.add_argument("--tries", type=int,
                        default=inference_defaults.tries,
                        help="Number of retries per tile")
    parser.add_argument("--embed_model_name", type=str,
                        default=inference_defaults.embed_model_name,
                        help="Embedding model identifier")
    parser.add_argument("--embed_model_path", type=str,
                        default=inference_defaults.embed_model_path,
                        help=("Path to optional pretrained foundation model; "
                              "set '' to disable"))
    parser.add_argument("--embed_model_chip_size", type=int,
                        default=inference_defaults.embed_model_chip_size,
                        help="Input size for embedding model")
    parser.add_argument("--embedding_batch_size", type=int,
                        default=inference_defaults.embedding_batch_size,
                        help="Batch size for embedding model inference")
    parser.add_argument("--geo_chip_size", type=int,
                        default=inference_defaults.geo_chip_size,
                        help="Chip size in pixels for cls_only stride cutting, or "
                             "per-window size for cls_patch (typically 224; must match "
                             "embed_model_chip_size when --embedding_strategy cls_patch)")
    parser.add_argument(
        "--embedding_strategy",
        type=str,
        choices=["cls_only", "cls_patch"],
        default=inference_defaults.embedding_strategy,
        help=(
            "cls_only: frozen FM, embed(), legacy *_embeddings.parquet. "
            "cls_patch: ViT intermediate layers, embed_dense(), *_embed_dense_*.parquet."
        ),
    )
    parser.add_argument("--embeddings_cache_dir", type=str,
                        default=inference_defaults.embeddings_cache_dir,
                        help=("Optional directory to save/reload embeddings. "
                              "Tiles read from cache skip fetching pixels; "
                              "with --run_sam2, inline SAM2 does not run for "
                              "those tiles (use standalone SAM2 masking if needed)."))
    parser.add_argument("--run_sam2", action="store_true",
                        default=inference_defaults.run_sam2,
                        help="Enable SAM2 segmentation after model predictions")
    parser.add_argument(
        "--inference_output_base",
        type=str,
        default=str(inference_defaults.inference_output_base),
        help=(
            "Base directory for prediction GeoJSON outputs "
            "(default: <repo>/data/outputs; subfolder per model version)"
        ),
    )

    # MaskConfig args
    parser.add_argument("--prior_sigma", type=float,
                        default=mask_defaults.prior_sigma,
                        help="Spatial prior falloff (pixels)")
    parser.add_argument("--smoothing_sigma", type=float,
                        default=mask_defaults.smoothing_sigma,
                        help="For Gaussian smoothing (pixels)")
    parser.add_argument("--sam2_repo_path", type=str,
                        default=mask_defaults.sam2_repo_path,
                        help="Path to SAM2 repository root")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default=mask_defaults.sam2_checkpoint,
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--finetuned_weights", type=str,
                        default=mask_defaults.finetuned_weights,
                        help="Path to fine-tuned SAM2 model weights")
    parser.add_argument("--sam2_model_cfg", type=str,
                        default=mask_defaults.sam2_model_cfg,
                        help="Hydra config name for SAM2 (e.g. configs/sam2.1/"
                             "sam2.1_hiera_s.yaml), or path to a YAML under "
                             "sam2_repo/sam2/configs/")
    parser.add_argument("--mask_dir", type=str,
                        default=mask_defaults.mask_dir,
                        help="Directory to save SAM2 outputs")

    # General args
    parser.add_argument("--region_path", type=Path,
                        default="../data/boundaries/amazon_basin.geojson",
                        help="Path to ROI geojson")
    parser.add_argument("--start_date", type=valid_date,
                        default="2023-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_date,
                        default="2023-12-31",
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--logdir", type=Path,
                        default=Path("../logs"),
                        help="Directory for log files")

    args = parser.parse_args()

    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.DataConfig)
    }
    data_config = gee.DataConfig(**config_dict)

    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.InferenceConfig)
    }
    config_dict["embed_model_path"] = str(args.embed_model_path) if args.embed_model_path else ""
    inference_config = gee.InferenceConfig(**config_dict)

    config_dict = {
        f.name: getattr(args, f.name, None) for f in fields(gee.MaskConfig)
    }
    mask_config = gee.MaskConfig(**config_dict)

    logpath = args.logdir / f"gee_{args.region_path.name}.log"
    logger = get_logger(logpath)
    logger.info(f"Inference {datetime.now().isoformat()}: {vars(args)}")

    main(data_config, inference_config, mask_config, args, logger)


