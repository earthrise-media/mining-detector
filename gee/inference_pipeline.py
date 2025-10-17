import argparse
from dataclasses import fields
from datetime import datetime
import logging
import logging.handlers
import re
from pathlib import Path

import geopandas as gpd
import tensorflow as tf
import torch

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

def get_outpath(model_path: Path, region_path: Path,
                start_date: str, end_date: str,
                pred_threshold: float) -> Path:
    model_version = '_'.join(model_path.stem.split('_')[:2])
    region_name = region_path.stem
    period = f"{start_date}_{end_date}"
    outdir = Path('../data/outputs') / model_version
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{region_name}_{model_version}_{pred_threshold:.2f}_{period}.geojson"


def main(data_config: gee.DataConfig,
         inference_config: gee.InferenceConfig,
         cli_args: argparse.Namespace,
         logger: logging.Logger):

    model = tf.keras.models.load_model(cli_args.model_path, compile=False)
    region = gpd.read_file(cli_args.region_path).geometry[0].__geo_interface__
    if cli_args.embed_model_path is not None:
        embed_model = torch.load(cli_args.embed_model_path, weights_only=False)
    else:
        embed_model = None

    tiles = tile_utils.create_tiles(
        region, data_config.tilesize, data_config.pad)
    logger.info(f"Created {len(tiles)} tiles")

    data_extractor = gee.GEE_Data_Extractor(
        cli_args.start_date,
        cli_args.end_date,
        config=data_config
    )

    outpath = get_outpath(
        cli_args.model_path,
        cli_args.region_path,
        cli_args.start_date,
        cli_args.end_date,
        inference_config.pred_threshold
    )
    engine = gee.InferenceEngine(
        data_extractor=data_extractor,
        model=model,
        config=inference_config,
        embed_model=embed_model,
        logger=logger
    )
    preds = engine.make_predictions(tiles, outpath=outpath)

    analyzed_area = len(tiles) * (data_config.tilesize / 100) ** 2
    logger.info(f"{analyzed_area} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above "
                f"{inference_config.pred_threshold}")


if __name__ == '__main__':
    data_defaults = gee.DataConfig()
    inference_defaults = gee.InferenceConfig()

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
    parser.add_argument("--embed_model_chip_size", type=int,
                        default=inference_defaults.embed_model_chip_size,
                        help="Input size for embedding model")
    parser.add_argument("--embedding_batch_size", type=int,
                        default=inference_defaults.embedding_batch_size,
                        help="Batch size for embedding model inference")
    parser.add_argument("--geo_chip_size", type=int,
                        default=inference_defaults.geo_chip_size,
                        help="Input size for embedding model")
    parser.add_argument("--embeddings_cache_dir", type=str,
                        default=inference_defaults.embeddings_cache_dir,
                        help="Optional directory to save/reload embeddings")
    # General args
    parser.add_argument("--model_path", type=Path,
                        default="../models/48px_v3.2-3.7ensemble_2024-02-13.h5",
                        help="Path to saved Keras model")
    parser.add_argument("--region_path", type=Path,
                        default="../data/boundaries/amazon_basin.geojson",
                        help="Path to ROI geojson")
    parser.add_argument("--embed_model_path", type=Path,
                        default=None,
                        help="Path to optional pretrained foundation model")
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
    inference_config = gee.InferenceConfig(**config_dict)

    logpath = args.logdir / f"gee_{args.region_path.name}.log"
    logger = get_logger(logpath)
    logger.info(f"Inference {datetime.now().isoformat()}: {vars(args)}")

    main(data_config, inference_config, args, logger)


