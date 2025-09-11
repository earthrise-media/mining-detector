import argparse
from dataclasses import fields
from datetime import datetime
import logging
import logging.handlers
from pathlib import Path

import geopandas as gpd
import tensorflow as tf

import gee
import tile_utils

def valid_datetime(s: str) -> datetime:
    if isinstance(s, datetime):
        return s
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
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
                start_date: datetime, end_date: datetime,
                pred_threshold: float) -> Path:
    model_version = '_'.join(model_path.stem.split('_')[:2])
    region_name = region_path.stem
    period = f"{start_date.date().isoformat()}_{end_date.date().isoformat()}"
    outdir = Path('../data/outputs') / model_version
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{region_name}_{model_version}_{pred_threshold:.2f}_{period}.geojson"

def main(data_config: gee.DataConfig,
         inference_config: gee.InferenceConfig,
         cli_args: argparse.Namespace,
         logger: logging.Logger):

    model = tf.keras.models.load_model(cli_args.model_path)
    region = gpd.read_file(cli_args.region_path).geometry[0].__geo_interface__

    tiles = tile_utils.create_tiles(
        region, data_config.tile_size, data_config.tile_padding)
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
        logger=logger
    )
    preds = engine.make_predictions(tiles, outpath=outpath)

    analyzed_area = len(tiles) * (data_config.tile_size / 100) ** 2
    logger.info(f"{analyzed_area} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above "
                f"{inference_config.pred_threshold}")


if __name__ == '__main__':
    data_defaults = gee.DataConfig()
    inference_defaults = gee.InferenceConfig()

    parser = argparse.ArgumentParser(description="Run bulk ML inference.")

    # DataConfig args
    parser.add_argument("--tile_size", type=int,
                        default=data_defaults.tile_size,
                        help="Tile width in pixels for requests to GEE")
    parser.add_argument("--tile_padding", type=int,
                        default=data_defaults.tile_padding,
                        help="Number of pixels to pad each tile")
    parser.add_argument("--collection", type=str,
                        default=data_defaults.collection,
                        choices=list(gee.BAND_IDS.keys()),
                        help="Satellite image collection")
    parser.add_argument("--clear_threshold", type=float,
                        default=data_defaults.clear_threshold,
                        help="Clear sky (cloud absence) threshold")
    parser.add_argument("--max_workers", type=int,
                        default=data_defaults.max_workers,
                        help="Maximum concurrent GEE requests")

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
    # General args
    parser.add_argument("--model_path", type=Path,
                        default="../models/48px_v3.2-3.7ensemble_2024-02-13.h5",
                        help="Path to saved Keras model")
    parser.add_argument("--region_path", type=Path,
                        default="../data/boundaries/amazon_basin.geojson",
                        help="Path to ROI geojson")
    parser.add_argument("--start_date", type=valid_datetime,
                        default=datetime(2023, 1, 1),
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_datetime,
                        default=datetime(2023, 12, 31),
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--logdir", type=Path,
                        default=Path("../logs"),
                        help="Directory for log files")

    args = parser.parse_args()

    config_dict = {
        f.name: getattr(args, f.name) for f in fields(gee.DataConfig)
    }
    data_config = gee.DataConfig(**config_dict)

    config_dict = {
        f.name: getattr(args, f.name) for f in fields(gee.InferenceConfig)
    }
    inference_config = gee.InferenceConfig(**config_dict)

    logpath = args.logdir / f"gee_{args.region_path.name}.log"
    logger = get_logger(logpath)
    logger.info(f"Inference {datetime.now().isoformat()}: {vars(args)}")

    main(data_config, inference_config, args, logger)


