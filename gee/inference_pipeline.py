import argparse
from dataclasses import dataclass, fields
from datetime import datetime
import logging
import logging.handlers
from pathlib import Path

import geopandas as gpd
import keras

import gee
import tile_utils

@dataclass
class InferenceConfig:
    model_path: Path = Path('../models/48px_v3.2-3.7ensemble_2024-02-13.h5')
    region_path: Path = Path('../data/boundaries/amazon_basin.geojson')
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2023, 12, 31)
    pred_threshold: float = 0.5
    clear_threshold: float = 0.6
    tile_size: int = 576
    tile_padding: int = 0
    stride_ratio: int = 1
    batch_size: int = 500
    max_workers: int = 8
    collection: str = 'S2L1C'
    tries: int = 2

    def get_outpath(self) -> Path:
        model_version = '_'.join(self.model_path.stem.split('_')[:2])
        region_name = self.region_path.stem
        period = f'{self.start_date.date().isoformat()}_{self.end_date.date().isoformat()}'
        outdir = Path('../data/outputs') / model_version
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f'{region_name}_{model_version}_{self.pred_threshold:.2f}_{period}.geojson'

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


def main(config: InferenceConfig, logger: logging.Logger):
    logger.info(f"Loading model from {config.model_path}")
    model = keras.models.load_model(config.model_path)

    logger.info(f"Loading region from {config.region_path}")
    region = gpd.read_file(config.region_path).geometry[0].__geo_interface__

    tiles = tile_utils.create_tiles(
        region, config.tile_size, config.tile_padding)
    logger.info(f"Created {len(tiles)} tiles")

    data_pipeline = gee.GEE_Data_Extractor(
        config.start_date,
        config.end_date,
        clear_threshold=config.clear_threshold,
        collection=config.collection,
        max_workers=config.max_workers
    )

    outpath = config.get_outpath()
    preds = data_pipeline.make_predictions(
        tiles,
        model,
        config.pred_threshold,
        config.stride_ratio,
        config.tries,
        config.batch_size,
        logger,
        outpath
    )

    analyzed_area = len(tiles) * (config.tile_size / 100) ** 2
    logger.info(f"{analyzed_area} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above {config.pred_threshold}")


if __name__ == '__main__':
    defaults = InferenceConfig()

    parser = argparse.ArgumentParser(description="Run bulk ML inference.")
    parser.add_argument("--model_path", type=Path,
                        default=defaults.model_path,
                        help="Path to saved Keras model")
    parser.add_argument("--region_path", type=Path,
                        default=defaults.region_path,
                        help="Path to ROI geojson")
    parser.add_argument("--start_date", type=valid_datetime,
                        default=defaults.start_date,
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=valid_datetime,
                        default=defaults.end_date,
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--pred_threshold", type=float,
                        default=defaults.pred_threshold,
                        help="Prediction threshold")
    parser.add_argument("--clear_threshold", type=float,
                        default=defaults.clear_threshold,
                        help="Clear sky (cloud absence) threshold")
    parser.add_argument("--tile_size", type=int,
                        default=defaults.tile_size,
                        help="Tile width in pixels for requests to GEE")
    parser.add_argument("--tile_padding", type=int,
                        default=defaults.tile_padding,
                        help="Number of pixels to pad each tile")
    parser.add_argument("--stride_ratio", type=int,
                        default=defaults.stride_ratio,
                        help="Stride is defined by tile_size//stride_ratio")
    parser.add_argument("--batch_size", type=int,
                        default=defaults.batch_size,
                        help="Number of tiles to process per batch")
    parser.add_argument("--max_workers", type=int,
                        default=defaults.max_workers,
                        help="Maximum concurrent GEE requests")
    parser.add_argument("--collection", type=str,
                        default=defaults.collection,
                        choices=list(gee.BAND_IDS.keys()),
                        help="Satellite image collection")
    parser.add_argument("--tries", type=int,
                        default=defaults.tries,
                        help="Number of retries per tile")
    parser.add_argument("--logdir", type=Path,
                        default=Path("../logs"),
                        help="Directory for log files")

    args = parser.parse_args()

    config_dict = {
        f.name: getattr(args, f.name) for f in fields(InferenceConfig)
    }
    config = InferenceConfig(**config_dict)

    logpath = args.logdir / f"gee_{config.region_path.name}.log"
    logger = get_logger(logpath)
    logger.info(f"{datetime.now().isoformat()}: {vars(config)}")

    main(config, logger)


