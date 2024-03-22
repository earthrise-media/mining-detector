
import argparse
from datetime import datetime
import logging
import logging.handlers
import os

import geopandas as gpd
import keras

import gee
import utils

def main(model_path, region_path, start_date, end_date, pred_threshold,
         clear_threshold, tile_size, tile_padding, batch_size, tries, logger):
    """Run model inference on specified region of interest."""
    model = keras.models.load_model(model_path)
    region = gpd.read_file(region_path).geometry[0].__geo_interface__
    
    tiles = utils.create_tiles(region, tile_size, tile_padding)
    logger.info(f"Created {len(tiles)} tiles")
    data_pipeline = gee.S2_Data_Extractor(
        tiles, start_date, end_date, clear_threshold, batch_size=batch_size)
    preds = data_pipeline.make_predictions(
        model, pred_threshold, tries, logger)
    
    logger.info(f"{len(tiles) * (tile_size / 100) ** 2} ha analyzed")
    logger.info(f"{len(preds)} chips with predictions above {pred_threshold}")

    if len(preds) > 0:
        outpath = get_outpath(
            model_path, region_path, start_date, end_date, pred_threshold)
        preds.to_file(outpath, index=False)

def get_outpath(model_path, region_path, start_date, end_date, pred_threshold):
    """Format an outpath from input parameters."""  
    model_version = '_'.join(os.path.basename(model_path).split('_')[:2])
    region_name = os.path.basename(region_path).split('.geojson')[0]
    period = f'{start_date.date().isoformat()}_{end_date.date().isoformat()}'
    outdir = f'../data/outputs/{model_version}'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(
        outdir,
        f'{region_name}_{model_version}_{pred_threshold:.2f}_{period}.geojson')
    return outpath

def valid_datetime(s):
    """Formulate a datetime object from an isoformat date string."""
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = f"Not a valid date: '{s}'."
        raise argparse.ArgumentTypeError(msg)

def get_logger(path, maxBytes=1e6, backupCount=5, level=logging.INFO):
    """Create a logging instance with a RotatingFileHandler."""
    logdir = os.path.dirname(path)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=maxBytes, backupCount=backupCount)

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default='../models/48px_v3.2-3.7ensemble_2024-02-13.h5',
        type=str, help="Path to saved Keras model")
    parser.add_argument(
        "--region_path",
        default='../data/boundaries/amazon_basin/amazon_2.geojson',
        type=str, help="Path to ROI geojson")
    parser.add_argument(
        "--start_date", default='2023-01-01', type=valid_datetime,
        help="Start date in YYYY-MM-DD format")
    parser.add_argument(
        "--end_date", default='2023-12-31', type=valid_datetime,
        help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--pred_threshold", default=0.5, type=float,
        help="Prediction threshold")
    parser.add_argument(
        "--clear_threshold", default=0.6, type=float,
        help="Clear sky (cloud absence) threshold")
    parser.add_argument(
        "--tile_size", default=576, type=int,
        help="Tile width in pixels for requests to GEE")
    parser.add_argument(
        "--tile_padding", default=24, type=int,
        help="Number of pixels to pad each tile")
    parser.add_argument(
        "--batch_size", default=500, type=int,
        help="Number of tiles to process between writes")
    parser.add_argument(
        "--tries", default=2, type=int,
        help="Number of times to try tiles in case of errors.")
    parser.add_argument(
        "--logdir",
        default=f"../logs/",
        type=str, help="Path to ROI geojson")
    args = parser.parse_args()

    logpath = os.path.join(
        args.logdir, f'gee_{os.path.basename(args.region_path)}.log')
    logger = get_logger(logpath)
    delattr(args, 'logdir')
    logger.info(f'{datetime.now().isoformat()}: {vars(args)}')
    main(logger=logger, **vars(args))
