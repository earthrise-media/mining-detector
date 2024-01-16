
import argparse
from datetime import datetime
import os

import geopandas as gpd
import keras

import gee_multimodel
import utils

def main(model_paths, region_path, start_date, end_date, pred_threshold,
         clear_threshold, tile_size, tile_padding, batch_size):
    """Run model inference on specified region of interest."""
    region = gpd.read_file(region_path).geometry[0].__geo_interface__
    
    tiles = utils.create_tiles(region, tile_size, tile_padding)
    print(f"Created {len(tiles)} tiles")
    data_pipeline = gee_multimodel.S2_Data_Extractor(
        tiles, start_date, end_date, clear_threshold, batch_size=batch_size)

    print(f"Analyzing {len(tiles) * (tile_size / 100) ** 2} ha ...")
    models = [keras.models.load_model(path) for path in model_paths]
    pred_sets = data_pipeline.make_predictions(models, pred_threshold)

    for model_path, preds in zip(model_paths, pred_sets):
        print(f"Model {os.path.basename(model_path)}")
        print(f"{len(preds)} chips with predictions above {pred_threshold}")
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths", nargs="*",
        default=['../models/48px_v3.7_2023-12-02.h5'],
        help="Paths to one or more saved Keras models")
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
    
    args = parser.parse_args()
    main(**vars(args))
