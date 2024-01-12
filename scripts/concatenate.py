
import argparse
import glob

import geopandas as gpd

def concatenate(inpaths, outpath):
    """Concatenate dataframes."""
    dfs = [gpd.read_file(p) for p in inpaths]
    df = gpd.pd.concat(dfs)
    df.to_file(outpath)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpaths", nargs="*", 
        help="List of model detections to concatenate into a single dataframe.",
        default=[])
    parser.add_argument(
        "--outpath", type=str, 
        help="Path at which to write concatenated detections",
        default='./combined_detections.geojson')

    args = parser.parse_args()
    concatenate(**vars(args))
