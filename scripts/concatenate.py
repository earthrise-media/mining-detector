
import argparse
from pathlib import Path

import geopandas as gpd


def concatenate(paths, outpath):
    """Concatenate FeatureCollections."""
    if not paths:
        raise ValueError("At least one input path is required.")

    gdf = gpd.pd.concat([gpd.read_file(p) for p in paths], ignore_index=True)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(outpath, driver="GeoJSON", index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*",
        help="Names of files with model detections to concatenate.",
        default=[])
    parser.add_argument(
        "--outpath", type=str,
        help="Path to write concatenated detections",
        default='./combined_detections.geojson')

    args = parser.parse_args()
    concatenate(**vars(args))
