
import argparse
from pathlib import Path

import geopandas as gpd


def filter(path, boundary_path):
    """Filter detections to those intersecting a boundary."""
    gdf = gpd.read_file(path)
    boundary = gpd.read_file(boundary_path)
    print(f'{len(gdf)} features prior to filtering')
    filtered = gdf[gdf.intersects(boundary.union_all())]
    print(f'{len(filtered)} features after filtering')
    outpath = Path(path).with_stem(f'{Path(path).stem}-filt')
    # Pinned so precision does not float with the GDAL/pyogrio version; see
    # core/postprocess.py and docs/design/persistence-planning.md.
    filtered.to_file(outpath, driver="GeoJSON", index=False,
                     COORDINATE_PRECISION=9)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str,
        help="Path to model detections")
    parser.add_argument(
        "boundary_path", type=str,
        help="Path to boundary GeoJSON")

    args = parser.parse_args()
    filter(**vars(args))
