
import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from postprocess import PostprocessConfig  # noqa: E402


def filter(path, boundary_path, outpath):
    """Filter detections to those intersecting a boundary.

    The caller names the destination. Earlier versions wrote a ``-filt`` sibling,
    which left the filtered file somewhere no downstream stage read from, so every
    caller had to move it afterwards -- and a consumer had to know that andes
    detections live under ``-filt`` while Amazon detections do not. The input is
    left alone as the pristine inference output.
    """
    gdf = gpd.read_file(path)
    boundary = gpd.read_file(boundary_path)
    print(f'{len(gdf)} features prior to filtering')
    filtered = gdf[gdf.intersects(boundary.union_all())]
    print(f'{len(filtered)} features after filtering')
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    # Pinned so precision does not float with the GDAL/pyogrio version; see
    # core/postprocess.py and docs/design/persistence-planning.md.
    filtered.to_file(outpath, driver="GeoJSON", index=False,
                     COORDINATE_PRECISION=PostprocessConfig.coordinate_precision)
    print(f'Wrote {len(filtered)} detections to {outpath}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str,
        help="Path to model detections")
    parser.add_argument(
        "boundary_path", type=str,
        help="Path to boundary GeoJSON")
    parser.add_argument(
        "--outpath", type=str, required=True,
        help=("Path to write filtered detections. Required: the destination is "
              "where downstream stages read from, so it is named, not guessed."))

    args = parser.parse_args()
    filter(**vars(args))
