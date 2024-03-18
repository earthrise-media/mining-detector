
import argparse

import geojson

def concatenate(paths, outpath):
    """Concatenate FeatureCollections."""
    features = []
    crss = []
    
    for path in paths:
        with open(path) as f:
            data = geojson.load(f)

        crss.append(data['crs'])
        features += data['features']

    common_crs = crss[0]
    for crs in crss:
        if crs != common_crs:
            raise ValueError(f'Inconsisent CRS: {crs} / {common_crs}')

    fc = geojson.FeatureCollection(features=features, crs=common_crs)
    with open(outpath, 'w') as f:
        geojson.dump(fc, f)
        
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
