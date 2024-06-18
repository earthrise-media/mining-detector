
import argparse

import geopandas as gpd

def dissolve(path, threshold, column, buffer_width):
    """Merge adjacent patch-wise mine detections into polygons."""
    df = gpd.read_file(path)
    print(f'{len(df)} features prior to filtering')
    if threshold is not None:
        df = df.loc[df[column] > threshold]
    print(f'{len(df)} features after to filtering')

    dissolved = df.geometry.buffer(buffer_width, join_style=2).unary_union
    df = gpd.GeoDataFrame(geometry=[dissolved]).explode().reset_index(drop=True)
    df.geometry = df.geometry.buffer(-buffer_width, join_style=2)
    outpath = path.split('.geojson')[0] + f'-dissolved-{threshold}.geojson'
    df.to_file(outpath)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, 
        default='../data/outputs/48px_v3.7/amazon_1_48px_v3.7_0.50_2023-01-01_2023-12-31.geojson',
        help="Path to model detections")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Prediction threshold between [0, 1]; if given, the dataframe must have a 'pred' column'")
    parser.add_argument(
        "--column", type=str, default='mean',
        help="Name of prediction value to threshold")
    parser.add_argument(
        "--buffer_width", type=float,
        help="Distance in dataframe CRS to buffer patches for smooth dissolve",
        default=0.00001)

    args = parser.parse_args()
    dissolve(**vars(args))

    
