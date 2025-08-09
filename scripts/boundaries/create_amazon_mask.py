# Creates an inverted polygon to mask out everything outside the Amazon boundaries.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/create_amazon_mask.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
#     "shapely",
# ]
# ///

import geopandas as gpd
from shapely.geometry import box

AMAZON_LIMITS_GEOJSON = "https://raw.githubusercontent.com/earthrise-media/mining-detector/ed/2025models/data/boundaries/Amazon_ACA.geojson"
OUTPUT_FILE = "data/boundaries/amazon_aca_mask.geojson"


def invert_polygons(gdf):
    """
    Invert polygon layer to cover the rest of the world.
    """
    world_boundary = box(-180, -90, 180, 90)
    world_crs = "EPSG:4326"

    # Ensure same CRS
    gdf_transformed = gdf.to_crs(world_crs) if gdf.crs != world_crs else gdf

    original_union = gdf_transformed.union_all()
    inverted_geom = world_boundary.difference(original_union)

    return gpd.GeoDataFrame(geometry=[inverted_geom], crs=world_crs)


if __name__ == "__main__":
    gdf = gpd.read_file(AMAZON_LIMITS_GEOJSON)
    inverted = invert_polygons(gdf)

    inverted.to_file(OUTPUT_FILE, driver="GeoJSON", encoding="utf-8")
