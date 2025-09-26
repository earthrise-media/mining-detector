# Standardizes national admin areas from source geojsons.
# The geojsons were downloaded from https://gadm.org/download_country.html.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/standardize_national_admin_areas.py`.

# to list all shapefiles in folder:
# find ./data/boundaries/national_admin/source_data -name "*.geojson" -type f | sed 's|^\./||' | sort

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
#     "pandas",
#     "shapely",
# ]
# ///
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from pathlib import Path

SOURCE_DATA_FOLDER = "data/boundaries/national_admin/source_data"
OUTPUT_DATA_FOLDER = "data/boundaries/national_admin/out"
AMAZON_LIMITS_GEOJSON = "https://raw.githubusercontent.com/earthrise-media/mining-detector/ed/2025models/data/boundaries/Amazon_ACA.geojson"
SIMPLIFY_TOLERANCE = 0.001


def combine_geojsons(files_metadata):
    # load frames
    frames = [
        gpd.read_file(f"{SOURCE_DATA_FOLDER}/{file['file']}") for file in files_metadata
    ]

    # combine frames
    combined_gdf = gpd.pd.concat(frames, ignore_index=True)
    output_combined_file = f"{OUTPUT_DATA_FOLDER}/national_admin"

    # lowercase column names
    combined_gdf.columns = [x.lower() for x in combined_gdf.columns]

    # ensure output directory exists
    output_path = Path(output_combined_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # create an unique ID
    combined_gdf["id"] = combined_gdf["gid_0"]
    assert len(combined_gdf) == combined_gdf["id"].nunique()

    # clip to amazon boundaries
    amazon_limits_gdf = gpd.read_file(AMAZON_LIMITS_GEOJSON)
    amazon_limits_gdf.to_crs("EPSG:4326")
    combined_gdf = combined_gdf.clip(amazon_limits_gdf)

    # create a single polygon for the whole Amazon area
    whole_amazon = {
        "gid_0": "AMAZ",
        "country": "Entire Amazon",
        "id": "AMAZ",
        "geometry": unary_union(amazon_limits_gdf.geometry),
    }
    whole_amazon_gdf = gpd.GeoDataFrame([whole_amazon], crs=combined_gdf.crs)
    # concatenate back to original gdf
    combined_gdf = pd.concat([whole_amazon_gdf, combined_gdf])

    # simplify
    combined_gdf["geometry"] = combined_gdf["geometry"].simplify(
        tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True
    )

    # save combined file
    combined_gdf.to_file(
        output_combined_file + ".geojson", driver="GeoJSON", encoding="utf-8"
    )
    print(f"Created: {output_combined_file}")


if __name__ == "__main__":
    files_metadata = [
        {"file": "gadm41_BOL_0.geojson"},
        {"file": "gadm41_BRA_0.geojson"},
        {"file": "gadm41_COL_0.geojson"},
        {"file": "gadm41_ECU_0.geojson"},
        {"file": "gadm41_GUF_0.geojson"},
        {"file": "gadm41_GUY_0.geojson"},
        {"file": "gadm41_PER_0.geojson"},
        {"file": "gadm41_SUR_0.geojson"},
        {"file": "gadm41_VEN_0.geojson"},
    ]
    combine_geojsons(files_metadata)
