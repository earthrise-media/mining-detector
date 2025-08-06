# Standardizes subnational admin areas from source shapefiles.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/standardize_subnational_admin_areas.py`.

# to list all shapefiles in folder:
# find ./data/boundaries/subnational_admin -name "*.shp" -type f | sed 's|^\./||' | sort

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "geopandas",
#     "numpy",
# ]
# ///
import chardet
import geopandas as gpd
import numpy as np
import json
from pathlib import Path

SOURCE_DATA_FOLDER = "data/boundaries/subnational_admin/source_data"
OUTPUT_DATA_FOLDER = "data/boundaries/subnational_admin/out"
SIMPLIFY_TOLERANCE = 0.001
AMAZON_LIMITS_GEOJSON = "https://raw.githubusercontent.com/earthrise-media/mining-detector/ed/2025models/data/boundaries/Amazon_ACA.geojson"

with open("scripts/boundaries/subnational_admin_files_metadata.json") as f:
    files_metadata = json.load(f)


def detect_shapefile_encoding(shapefile_path):
    # get the .dbf file path
    dbf_path = shapefile_path.replace(".shp", ".dbf")

    with open(dbf_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result["encoding"]


def combine_and_save_frames(all_frames, output_folder, filename, simplify):
    # combine frames
    combined_gdf = gpd.pd.concat(all_frames, ignore_index=True)
    output_combined_file = f"{output_folder}/{filename}"

    # ensure output directory exists
    output_path = Path(output_combined_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # simplify geometries
    if simplify:
        combined_gdf["geometry"] = combined_gdf["geometry"].simplify(
            tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True
        )

    # create an unique ID
    combined_gdf = combined_gdf.reset_index(drop=False)
    combined_gdf = combined_gdf.rename(columns={"index": "id"})

    # save combined file
    # combined_gdf.to_file(output_combined_file + ".geojson", driver="GeoJSON", encoding="utf-8")
    combined_gdf.to_file(output_combined_file + ".gpkg", driver="GPKG", layer="admin_areas")
    print(f"Created: {output_combined_file}")


def standardize_and_combine_shapefiles(files_metadata):
    admin_areas = []
    amazon_limits_gdf = gpd.read_file(AMAZON_LIMITS_GEOJSON)
    amazon_limits_gdf.to_crs("EPSG:4326")
    for file in files_metadata:
        # a dictionary to rename the columns in the data
        field_cols_rename_dict = {
            v: k for k, v in file.items() if k.endswith("_field") and v != ""
        }

        # read file
        file_path = f"{SOURCE_DATA_FOLDER}/{file['file']}"
        print(f"Reading {file_path}")

        try:
            # assume first it's in utf-8
            gdf = gpd.read_file(file_path, encoding="utf-8")
        except Exception:
            # if gpd can't read the file with utf-8, try to detect the encoding
            detected_encoding = detect_shapefile_encoding(file_path)
            print(f"Detected encoding: {detected_encoding}")
            gdf = gpd.read_file(file_path, encoding=detected_encoding)

        # rename
        gdf = gdf.rename(columns=field_cols_rename_dict)

        # all of the required columns
        cols_to_export = [k for k in file.keys() if k.endswith("_field")]
        # add missing columns
        missing_cols = {col: np.nan for col in cols_to_export if col not in gdf.columns}
        gdf = gdf.assign(**missing_cols)

        # standardize crs
        gdf = gdf.to_crs("EPSG:4326")

        # only keep the ones that intersect the Amazon boundaries
        intersecting_mask = gdf.intersects(amazon_limits_gdf.union_all())
        gdf = gdf[intersecting_mask]

        # include country name and code
        gdf["country"] = file["country"]
        gdf["country_code"] = file["country_code"]

        # include other columns
        cols_to_export = ["country", "country_code"] + cols_to_export + ["geometry"]

        # add to list
        admin_areas.append(gdf[cols_to_export])

    # combine frames and save to file
    combine_and_save_frames(
        admin_areas,
        OUTPUT_DATA_FOLDER,
        "admin_areas",
        simplify=True,
    )


if __name__ == "__main__":
    standardize_and_combine_shapefiles(files_metadata)
