# Standardizes Indigenous Territories and
# Protected Areas from source shapefiles.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/standardize_it_and_pa_areas.py`.

# to list all shapefiles in folder:
# find ./data/boundaries/protected_areas_and_indigenous_territories -name "*.shp" -type f | sed 's|^\./||' | sort

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
from shapely import set_precision
from pathlib import Path

SOURCE_DATA_FOLDER = (
    "data/boundaries/protected_areas_and_indigenous_territories/source_data"
)
OUTPUT_DATA_FOLDER = "data/boundaries/protected_areas_and_indigenous_territories/out"
SIMPLIFY_TOLERANCE = 0.00025
AMAZON_LIMITS_GEOJSON = "https://raw.githubusercontent.com/earthrise-media/mining-detector/ed/2025models/data/boundaries/Amazon_ACA.geojson"

with open("scripts/boundaries/it_and_pa_files_metadata.json") as f:
    FILES_METADATA = json.load(f)


def detect_shapefile_encoding(shapefile_path):
    # get the .dbf file path
    dbf_path = shapefile_path.replace(".shp", ".dbf")

    with open(dbf_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result["encoding"]


def combine_and_save_frames(
    all_frames, output_folder, filename, simplify, dissolve_by_attributes
):
    # combine frames
    combined_gdf = gpd.pd.concat(all_frames, ignore_index=True)
    output_combined_file = f"{output_folder}/{filename}"
    print(f"Creating: {output_combined_file}")

    # ensure output directory exists
    output_path = Path(output_combined_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # simplify geometries
    if simplify:
        combined_gdf["geometry"] = combined_gdf["geometry"].simplify(
            tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True
        )
        combined_gdf["geometry"] = combined_gdf["geometry"].apply(
            lambda geom: set_precision(geom, grid_size=1e-6)
        )

    # Dissolve based on country, name, and status
    if dissolve_by_attributes:
        # we'll ignore those that don't have country and name fields
        combined_gdf_missing = combined_gdf[
            (combined_gdf["name_field"].isna()) | (combined_gdf["country_code"].isna())
        ]

        to_dissolve = combined_gdf[
            (combined_gdf["name_field"].notna())
            & (combined_gdf["country_code"].notna())
        ]
        # replace NaNs with a unique placeholder
        to_dissolve["_status_field"] = to_dissolve["status_field"].fillna("__nan__")

        # dissolve using the temporary columns
        dissolved_gdf = to_dissolve.dissolve(
            by=["country_code", "name_field", "_status_field"], as_index=False
        )
        dissolved_gdf = dissolved_gdf.drop(columns=["_status_field"])

        # concat again into single gdf
        combined_gdf = gpd.pd.concat(
            [combined_gdf_missing, dissolved_gdf], ignore_index=True
        )

    # create an ID
    combined_gdf["id_field_str"] = np.where(
        # fill na with index to avoid duplicates
        combined_gdf["id_field"].notna(),
        combined_gdf["id_field"],
        combined_gdf.index,
    ).astype(str)
    combined_gdf["status_field_filled"] = combined_gdf["status_field"].fillna("unknown")
    # generate disambiguation combination for different country_code, id, and status_field_filled
    combined_gdf["disambig_num"] = combined_gdf.groupby(
        ["country_code", "id_field_str", "status_field_filled"]
    ).cumcount()
    combined_gdf["id"] = (
        combined_gdf["country_code"]
        + combined_gdf["id_field_str"]
        + "_"
        + combined_gdf["disambig_num"].astype(str)
    )
    del combined_gdf["disambig_num"]
    del combined_gdf["id_field_str"]
    del combined_gdf["status_field_filled"]

    print(combined_gdf["id"].nunique())
    print(len(combined_gdf))

    # # save duplicates to file
    # dupes = combined_gdf[combined_gdf["id"].duplicated(keep=False)]
    # dupes = dupes.sort_values("id")
    # dupes.to_file(
    #     output_combined_file.replace(".geojson", "_dupes.geojson"),
    #     driver="GeoJSON",
    #     encoding="utf-8",
    # )

    # identify all duplicated IDs
    dupes_mask = combined_gdf["id"].duplicated(keep=False)
    # for each duplicated row, append the index to make the ID unique
    combined_gdf.loc[dupes_mask, "id"] = (
        combined_gdf.loc[dupes_mask, "id"]
        + "_"
        + combined_gdf.loc[dupes_mask].index.astype(str)
    )
    print(combined_gdf["id"].nunique())
    print(len(combined_gdf))
    assert len(combined_gdf) == combined_gdf["id"].nunique()

    # save combined file
    combined_gdf.to_file(output_combined_file, driver="GeoJSON", encoding="utf-8")
    print(f"Created: {output_combined_file}")


def standardize_and_combine_shapefiles(files_metadata):
    indigenous_territories = []
    protected_areas = []
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

        # include area units and country name from file metadata
        gdf["area_units"] = file["area_units"]
        gdf["country"] = file["country"]
        gdf["country_code"] = file["country_code"]

        # include other columns
        cols_to_export = (
            ["country", "country_code"] + cols_to_export + ["area_units", "geometry"]
        )

        # add to lists
        if file["type"] == "indigenous-territory":
            indigenous_territories.append(gdf[cols_to_export])
        elif file["type"] == "protected-area":
            protected_areas.append(gdf[cols_to_export])
        else:
            print(f"Error, unrecognized data type: {file['type']}")
            raise ValueError

    # combine frames and save to file
    combine_and_save_frames(
        indigenous_territories,
        OUTPUT_DATA_FOLDER,
        "indigenous_territories.geojson",
        simplify=True,
        dissolve_by_attributes=True,
    )
    combine_and_save_frames(
        protected_areas,
        OUTPUT_DATA_FOLDER,
        "protected_areas.geojson",
        simplify=True,
        dissolve_by_attributes=True,
    )


if __name__ == "__main__":
    standardize_and_combine_shapefiles(FILES_METADATA)
