# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/check_duplicated_areas.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
#     "shapely",
# ]
# ///

import geopandas as gpd

OUTPUT_DATA_FOLDER = "data/boundaries/protected_areas_and_indigenous_territories/out"
INDIGENOUS_TERRITORIES = f"{OUTPUT_DATA_FOLDER}/indigenous_territories.geojson"
PROTECTED_AREAS = f"{OUTPUT_DATA_FOLDER}/protected_areas.geojson"


def check_duplicated_areas(file_path, output_file_prefix):
    gdf = gpd.read_file(file_path)

    # count by country and name
    gdf.value_counts(["country", "name_field"]).loc[lambda x: x > 1].to_csv(
        f"{OUTPUT_DATA_FOLDER}/{output_file_prefix}_country_name_counts.csv",
        header=True,
    )

    # count by country, name and status
    gdf.value_counts(["country", "name_field", "status_field"]).loc[
        lambda x: x > 1
    ].to_csv(
        f"{OUTPUT_DATA_FOLDER}/{output_file_prefix}_country_name_status_counts.csv",
        header=True,
    )


if __name__ == "__main__":
    check_duplicated_areas(INDIGENOUS_TERRITORIES, "indigenous")
    check_duplicated_areas(PROTECTED_AREAS, "protected")
