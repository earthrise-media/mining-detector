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


def check_duplicated_areas():
    gdf = gpd.read_file(INDIGENOUS_TERRITORIES)

    # count by country and name
    gdf.value_counts(["country", "name_field"]).loc[lambda x: x > 1].to_csv(
        f"{OUTPUT_DATA_FOLDER}/country_name_counts.csv", header=True
    )

    # count by country, name and status
    gdf.value_counts(["country", "name_field", "status_field"]).loc[
        lambda x: x > 1
    ].to_csv(f"{OUTPUT_DATA_FOLDER}/country_name_status_counts.csv", header=True)


if __name__ == "__main__":
    check_duplicated_areas()
