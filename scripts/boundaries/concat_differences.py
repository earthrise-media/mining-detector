# Concatenates all the geojsons that contain mining detection differences
# into a single file, with each detection tagged with the year, for use
# in the website.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/concat_differences.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
# ]
# ///
import geopandas as gpd
from shapely import set_precision
from pathlib import Path
from constants import (
    MINING_DIFFERENCES_FILES,
    MINING_YEARS_QUARTERS,
    MINING_COMBINED_FILE,
    GENERATE_MINING_SIMPLIFIED_FILENAME,
)

def simplify_gdf_and_save(gdf, output_file):
    # create a copy with simplified geometries and columns, for display in the website
    gdf_simplified = gdf.copy()
    gdf_simplified["geometry"] = gdf_simplified["geometry"].simplify(
        tolerance=0.0001, preserve_topology=True
    )
    gdf_simplified["geometry"] = gdf_simplified["geometry"].apply(
        lambda geom: set_precision(geom, grid_size=1e-6)
    )

    gdf_simplified = gdf_simplified.drop(columns=["Polygon area (ha)"])

    # ensure output folder exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_simplified.to_file(output_file, driver="GeoJSON")
    print(f"Created: {output_file}")


def concat_differences():
    """Combines and saves all years into a single GeoJSON,
    tagged with a 'year' property, also saves one file per year too."""
    all_differences = []

    for i in range(0, len(MINING_YEARS_QUARTERS)):
        # load geodataframes
        current_year = MINING_YEARS_QUARTERS[i]
        current_gdf = gpd.read_file(MINING_DIFFERENCES_FILES[current_year])
        current_gdf["year"] = current_year

        # simplify and save
        simplify_gdf_and_save(
            current_gdf, GENERATE_MINING_SIMPLIFIED_FILENAME(current_year)
        )

        all_differences.append(current_gdf)

    # combine frames
    combined_gdf = gpd.pd.concat(all_differences, ignore_index=True)

    # save combined file
    combined_gdf.to_file(MINING_COMBINED_FILE, driver="GeoJSON")
    print(f"Created: {MINING_COMBINED_FILE}")

    # simplify and save
    simplify_gdf_and_save(combined_gdf, MINING_COMBINED_FILE)


if __name__ == "__main__":
    concat_differences()
