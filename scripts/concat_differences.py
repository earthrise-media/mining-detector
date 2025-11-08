# Concatenates all the geojsons that contain mining detection differences
# into a single file, with each detection tagged with the year, for use
# in the website.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/concat_differences.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
# ]
# ///
import geopandas as gpd
from shapely import set_precision
from pathlib import Path

# FIXME: use right folder after Ed's PR gets merged
# base_folder = "data/outputs/48px_v3.2-3.7ensemble/cumulative"
base_folder = "data/outputs/test-data"
output_folder = base_folder.replace("cumulative", "difference")

files = {
    202503: "amazon_basin_48px_v0.X_SSL4EO-MLPensemble2025Q3diff-clean-gt11ha.geojson",
    202502: "amazon_basin_48px_v0.X_SSL4EO-MLPensemble2025Q2diff-clean-gt11ha.geojson",
    202400: "amazon_basin_48px_v0.X_SSL4EO-MLPensemble2024-clean-diff-gt11ha.geojson",
    202300: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2023diff.geojson",
    202200: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2022diff.geojson",
    202100: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2021diff.geojson",
    202000: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2020diff.geojson",
    201900: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2019diff.geojson",
    201800: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2018cumulative.geojson",
}


def simplify_gdf_and_save(gdf, output_file):
    # create a copy with simplified geometries and columns, for display in the website
    gdf_simplified = gdf.copy()
    gdf_simplified["geometry"] = gdf_simplified["geometry"].simplify(
        tolerance=0.0001, preserve_topology=True
    )
    gdf_simplified["geometry"] = gdf_simplified["geometry"].apply(
        lambda geom: set_precision(geom, grid_size=1e-6)
    )

    # gdf_simplified['Polygon area (ha)'] = gdf_simplified['Polygon area (ha)'].round(2)
    # gdf_simplified['Mined area (ha)'] = gdf_simplified['Mined area (ha)'].round(2)
    gdf_simplified = gdf_simplified.drop(
        columns=["Polygon area (ha)", "Mined area (ha)"]
    )

    output_simplified_file = output_file.replace(".geojson", "_simplified.geojson")
    gdf_simplified.to_file(output_simplified_file, driver="GeoJSON")
    print(f"Created: {output_simplified_file}")


def concat_differences():
    """Combines and saves all years into a single GeoJSON,
    tagged with a 'year' property, also saves one file per year too."""
    years = sorted(files.keys())
    all_differences = []

    for i in range(0, len(years)):
        # load geodataframes
        current_year = years[i]
        current_gdf = gpd.read_file(f"{base_folder}/{files[current_year]}")
        current_gdf["year"] = current_year

        # simplify and save
        simplify_gdf_and_save(
            current_gdf, f"{base_folder}/mining_{current_year}.geojson"
        )

        all_differences.append(current_gdf)

    # combine frames
    combined_gdf = gpd.pd.concat(all_differences, ignore_index=True)
    output_combined_file = f"{output_folder}/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2025Q3_all_differences.geojson"
    # ensure output directory exists
    output_path = Path(output_combined_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # save combined file
    combined_gdf.to_file(output_combined_file, driver="GeoJSON")
    print(f"Created: {output_combined_file}")

    # simplify and save
    simplify_gdf_and_save(combined_gdf, output_combined_file)


if __name__ == "__main__":
    concat_differences()
