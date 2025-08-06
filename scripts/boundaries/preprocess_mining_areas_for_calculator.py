# Preprocesses the mining areas, intersecting them with the admin regions,
# then intersects them with areas of interest (Indigenous territories and
# protected areas) and summarizes them, preparing for querying the Mining
# Calculator API.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/preprocess_mining_areas_for_calculator.py`.

# to list all geojsons in folder:
# find ./data/outputs/48px_v3.2-3.7ensemble/cumulative/ -name "*.geojson" -type f | sed 's|^\./||' | sort

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
# ]
# ///
import geopandas as gpd
from pathlib import Path

MINING_GEOJSONS_FOLDER = "data/outputs/48px_v3.2-3.7ensemble/cumulative"
MINING_GEOJSONS = [
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2018cumulative.geojson",
        2018,
    ),
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2019cumulative.geojson",
        2019,
    ),
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2020cumulative.geojson",
        2020,
    ),
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2021cumulative.geojson",
        2021,
    ),
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2022cumulative.geojson",
        2022,
    ),
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2023cumulative.geojson",
        2023,
    ),
    (
        "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2024cumulative.geojson",
        2024,
    ),
]
ADMIN_AREAS_GPKG = "data/boundaries/subnational_admin/out/admin_areas.gpkg"
ADMIN_OUTPUT_FOLDER = "data/boundaries/subnational_admin/out/mining_by_admin_areas"
PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER = (
    "data/boundaries/protected_areas_and_indigenous_territories/out"
)
INDIGENOUS_TERRITORIES_GEOJSON = f"{PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER}/indigenous_territories.geojson"
PROTECTED_AREAS_GEOJSON = (
    f"{PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER}/protected_areas.geojson"
)


def calculate_area_using_utm(gdf, area_col_name="area", unit="hectares"):
    # units can be "hectares", "square_km" or "acres"
    zone_min = 32718
    lon_min = -84
    delta_lon = 6
    n_zones = 9

    gdf_copy = gdf.copy()
    print("Calculating areas...")
    for i in range(n_zones):
        idx = gdf_copy.cx[
            lon_min + i * delta_lon : lon_min + (i + 1) * delta_lon, :
        ].index
        gdf_copy.loc[idx, area_col_name] = (
            gdf_copy.loc[idx]
            .to_crs(f"epsg:{zone_min + i}")
            .apply(lambda x: x.geometry.area / 1e4, axis=1)
        )

    if unit == "hectares":
        pass
    elif unit == "square_km":
        gdf_copy[area_col_name] = gdf_copy[area_col_name] / 100
    elif unit == "acres":
        gdf_copy[area_col_name] = gdf_copy[area_col_name] * 2.471054
    else:
        print(f"Error, unrecognized unit: {unit}")
        raise ValueError

    return gdf_copy


def intersect_and_calculate_areas(mining_gdf, gdf_to_intersect):
    if mining_gdf.crs != gdf_to_intersect.crs:
        print(
            f"CRS mismatch: mining_gdf ({mining_gdf.crs}) vs gdf_to_intersect ({gdf_to_intersect.crs})"
        )
        print("Reprojecting gdf_to_intersect to match mining_gdf...")
        gdf_to_intersect = gdf_to_intersect.to_crs(mining_gdf.crs)

    # calculate original areas (before split)
    # mining_gdf["original_area"] = mining_gdf.geometry.area
    mining_gdf = calculate_area_using_utm(mining_gdf, "original_area_ha", "hectares")

    print("Performing intersection...")
    intersected = gpd.overlay(mining_gdf, gdf_to_intersect, how="intersection")

    # calculate areas after intersection
    # intersected["intersected_area"] = intersected.geometry.area
    intersected = calculate_area_using_utm(
        intersected, "intersected_area_ha", "hectares"
    )

    # calculate area statistics
    intersected["area_ratio"] = (
        intersected["intersected_area_ha"] / intersected["original_area_ha"]
    )

    return intersected


def save_to_geojson(gdf, output_file):
    # ensure output directory exists
    output_path = Path(output_file)
    # save to file
    print(f"Saving {output_file}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_file, driver="GeoJSON", encoding="utf-8")


def intersect_with_it_or_pa_and_summarize(mining_admin_intersect_gdf, areas_of_interest_gdf, col_prefix, output_folder):
    """
    Intersects either Indigenous territories (IT) or protected areas (PA)
    with the already admin-intersected mining areas. Summarizes the areas 
    by IT/PA and admin boundaries, to use in the Mining Calculator requests.
    """
    # intersect with IT/PA, calculate areas
    intersected_with_areas_of_interest = intersect_and_calculate_areas(
        mining_admin_intersect_gdf, areas_of_interest_gdf
    )
    # prefix columns with it_
    intersected_with_areas_of_interest.columns = [
        f"{col_prefix}_{col}" if col != "geometry" and not col.startswith("admin") else col
        for col in intersected_with_areas_of_interest.columns
    ]

    # save to geojson
    save_to_geojson(
        intersected_with_areas_of_interest,
        f"{output_folder}/{file}",
    )
    # get summary statistics
    summary = intersected_with_areas_of_interest.groupby(
        [f"{col_prefix}_id", "admin_country", "admin_country_code", "admin_name_field"]
    )[[f"{col_prefix}_intersected_area_ha"]].sum()
    # save to csv
    summary.to_csv(
        f"{output_folder}/{file.replace('.geojson', '.csv')}"
    )


if __name__ == "__main__":
    admin_areas_gdf = gpd.read_file(ADMIN_AREAS_GPKG)
    indigenous_territories_gdf = gpd.read_file(INDIGENOUS_TERRITORIES_GEOJSON)
    protected_areas_gdf = gpd.read_file(PROTECTED_AREAS_GEOJSON)

    for file, year in MINING_GEOJSONS:
        # read mining file
        mining_file = f"{MINING_GEOJSONS_FOLDER}/{file}"
        print(f"Reading: ${mining_file}")
        mining_gdf = gpd.read_file(mining_file)

        # intersect mining with admin boundaries and calculate areas
        intersected_with_admin = intersect_and_calculate_areas(
            mining_gdf, admin_areas_gdf
        )
        # prefix columns with admin_
        intersected_with_admin.columns = [
            "admin_" + col if col != "geometry" else col
            for col in intersected_with_admin.columns
        ]
        # save to geojson
        save_to_geojson(intersected_with_admin, f"{ADMIN_OUTPUT_FOLDER}/{file}")

        # intersect with indigenous territories, calculate summary
        intersect_with_it_or_pa_and_summarize(
            mining_admin_intersect_gdf=intersected_with_admin,
            areas_of_interest_gdf=indigenous_territories_gdf,
            col_prefix="it",
            output_folder=f"{PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER}/mining_by_indigenous_territories",
        )

        # do the same for protected areas
        intersect_with_it_or_pa_and_summarize(
            mining_admin_intersect_gdf=intersected_with_admin,
            areas_of_interest_gdf=protected_areas_gdf,
            col_prefix="pt",
            output_folder=f"{PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER}/mining_by_protected_areas",
        )
