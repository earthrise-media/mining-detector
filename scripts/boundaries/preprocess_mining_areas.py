"""
Preprocess mining areas for the website:
- Prepares the mining data, concatenaing it into single gdf for use in this script
- Saves the mining data as one file per year, with simplified geometries
- Intersects mining polygons with administrative boundaries.
- Intersects these with areas of interest (indigenous territories, protected areas).
- Calculates area summaries, yearly timeseries.
- Overlays mining polygons with illegality categories.
"""

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/preprocess_mining_areas.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
#     "numpy",
#     "pandas",
# ]
# ///

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from constants import (
    MINING_DIFFERENCES_FILES,
    MINING_YEARS_QUARTERS,
    generate_mining_simplified_filename,
)
from shapely import set_precision

ADMIN_AREAS_GEOJSON = "data/boundaries/subnational_admin/out/admin_areas.geojson"
ILLEGALITY_AREAS_GEOJSON = "data/boundaries/illegality/out/illegality_v1_areas.geojson"
PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER = (
    "data/boundaries/protected_areas_and_indigenous_territories/out"
)
INDIGENOUS_TERRITORIES_GEOJSON = f"{PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER}/indigenous_territories.geojson"
PROTECTED_AREAS_GEOJSON = (
    f"{PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER}/protected_areas.geojson"
)
NATIONAL_ADMIN_FOLDER = "data/boundaries/national_admin/out"
NATIONAL_ADMIN_GEOJSON = f"{NATIONAL_ADMIN_FOLDER}/national_admin.geojson"
SUBNATIONAL_ADMIN_FOLDER = "data/boundaries/subnational_admin/out"
SUBNATIONAL_ADMIN_GEOJSON = f"{SUBNATIONAL_ADMIN_FOLDER}/admin_areas_display.geojson"


def simplify_gdf(gdf):
    # create a copy with simplified geometries and columns, for display in the website
    gdf_simplified = gdf.copy()
    gdf_simplified["geometry"] = gdf_simplified["geometry"].simplify(
        tolerance=0.0001, preserve_topology=True
    )
    gdf_simplified["geometry"] = gdf_simplified["geometry"].apply(
        lambda geom: set_precision(geom, grid_size=1e-6)
    )
    return gdf_simplified


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


def intersect_and_calculate_areas(mining_gdf, gdf_to_intersect, mining_area_col_name):
    if mining_gdf.crs != gdf_to_intersect.crs:
        print(
            f"CRS mismatch: mining_gdf ({mining_gdf.crs}) vs gdf_to_intersect ({gdf_to_intersect.crs})"
        )
        print("Reprojecting gdf_to_intersect to match mining_gdf...")
        gdf_to_intersect = gdf_to_intersect.to_crs(mining_gdf.crs)

    # calculate original areas (before split)
    mining_gdf = calculate_area_using_utm(mining_gdf, "original_area_ha", "hectares")
    print("Total mining area sum (ha):")
    print(mining_gdf["original_area_ha"].sum())

    print("Performing intersection...")
    intersected = gpd.overlay(mining_gdf, gdf_to_intersect, how="intersection")

    # calculate areas after intersection
    intersected = calculate_area_using_utm(
        intersected, "intersected_area_ha", "hectares"
    )
    print("Intersected mining area sum (ha):")
    print(intersected["intersected_area_ha"].sum())

    # calculate area statistics
    intersected["area_ratio"] = (
        intersected["intersected_area_ha"] / intersected["original_area_ha"]
    )

    intersected[mining_area_col_name] = (
        intersected[mining_area_col_name] * intersected["area_ratio"]
    )

    return intersected


def ensure_output_path_exists(output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def save_to_geojson(gdf, output_file, id_column="id"):
    ensure_output_path_exists(output_file)
    print(f"Saving {output_file}")

    # we're doing the steps below to add a top-level id property to the geojson,
    # instead of just `gdf.to_file(output_file, driver="GeoJSON", encoding="utf-8")`

    # convert to GeoJSON dictionary
    geojson_dict = json.loads(gdf.to_json())

    # make sure ids are unique
    print(len(gdf))
    print(gdf[id_column].nunique())
    duplicates = gdf[gdf.duplicated(subset=id_column, keep=False)]
    print(duplicates)
    assert len(gdf) == gdf[id_column].nunique()

    # move id from properties to top level
    for feature in geojson_dict["features"]:
        if id_column in feature["properties"]:
            feature[id_column] = feature["properties"][id_column]

    # save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, ensure_ascii=False)


def intersect_with_areas_of_interest_and_summarize(
    mining_admin_intersect_gdf,
    areas_of_interest_gdf,
    ignore_if_outside_country,
):
    """
    Intersects areas of interest (countries, Indigenous Territories, protected areas)
    with the already admin-intersected mining areas. Summarizes the areas
    by area of interest and admin boundaries, to use in the Mining Calculator requests.

    The ignore_if_outside_country argument makes the function ignore in case the area's
    country code doesn't match the parent country code. This is useful for areas that are
    on the border and might fall outside of the country's boundaries.
    """
    # intersect with area of interest, calculate areas
    intersected_with_areas_of_interest = intersect_and_calculate_areas(
        mining_admin_intersect_gdf, areas_of_interest_gdf, "admin_Mined area (ha)"
    )
    if ignore_if_outside_country:
        intersected_with_areas_of_interest = intersected_with_areas_of_interest[
            # this covers the entire Amazon case, which has no country_code
            (intersected_with_areas_of_interest["country_code"].isna())
            | (
                intersected_with_areas_of_interest["admin_country_code"]
                == intersected_with_areas_of_interest["country_code"]
            )
        ]

    # rename mined area col to use it instead of old intersected_area_ha col
    intersected_with_areas_of_interest["intersected_area_ha"] = (
        intersected_with_areas_of_interest["admin_Mined area (ha)"]
    )

    # get summary statistics
    summary = intersected_with_areas_of_interest.groupby(
        [
            "id",
            "admin_country",
            "admin_country_code",
            "admin_name_field",
            "admin_id_field",
            "admin_year",
            "admin_illegality_max",
        ]
    )[["intersected_area_ha"]].sum()

    return summary


def calculate_mining_area_timeseries(summary):
    # calculate mining area affected per year
    summary_mining_affected_area_ha_yearly = (
        summary.groupby(["id", "admin_year"])["intersected_area_ha"]
        .sum()
        .reset_index()
        .sort_values(by=["id", "admin_year"])
    )
    # cumulative sum for each id
    summary_mining_affected_area_ha_yearly["intersected_area_ha_cumulative"] = (
        summary_mining_affected_area_ha_yearly.groupby("id")[
            "intersected_area_ha"
        ].cumsum()
    )
    years_range = summary_mining_affected_area_ha_yearly["admin_year"].unique().tolist()
    years_range.sort()
    complete_years = pd.DataFrame(
        {
            "id": summary_mining_affected_area_ha_yearly["id"].unique(),
        }
    )
    # create a cartesian product of all id's with the complete set of years
    complete_years = complete_years.merge(
        pd.DataFrame({"admin_year": years_range}), on=None, how="cross"
    )
    # merge the complete years dataframe with the original summary
    summary_mining_affected_area_ha_yearly = pd.merge(
        complete_years,
        summary_mining_affected_area_ha_yearly,
        on=["id", "admin_year"],
        how="left",
    )
    summary_mining_affected_area_ha_yearly["intersected_area_ha_cumulative"] = (
        summary_mining_affected_area_ha_yearly.groupby("id")[
            "intersected_area_ha_cumulative"
        ]
        # fill missing nans by carrying forward the previous year's value
        .ffill()
        # then fillna zero for years before any detection
        .fillna(0)
    )

    return summary_mining_affected_area_ha_yearly


def prepare_for_mining_calculator_and_save(summary):
    def cleanup_region_id(region_id, country_code):
        # cleanup region_id to match mining calculator API standard
        return int(region_id.replace(country_code, ""))

    def cleanup_country_code(country_code):
        # cleanup country code to match mining calculator API standard
        return {"SR": "SU", "GY": "GU"}.get(country_code, country_code)

    def create_locations_dict(group):
        locations = []
        for _, row in group.iterrows():
            country_clean = cleanup_country_code(row["admin_country_code"])
            region_id_clean = cleanup_region_id(
                row["admin_id_field"], row["admin_country_code"]
            )

            # ignore if location has no affected area
            if row["intersected_area_ha"] <= 0:
                continue

            locations.append(
                {
                    "country": country_clean,
                    "regionId": region_id_clean,
                    "affectedArea": row["intersected_area_ha"],
                }
            )
        return {"locations": locations}

    result = (
        summary.reset_index()
        .groupby("id")
        .apply(create_locations_dict, include_groups=False)
        .to_dict()
    )

    return result


def overlay_max_category(
    illegality_gdf: gpd.GeoDataFrame,
    mining_gdf: gpd.GeoDataFrame,
    category_col: str,
) -> gpd.GeoDataFrame:
    """
    Overlays illegality_gdf with mining_gdf and assigns to each polygon in mining_gdf the maximum value
    of `category_col` from overlapping polygons in illegality_gdf.

    Parameters
    ----------
    illegality_gdf : GeoDataFrame
        Source GeoDataFrame containing the illegality data column.
    mining_gdf : GeoDataFrame
        Target GeoDataFrame to which max values will be added.
    category_col : str
        Column name in illegality_gdf containing numeric category values.

    Returns
    -------
    GeoDataFrame
        mining_gdf with an additional column '{category_col}_max' containing the max values.
    """

    # Ensure both GeoDataFrames share the same CRS
    if illegality_gdf.crs != mining_gdf.crs:
        illegality_gdf = illegality_gdf.to_crs(mining_gdf.crs)

    # Create spatial index for illegality_gdf if it doesn't exist
    illegality_sindex = illegality_gdf.sindex

    # Pre-extract geometries and values for faster access
    illegality_geoms = illegality_gdf.geometry.values
    illegality_vals = illegality_gdf[category_col].values

    # Initialize result array with NaN
    max_vals = np.full(len(mining_gdf), np.nan)

    # Process each mining polygon
    for idx, mining_geom in enumerate(mining_gdf.geometry):
        # Use spatial index to find potential matches (bounding box intersection)
        possible_matches_idx = list(illegality_sindex.intersection(mining_geom.bounds))

        if not possible_matches_idx:
            continue

        # Check actual intersections and find max value
        max_val = np.nan
        for ill_idx in possible_matches_idx:
            if mining_geom.intersects(illegality_geoms[ill_idx]):
                val = illegality_vals[ill_idx]
                if np.isnan(max_val) or val > max_val:
                    max_val = val

        max_vals[idx] = max_val

    # Copy mining_gdf and assign new column
    mining_gdf_out = mining_gdf.copy()
    mining_gdf_out[f"{category_col}_max"] = max_vals
    # We need to fill with 0 because the dataframe gets grouped by this column later,
    # and if it is null it will dissappear
    mining_gdf_out[f"{category_col}_max"] = mining_gdf_out[
        f"{category_col}_max"
    ].fillna(0)
    return mining_gdf_out


if __name__ == "__main__":
    admin_areas_gdf = gpd.read_file(ADMIN_AREAS_GEOJSON)
    illegality_areas_gdf = gpd.read_file(ILLEGALITY_AREAS_GEOJSON)

    # load all mining data
    all_mining_gdfs = []
    for i in range(0, len(MINING_YEARS_QUARTERS)):
        # load geodataframes
        current_year = MINING_YEARS_QUARTERS[i]
        current_gdf = gpd.read_file(MINING_DIFFERENCES_FILES[current_year])
        current_gdf["year"] = current_year  # add year column

        # simplify
        gdf_simplified = simplify_gdf(current_gdf)

        # cleanup and save
        gdf_simplified = gdf_simplified.drop(columns=["Polygon area (ha)"])
        output_file = generate_mining_simplified_filename(current_year)
        ensure_output_path_exists(output_file)
        gdf_simplified.to_file(output_file, driver="GeoJSON")
        print(f"Created: {output_file}")

        all_mining_gdfs.append(current_gdf)

    # combine individual frames (one per year quarter) into single gdf
    mining_gdf = gpd.pd.concat(all_mining_gdfs, ignore_index=True)

    # overlay illegality data
    mining_gdf = overlay_max_category(illegality_areas_gdf, mining_gdf, "illegality")

    # intersect mining with admin boundaries and calculate areas (once per mining file)
    intersected_with_admin = intersect_and_calculate_areas(
        mining_gdf, admin_areas_gdf, "Mined area (ha)"
    )
    # prefix columns with admin_
    intersected_with_admin.columns = [
        "admin_" + col if col != "geometry" else col
        for col in intersected_with_admin.columns
    ]

    datasets_to_process = [
        {
            "name": "national_admin",
            "file": NATIONAL_ADMIN_GEOJSON,
            # "output_folder": NATIONAL_ADMIN_FOLDER,
            # "output_subfolder": "mining_by_national_admin",
            "ignore_if_outside_country": True,
        },
        {
            "name": "subnational_admin",
            "file": SUBNATIONAL_ADMIN_GEOJSON,
            # "output_folder": SUBNATIONAL_ADMIN_FOLDER,
            # "output_subfolder": "mining_by_subnational_admin",
            "ignore_if_outside_country": True,
        },
        {
            "name": "indigenous_territories",
            "file": INDIGENOUS_TERRITORIES_GEOJSON,
            # "output_folder": PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER,
            # "output_subfolder": "mining_by_indigenous_territories",
            "ignore_if_outside_country": True,
        },
        {
            "name": "protected_areas",
            "file": PROTECTED_AREAS_GEOJSON,
            # "output_folder": PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER,
            # "output_subfolder": "mining_by_protected_areas",
            "ignore_if_outside_country": True,
        },
    ]

    # process each dataset
    for dataset in datasets_to_process:
        gdf = gpd.read_file(dataset["file"])

        summary = intersect_with_areas_of_interest_and_summarize(
            mining_admin_intersect_gdf=intersected_with_admin,
            areas_of_interest_gdf=gdf,
            ignore_if_outside_country=dataset["ignore_if_outside_country"],
        )
        summary_illegality = (
            summary.groupby(["id", "admin_illegality_max"])["intersected_area_ha"]
            .sum()
            .round(2)
            .reset_index()
            .rename(columns={"intersected_area_ha": "mining_affected_area"})
        )
        illegality_by_id = (
            summary_illegality.groupby("id")
            .apply(
                lambda g: g[["admin_illegality_max", "mining_affected_area"]].to_dict(
                    "records"
                )
            )
            .to_dict()
        )

        summary_mining_affected_area_ha_yearly = calculate_mining_area_timeseries(
            summary
        )
        # save yearly summary to a json
        summary_mining_affected_area_ha_yearly.to_json(
            dataset["file"].replace(".geojson", "_yearly.json"),
            index=False,
            orient="records",
        )

        result = prepare_for_mining_calculator_and_save(summary)

        # transform json result into dataframe
        summary_mining_affected_area_ha = summary.groupby("id")[
            "intersected_area_ha"
        ].sum()
        result_df = pd.DataFrame(
            [
                {
                    "id": id,
                    "locations": v["locations"],
                    "mining_affected_area_ha": summary_mining_affected_area_ha[id],
                    "illegality_areas": [
                        {
                            **x,
                            "mining_affected_area_pct": round(
                                x["mining_affected_area"]
                                / summary_mining_affected_area_ha[id],
                                3,
                            ),
                        }
                        for x in illegality_by_id.get(id, [])
                    ],
                }
                for id, v in result.items()
            ]
        )

        def group_and_sum_locations(locations):
            # group by country and regionId, sum affectedArea to reduce repetitions
            grouped = {}
            for loc in locations:
                key = (loc["country"], loc["regionId"])
                if key in grouped:
                    grouped[key]["affectedArea"] += loc["affectedArea"]
                else:
                    grouped[key] = loc.copy()

            # round affectedArea to 2 decimal places
            for loc in grouped.values():
                loc["affectedArea"] = round(loc["affectedArea"], 2)

            return list(grouped.values())

        result_df["locations"] = result_df["locations"].apply(group_and_sum_locations)

        # round results
        result_df["mining_affected_area_ha"] = result_df[
            "mining_affected_area_ha"
        ].round(2)
        # merge back to original gdf and save
        gdf_merged = gdf.merge(result_df, on="id", how="left")

        # save unfiltered
        save_to_geojson(
            gdf_merged,
            dataset["file"].replace(".geojson", "_impacts_unfiltered.geojson"),
            id_column="id",
        )
        # filter and save only areas with impact
        gdf_merged = gdf_merged[gdf_merged["mining_affected_area_ha"] > 0]
        save_to_geojson(
            gdf_merged,
            dataset["file"].replace(".geojson", "_impacts.geojson"),
            id_column="id",
        )
