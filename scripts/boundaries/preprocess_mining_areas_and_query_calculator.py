# Preprocesses the mining areas, intersecting them with the admin regions,
# then intersects them with areas of interest (Indigenous territories and
# protected areas) and summarizes them, then queries the Mining
# Calculator API.

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/preprocess_mining_areas_and_query_calculator.py`.

# to list all geojsons in folder:
# find ./data/outputs/48px_v3.2-3.7ensemble/cumulative/ -name "*.geojson" -type f | sed 's|^\./||' | sort

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dotenv",
#     "geopandas",
#     "numpy",
#     "pandas",
#     "requests",
# ]
# ///
from dotenv import load_dotenv
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import time
import random
import hashlib
import glob

load_dotenv()

mining_calculator_api_key = os.getenv("MINING_CALCULATOR_API_KEY")

MINING_GEOJSONS_FOLDER = "data/outputs/48px_v3.2-3.7ensemble/difference"
MINING_GEOJSONS = [
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2018cumulative.geojson",
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2019cumulative.geojson",
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2020cumulative.geojson",
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2021cumulative.geojson",
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2022cumulative.geojson",
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2023cumulative.geojson",
    # "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2024cumulative.geojson",
    "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2024_all_differences.geojson",
]
ADMIN_AREAS_GEOJSON = "data/boundaries/subnational_admin/out/admin_areas.geojson"
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

with open("scripts/boundaries/mining_calculator_ignore.json") as f:
    # these are regions which are missing in the mining calculator and should be ignored,
    # otherwise the calculator throws an error when making a request
    REGIONS_TO_IGNORE = json.load(f)


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

    return intersected


def ensure_output_path_exists(output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def save_to_geojson(gdf, output_file, id_column):
    ensure_output_path_exists(output_file)
    print(f"Saving {output_file}")

    # we're doing the steps below to add a top-level id property to the geojson,
    # instead of just `gdf.to_file(output_file, driver="GeoJSON", encoding="utf-8")`

    # convert to GeoJSON dictionary
    geojson_dict = json.loads(gdf.to_json())

    # make sure ids are unique
    print(len(gdf))
    print(gdf["id"].nunique())
    duplicates = gdf[gdf.duplicated(subset="id", keep=False)]
    print(duplicates)
    # assert len(gdf) == gdf["id"].nunique()

    # move id from properties to top level
    for feature in geojson_dict["features"]:
        if "id" in feature["properties"]:
            feature["id"] = feature["properties"]["id"]

    # save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, ensure_ascii=False)


def get_cache_filename(locations):
    # create a hash of the locations data for consistent filename
    locations_str = json.dumps(locations, sort_keys=True)
    hash_obj = hashlib.md5(locations_str.encode())
    return f"{hash_obj.hexdigest()}.json"


def load_from_cache(cache_file):
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading cache file {cache_file}: {e}")
    return None


def save_to_cache(cache_file, data):
    ensure_output_path_exists(cache_file)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def save_error_to_cache(locations, cache_filename, error_info):
    error_file = f"./.tmp/mining_calculator/errors/{cache_filename}"
    ensure_output_path_exists(error_file)

    error_data = {
        "locations": locations,
        "error": error_info,
        "timestamp": time.time(),
    }

    with open(error_file, "w") as f:
        json.dump(error_data, f, indent=2)


def compile_error_report():
    """Read all error files and create a CSV of unique country-regionId combinations"""
    errors_dir = "./.tmp/mining_calculator/errors"
    error_files = glob.glob(f"{errors_dir}/*.json")

    if not error_files:
        print("No error files found")
        return

    all_locations = []
    for error_file in error_files:
        with open(error_file, "r") as f:
            error_data = json.load(f)
        all_locations.extend(error_data.get("locations", []))

    df = pd.DataFrame(all_locations)
    unique_combinations = (
        df[["country", "regionId"]]
        .drop_duplicates()
        .sort_values(["country", "regionId"])
    )

    csv_file = "./.tmp/mining_calculator/error_combinations.csv"
    unique_combinations.to_csv(csv_file, index=False)

    print(f"Error report saved to {csv_file}")
    print(
        f"Found {len(unique_combinations)} unique country-regionId combinations with errors"
    )


def get_mining_calculator_data(locations):
    cache_filename = get_cache_filename(locations)
    cache_file = f"./.tmp/mining_calculator/{cache_filename}"

    # try to load from cache first
    cached_data = load_from_cache(cache_file)
    if cached_data is not None:
        return cached_data.get("totalImpact")

    try:
        print("Making a request to the mining calculator...")
        response = requests.post(
            "https://miningcalculator.conservation-strategy.org/api/calculate",
            headers={
                "Content-Type": "application/json",
                "x-api-key": mining_calculator_api_key,
            },
            json={"locations": locations},
        )

        # wait if rate limited
        if response.status_code == 429:
            delay = random.uniform(35, 45)
            print(f"Rate limit hit (429). Waiting {delay:.2f} seconds...")
            time.sleep(delay)
            # retry
            return get_mining_calculator_data(locations)

        # raise an exception for other bad status codes
        response.raise_for_status()

        data = response.json()
        save_to_cache(cache_file, data)
        return data["totalImpact"]

    except requests.exceptions.RequestException as error:
        error_info = {"type": "RequestException", "message": str(error)}
        print(f"Error: {error} for {locations}")
        save_error_to_cache(locations, cache_filename, error_info)
    except KeyError as error:
        error_info = {"type": "KeyError", "message": str(error)}
        print(f"Key error: {error} for {locations}")
        save_error_to_cache(locations, cache_filename, error_info)
    except json.JSONDecodeError as error:
        error_info = {"type": "JSONDecodeError", "message": str(error)}
        print(f"JSON decode error: {error} for {locations}")
        save_error_to_cache(locations, cache_filename, error_info)


def intersect_with_areas_of_interest_and_summarize(
    mining_admin_intersect_gdf, areas_of_interest_gdf, output_file
):
    """
    Intersects areas of interest (countries, Indigenous Territories, protected areas)
    with the already admin-intersected mining areas. Summarizes the areas
    by area of interest and admin boundaries, to use in the Mining Calculator requests.
    """
    # intersect with area of interest, calculate areas
    intersected_with_areas_of_interest = intersect_and_calculate_areas(
        mining_admin_intersect_gdf, areas_of_interest_gdf
    )

    ensure_output_path_exists(output_file)
    # # save to geojson
    # save_to_geojson(
    #     intersected_with_areas_of_interest,
    #     output_file,
    # )
    # get summary statistics
    summary = intersected_with_areas_of_interest.groupby(
        [
            "id",
            "admin_country",
            "admin_country_code",
            "admin_name_field",
            "admin_id_field",
            "admin_year",
        ]
    )[["intersected_area_ha"]].sum()
    # save to csv
    summary.to_csv(output_file.replace(".geojson", ".csv"))

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


def enrich_summary_with_mining_calculator_and_save(summary, output_file):
    def cleanup_region_id(region_id, country_code):
        # cleanup region_id to match mining calculator API standard
        return int(region_id.replace(country_code, ""))

    def cleanup_country_code(country_code):
        # cleanup country code to match mining calculator API standard
        return {"SR": "SU", "GY": "GU"}.get(country_code, country_code)

    def create_locations_dict(group):
        locations = []
        exclusion_set = {
            (item["countryCode"], item["regionId"]) for item in REGIONS_TO_IGNORE
        }

        for _, row in group.iterrows():
            country_clean = cleanup_country_code(row["admin_country_code"])
            region_id_clean = cleanup_region_id(
                row["admin_id_field"], row["admin_country_code"]
            )

            # skip if this combination exists in the exclusion list
            if (country_clean, region_id_clean) in exclusion_set:
                continue

            # ignore if location has no affected area, areas <= 0 break the calculator
            if row["intersected_area_ha"] <= 0:
                continue

            # ignore Venezuala and French Guyana, not included in mining calculator
            if country_clean == "VE" or country_clean == "GF":
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

    # for key in result:
    #     locations = [
    #         x
    #         for x in result[key]["locations"]
    #         # calculator doesn't include Venezuela and French Guyana
    #         if x["country"] != "VE" and x["country"] != "GF"
    #     ]
    #     if len(locations):
    #         total_impact = get_mining_calculator_data(locations)
    #         result[key]["totalImpact"] = total_impact

    output_json = output_file.replace(".geojson", ".json")
    print(f"Saving to: {output_json}")
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    admin_areas_gdf = gpd.read_file(ADMIN_AREAS_GEOJSON)

    for file in MINING_GEOJSONS:
        mining_file = f"{MINING_GEOJSONS_FOLDER}/{file}"
        print(f"Reading: {mining_file}")
        mining_gdf = gpd.read_file(mining_file)

        # intersect mining with admin boundaries and calculate areas (once per mining file)
        intersected_with_admin = intersect_and_calculate_areas(
            mining_gdf, admin_areas_gdf
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
                "output_folder": NATIONAL_ADMIN_FOLDER,
                "output_subfolder": "mining_by_national_admin",
            },
            {
                "name": "subnational_admin",
                "file": SUBNATIONAL_ADMIN_GEOJSON,
                "output_folder": SUBNATIONAL_ADMIN_FOLDER,
                "output_subfolder": "mining_by_subnational_admin",
            },
            {
                "name": "indigenous_territories",
                "file": INDIGENOUS_TERRITORIES_GEOJSON,
                "output_folder": PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER,
                "output_subfolder": "mining_by_indigenous_territories",
            },
            {
                "name": "protected_areas",
                "file": PROTECTED_AREAS_GEOJSON,
                "output_folder": PROTECTED_AREAS_AND_INDIGENOUS_TERRITORIES_FOLDER,
                "output_subfolder": "mining_by_protected_areas",
            },
        ]

        # process each dataset
        for dataset in datasets_to_process:
            gdf = gpd.read_file(dataset["file"])
            output_file = (
                f"{dataset['output_folder']}/{dataset['output_subfolder']}/{file}"
            )

            summary = intersect_with_areas_of_interest_and_summarize(
                mining_admin_intersect_gdf=intersected_with_admin,
                areas_of_interest_gdf=gdf,
                output_file=output_file,
            )
            summary_mining_affected_area_ha = summary.groupby("id")[
                "intersected_area_ha"
            ].sum()

            summary_mining_affected_area_ha_yearly = calculate_mining_area_timeseries(
                summary
            )
            # save yearly summary to a json
            summary_mining_affected_area_ha_yearly.to_json(
                dataset["file"].replace(".geojson", "_yearly.json"),
                index=False,
                orient="records",
            )

            result = enrich_summary_with_mining_calculator_and_save(
                summary, output_file
            )

            # transform json result into dataframe
            result_df = pd.DataFrame(
                [
                    {
                        "id": id,
                        # "economic_impact_usd": (
                        #     v["totalImpact"] if "totalImpact" in v else None
                        # ),
                        "locations": v["locations"],
                        # we need to get the affected area from the original summary df,
                        # since the mining calculator doesn't return results for every possible area
                        "mining_affected_area_ha": summary_mining_affected_area_ha[id],
                    }
                    for id, v in result.items()
                ]
            )
            # round results
            # result_df["economic_impact_usd"] = result_df["economic_impact_usd"].round(2)
            result_df["mining_affected_area_ha"] = result_df[
                "mining_affected_area_ha"
            ].round(2)
            # merge back to original gdf and save
            gdf_merged = gdf.merge(result_df, on="id", how="left")
            # # Remove mining calculations from countries to ignore, since they don't have
            # # any mining calculator data in the API. Any mining calculations in them are artifacts of
            # # areas from other countries that might have overlapped.
            # gdf_merged["economic_impact_usd"] = np.where(
            #     gdf_merged["country"].isin(
            #         ["Venezuela", "FrenchGuiana", "French Guiana"]
            #     ),
            #     np.nan,
            #     gdf_merged["economic_impact_usd"],
            # )

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

    compile_error_report()
