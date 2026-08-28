"""
Preprocess mining areas for the website:
- Prepares the mining data, concatenaing it into single gdf for use in this script
- Saves the mining data as one file per year, with simplified geometries
- Intersects mining polygons with administrative boundaries.
- Intersects these with areas of interest (indigenous territories, protected areas).
- Calculates area summaries, yearly timeseries.
- Overlays mining polygons with illegality categories.

Note: raster snapshots are used as-is (cumulative extent at each period).
Per-period differences are derived at the timeseries step instead of via
vector overlay differences.
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
#     "rasterio",
# ]
# ///

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from constants import (
    COMBINED_MINING_FILE,
    ILLEGALITY_AREAS_GEOJSON,
    ILLEGALITY_DATA_UPDATED_AT,
    MINING_DIFFERENCES_FILES,
    MINING_RASTER_YEARS_QUARTERS,
    MINING_YEARS_QUARTERS,
    generate_mining_simplified_filename,
    generate_vectorized_raster_filename,
)
from rasterio.features import shapes
from shapely import set_precision
from shapely.geometry import shape

ADMIN_AREAS_GEOJSON = "data/boundaries/subnational_admin/out/admin_areas.geojson"
INDIGENOUS_TERRITORIES_GEOJSON = "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories.geojson"
PROTECTED_AREAS_GEOJSON = "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas.geojson"
NATIONAL_ADMIN_GEOJSON = "data/boundaries/national_admin/out/national_admin.geojson"
SUBNATIONAL_ADMIN_GEOJSON = (
    "data/boundaries/subnational_admin/out/admin_areas_display.geojson"
)

# Global cylindrical equal-area on WGS84. Area is exact at every latitude, so one
# CRS covers the whole basin and a geometry measures the same however it was split
# -- unlike per-zone UTM, where the answer depends on which zone a piece lands in.
# Shape and distance are badly distorted here; only ever measure area with it.
EQUAL_AREA_CRS = "EPSG:6933"

ENTIRE_AMAZON_ID = "AMAZ"

# How far a fragment may be from the nearest subnational area *in its own country*
# before we decline to re-attribute it. The layers disagree over a sliver roughly
# 111 m wide, so 5 km is generous; past it we would be guessing which region the
# mining belongs to rather than correcting a border artefact.
REPAIR_MAX_DISTANCE_M = 5_000

# The subnational layers do not quite reach their countries' national outlines, so a
# fragment can land in a hole with no region to attribute it to. Those are dropped,
# which is tolerable only while they stay a rounding error -- past this share of all
# mining area, stop rather than drop quietly. Typically ~0.0004%.
#
# Measured against the whole dataset, not against the fragments needing repair: that
# population shrinks as attribution improves, so using it would make this guard fire
# more readily the better the pipeline got.
UNATTRIBUTABLE_SHARE_LIMIT = 0.0001


def load_vectorized_raster(year):
    """
    Loads the pre-vectorized mining raster for a given year/quarter.
    These are produced by convert_rasters_to_vector.py.
    """
    vector_file = generate_vectorized_raster_filename(year)
    if not Path(vector_file).exists():
        raise FileNotFoundError(
            f"Missing vectorized raster: {vector_file}. "
            "Run convert_rasters_to_vector.py first."
        )
    print(f"Loading {vector_file}")
    gdf = gpd.read_file(vector_file)
    gdf["year"] = year  # add year column
    return gdf


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


def calculate_area(gdf, area_col_name="area", unit="hectares"):
    # units can be "hectares", "square_km" or "acres"
    gdf_copy = gdf.copy()
    print("Calculating areas...")
    gdf_copy[area_col_name] = gdf_copy.to_crs(EQUAL_AREA_CRS).area / 1e4

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


JURISDICTION_OUTPUT_SUFFIXES = (
    "_yearly.json",
    "_impacts.geojson",
    "_impacts_unfiltered.geojson",
    "_impacts_unfiltered_dict.json",
    "_impacts_yearly.csv",
)


def quarantine_previous_outputs(dataset_files):
    """Move the previous run's jurisdiction outputs aside before regenerating.

    A run that dies must not leave outputs that look current: `upload_data_to_s3.py`
    would publish them without complaint, alongside the display geometry this script
    writes earlier. Absent files make it fail loudly instead.

    Kept as `.stale` rather than deleted, so a failed run costs nothing.
    """
    moved = 0
    for dataset_file in dataset_files:
        for suffix in JURISDICTION_OUTPUT_SUFFIXES:
            path = Path(dataset_file.replace(".geojson", suffix))
            if path.exists():
                path.replace(path.with_suffix(path.suffix + ".stale"))
                moved += 1
    print(f"Set aside {moved} outputs from the previous run as .stale")


def split_by_national_boundaries(mining_gdf, national_admin_gdf):
    """Split mining so every fragment lies in exactly one country.

    Adds `country_code_auth`: the country from the national layer, which is the
    authority for country attribution. The subnational layer also carries a
    country code, but it is a different source and the two disagree along
    international borders -- and a fragment bounded only by the subnational layer
    can straddle a national border, leaving no single country to attribute it to.
    Splitting here makes that well-defined before anything downstream asks.

    Mining outside every national polygon is dropped: a thin coastal fringe where
    the basin outline reaches past the land boundaries, and false positives over
    water.
    """
    countries = national_admin_gdf[national_admin_gdf["id"] != ENTIRE_AMAZON_ID]
    countries = countries[["country_code", "geometry"]].rename(
        columns={"country_code": "country_code_auth"}
    )
    split = gpd.overlay(mining_gdf, countries, how="intersection")
    # identifies a fragment across the overlays that follow, so the several rows
    # one fragment produces can be recognised as the same piece of ground
    split["mining_fragment_id"] = range(len(split))
    print(f"Split {len(mining_gdf):,} mining fragments into {len(split):,} by country")
    return split


def prefer_domestic_rows(gdf, fragment_col, area_col, domestic, always_keep=None):
    """Pick the rows whose area of interest lies in the fragment's own country.

    Subnational and jurisdiction polygons overlap across international borders, so
    one fragment can appear under two countries' polygons and be counted twice.
    Keeping the domestic row resolves that. Where nothing domestic covers the
    fragment its largest row is kept, so it is de-duplicated rather than dropped.
    """
    if always_keep is None:
        always_keep = pd.Series(False, index=gdf.index)
    keep = domestic | always_keep
    orphaned = ~domestic.groupby(gdf[fragment_col]).transform("any") & ~always_keep
    if orphaned.any():
        largest = gdf.loc[orphaned].groupby(fragment_col)[area_col].idxmax()
        keep.loc[largest] = True
    return keep


def repair_admin_country_codes(intersected_with_admin, admin_areas_gdf):
    """Re-attribute fragments whose subnational and national country codes differ.

    Along international borders the subnational layer (partner sources) and the
    national layer (GADM level 0) disagree about which country a point is in. The
    country becomes the one whose national polygon contains the fragment, and the
    region the nearest subnational area *within that country*, so
    `ignore_if_outside_country` passes for every row and stays in place as a guard
    instead of discarding the fragment.
    """
    auth = intersected_with_admin["admin_country_code_auth"]
    mismatched = intersected_with_admin["admin_country_code"] != auth
    count = int(mismatched.sum())
    area = float(intersected_with_admin.loc[mismatched, "admin_intersected_area_ha"].sum())
    if count == 0:
        print("Country attribution: layers agree everywhere, nothing to repair")
        return intersected_with_admin

    # Logged every run: repairing the disagreement conceals it, so this is the only
    # signal of how far the two boundary sources have drifted. Summed over periods,
    # so the same ground counts once per snapshot.
    print(
        f"Country attribution: repairing {count:,} fragments ({area:,.2f} ha, summed "
        "across periods) where the subnational and national layers disagree"
    )

    # Distances need a metric CRS. Equal-area is not conformal, so lengths here are
    # off by up to ~15% at these latitudes -- irrelevant for a 5 km sanity bound.
    candidates_by_country = {
        cc: g.to_crs(EQUAL_AREA_CRS)
        for cc, g in admin_areas_gdf[
            ["country", "country_code", "id_field", "name_field", "geometry"]
        ].groupby("country_code")
    }

    repaired = intersected_with_admin.copy()
    unattributable = []
    for country_code, group in repaired[mismatched].groupby("admin_country_code_auth"):
        candidates = candidates_by_country.get(country_code)
        if candidates is None:
            unattributable.extend(group.index)
            continue

        nearest = gpd.sjoin_nearest(
            group[["geometry"]].to_crs(EQUAL_AREA_CRS),
            candidates,
            how="left",
            max_distance=REPAIR_MAX_DISTANCE_M,
        )
        # ties can return several matches for one fragment; one is enough, and
        # keeping both would double-count its area downstream
        nearest = nearest[~nearest.index.duplicated(keep="first")]

        unattributable.extend(nearest.index[nearest["country_code"].isna()])
        found = nearest.index[nearest["country_code"].notna()]
        repaired.loc[found, "admin_country"] = nearest.loc[found, "country"]
        repaired.loc[found, "admin_country_code"] = nearest.loc[found, "country_code"]
        repaired.loc[found, "admin_id_field"] = nearest.loc[found, "id_field"]
        repaired.loc[found, "admin_name_field"] = nearest.loc[found, "name_field"]

    if unattributable:
        lost = float(repaired.loc[unattributable, "admin_intersected_area_ha"].sum())
        all_area = float(repaired["admin_intersected_area_ha"].sum())
        share = lost / all_area if all_area else 0.0
        if share > UNATTRIBUTABLE_SHARE_LIMIT:
            by_country = (
                repaired.loc[unattributable]
                .groupby("admin_country_code_auth")["admin_intersected_area_ha"]
                .agg(["size", "sum"])
            )
            detail = "; ".join(
                f"{cc}: {int(r['size'])} fragments, {r['sum']:,.2f} ha"
                for cc, r in by_country.iterrows()
            )
            raise ValueError(
                f"{lost:,.2f} ha ({share:.3%} of all mining area) has no subnational "
                f"area within {REPAIR_MAX_DISTANCE_M} m ({detail}). The "
                "boundary layers have diverged by more than a border sliver -- check the "
                "incoming boundaries rather than raising the limit, which would drop real "
                "mining without saying so."
            )
        print(
            f"  {len(unattributable):,} fragments ({lost:,.2f} ha, {share:.4%} of all "
            "mining area) sit in a gap in their country's subnational coverage, dropped"
        )
        repaired = repaired.drop(index=unattributable)

    # Localises a partial repair here, instead of letting the totals assertion fail
    # later and point at the wrong stage.
    remaining = int(
        (repaired["admin_country_code"] != repaired["admin_country_code_auth"]).sum()
    )
    if remaining:
        raise ValueError(
            f"{remaining:,} fragments still disagree with their country after "
            "re-attribution. The repair itself is at fault, not anything downstream."
        )
    return repaired


def assert_country_totals_tie(summary_yearly, tolerance_ha=50.0):
    """The countries must sum to the Entire Amazon row, in every period.

    Raises rather than warns: a break here still yields plausible numbers and a
    clean log, so a warning would go unnoticed.
    """
    totals = summary_yearly.groupby(["admin_year", summary_yearly["id"] == ENTIRE_AMAZON_ID])[
        "intersected_area_ha_cumulative"
    ].sum()
    for year in summary_yearly["admin_year"].unique():
        basin = totals.get((year, True), 0.0)
        countries = totals.get((year, False), 0.0)
        if abs(countries - basin) > tolerance_ha:
            raise ValueError(
                f"Country totals do not tie to the basin total for {year}: "
                f"countries {countries:,.2f} ha vs {ENTIRE_AMAZON_ID} {basin:,.2f} ha "
                f"({countries - basin:+,.2f} ha, tolerance {tolerance_ha})"
            )
    print(f"Country totals tie to {ENTIRE_AMAZON_ID} within {tolerance_ha} ha in every period")


def intersect_and_calculate_areas(mining_gdf, gdf_to_intersect, mining_area_col_name):
    if mining_gdf.crs != gdf_to_intersect.crs:
        print(
            f"CRS mismatch: mining_gdf ({mining_gdf.crs}) vs gdf_to_intersect ({gdf_to_intersect.crs})"
        )
        print("Reprojecting gdf_to_intersect to match mining_gdf...")
        gdf_to_intersect = gdf_to_intersect.to_crs(mining_gdf.crs)

    # calculate original areas (before split)
    mining_gdf = calculate_area(mining_gdf, "original_area_ha", "hectares")
    print("Total mining area sum (ha):")
    print(mining_gdf["original_area_ha"].sum())

    print("Performing intersection...")
    intersected = gpd.overlay(mining_gdf, gdf_to_intersect, how="intersection")

    # calculate areas after intersection
    intersected = calculate_area(intersected, "intersected_area_ha", "hectares")
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
    ignore_if_outside_of_state,
):
    """
    Intersects areas of interest (countries, Indigenous Territories, protected areas)
    with the already admin-intersected mining areas. Summarizes the areas
    by area of interest and admin boundaries, to use in the Mining Calculator requests.

    Note: the mining areas here are snapshots (cumulative extent per period), so the
    resulting summary is cumulative per year rather than incremental.

    The ignore_if_outside_country argument makes the function ignore in case the area's
    country code doesn't match the parent country code. This is useful for areas that are
    on the border and might fall outside of the country's boundaries.

    The ignore_if_outside_of_state does the same, by checking if the area being intersected
    starts with the same code as the parent area.
    """
    # intersect with area of interest, calculate areas
    intersected_with_areas_of_interest = intersect_and_calculate_areas(
        mining_admin_intersect_gdf, areas_of_interest_gdf, "admin_Mined area (ha)"
    )
    if ignore_if_outside_country:
        # Areas of interest overlap across international borders, so one mining
        # fragment can appear under two countries' polygons. Prefer the row for the
        # fragment's own country and drop the duplicate -- but where *no* domestic
        # polygon covers it, keep its largest row rather than discarding it.
        aoi = intersected_with_areas_of_interest
        intersected_with_areas_of_interest = aoi[
            prefer_domestic_rows(
                aoi,
                fragment_col="admin_mining_fragment_id",
                area_col="intersected_area_ha",
                domestic=aoi["admin_country_code"] == aoi["country_code"],
                # the entire Amazon has no country code, and is always kept
                always_keep=aoi["country_code"].isna(),
            )
        ]
    if ignore_if_outside_of_state:
        mask = [
            id_val.startswith(admin_val)
            for id_val, admin_val in zip(
                intersected_with_areas_of_interest["admin_id_field"],
                intersected_with_areas_of_interest["id_field"],
                strict=True,
            )
        ]
        intersected_with_areas_of_interest = intersected_with_areas_of_interest[mask]

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
    """
    The input summary comes from snapshot (cumulative) mining extents, so the
    per-year value is already the cumulative area. The incremental area for each
    year is derived here by subtracting the previous period's cumulative value.
    """
    # cumulative mining area affected at each period
    summary_mining_affected_area_ha_yearly = (
        summary.groupby(["id", "admin_year"])["intersected_area_ha"]
        .sum()
        .reset_index()
        .sort_values(by=["id", "admin_year"])
        .rename(columns={"intersected_area_ha": "intersected_area_ha_cumulative"})
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
    ).sort_values(by=["id", "admin_year"])

    summary_mining_affected_area_ha_yearly["intersected_area_ha_cumulative"] = (
        summary_mining_affected_area_ha_yearly.groupby("id")[
            "intersected_area_ha_cumulative"
        ]
        # fill missing nans by carrying forward the previous period's value
        .ffill()
        # then fillna zero for years before any detection
        .fillna(0)
    )

    # derive the per-period difference from the cumulative snapshots
    summary_mining_affected_area_ha_yearly["intersected_area_ha"] = (
        summary_mining_affected_area_ha_yearly.groupby("id")[
            "intersected_area_ha_cumulative"
        ]
        .diff()
        # the first period has no previous snapshot, so its increment is its own total
        .fillna(summary_mining_affected_area_ha_yearly["intersected_area_ha_cumulative"])
    )
    # snapshots can shrink slightly between periods (reclassification/noise),
    # clip negatives so the increments stay meaningful
    summary_mining_affected_area_ha_yearly["intersected_area_ha"] = (
        summary_mining_affected_area_ha_yearly["intersected_area_ha"].clip(lower=0)
    )

    summary_mining_affected_area_ha_yearly = summary_mining_affected_area_ha_yearly[
        ["id", "admin_year", "intersected_area_ha", "intersected_area_ha_cumulative"]
    ].reset_index(drop=True)

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


def summarize_latest_snapshot(summary):
    """
    Since the summary is built from cumulative snapshots, summing across all years
    would double-count. Reduce to the latest snapshot per group instead.
    """
    flat = summary.reset_index()
    group_cols = [
        "id",
        "admin_country",
        "admin_country_code",
        "admin_name_field",
        "admin_id_field",
        "admin_illegality_max",
    ]
    latest = (
        flat.sort_values("admin_year")
        .groupby(group_cols, as_index=False)
        .last()
        .drop(columns=["admin_year"])
    )
    return latest.set_index(group_cols)


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
    print("Overlaying with illegality data...")

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
    # before any expensive work, so a run that dies leaves no outputs that could
    # be mistaken for current ones
    quarantine_previous_outputs(
        [
            NATIONAL_ADMIN_GEOJSON,
            SUBNATIONAL_ADMIN_GEOJSON,
            INDIGENOUS_TERRITORIES_GEOJSON,
            PROTECTED_AREAS_GEOJSON,
        ]
    )

    admin_areas_gdf = gpd.read_file(ADMIN_AREAS_GEOJSON)
    illegality_areas_gdf = gpd.read_file(ILLEGALITY_AREAS_GEOJSON)

    # load all mining data
    all_mining_gdfs = []
    full_resolution_mining_gdfs = []
    for i in range(0, len(MINING_YEARS_QUARTERS)):
        # load geodataframes
        current_year = MINING_YEARS_QUARTERS[i]
        current_gdf = gpd.read_file(MINING_DIFFERENCES_FILES[current_year])
        current_gdf["year"] = current_year  # add year column

        full_resolution_mining_gdfs.append(current_gdf)
        # simplify
        gdf_simplified = simplify_gdf(current_gdf)

        # cleanup and save
        output_file = generate_mining_simplified_filename(current_year)
        ensure_output_path_exists(output_file)
        gdf_simplified.to_file(output_file, driver="GeoJSON")
        print(f"Created: {output_file}")

        all_mining_gdfs.append((current_year, current_gdf))

    combined_mining_gdf = gpd.GeoDataFrame(pd.concat(full_resolution_mining_gdfs, ignore_index=True))
    ensure_output_path_exists(COMBINED_MINING_FILE)
    combined_mining_gdf.to_file(COMBINED_MINING_FILE, driver="GeoJSON")

    start = time.time()
    # rasters are vectorized ahead of time by convert_rasters_to_vector.py
    all_mining_raster_gdfs = [
        load_vectorized_raster(year) for year in MINING_RASTER_YEARS_QUARTERS
    ]
    print(f"Loading vectorized rasters took {time.time() - start:.1f}s")

    # use the rasterized snapshots as they are: each one is the full mining extent
    # for that period. Differences between periods are derived later, at the
    # timeseries step, instead of via vector overlay differences.
    mining_gdf = gpd.pd.concat(all_mining_raster_gdfs, ignore_index=True)
    # split by country first, so each fragment has one authoritative country to
    # attribute against once the subnational layer disagrees further down
    mining_gdf = split_by_national_boundaries(
        mining_gdf, gpd.read_file(NATIONAL_ADMIN_GEOJSON)
    )
    # need to calculate area as it is not present in original raster files
    mining_gdf = calculate_area(mining_gdf, "Mined area (ha)", "hectares")

    # for illegality, use a cutoff date, which is when illegality data was produced
    mining_gdf_for_illegality = mining_gdf[mining_gdf.year <= ILLEGALITY_DATA_UPDATED_AT]
    # take the rest of the mining data and store in variable
    mining_gdf_rest = mining_gdf[mining_gdf.year > ILLEGALITY_DATA_UPDATED_AT]

    # overlay illegality data
    mining_gdf_with_illegality = overlay_max_category(
        illegality_areas_gdf, mining_gdf_for_illegality, "illegality"
    )
    # ensure the illegality_max column exists in rest gdf with a -1 value, to be ignored
    if len(mining_gdf_rest) > 0:
        mining_gdf_rest["illegality_max"] = -1

    # concat with rest of data
    mining_gdf = gpd.pd.concat(
        [mining_gdf_with_illegality, mining_gdf_rest], ignore_index=True
    )

    # intersect mining with admin boundaries and calculate areas (once per mining file)
    intersected_with_admin = intersect_and_calculate_areas(
        mining_gdf, admin_areas_gdf, "Mined area (ha)"
    )
    # subnational polygons from neighbouring countries overlap along borders, so a
    # fragment can land in two of them; take the one in its own country
    intersected_with_admin = intersected_with_admin[
        prefer_domestic_rows(
            intersected_with_admin,
            fragment_col="mining_fragment_id",
            area_col="intersected_area_ha",
            domestic=intersected_with_admin["country_code"]
            == intersected_with_admin["country_code_auth"],
        )
    ]
    # prefix columns with admin_
    intersected_with_admin.columns = [
        "admin_" + col if col != "geometry" else col
        for col in intersected_with_admin.columns
    ]
    # correct the subnational attribution where it disagrees with the country the
    # fragment is actually in, rather than letting the filter discard it later
    intersected_with_admin = repair_admin_country_codes(
        intersected_with_admin, admin_areas_gdf
    )

    datasets_to_process = [
        {
            "name": "national_admin",
            "file": NATIONAL_ADMIN_GEOJSON,
            "ignore_if_outside_country": True,
            "ignore_if_outside_of_state": False,
        },
        {
            "name": "subnational_admin",
            "file": SUBNATIONAL_ADMIN_GEOJSON,
            "ignore_if_outside_country": True,
            "ignore_if_outside_of_state": True,
        },
        {
            "name": "indigenous_territories",
            "file": INDIGENOUS_TERRITORIES_GEOJSON,
            "ignore_if_outside_country": True,
            "ignore_if_outside_of_state": False,
        },
        {
            "name": "protected_areas",
            "file": PROTECTED_AREAS_GEOJSON,
            "ignore_if_outside_country": True,
            "ignore_if_outside_of_state": False,
        },
    ]

    # process each dataset
    for dataset in datasets_to_process:
        gdf = gpd.read_file(dataset["file"])

        summary = intersect_with_areas_of_interest_and_summarize(
            mining_admin_intersect_gdf=intersected_with_admin,
            areas_of_interest_gdf=gdf,
            ignore_if_outside_country=dataset["ignore_if_outside_country"],
            ignore_if_outside_of_state=dataset["ignore_if_outside_of_state"],
        )

        # the yearly timeseries needs the per-year snapshots, so it is computed
        # before collapsing the summary to the latest snapshot
        summary_mining_affected_area_ha_yearly = calculate_mining_area_timeseries(
            summary
        )
        if dataset["name"] == "national_admin":
            assert_country_totals_tie(summary_mining_affected_area_ha_yearly)

        # snapshots are cumulative, so totals come from the latest period only
        summary_latest = summarize_latest_snapshot(summary)

        summary_illegality = (
            summary_latest.groupby(["id", "admin_illegality_max"])[
                "intersected_area_ha"
            ]
            .sum()
            .round(2)
            .reset_index()
            .rename(columns={"intersected_area_ha": "mining_affected_area"})
        )

        # filter out -1 (no illegality data) from the illegality breakdown
        summary_illegality_filtered = summary_illegality[
            summary_illegality["admin_illegality_max"] != -1
        ]

        # calculate the denominator for percentages (only area WITH illegality data)
        illegality_area_totals = summary_illegality_filtered.groupby("id")[
            "mining_affected_area"
        ].sum()

        illegality_by_id = (
            summary_illegality_filtered.groupby("id")
            .apply(
                lambda g: g[["admin_illegality_max", "mining_affected_area"]].to_dict(
                    "records"
                ),
                include_groups=False,
            )
            .to_dict()
        )

        # save yearly summary to a json
        summary_mining_affected_area_ha_yearly.to_json(
            dataset["file"].replace(".geojson", "_yearly.json"),
            index=False,
            orient="records",
        )

        result = prepare_for_mining_calculator_and_save(summary_latest)

        # transform json result into dataframe
        summary_mining_affected_area_ha = summary_latest.groupby("id")[
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
                                / illegality_area_totals.get(
                                    id, 1
                                ),  # use filtered totals, excluding -1 values
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
        # save as a simple json, no geometry data
        gdf_merged_dict = gdf_merged.copy()
        gdf_merged_dict["bbox"] = gdf_merged_dict.geometry.apply(
            lambda g: list(g.bounds)
        )  # add bbox column
        gdf_merged_dict = gdf_merged_dict.drop(columns="geometry")
        gdf_merged_dict.to_json(
            dataset["file"].replace(".geojson", "_impacts_unfiltered_dict.json"),
            orient="records",
        )

        # merge with yearly and save to csv for reference
        ref = gdf_merged.merge(
            summary_mining_affected_area_ha_yearly, on="id", how="left"
        )
        ref = ref.drop(
            columns=[
                "locations",
                "mining_affected_area_ha",
                "geometry",
                "illegality_areas",
                "area_field",
                "area_units",
            ],
            errors="ignore",
        )
        ref = ref.rename(
            columns={
                "intersected_area_ha": "mining_affected_area_ha",
                "intersected_area_ha_cumulative": "mining_affected_area_ha_cumulative",
            }
        )
        ref.to_csv(
            dataset["file"].replace(".geojson", "_impacts_yearly.csv"), index=False
        )

        # filter and save only areas with impact
        gdf_merged = gdf_merged[gdf_merged["mining_affected_area_ha"] > 0]
        save_to_geojson(
            gdf_merged,
            dataset["file"].replace(".geojson", "_impacts.geojson"),
            id_column="id",
        )
