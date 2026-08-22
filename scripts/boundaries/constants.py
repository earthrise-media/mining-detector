# constants.py

from pathlib import Path

OUTPUTS_FOLDER = "data/outputs"
WEBSITE_OUTPUTS_FOLDER = f"{OUTPUTS_FOLDER}/website"

DATA_UPDATED_AT = "20260820"

MINING_DIFFERENCES_FILES = {
    202602: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_Q226-dissolved.geojson",
    202601: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_Q126-dissolved.geojson",
    202504: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_Q425-dissolved.geojson",
    202503: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_Q325-dissolved.geojson",
    202502: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_Q225-dissolved.geojson",
    202501: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_Q125-dissolved.geojson",
    202400: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2024-dissolved.geojson",
    202300: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2023-dissolved.geojson",
    202200: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2022-dissolved.geojson",
    202100: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2021-dissolved.geojson",
    202000: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2020-dissolved.geojson",
    201900: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2019-dissolved.geojson",
    201800: f"{WEBSITE_OUTPUTS_FOLDER}/cumulative_dissolved/diffs/amazon_basin_growth_2018-dissolved.geojson",
}
MINING_YEARS_QUARTERS = sorted(MINING_DIFFERENCES_FILES.keys())
first_mining_year_quarter, *_, last_mining_year_quarter = MINING_YEARS_QUARTERS

# downloaded from https://source.coop/earthgenome/amazon-mining-watch/mining_scar_masks
# Single raster where each pixel's value is the first-detection year/quarter
# (e.g. 2018, 20262). 0 is nodata. Note that format is different from the standard
# in the rest of the scripts.
MINING_FIRST_YEAR_RASTER_FILE = (
    "data/outputs/rasters/amazon_basin_mining_scar_masks_first_year.tif"
)
# The period values that appear as pixel values in the raster above; one
# vectorized output file is produced per value.
MINING_RASTER_YEARS_QUARTERS = MINING_YEARS_QUARTERS

def generate_mining_simplified_filename(year_quarter):
    return f"{WEBSITE_OUTPUTS_FOLDER}/mining_{year_quarter}_simplified.geojson"

def generate_vectorized_raster_filename(year):
    raster_path = Path(MINING_FIRST_YEAR_RASTER_FILE)
    return str(raster_path.parent / "vectorized" / f"{raster_path.stem}_{year}.geojson")

MINING_SIMPLIFIED_FILES = [
    generate_mining_simplified_filename(yq) for yq in MINING_YEARS_QUARTERS
]
COMBINED_MINING_FILE = f"{WEBSITE_OUTPUTS_FOLDER}/mining_combined_full.geojson"

ILLEGALITY_AREAS_GEOJSON = "data/boundaries/illegality/out/illegality_v2_areas_simplified.geojson"
ILLEGALITY_DATA_UPDATED_AT = 202503
