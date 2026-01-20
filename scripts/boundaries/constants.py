OUTPUTS_FOLDER = "data/outputs"
WEBSITE_OUTPUTS_FOLDER = f"{OUTPUTS_FOLDER}/website"

MINING_DIFFERENCES_FILES = {
    202503: f"{OUTPUTS_FOLDER}/48px_v0.X-SSL4EO-MLPensemble/cumulative/amazon_basin_48px_v0.X-SSL4EO-MLPensemble2025Q3diff-clean-gt11ha.geojson",
    202502: f"{OUTPUTS_FOLDER}/48px_v0.X-SSL4EO-MLPensemble/cumulative/amazon_basin_48px_v0.X-SSL4EO-MLPensemble2025Q2diff-clean-gt11ha.geojson",
    202400: f"{OUTPUTS_FOLDER}/48px_v0.X-SSL4EO-MLPensemble/cumulative/amazon_basin_48px_v0.X-SSL4EO-MLPensemble2024-clean-diff-gt11ha.geojson",
    202300: f"{OUTPUTS_FOLDER}/48px_v3.2-3.7ensemble/cumulative/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2023diff.geojson",
    202200: f"{OUTPUTS_FOLDER}/48px_v3.2-3.7ensemble/cumulative/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2022diff.geojson",
    202100: f"{OUTPUTS_FOLDER}/48px_v3.2-3.7ensemble/cumulative/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2021diff.geojson",
    202000: f"{OUTPUTS_FOLDER}/48px_v3.2-3.7ensemble/cumulative/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2020diff.geojson",
    201900: f"{OUTPUTS_FOLDER}/48px_v3.2-3.7ensemble/cumulative/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2019diff.geojson",
    201800: f"{OUTPUTS_FOLDER}/48px_v3.2-3.7ensemble/cumulative/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2018cumulative.geojson",
}

MINING_YEARS_QUARTERS = sorted(MINING_DIFFERENCES_FILES.keys())
first_mining_year_quarter, *_, last_mining_year_quarter = MINING_YEARS_QUARTERS

MINING_COMBINED_FILE = f"{WEBSITE_OUTPUTS_FOLDER}/amazon_basin_{first_mining_year_quarter}-{last_mining_year_quarter}_all_differences.geojson"


def GENERATE_MINING_SIMPLIFIED_FILENAME(year_quarter):
    return f"{WEBSITE_OUTPUTS_FOLDER}/mining_{year_quarter}_simplified.geojson"


MINING_SIMPLIFIED_FILES = [
    GENERATE_MINING_SIMPLIFIED_FILENAME(yq) for yq in MINING_YEARS_QUARTERS
]
