OUTPUTS_FOLDER = "data/outputs"
WEBSITE_OUTPUTS_FOLDER = f"{OUTPUTS_FOLDER}/website"

DATA_UPDATED_AT = "20260722"

MINING_DIFFERENCES_FILES = {
    202602: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-Q226-diff.geojson",
    202601: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-Q126-diff.geojson",
    202504: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-Q425-diff.geojson",
    202503: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-Q325-diff.geojson",
    202502: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-Q225-diff.geojson",
    202501: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-Q125-diff.geojson",
    202400: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2024-diff.geojson",
    202300: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2023-diff.geojson",
    202200: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2022-diff.geojson",
    202100: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2021-diff.geojson",
    202000: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2020-diff.geojson",
    201900: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2019-diff.geojson",
    201800: f"{OUTPUTS_FOLDER}/cumulative_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2018-diff.geojson",
}
MINING_YEARS_QUARTERS = sorted(MINING_DIFFERENCES_FILES.keys())
first_mining_year_quarter, *_, last_mining_year_quarter = MINING_YEARS_QUARTERS

# downloaded from https://source.coop/earthgenome/amazon-mining-watch/mining_scar_masks
MINING_DIFFERENCES_RASTER_FILES = {
    202602: "data/outputs/rasters/mining_mask_Q226_full.tif",
    202601: "data/outputs/rasters/mining_mask_Q126_full.tif",
    202504: "data/outputs/rasters/mining_mask_Q425_full.tif",
    202503: "data/outputs/rasters/mining_mask_Q325_full.tif",
    202502: "data/outputs/rasters/mining_mask_Q225_full.tif",
    202501: "data/outputs/rasters/mining_mask_Q125_full.tif",
    202400: "data/outputs/rasters/mining_mask_2024-01-01_2024-12-31_epsg4326.tif",
    202300: "data/outputs/rasters/mining_mask_2023-01-01_2023-12-31_epsg4326.tif",
    202200: "data/outputs/rasters/mining_mask_2022-01-01_2022-12-31_epsg4326.tif",
    202100: "data/outputs/rasters/mining_mask_2021-01-01_2021-12-31_epsg4326.tif",
    202000: "data/outputs/rasters/mining_mask_2020-01-01_2020-12-31_epsg4326.tif",
    201900: "data/outputs/rasters/mining_mask_2019-01-01_2019-12-31_epsg4326.tif",
    201800: "data/outputs/rasters/mining_mask_2018-01-01_2018-12-31_epsg4326.tif",
}
MINING_RASTER_YEARS_QUARTERS = sorted(MINING_DIFFERENCES_RASTER_FILES.keys())

def generate_mining_simplified_filename(year_quarter):
    return f"{WEBSITE_OUTPUTS_FOLDER}/mining_{year_quarter}_simplified.geojson"


MINING_SIMPLIFIED_FILES = [
    generate_mining_simplified_filename(yq) for yq in MINING_YEARS_QUARTERS
]
COMBINED_MINING_FILE = f"{WEBSITE_OUTPUTS_FOLDER}/mining_combined_full.geojson"

ILLEGALITY_AREAS_GEOJSON = "data/boundaries/illegality/out/illegality_v2_areas_simplified.geojson"
ILLEGALITY_DATA_UPDATED_AT = 202503
