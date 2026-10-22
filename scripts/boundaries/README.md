# Boundaries scripts

These scripts prepare boundaries for use in the AMW website:

1. Concatenates all the mining detection geojsons into a single file, and also saves copies of the individual files while simplifying the geometry
2. Standardizes all subnational and national admin areas, as well as Indigienous Territories and protected areas from different shapefile sources
3. Preprocesses mining areas, atrributing admin jurisdiction data to them (for use in the Mining Calculator API); pre-calculates timeseries for admin areas; overlays with mining illegality layer
4. Converts the geojsons to pmtiles for use in the website
5. Uploads data to S3 for use in the website

## Requirements

Besides the python libraries required in each of the scripts, you'll need to have [tippecanoe](https://github.com/mapbox/tippecanoe) installed.

## Updating mining data

If you are updating mining data:

1. Update the `DATA_UPDATED_AT` variable in `scripts/boundaries/constants.py`, using the `YYYYMMDD` format. This ensures you will not overwrite previous data when uploading to S3
2. Update the references to your mining files in `scripts/boundaries/constants.py`, `MINING_DIFFERENCES_FILES` variable
3. Run the scrips below, skipping the `standardize_` scripts if admin areas, ITs, PAs, and illegality areas have not changed

## Pipeline

To get a full refresh of the data, run the following scripts sequentially:

```bash
# If you're starting from a fresh repo clone, you'll need too bring in the source data
# For more details on this see the "Source data and outputs" section further below.
python scripts/boundaries/sync_source_data_to_s3.py --download

# These are only required if the admin areas, ITs, PAs, or illegality areas have changed.
python scripts/boundaries/standardize_subnational_admin_areas.py
python scripts/boundaries/standardize_national_admin_areas.py
python scripts/boundaries/standardize_it_and_pa_areas.py
python scripts/boundaries/standardize_illegality_areas.py

# if there are new datasets, you can copy them from
gcloud storage cp gs://amw-published/amazon_basin_mining_scar_masks.tif data/outputs/rasters/amazon_basin_mining_scar_masks.tif # tif
gcloud storage rsync --recursive gs://amw-published/cumulative_dissolved/diffs/ data/outputs/website/cumulative_dissolved/diffs/ # geojson diffs

python scripts/boundaries/convert_rasters_to_vector.py --overwrite
python scripts/boundaries/preprocess_mining_areas.py
# check the outputs before publishing them, see "Validating outputs" below
python scripts/boundaries/test_preprocess_mining_outputs.py
python scripts/boundaries/convert_geojsons_to_pmtiles.py
python scripts/boundaries/upload_data_to_s3.py
python scripts/boundaries/upload_tiles_to_s3.py
```

## Validating outputs

`test_preprocess_mining_outputs.py` runs data-quality tests on the outputs of
`preprocess_mining_areas.py` (the `_yearly.json`, `_impacts_unfiltered_dict.json`,
`.geojson` and `_impacts_yearly.csv` files of every dataset, plus the simplified mining
files). Run it after preprocessing and before uploading:

```bash
uv run scripts/boundaries/test_preprocess_mining_outputs.py
# or, with pytest arguments, e.g. only the yearly checks, stopping at the first failure
uv run scripts/boundaries/test_preprocess_mining_outputs.py -k yearly -x
```

Besides checking each file on its own (unique ids, every period present, no negative
or missing values), it ties the files to each other: the totals, the illegality
breakdown and the Mining Calculator locations all come from the same per-period
snapshots as the yearly timeseries, so they must agree up to rounding. Small
discrepancies that can have legitimate causes, such as a cumulative value dipping by a
fraction of a hectare, show up as warnings in the summary instead of failing the run.
The thresholds are constants at the top of the file.

The console shows a short message per failure and warning, with the first few areas
involved (id, name, type). Everything else goes to a Markdown report,
`logs/preprocess_outputs_report.md` (gitignored; set `AMW_REPORT_PATH` to write it
elsewhere): a summary, every failure and warning with the test's description and the
complete tables, and a checklist of all tests. The console formatting lives in
`conftest.py`, next to the tests.

To also compare against the previously published data on the CDN (past periods
shouldn't move much, and the newest total shouldn't drop):

```bash
AMW_COMPARE_PUBLISHED=latest uv run scripts/boundaries/test_preprocess_mining_outputs.py
# or a specific publish folder
AMW_COMPARE_PUBLISHED=20261005 uv run scripts/boundaries/test_preprocess_mining_outputs.py
```

`latest` is the newest publish folder older than `DATA_UPDATED_AT`, so a run that was
already uploaded isn't compared with itself. The folder used, and the newest period in
it, are printed at the end of the run and in the report.

## Collated area summaries

To get one flat CSV of mined area per jurisdiction per year — for sharing with
people who won't be querying the API — run:

```bash
python scripts/boundaries/collate_jurisdiction_areas.py
```

It reads the published data back off the CDN, so run it *after*
`upload_data_to_s3.py`. It finds the newest publish folder by probing the CDN
(the bucket doesn't allow listing) between today and `DATA_UPDATED_AT`. Pass
`--data-date YYYYMMDD` to pin an older folder.

The output path is fixed at `data/public/mined_areas_by_jurisdiction.csv`
— the AMW website links directly to it, so the name must not change between
data updates. Commit the refreshed file to get a versioned history of it.

Every row carries the `date_published` of the publish it was built from, so a
copy that has been downloaded and passed around still says which vintage it is.
`--data-date` takes that same dashed form as well as the `YYYYMMDD` folder name,
so a date read off the CSV can be pasted straight back in to rebuild it.

## Source data and outputs

We've stopped saving the outputs (and never saved source data) to the Github repo because it was too large and changed too often. That now includes everything under `data/boundaries/*/out/` — the standardized boundary layers as well as the impacts and timeseries derived from them — and everything under `data/outputs/website/`. They are gitignored, so keep your local copies current by syncing; a stale checkout will silently produce stale numbers. The stable reference boundaries outside those folders (`Amazon_ACA.geojson` and similar) stay in the repo.

Instead, you can use our S3 bucket to sync it with your local dev folder. The sync script covers `data/boundaries`, `data/outputs/website` and `data/outputs/rasters`, skips `.DS_Store` and `.pmtiles` files, and requires you to choose a direction explicitly — running it with no flag is an error.

To download (pulls files you don't have and refreshes any local file whose size or ETag differs from the bucket copy; never uploads):

```bash
python scripts/boundaries/sync_source_data_to_s3.py --download
# add --dry-run to preview the changes
```

To upload (pushes new or modified local files, compared by size; never downloads):

```bash
python scripts/boundaries/sync_source_data_to_s3.py --upload
# add --dry-run to preview the changes
```

Always run `--download` before `--upload`. Uploading overwrites the bucket copy with whatever you have locally, so uploading from a stale checkout will clobber the backup with old data.

Uploads compare files by size only, so an edited file that happens to keep the same size will not be uploaded.

To fetch a single file without AWS credentials, read it from the CDN:

```bash
curl -fSL --create-dirs -o data/boundaries/<path> \
  https://media-amw.earthgenome.org/mining-detector-repo-backups/data/boundaries/<path>
```
