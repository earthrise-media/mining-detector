# Boundaries scripts

These scripts prepare boundaries for use in the AMW website:

1. Concatenates all the mining detection geojsons into a single file, and also saves copies of the individual files while simplifying the geometry
2. Standardizes all subnational and national admin areas, as well as Indigienous Territories and protected areas from different shapefile sources
3. Preprocesses mining areas, atrributing admin jurisdiction data to them (for use in the Mining Calculator API); pre-calculates timeseries for admin areas; overlays with mining illegality layer
4. Converts the geojsons to pmtiles for use in the website
5. Uploads data to S3 for use in the website

## Requirements

Besides the python libraries required in each of the scripts, you'll need to have [tippecanoe](https://github.com/mapbox/tippecanoe) installed.

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

python scripts/boundaries/convert_rasters_to_vector.py
python scripts/boundaries/preprocess_mining_areas.py
python scripts/boundaries/convert_geojsons_to_pmtiles.py
python scripts/boundaries/upload_data_to_s3.py
python scripts/boundaries/upload_tiles_to_s3.py
```

## Updating mining data

If you are updating mining data:

1. Update the `DATA_UPDATED_AT` variable in `scripts/boundaries/constants.py`, using the `YYYYMMDD` format. This ensures you will not overwrite previous data when uploading to S3
2. Update the references to your mining files in `scripts/boundaries/constants.py`, `MINING_DIFFERENCES_FILES` variable
3. Run the scrips above, skipping the `standardize_` scripts if admin areas, ITs, PAs, and illegality areas have not changed

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

Instead, you can use our S3 bucket to sync it with your local dev folder:

```bash
aws s3 sync ./data/boundaries s3://AWS_BUCKET_NAME_HERE/mining-detector-repo-backups/data/boundaries --exclude "*/.DS_Store" --exclude ".DS_Store"
aws s3 sync ./data/outputs/website s3://AWS_BUCKET_NAME_HERE/mining-detector-repo-backups/data/outputs/ --exclude "*/.DS_Store" --exclude ".DS_Store"website
```

Or you can use the python script instead to upload:

```bash
python scripts/boundaries/sync_source_data_to_s3.py
```

Or download:

```bash
python scripts/boundaries/sync_source_data_to_s3.py --download
```
