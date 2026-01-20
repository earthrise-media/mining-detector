# Boundaries scripts

These scripts prepare boundaries for use in the AMW website:

1. Concatenates all the mining detection geojsons into a single file, and also saves copies of the individual files while simplifying the geometry
2. Standardizes all subnational and national admin areas, as well as Indigienous Territories and protected areas from different shapefile sources
3. Preprocesses mining areas, atrributing admin jurisdiction data to them (for use in the Mining Calculator API); pre-calculates timeseries for admin areas; overlays with mining illegality layer
4. Uploads data to S3 for use in the website

## Pipeline

To get a full refresh of the data, run the following scripts sequentially:

```bash
python scripts/boundaries/concat_differences.py

# these are only required if the admin areas, ITs or PAs have changed
python scripts/boundaries/standardize_subnational_admin_areas.py
python scripts/boundaries/standardize_national_admin_areas.py
python scripts/boundaries/standardize_it_and_pa_areas.py

python scripts/boundaries/preprocess_mining_areas.py
python scripts/upload_data_to_s3.py
```

## Updating mining data

If you are updating mining data:

1. Update the references to your mining files in `scripts/boundaries/constants.py`, `MINING_DIFFERENCES_FILES` variable
2. Run the scrips above, skipping the `standardize_` scripts if admin areas, ITs and PAs have not changed