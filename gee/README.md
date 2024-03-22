
Data generation and model inference run from code in this folder. 

### Setup

In March 2024, inference ran on Python 3.9.12 in an anaconda3-2022.05 environment installed via Pyenv. We split the Amazon basin into six subregions and ran up to four regions concurrently on two Google Cloud Platform (GCP) n2-standard virtual machines, one with 8 CPUs and the other with 16 CPUs, each with 50 GB RAM. It takes about a day and a half to process the full Amazon basin for one time period in this setup. 

After installing python and this repository: 

```
sudo apt-get install tmux
pip install -r requirements.txt
gcloud auth login
gcloud auth application-default login
```

### Model inference

```
cd gee
tmux new -s a2
python gee_pipeline.py --model_path ../models/48px_v3.2-3.7ensemble_2024-02-13.h5 --region_path ../data/boundaries/amazon_basin/amazon_2.geojson --start_date 2023-01-01 --end_date 2023-12-31

# etc. for other Amazon subregions 

# Merge outputs for the six Amazon subregions 
cd ../data/outputs/48px_v3.2-3.7ensemble
python ../../../scripts/concatenate.py amazon_?_48px_v3.2-3.7ensemble_0.50_2023-01-01_2023-12-31.geojson --outpath amazon_basin_48px_v3.2-3.7ensemble_0.50_2023-01-01_2023-12-31.geojson

# Optional: dissolve adjacent patches to polygons, with a higher prediction threshold
python ../../../scripts/dissolve.py amazon_basin_48px_v3.2-3.7ensemble_0.50_2023-01-01_2023-12-31.geojson --threshold 0.6
```

### Notes

Environment variable **EE_PROJECT**: Google Earth Engine (GEE) requires a 'project' to be specified within the Google cloud ecosystem, which must be accessible from the provided gcloud auth. In `gee.py` the code tries to read an EE_PROJECT environment variable and, failing that, defaults to ours. Currently satellite data comes free from GEE, although they have said they will start charging at standard GCP data egress rates sometime in 2024. 

We saw some rate limit errors from GEE, which seemed worse when running all six Amazon subregions at once. With the above setup, the basic retry logic implemented in `gee.py` was sufficient to see all tiles processed.

Regions amazon_2 and amazon_5 require almost double the processing time of the others and accordingly ran on the 16 CPU machine.


