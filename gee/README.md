
Data generation and model inference run from code in this folder. 

### Setup

In March 2024, inference ran on Python 3.9.12 in an anaconda3-2022.05 environment installed via Pyenv. On a Google Cloud Platform (GCP) n2-standard virtual machine with 16 CPUs it takes about 48 hours to process the full Amazon basin for one time period. 

After installing python and this repository: 

```
sudo apt-get install tmux
pip install -r requirements.txt  # Might require some tweaking - requirements.txt was created after the fact
gcloud auth login
gcloud auth application-default login
```

### Model inference

```
cd gee
tmux new 
python gee_pipeline.py --model_path ../models/48px_v3.2-3.7ensemble_2024-02-13.h5 --region_path ../data/boundaries/amazon_basin.geojson --start_date 2023-01-01 --end_date 2023-12-31

# Optional: dissolve adjacent patches to polygons, with a higher prediction threshold
cd ..
python scripts/dissolve.py data/outputs/48px_v3.2-3.7ensemble/amazon_basin_48px_v3.2-3.7ensemble_0.50_2023-01-01_2023-12-31.geojson --threshold 0.6
```

### Notes

##### Environment variable EE_PROJECT

Google Earth Engine (GEE) requires a 'project' to be specified within the Google cloud ecosystem, which must be accessible from the provided gcloud auth. In `gee.py` the code tries to read an EE_PROJECT environment variable and, failing that, defaults to our project name. Currently satellite data comes free from GEE, although they have said they will start charging at standard GCP data egress rates sometime in 2024. 

##### Compute resources and rate limit errors

The code leverages the available CPU cores, so there is essentially nothing to be gained by splitting the Amazon region to run multiple inference processes simultaneously on the same machine. However, splitting the region (cf. six subregions in `data/boundaries/amazon_basin/`) and running on multiple machines gives a linear speed up if needed. The RAM requirement looks to be roughly 12GB per process. 

Running multiple regions concurrently we did begin to see rate limit errors from GEE. With the above setup and up to four concurrent processes, the basic retry logic implemented in `gee.py` was sufficient to see all tiles processed.

