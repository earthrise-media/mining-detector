
Data generation and model inference run from code in this folder. 

### Setup

Various iterations of the code have worked with Pythons 3.9-3.11. 

```
pip install -r requirements.txt  # May require tweaking - only critical version numbers are pinned.
gcloud auth login
gcloud auth application-default login
```

2025 models are built on the [SSL4EO ViT DINO S/16](https://github.com/zhu-xlab/SSL4EO-S12) foundation model. The model is also available through TorchGeo and HuggingFace. We caution that filenames, and therefore potentially the model, have been updated since we downloaded the file `dino_vit_small_patch16_224.pt`.

### Model training workflow

* `collect_sampling_locations.ipynb`: Merge selected training data files in ```sampling_data/```.
* `get_training_data.ipynb`: Download the training data.
* `cloud_mask_filter.ipynb`: Optional review of cloud masking. We keep some clouds and cloud masked images in the negative training set.
* `embed.ipynb`: Run foundation model inference and train a classification head.
  - `train_model.ipynb`: Alternatively, train a neural network from scratch. A few options can be loaded from model_library.py 
* `ensemble.ipynb`: Merge trained models into a single ensemble model.
* `inference.ipynb`: Run a model on a test area.
* `inference_pipeline.py`: For large-scale inference.

### Model inference

Q3 2025 model inference example on the Amazon_6 subregion:  
```
tmux new
cd gee
python inference_pipeline.py \
    --model ../models/48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5 \
    --region_path ../data/boundaries/Amazon_ACA/Amazon_ACA_6.geojson \
    --embed_model_path SSL4EO/pretrained/dino_vit_smal_patch16_224.pt \
    --geo_chip_size 48 \
    --pred_threshold 0.85 \
    --start_date 2025-07-01 \
    --end_date 2025-09-30
```

On an A2 GPU-equipped virtual machine on Google Cloud Platform (GCP), this runs in a little over 24 hours. 

Then post-process to impose higher thresholds, aggregate patches to polygons, and compute areas using an NDVI mask: 

```
cd data/outputs/48px_v0.X-SSL4EO-MLPensemble
python ../../../gee/postprocess.py \
    Amazon_ACA_*_48px_v0.X-SSL4EO-MLPensemble_0.85_2025-07-01_2025-09-30.geojson \
    --ndvi_threshold 0.5 \
    --threshold 0.925 \
    --low_area_conf_threshold 0.975 \
    --outpath amazon_basin_48px_v0.X-SL4EO-MLP0.85_2025-07-01_2025-09-30post.geojson \
    --start_date 2025-07-01 \
    --end_date 2025-09-30
```

### Notes

##### Environment variable EE_PROJECT

Google Earth Engine (GEE) requires a 'project' to be specified within the Google cloud ecosystem, which must be accessible from the provided gcloud auth. In `gee.py` the code tries to read an EE_PROJECT environment variable and, failing that, defaults to our project name. Currently satellite data comes free from GEE, although they have said they will start charging at standard GCP data egress rates sometime in 2024. 

##### Rate limit errors

Running multiple regions concurrently we begin to see rate limit errors from GEE. With the above setup and up to four concurrent processes, the basic retry logic implemented in `gee.py` was sufficient to see all tiles processed.

