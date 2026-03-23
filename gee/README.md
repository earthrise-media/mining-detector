
Data generation and model inference run from code in this folder. 

### Setup

1. **Google Cloud CLI**: This repo uses Application Default Credentials (ADC). 
   Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) for your OS.
   
2. **Authentication**: Run the following commands in your terminal:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   export EE_PROJECT=YOUR_PROJECT_ID  # Your Google Cloud project, required even if no data egress charges applied. (Check this: Google has announced coming charges.)
   ```
   
Various iterations of the code have worked with Pythons 3.9-3.11. 
```
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt  # May require tweaking - only critical version numbers are pinned.
```

2025 models are built on the [SSL4EO ViT DINO S/16](https://github.com/zhu-xlab/SSL4EO-S12) foundation model. The model is also available through TorchGeo and HuggingFace. We caution that filenames, and therefore potentially the model, have been updated since we downloaded the file `dino_vit_small_patch16_224.pt`. 

Default paths in `gee.py` are anchored to the **repository root** (`REPO_ROOT`): SSL4EO weights `models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt` (`SSL4EO_PATH`), SAM2 checkout  `models/sam2` (`SAM2_PATH`), and mask output dir `data/outputs/sam2` (`DEFAULT_MASK_DIR`). Run scripts from the `gee/` folder; other CLI paths are relative to the current working directory unless you pass absolutes.


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
    --pred_threshold 0.85 \
    --start_date 2025-07-01 \
    --end_date 2025-09-30
```

On an A2 GPU-equipped virtual machine on Google Cloud Platform (GCP), this runs in a little over 24 hours. 

### Post-processing and masking

Post-processing is in a state of flux as of spring 2026. NDVI masking is disabled, but the existing script should be useable to impose higher thresholds and aggregate patches to polygons: 

```
python postprocess.py \
    ../data/outputs/48px_v0.X-SSL4EO-MLPensemble/Amazon_ACA_*_48px_v0.X-SSL4EO-MLPensemble_0.85_2025-07-01_2025-09-30.geojson \
    --threshold 0.925 \
    --low_area_conf_threshold 0.975 \
    --outpath amazon_basin_48px_v0.X-SL4EO-MLP0.85_2025-07-01_2025-09-30post.geojson \
```

Masking of the mine scars around the detection polygons is now handled by a fine-tuned SAM2 segmentation model, which requires additional set-up. 
```
# From repo root
cd models/
git clone https://github.com/facebookresearch/sam2.git
cd sam2/
pip install -e .
./checkpoints/download_ckpts.sh
gsutil cp --billing-project=YOUR_PROJECT_ID gs://amazon-mining-watch/sam2/SAM_model_96_px_final.pth .   # 176MB file, expected cost is pennies 
```

By default the `sam2` repository is expected to be found in `models/`, but the path can also be set at run time. 

SAM2 masking can be run in-line with inference (setting run_sam2 -> True) or after inference (from `gee/`): 
```
python sam2_mask.py \
    ../data/outputs/48px_v0.X-SSL4EO-MLPensemble/cumulative/amazon_basin-2018-2025Q3cumulative-clean.geojson \
    --start_date 2025-07-01 \
    --end_date 2025-09-30 \
    --cog
```
