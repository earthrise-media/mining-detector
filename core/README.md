
Data generation and model inference run from code in this folder. 

### Setup

Earth Engine access requires a Google Cloud project and credentials. In each shell session (or in your shell profile):

```bash
export EE_PROJECT=YOUR_PROJECT_ID
```

1. **Google Cloud CLI** (optional for service-account auth): Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) if you use user credentials below.

2. **Authentication** — pick one:

   **Option A — user credentials (Application Default Credentials)**  
   Interactive login; good for local development:

   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

   **Option B — service account key file**  
   Point `GOOGLE_APPLICATION_CREDENTIALS` at a JSON key before running scripts. When this variable is set, `gee.py` uses the service account instead of ADC:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/home/<user>/.amw-gee-data-pull-key.json
   ```

   For long jobs (e.g. `tmux` inference), export both variables in that session before starting.
   
Various iterations of the code have worked with Pythons 3.9-3.11. 
```
python -m venv venv  # Consider whether you want to include --system-site-packages 
source venv/bin/activate
pip install -r requirements.txt  # May require tweaking - only critical version numbers are pinned.
```

Run scripts from the `core/` folder; CLI paths are interpreted relative to the current working directory unless you pass absolutes.

### Model training workflow

* `collect_sampling_locations.ipynb`: Merge selected training data files in ```sampling_data/```.
* `get_training_data.ipynb`: Download the training data.
* `cloud_mask_filter.ipynb`: Optional review of cloud masking. We keep some clouds and cloud masked images in the negative training set.
* `train_model.ipynb`: Train a neural network model. A few basic architectures can be loaded from model_library.py.
  - `embed.ipynb`: Alternately, run foundation model inference and train a classification head.
* `ensemble.ipynb`: Merge trained models into a single ensemble model.
* `inference.ipynb`: Run a model on a test area.
* `inference_pipeline.py`: For large-scale inference.

### Model inference

2025 model inference example: 
```
tmux new
cd core
python inference_pipeline.py \
    --model ../models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 \
    --region_path ../data/boundaries/Amazon_ACA.geojson \
    --start_date 2025-01-01 --end_date 2025-12-31 \
    --image_cache_dir /mnt/tempdisk/amw_image_cache2025_552-12/ \
    --pred_threshold 0.4 \
    --tries 3
```

### Post-processing

Patch detections can be filtered with a dual confidence threshold, with higher confidence required of isolated candidate detections (distance to the k-th nearest neighbor above a cutoff, by default 3 km). The folds into the analysis a rough spatial prior, that patches with mine scars tend to cluster. 

```
python postprocess.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2024-01-01_2024-12-31.geojson \
    --t-main 0.43 --k 5 --D 3 --t-iso 0.75
```

Defaults match the above. Add `--dissolve` to also write merged polygons.

### Masking

Masking of the mine scars is now handled by a fine-tuned SAM2 segmentation model, which requires additional set-up. 
```
# From repo root with venv activated
cd models/
git clone https://github.com/facebookresearch/sam2.git
cd sam2/
pip install -e .
./checkpoints/download_ckpts.sh
gsutil cp --billing-project=YOUR_PROJECT_ID gs://amazon-mining-watch/sam2/SAM_model_96_px_final.pth .   # 176MB file, expected cost is pennies 
```

By default the `sam2` repository is expected to be found in `models/`, but the path can also be set at run time. 

```
python sam2_mask.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2025-01-01_2025-12-31_t0.43_d5_3km_t0.75-dissolved.geojson \
    --start_date 2025-01-01 \
    --end_date 2025-12-31 \
    --cog
```

### Embedding-based models

As of May 2026, our best detection model remains an ensemble of CNNs trained from scratch. We experimented extensively with models constructed as (ensembles of) probes trained on top of geo-foundation model embeddings, and the code still supports this alternate paradigm. You would use `embed.ipynb` in place of `train_model.ipynb` and otherwise follow the same workflow.

Our foundation model of choice was the [SSL4EO ViT DINO S/16](https://github.com/zhu-xlab/SSL4EO-S12), also available through TorchGeo and HuggingFace. We caution that filenames, and therefore potentially the model, have been updated since we downloaded the file `dino_vit_small_patch16_224.pt`. The code looks for this checkpoint at `models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt`. 
