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
  - For Earth Genome only: May 2026 modeling patches are at `gs://amw-dev/training_patches2026-05-04T09:47_48px/`.
* `cloud_mask_filter.ipynb`: Optional review of cloud masking. We keep some clouds and cloud masked images in the negative training set.
* `train_model.ipynb`: Train a neural network model. A few basic architectures can be loaded from model_library.py.
  - Alternate foundation-model track (two steps): `embed.ipynb` or `embed_cls_patch.ipynb` (extract embeddings), then `train_probe.ipynb` (train a classification head on those embeddings).
* `ensemble.ipynb`: Merge trained models into a single ensemble model.
* `model_evaluation.ipynb`: Extra evaluation protocols (dual-threshold, `t_iso` sweeps, cumulative ∩ chips) beyond the metrics in `train_model.ipynb`.
* `inference.ipynb`: Run a model on a test area.
* `inference_pipeline.py`: For large-scale inference.

### Model inference

Large jobs are typically run on a VM with a local Sentinel-2 image cache.

* For 2026 runs we used two GCP `n2-32-standard` machines, running multiple processes simultaneously on each.
* Earth Genome: cached imagery for 2018 through 2026Q2 can be pulled from `gs://amw-dev/`.

Example Amazon ACA year (CNN ensemble):

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

Andes supplemental regions use the same CNN ensemble at a lower raw threshold (`0.2`):

```
python inference_pipeline.py \
    --model ../models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 \
    --region_path ../data/boundaries/andes_supplemental.geojson \
    --start_date 2025-01-01 --end_date 2025-12-31 \
    --image_cache_dir /mnt/tempdisk/amw_image_cache2025_552-12/ \
    --pred_threshold 0.2 \
    --tries 3
```

If Amazon was split across multiple jobs/sections, concatenate on your local machine:

```
# from repo root
python scripts/concatenate.py path/to/part_a.geojson path/to/part_b.geojson \
    --outpath data/outputs/.../Amazon_ACA_....geojson
```

### Post-processing

Patch detections can be filtered with a dual confidence threshold, with higher confidence required of isolated candidate detections (distance to the k-th nearest neighbor above a cutoff, by default 3 km). This folds into the analysis a rough spatial prior, that patches with mine scars tend to cluster.

CLI defaults match the **relaxed** single-period postprocess (`t_main=0.43`, `t_iso=0.75`, `k=5`, `D=3`):

```
python postprocess.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2025-01-01_2025-12-31.geojson
```

For the **stringent** settings used in website cumulatives, pass explicit thresholds:

```
python postprocess.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2025-01-01_2025-12-31.geojson \
    --t-main 0.55 --k 5 --D 3 --t-iso 0.8
```

Add `--dissolve` to also write merged polygons.

Clip Andes supplemental detections to the supplemental boundary (writes a `*-filt.geojson` sibling):

```
# from repo root
python scripts/geo_filter.py \
    data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/andes_supplemental_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.20_2025-01-01_2025-12-31.geojson \
    data/boundaries/andes_supplemental.geojson
```

### Cumulatives and diffs

Build year-end / quarterly cumulatives and period diffs in `cumulatives_and_diffs.ipynb` (Amazon postprocessed at the stringent dual threshold, unioned with Andes supplemental). Rules and published path layout are summarized in `data/outputs/MANIFEST.yaml` under `cumulative_rules` and `products`.

Typical local folder layout under `data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/`:

* `raw_detections/`
* `postprocessed_t0.43_d5_3km_t-iso0.75/`
* `postprocessed_t0.55_d5_3km_t-iso0.8/`
* `cumulative_t0.55_d5_3km_t-iso0.8/` (includes a `diffs/` subfolder)

### Masking

Masking of the mine scars is handled by a fine-tuned SAM2 segmentation model, which requires additional set-up.

```
# From repo root with venv activated
cd models/
git clone https://github.com/facebookresearch/sam2.git
cd sam2/
pip install -e .
./checkpoints/download_ckpts.sh
gsutil cp --billing-project=YOUR_PROJECT_ID gs://amazon-mining-watch/sam2/SAM_model_96_px_final.pth .   # 176MB file, expected cost is pennies
```

By default `sam2_mask.py` expects the `sam2` checkout under `models/sam2` (re-run `pip install -e .` after moving the folder there). The path can also be set at run time.

Run SAM2 masks on **full-year detections** and on **quarterly diffs** (not on quarter-only cumulatives alone). Example: 2025 imagery window on the through-2025 cumulative / year-end product:

```
python sam2_mask.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/cumulative_t0.55_d5_3km_t-iso0.8/diffs/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-2025.geojson \
    --start_date 2025-01-01 \
    --end_date 2025-12-31 \
    --cog
```

#### Saved logits: the smoothing must be replayed

Each masked tile writes two products: `*-msk.tif` and `*-logits.tif`. **They are not two views of the same array, and thresholding the logits does not reproduce the mask.**

The production mask is `smooth(upsampled_log_odds) > 0`. The saved logits are the same field *before* the Gaussian smoothing (and clamped to ±16), stored that way deliberately so `smoothing_sigma` stays retunable without re-running SAM2. Thresholding them directly gives a measured IoU of about **0.84** against the real product — expected, not a bug.

Use the helper, which replays the smoothing and reproduces the product exactly:

```python
from sam2_logits import mask_from_logits

mask = mask_from_logits(logits_array)                  # == the production mask
mask = mask_from_logits(logits_array, threshold=1.5)   # a stricter provisional mask
```

Two rules that are easy to get wrong:

* **Replay per tile, before mosaicking.** Never threshold a logits *mosaic*. Smoothing does not commute with the max-reduce that merges overlapping tiles — max-reduce on raw logits is biased upward, and smoothing spreads that inflated max across the seam, inflating area by up to 1.8% at a 12 px overlap. Correct order: per-tile smooth → threshold → union-merge (`sam2_logits.mosaic_masks`). The logits mosaic from `sam2_build_cog.py` is for inspection, not for deriving masks.
* **`smoothing_sigma` must match the run that produced the logits.** Changing it is legitimate — that is the point of storing them unsmoothed — but it yields a different product.

The ±16 clamp is lossless with respect to the mask: it was chosen so the re-derived mask is bit-identical to the unclamped one after the replay. Background and measurements in [`docs/design/persistence-planning.md`](../docs/design/persistence-planning.md).

**Logits written before this change are worse, and `mask_from_logits` alone will not rescue them.** Everything published through July 2026 stores raw SAM2 `log_odds` — prior included, but neither upsampled nor smoothed — so those rasters are not even co-registered with their own masks: the logits are coarser by exactly 35/32 in every UTM/lat band. Deriving a mask from them needs an upsample to the mask grid *as well as* the smoothing replay, and still will not match bit-for-bit, since the bilinear upsample cannot be reproduced exactly outside the original torch path. Treat pre-rerun logits as diagnostic only.

#### Quarterly accumulation

Quarterly mosaics have large cloud gaps, so a quarter-only mask under-covers known scars. After masking the quarterly diffs, **accumulate** each quarter’s diff mask onto the prior full-year (or prior `*_full`) mask with `sam2_combine_masks.py` (OR merge) to produce that quarter’s `*_full` mask. On VMs that use GDAL’s Python pixel function for the VRT step, point `PYTHONSO` at the system libpython first:

```
export PYTHONSO=/usr/lib/x86_64-linux-gnu/libpython3.9.so.1.0
python sam2_combine_masks.py \
    mining_mask_2024-01-01_2024-12-31_epsg4326.tif \
    mining_mask_2025-01-01_2025-03-31diff_epsg4326.tif \
    mining_mask_2025Q1_full_epsg4326.tif
```

### Publishing outputs

Artifacts are **not** checked into git. Keep `data/outputs/MANIFEST.yaml` current as the in-repo catalog of what lives where.

**Internal mirror** (`gs://amw-dev/outputs/`), from the model output directory:

```
# Detection folders (layout matches MANIFEST path_map / gcs keys)
gsutil -m rsync -r cumulative_t0.55_d5_3km_t-iso0.8 \
    gs://amw-dev/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/cumulative_t0.55_d5_3km_t-iso0.8/
gsutil -m rsync -r postprocessed_t0.43_d5_3km_t-iso0.75 \
    gs://amw-dev/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/postprocessed_t0.43_d5_3km_t-iso0.75/
gsutil -m rsync -r postprocessed_t0.55_d5_3km_t-iso0.8 \
    gs://amw-dev/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/postprocessed_t0.55_d5_3km_t-iso0.8/
gsutil -m rsync -r raw_detections \
    gs://amw-dev/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/raw_detections/
```

SAM2 COGs (example sync from local `data/outputs/sam2/` naming):

```
for f in Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_t0.55_d5_3km_t-iso0.8_cumulative2018-* ; do
  gsutil -m rsync -r "$f/cog_outputs/" "gs://amw-dev/outputs/sam2/$f/cog_outputs/"
done
```

Also upload yearly and combined quarterly full masks (e.g. `mining_mask_Q325_full.tif`) under the model’s `mining_scar_masks/` prefix on GCS.

**Public** ([Source Cooperative](https://source.coop/earthgenome/amazon-mining-watch)): paste temporary AWS-compatible credentials into the shell, then upload these product trees (local folder → bucket key):

```
aws s3 ls s3://earthgenome/amazon-mining-watch/

# Cumulatives (+ diffs/) → cumulative_detections/
aws s3 cp --recursive cumulative_t0.55_d5_3km_t-iso0.8/ \
    s3://earthgenome/amazon-mining-watch/cumulative_detections/

# Single-period detections → single_periods/
aws s3 cp --recursive raw_detections/ \
    s3://earthgenome/amazon-mining-watch/single_periods/raw_detections/
aws s3 cp --recursive postprocessed_t0.43_d5_3km_t-iso0.75/ \
    s3://earthgenome/amazon-mining-watch/single_periods/postprocessed_t0.43_d5_3km_t-iso0.75/
aws s3 cp --recursive postprocessed_t0.55_d5_3km_t-iso0.8/ \
    s3://earthgenome/amazon-mining-watch/single_periods/postprocessed_t0.55_d5_3km_t-iso0.8/

# SAM2 scar masks
aws s3 cp --recursive mining_scar_masks/ \
    s3://earthgenome/amazon-mining-watch/mining_scar_masks/

# optional: refresh the product README on the bucket
aws s3 cp /path/to/README-source.coop.md s3://earthgenome/amazon-mining-watch/README.md
```

After publishing, update `data/outputs/MANIFEST.yaml` (`updated` date, periods, and any new path notes).

### Embedding-based models

As of May 2026, our best detection model remains an ensemble of CNNs trained from scratch. We experimented extensively with models constructed as (ensembles of) probes trained on top of geo-foundation model embeddings, and the code still supports this alternate paradigm. Use an embedding notebook then `train_probe.ipynb` in place of `train_model.ipynb`, and otherwise follow the same workflow.

Two embedding exports are supported (SSL4EO ViT-S/16):

* **Class token only** — `embed.ipynb` (`embedding_strategy=cls_only`). Legacy path; one vector per chip from the ViT CLS token.
* **CLS + ViT patch token** — `embed_cls_patch.ipynb` (backed by `embed_cls_patch.py`; `embedding_strategy=cls_patch`). Concatenates the CLS token with one selected spatial patch token (768-d for ViT-S/16). We experimented with this denser representation; it remains supported for training and for bulk inference via `--embedding_strategy cls_patch`.

In `train_probe.ipynb`, point at the parquet from the chosen embed step and set `EMBEDDING_STRATEGY` to match (`cls_only` vs `cls_patch`). For `cls_patch`, the probe expects feature columns `cls*` then `spatial*` (see `model_library.MLP_with_targeted_dropout`).

Our foundation model of choice was the [SSL4EO ViT DINO S/16](https://github.com/zhu-xlab/SSL4EO-S12), also available through TorchGeo and HuggingFace. We caution that filenames, and therefore potentially the model, have been updated since we downloaded the file `dino_vit_small_patch16_224.pt`. The code looks for this checkpoint at `models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt`.
