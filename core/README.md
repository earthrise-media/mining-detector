Data generation and model inference run from code in this folder.
`scripts/pipeline.py` drives the whole chain end to end; the sections below
document the stages it calls and how to run them one at a time.

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

### Inference: Driving the pipeline

The full production pipeline now encompasses model inference for patch
detections and segmentation along with numerous post-processing
steps. These are integrated under a master pipeline orchestrator. The
underlying calls for each step are provided in more detail below.

`scripts/pipeline.py` is the single entry point. It knows the stage order and
derives every path and date from the period tags, so a quarterly refresh is the
same sequence of commands as a bulk rewrite — with one more tag in the period list.

```
python ../scripts/pipeline.py --list                    # stage order, and who runs each
python ../scripts/pipeline.py review-periods            # stage 0: the period list
python ../scripts/pipeline.py <stage> --periods Q226    # work on one period
python ../scripts/pipeline.py <stage> --all --dry-run   # show what a rebuild would do
```

**`--periods` is required, and every value must be listed in
[`core/periods.py`](periods.py) as `ALL_CURRENT_PERIODS`.** That list is the only
variable this pipeline needs a human to set; everything else defaults correctly.
Stage 0 exists to put it in front of you before anything runs.

The requirement is not bureaucratic. `persist-detections`, `persist-masks` and
`stage` recompute from the whole history rather than from the periods you name, so
they read `ALL_CURRENT_PERIODS` directly — a period absent from it is invisible to
them and cannot reach the product. `pipeline.py` refuses rather than half-running
it.

**Full refresh.** Five steps, alternating between what you run on a VM and what
the pipeline runs for you. The alternation is not cosmetic — each group needs the
one before it. Scripted stages skip what already exists, so re-running a
step is how you resume after an interruption.

```
cd core

# 0. the period list -- add what you are about to run, then carry on.
python ../scripts/pipeline.py review-periods

# 1. patch detections. Long VM job; launch under tmux and watch.
python ../scripts/pipeline.py inference --all

# 2. scripted: concatenate subregions, clip andes, postprocess, corroborate.
python ../scripts/pipeline.py concat filter postprocess persist-detections --all

# 3. segmentation. Long VM jobs, and they need step 2's output.
python ../scripts/pipeline.py mask-annual --all
python ../scripts/pipeline.py mask-quarterly --all

# 4. scripted: cog the masks, corroborate them, assemble both staging trees.
python ../scripts/pipeline.py cog persist-masks stage manifest --all

# 5. review the rasters, then push.
python ../scripts/pipeline.py publish --all
```

**Quarterly update.** Add the new tag to `ALL_CURRENT_PERIODS` in
`core/periods.py`, then walk the same steps with `--periods Q326` in place of
`--all` — or `--periods 2026 Q127` in a January, when a new annual witness lands
alongside the quarter. Only the new period has work
to do; everything else is skipped.

In Q2, Q3 and Q4 step 3 needs `mask-quarterly` only — the quarter is segmented from
`patch_diffs/`. **In Q1 you need `mask-annual` as well**, for the year just
completed: that year's annual mask is a witness for every later onset, even though
its own layer is not published while quarters still cover the year. Note that
`mask-annual` emits a command for every annual period, not just the new
one, so take the pair you need.

Stages run in this order:

```
inference → concat → filter → postprocess → persist-detections
  → mask-annual → mask-quarterly → cog → persist-masks
  → stage → manifest → publish
```

`publish` emits three numbered steps — push to `gs://amw-published`, mirror to
`gs://amw-dev/published/`, then Source Cooperative — each followed by a count
check.

### Detection model inference

Large jobs are typically run on a VM with a local Sentinel-2 image cache.

* For 2026 runs we used two GCP `n2-32-standard` machines, running multiple processes simultaneously on each.
* Earth Genome: cached imagery for 2018 through 2026Q2 can be pulled from `gs://amw-dev/`.

Example Amazon ACA year (CNN ensemble):

```
tmux new
source venv/bin/activate
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

#### Concatenating subregions — and why it deduplicates

The basin is run as the six subregions in `data/boundaries/Amazon_ACA/Amazon_ACA_{1..6}.geojson`, split at lon −66/−56 and lat −5. This is the normal path, not a fallback: whole-basin runs proved unreliable, so expect to concatenate on every full-basin pass. Do it on your local machine:

```
# from repo root
python scripts/concatenate.py path/to/part_a.geojson path/to/part_b.geojson \
    --outpath data/outputs/.../Amazon_ACA_....geojson
```

**Always use this script — not `ogrmerge`, `cat`, or a plain `pd.concat`.** Tiles straddling a subregion boundary are generated by *both* adjacent runs, and chips are cut across the whole tile, so those tiles emit the same detections twice. Plain concatenation double-counts a band one tile wide (0.05°) along every seam. Because the six-way split is permanent, so is the duplication: it is a property of the workflow, not a one-off accident.

**Dedupe before postprocessing, never after.** This is not cosmetic tidying: a duplicate sits at distance zero from its twin, which depresses the k-th-nearest-neighbour distance `postprocess.py` uses to judge isolation, so a pair can slip past the stricter `t_iso` cutoff that should have rejected it. Measured on 2023: 875 duplicate records (0.48%), plus 11 locations at `t0.43` and 3 at `t0.55` that survived the isolation test only because their twin was sitting on top of them.


### Post-processing

Patch detections can be filtered with a dual confidence threshold, with higher confidence required of isolated candidate detections (distance to the k-th nearest neighbor above a cutoff, by default 3 km). This folds into the analysis a rough spatial prior, that patches with mine scars tend to cluster.

CLI defaults match the **relaxed** single-period postprocess (`t_main=0.43`, `t_iso=0.75`, `k=5`, `D=3`):

```
python postprocess.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/raw_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2025-01-01_2025-12-31.geojson
```

For the **stringent** settings used for provisional detections, pass explicit thresholds:

```
python postprocess.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/raw_detections/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2025-01-01_2025-12-31.geojson \
    --t-main 0.55 --k 5 --D 3 --t-iso 0.8
```

Clip Andes supplemental detections to the supplemental boundary. `--outpath` is
required; write it to `raw_detections/andes_supplemental/` under the `_0.2_` name,
where `persistence.py` and staging both expect it.

```
# from repo root
python scripts/geo_filter.py \
    data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/andes_supplemental_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.20_2025-01-01_2025-12-31.geojson \
    data/boundaries/andes_supplemental.geojson \
    --outpath data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/raw_detections/andes_supplemental/andes_supplemental_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.2_2025-01-01_2025-12-31.geojson
```

### Persistence: cumulatives and first-year products

Temporal corroboration replaced threshold-tightening as the way cumulatives
control false positives. `persistence.py` requires a detection to appear in
**k = 2 annual periods within a 2-period window** before it is published as
confirmed; the stringent `t0.55` set is retained only to stand in for a witness
at the provisional edge, where the corroborating year does not exist yet.

Onset is a pure function of the whole period stack, not an incremental patch, so
a refresh recomputes and cannot drift. That is the property the rest of the
pipeline leans on.

```
python persistence.py --base ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble \
    --years 2018 2019 2020 2021 2022 2023 2024 2025 \
    --quarters Q125 Q225 Q325 Q425 Q126 Q226 \
    --dissolve
```

Writes, under `--outdir` (defaults beside `--base`):

* `cumulative/` — per-period cumulative patches, plus `patch_diffs/`, each
  period's newly confirmed patches. The quarterly SAM2 prompt set.
* `cumulative_dissolved/` — display polygons, with geometric yearly increments
  under `diffs/`. An 11 ha minimum applies to increments only, never to the
  cumulative itself.

Defaults are recipe A (`--k 2 --window 2 --witnesses annual`). `--witnesses all`
admits quarters as witnesses, which is deliberately *not* the default: quarters
do not exist before 2025, so admitting them makes early and late years
incomparable. Rules and published layout are catalogued in
[`data/outputs/MANIFEST.yaml`](../data/outputs/MANIFEST.yaml) under
`persistence_rules` and `path_map`.

Consumer-facing folders carry no threshold tag. The parameters that produced a
directory live in its `config.txt` sidecar, so a path stays stable when a
threshold is retuned.

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

Two prompt sets, matching the two persistence cadences.

**Annual** — the loose single-period postprocess for that year. Annual masks are
prompted from the full year's detections, not from a cumulative, so each year's
mask is independent evidence and the k-of-n rule has something to corroborate:

```
python sam2_mask.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/postprocessed_t0.43_d5_3km_t-iso0.75/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2025-01-01_2025-12-31_t0.43_d5_3km_t-iso0.75.geojson \
    --start_date 2025-01-01 --end_date 2025-12-31 \
    --cog
```

**Quarterly** — that quarter's newly confirmed patches from `patch_diffs/`, because quarterly imagery has gaps and cannot support a reliable full segmentation:

```
python sam2_mask.py \
    ../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/cumulative/patch_diffs/Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_growth_Q226.geojson \
    --start_date 2026-04-01 --end_date 2026-06-30 \
    --cog
```

#### Saved logits reproduce the mask

Each masked tile writes `*-msk.tif` and `*-logits.tif`. The logits store
`clip(smooth(upsampled_log_odds), ±16)` — the *smoothed* field, on the mask grid
— so `logits > 0` **is** the mask, bit for bit.

```python
from sam2_logits import mask_from_logits

mask = mask_from_logits(logits_array)                  # == the production mask
mask = mask_from_logits(logits_array, threshold=1.5)   # a stricter provisional mask
```


#### Mask persistence

`sam2_persistence.py` applies the same k-of-n rule at the pixel level
as for patch detections, and it writes one **onset raster** per
UTM/latitude band group: uint16, where each pixel holds the period in
which mining was first confirmed there.

```
# one band group
python sam2_persistence.py --group utm19_lat_-8_0 \
    --quarters Q125 Q225 Q325 Q425 Q126 Q226 \
    --outdir ../data/outputs/sam2/persistence_masks

# then mosaic the groups into the master COG
python sam2_persistence.py --mosaic \
    ../data/outputs/sam2/persistence_masks/amazon_basin_mining_scar_masks.tif \
    --outdir ../data/outputs/sam2/persistence_masks
```

Onset is recomputed from the full stack on every run. A new period supersedes an
earlier provisional call structurally, with no incremental state to go stale.

### Publishing outputs

Artifacts are **not** checked into git.
[`data/outputs/MANIFEST.yaml`](../data/outputs/MANIFEST.yaml) is the in-repo
catalogue of what lives where; keep it current.

Three buckets, with distinct jobs:

| bucket | holds | class |
| --- | --- | --- |
| `gs://amw-published` | the data store of record — published outputs only, object versioning on | Standard |
| `gs://amw-dev/published/` | server-side copy of the above | Standard |
| `gs://amw-image-caches` | Sentinel-2 caches, needed only to re-run a model or SAM2 | Archive |

Source Cooperative mirrors the public subset *from* `gs://amw-published`, never
the reverse. If the two disagree, the bucket is correct.

Staging is scripted. `scripts/stage_outputs.py` assembles both trees under
`data/`, with clean consumer-facing filenames, a `config.txt` in each directory,
and the bucket README rendered from
`scripts/templates/amw_published_README.md`:

* `data/staging_gs/` — everything, for `gs://amw-published`
* `data/staging_source-coop/` — the public subset only

### Footnote: Embedding-based models

As of May 2026, our best detection model remains an ensemble of CNNs trained from scratch. We experimented extensively with models constructed as (ensembles of) probes trained on top of geo-foundation model embeddings, and the code still supports this alternate paradigm. Use an embedding notebook then `train_probe.ipynb` in place of `train_model.ipynb`, and otherwise follow the same workflow.

Two embedding exports are supported (SSL4EO ViT-S/16):

* **Class token only** — `embed.ipynb` (`embedding_strategy=cls_only`). Legacy path; one vector per chip from the ViT CLS token.
* **CLS + ViT patch token** — `embed_cls_patch.ipynb` (backed by `embed_cls_patch.py`; `embedding_strategy=cls_patch`). Concatenates the CLS token with one selected spatial patch token (768-d for ViT-S/16). We experimented with this denser representation; it remains supported for training and for bulk inference via `--embedding_strategy cls_patch`.

In `train_probe.ipynb`, point at the parquet from the chosen embed step and set `EMBEDDING_STRATEGY` to match (`cls_only` vs `cls_patch`). For `cls_patch`, the probe expects feature columns `cls*` then `spatial*` (see `model_library.MLP_with_targeted_dropout`).

Our foundation model of choice was the [SSL4EO ViT DINO S/16](https://github.com/zhu-xlab/SSL4EO-S12), also available through TorchGeo and HuggingFace. We caution that filenames, and therefore potentially the model, have been updated since we downloaded the file `dino_vit_small_patch16_224.pt`. The code looks for this checkpoint at `models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt`.
