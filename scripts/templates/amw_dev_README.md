<!-- Static; upload to the bucket root by hand after editing:
     gcloud storage cp scripts/templates/amw_dev_README.md gs://amw-dev/README.md -->
# gs://amw-dev — Amazon Mining Watch development store

The working store for model development. Five stores divide the project:

| | |
| --- | --- |
| `gs://amw-dev` | **this bucket** — training data, model weights, experimental and alternate outputs, and a backup of the published set |
| `gs://amw-published` | the **data store of record**. Carries its own README. |
| `gs://amw-archived` | the Sentinel-2 mosaics inference ran on, in ARCHIVE class for a future rebuild |
| `gs://amw-models` | public read, requester pays; the fine-tuned SAM2 weights, copied from here |
| Source Cooperative | public mirror of a **subset** of `amw-published` — not a full mirror; the published bucket deliberately holds files that are not made public |

Nothing in this bucket is the data of record. That does not make it expendable —
see the training data below.

## Size, measured September 2026

| | | |
| --- | --- | --- |
| `training_patches*/` | 93.7 GB | 91 GB of it the 433/443 px sets; the 48 px sets the production model uses are only ~1.2 GB |
| `published/` | 19.6 GB | 257k objects, nearly all per-tile masks |
| `outputs/` | 3.0 GB | before the redundancy cleanup described below |
| `SAM2_finetuned_weights/` | 0.2 GB | |
| **total** | **~116 GB** | about $3/month at Standard, US multi-region |

## Contents

### `training_patches*/` — our training data - apply caution before a delete.

Extraction date and patch size are in each directory name. The data is regenerable with a Google Earth Engine connection, from `core/get_training_data.ipynb` and a collected_locations file in `data/sampling_locations/`.

### `SAM2_finetuned_weights/`

The weights that adapt generic SAM2 to mining-scar segmentation, and the only
surviving artifact of that fine-tuning run. This is the master copy; the public
one at `gs://amw-models/` is pushed from here by hand, so a delete there is
recoverable and a delete here is not.

### `published/`

A straight backup of `gs://amw-published`. Redundant with it by design — that is
the point, and it is not a cleanup target.

### `outputs/`

Two unrelated things sharing a directory:

- `48px_v0.X-SSL4EO-MLPensemble/` — outputs from the **experimental SSL4EO-MLP
  model.** These exist nowhere else; the published store has never carried this
  model's results.
- `48px_v4.10b-18d-20g-21a-22bc-ensemble/` — Hot-off-the-press model outputs, most of it duplicated exactly in `published/`, after being renamed.

## Filenames here do not match `published/`

Files in `outputs/` carry the model and threshold parameters in full:

    Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40_2018-01-01_2018-12-31_t0.43_d5_3km_t-iso0.75.geojson

`amw-published` renames the same content to consumer-facing form, deliberately
dropping those parameters:

    amazon_basin_2018_t0.43_t-iso0.75.geojson

Matching on name and modification date across `outputs/` and
`published/` returns zero overlap; matching on content hash returns well over a
gigabyte of it. Compare with `md5Hash` from the JSON objects API, using size
equality as a cheap pre-filter, and never conclude "no overlap" from names
alone.
