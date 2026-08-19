<!-- Rendered into the published bucket by scripts/stage_outputs.py.
     Edit this template, not the copy in data/staging_gs/. -->
# gs://amw-published — Amazon Mining Watch data store of record

**Updated ${updated}.** Model `${model}`, annual periods ${years} and quarters ${quarters}.

This bucket holds the published detection and mine-scar-segmentation outputs and
is the **data store of record**. Source Cooperative mirrors the public subset
from here; never the reverse. If the two disagree, this bucket is correct.

Nothing here is in the GitHub repo. Layout, filename grammar and threshold
parameters are documented there in `data/outputs/MANIFEST.yaml`, and each
directory carries a `config.txt` sidecar recording the model and parameters that
produced it — consumer-facing filenames deliberately omit them.

This bucket holds published outputs only. Image caches, training patches and
prior-model outputs live in other buckets; `MANIFEST.yaml` records which.

**One model vintage at a time.** Paths carry no model or SAM2 version, because the
bucket never holds two: when the detection model or the segmentation model
changes, the whole contents are moved to another bucket and replaced wholesale.
So a path means the same thing across every file here, and the sidecars say which
vintage that is.

## Contents

| | |
| --- | --- |
| `amazon_basin_detections_first_year.geojson` | The authoritative detection product: one row per location, the year mining was confirmed to have begun, and confirmed/provisional status. Every cumulative derives from it. |
| `amazon_basin_mining_scar_masks_first_year.tif` | The raster analogue. uint16; years are the year, quarters are `year*10+quarter` (Q125 = 20251), so any cumulative is `0 < onset <= code`. 0 means no confirmed mining. |
| `amazon_basin_mining_scar_masks_first_year.tif.aux.xml` | GDAL statistics sidecar for the COG above. Keep it. Onset pixels are 0.008% of the raster, so without precomputed statistics QGIS's approximate sampling finds none of them and renders the layer blank. Regenerate with `gdalinfo -stats` if it is ever lost. |
| `raw_detections/` | Per-period, unpostprocessed. Basin at t0.4, andes supplemental at t0.2, flat in one directory. |
| `postprocessed/` | Per-period at t0.43 / t-iso0.75. The loose set; SAM2 is prompted from this. |
| `postprocessed_t0.55_d5_3km_t-iso0.8/` | Stringent set. Stands in for temporal corroboration at the provisional edge only. |
| `cumulative/` | Per-period cumulative patches, plus `patch_diffs/` — each period's new patches, the SAM2 prompt set for quarterly segmentation. |
| `cumulative_dissolved/` | Display polygons and yearly increments under `diffs/`. The front end converts these to pmtiles. |
| `mining_scar_masks/` | Per-run tile masks and logits, one directory per segmentation run, each with a `mask_config.txt`; a folder-level `config.txt` names the SAM2 checkpoint and fine-tuned weights common to all of them. Plus `persistence_masks/` holding the per-band onset rasters the master COG is mosaicked from. |

## The non-public part is not archival

Source Cooperative carries only the first-year layers, `postprocessed/` and
`raw_detections/`. **The rest is required to produce the next update.** Temporal
persistence is recomputed from the full period stack at every refresh rather than
patched incrementally — that is what keeps it from drifting — so deleting the
supporting series breaks future quarterly and yearly updates, not just history.

## What it costs to lose

Two things are expensive or impossible to recreate:

- **`raw_detections/`** — hours of Earth Engine inference per period, against
  composites that may not be reproducible later.
- **`mining_scar_masks/`** per-tile masks and logits — roughly 3.5 hours of
  CPU segmentation per annual period.

Everything else regenerates from those in under two hours with the code in the
repo. If storage has to be reclaimed, reclaim derived products.

## Backup

Object versioning is enabled on this bucket, with noncurrent versions expiring
after 90 days: the realistic risk is an accidental overwrite or delete, not media
failure. A second copy lives at `gs://amw-dev/published/`, synced server-side.

**Verify counts after any bulk transfer.** `gsutil cp -I` has been observed
reporting success while copying 2 of 15,752 files, and an unquoted shell glob
exceeds `ARG_MAX` and fails part-way. Compare
`gsutil ls 'gs://amw-published/**' | wc -l` against the source file count.

## Contact

info@earthgenome.org — Earth Genome. Data licensed CC-BY-4.0.
