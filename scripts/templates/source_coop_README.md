<!-- Rendered into the public mirror by scripts/stage_outputs.py.
     Edit this template, not the copy in data/staging_source-coop/. -->

![EarthGenome](https://data.source.coop/earthgenome/earthindexembeddings/logo.png)
# Amazon Mining Watch

[![CC-4 license](https://img.shields.io/badge/License-CC--4-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Sensor](https://img.shields.io/badge/Sensor%F0%9F%9B%B0%EF%B8%8F-Sentinel2-blue)](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2)
[![Format](https://img.shields.io/badge/Format-GeoJSON%20%F0%9F%8C%8E%EF%B8%8F-blue)](https://geojson.org/)

**Updated ${updated}.** Model `${model}`, annual periods ${years} and quarters ${quarters}.

This repository contains automated detections of artisanal gold mine scars in
Sentinel-2 satellite imagery.

The data is licensed under the [creative commons 4.0](https://creativecommons.org/licenses/by/4.0/)
international license which, to summarize, only requires attribution.

Code, description of the machine-learning models that generated the detections,
and additional versions of this data are available at the
[Amazon Mining Watch Github repository](https://github.com/earthrise-media/mining-detector).

### Data

Coverage is yearly from 2018, and both yearly and quarterly from 2025.

**Patch detections are the primary product.** A machine-learned model
 makes probabilistic assessments on 480 m × 480 m square patches of
 the Earth's surface as to whether traces of mining activity are
 visible within a patch. These are its outputs. The model canvases
 over 100 million patches across the Amazon basin at each time step.

| file | what it is |
| --- | --- |
| `amazon_basin_detections_first_year.geojson` | One row per location, carrying the period in which mining was first confirmed there, and whether that call is confirmed or provisional. |

This file replaces the per-period cumulative files published earlier in 2026, which have been removed.


**Mining scar masks are at a relatively early stage of
  development**. They provide footprints of mining activity in and
  around the patches detected by the primary model. From the
  footprints, total mined area estimates for various jurisdictions are
  surfaced on the Amazon Mining Watch website.

| file | what it is |
| --- | --- |
| `amazon_basin_mining_scar_masks_first_year.tif` | Pixelwise mine-scar extent at 10 m Sentinel-2 resolution, from a fine-tuned [SAM2](https://ai.meta.com/research/sam2/) model prompted by the detections above. |


#### Confirmed and provisional detections

The patch detections are an accumulated historical record of lands
that have been mined since the beginning of monitoring in 2018. A
detection is published as **confirmed** once it appears in two annual
periods within a two-year window. The period recorded is the first of
those.

Recent quarterly data are marked **provisional** and are held to a
stricter confidence threshold instead. Provisional data will be
replaced, not amended, once the corroborating year arrives. A
confirmed detection is never withdrawn by a later update.

For the mining scar masks, pixel values indicate the time period: a
year is the year, a quarter is `year*10 + quarter`, so 2024 takes the
raster value `2024` and Q1 2025 takes the value `20251`. Zero
means no confirmed mining. The two-year persistence rule is applied to
annual data also at the pixel level.

Segmentation is the least mature part of this pipeline and areas are
known to run large compared to hand-drawn
boundaries. Comparisons *between* periods are more trustworthy than
any single total. More details can be found on our [Github
repository](https://github.com/earthrise-media/mining-detector).

`amazon_basin_mining_scar_masks_first_year.tif.aux.xml` sits beside the
raster, holds precomputed statistics, and may help rendering in GIS
software.

#### Single-period data

The individual yearly and quarterly assessments are for users who want the
unaggregated model output:

- `raw_detections/` — direct model output, every patch at or above a confidence of
  0.4, before any filtering.
- `postprocessed/` — the recommended single-period product. Mining activity
  clusters in space, so isolated candidate detections are held to a higher
  confidence than clustered ones (`t_main = 0.43`, `t_iso = 0.75`). This is also
  the set the segmentation model is prompted from.

#### `archived/`

`archived/` data were published here in January, 2026, from an
experimental model that has since been superseded by the current model
version. We include them here for the record. The data have the limitation
and the virtue of having been extensively cleaned by hand prior to
publication.

### Models

The primary mining activity detector is [an ensemble of convolutional neural
networks](https://github.com/earthrise-media/mining-detector/blob/main/models/${model}.h5),
trained on a dataset of approximately 20,000 locations labeled as mines or
mine-free areas. This model was wholly redeveloped in 2026 and the data recomputed
starting from year 2018.

Mining scar masks are computed by a fine-tuned [SAM2 (Segment
Anything)](https://ai.meta.com/research/sam2/) model applied in the
near field of view surrounding the cumulative detections.

### Contact

Using this? Want to work together? Ping us @ info@earthgenome.org or info@earthindex.ai
