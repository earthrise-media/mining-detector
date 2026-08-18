# Pipeline: from inference to published product

**Recorded 2026-08-18.** Target design for running the whole product end to end,
by someone who did not build it. Supersedes the step-by-step in `core/README.md`,
which predates persistence and is out of date in several places.

Nothing here is built yet except the individual tools it composes. The build order
is at the end.

## Constraints that shape the design

- **Everything runs on the VM.** Transfers to a laptop are slow, so the pipeline
  produces its outputs on the VM and pushes once, at the end, to
  `gs://amw-dev/`. Local review pulls from there.
- **Two cadences, not one.** The bulk rewrite happening now and the quarterly
  refresh are the same pipeline with a different period list. A yearly refresh
  additionally *replaces* the previous year's provisional data.
- **Inference is human-orchestrated.** Detection and SAM2 inference run for hours
  across parallel tmux sessions on 2–3 VMs; GEE tolerates about a dozen
  concurrent sessions before error rates climb. The pipeline does not run these,
  it *emits the commands* for them.
- **Persistence is stateless.** Onset is a pure function of the full period
  stack, so a refresh recomputes rather than patches, and cannot drift. This is
  what makes the incremental case cheap: seconds for detections, ~1 hour for the
  mask side.

## Naming and provenance

Three audiences, three mechanisms, and they are not interchangeable:

| | answers | lives |
| --- | --- | --- |
| **short filenames** | which period, which threshold | the filename |
| **`config.txt` sidecar** | what produced this directory | beside the data |
| **`MANIFEST.yaml`** | what the product contains, and where it mirrors | the repo |

Filenames drop the model version and the isolation parameters, so a sidecar per
output directory records model version, thresholds, `k`, isolation distance, and
for SAM2 the smoothing sigma, clamp and prior sigma. This is not redundant with
the manifest: **the data leaves the repo and the manifest does not.** Two
vintage-ambiguity incidents in one week — the unsmoothed logits reading as a valid
mask, and the pre-upsample logits at a different grid — were both resolved by
provenance travelling with the data. `core/inference_engine.py` already does this
via `MaskConfig.write_config()`; `postprocess` and the raw-detection write need
the same.

## Target layout

Everything lands under two staging trees, neither ever checked in:

```
data/staging_source-coop/          # public: patches only
  amazon_basin_detections_first_year.geojson
  amazon_basin_mining_scar_masks_first_year.tif
  postprocessed/
    amazon_basin_2018_t0.43_t-iso0.75.geojson   ... 2025, Q125 ... Q226
    config.txt
  raw_detections/                  # flat, no andes nesting
    amazon_basin_2018_t0.4.geojson ...
    andes_supplemental_2018_t0.2.geojson ...
    config.txt

data/staging_gs/                   # internal: the above, plus
  postprocessed_t0.55_d5_3km_t-iso0.80/
  cumulative/                      # patch detections per period end
    amazon_basin_cumulative_2018-2018.geojson ... 2018-Q226
    patch_diffs/                   # SAM2 prompt set for quarters
  cumulative_dissolved/            # display product; front end converts to pmtiles
    diffs/
  sam2/
    <run>/                         # tile-wise *-msk.tif, *-logits.tif, mask_config.txt
    persistence_masks/             # per-band onset rasters
```

**Published data is patches, not dissolved polygons.** The dissolved layers stay
internal; the front end pulls them from `gs://amw-dev/` and converts to pmtiles.

**Per-band onset rasters are kept**, not just the master COG: 25 MB against
61 minutes to rebuild, and they are the intermediate the master is mosaicked
from, so a single band can be redone without redoing 23.

## Stages

| stage | run by | tool |
| --- | --- | --- |
| `inference` | **human** | `inference_pipeline.py`, per subregion × period |
| `concat` | pipeline | `scripts/concatenate.py` — dedupes seam duplicates |
| `filter` | pipeline | `scripts/geo_filter.py` — clips andes to its boundary |
| `postprocess` | pipeline | `postprocess.py` at t0.43 and t0.55 |
| `persist-detections` | pipeline | `persistence.py --dissolve` |
| `mask-annual` | **human** | `sam2_mask.py` on postprocessed t0.43 |
| `mask-quarterly` | **human** | `sam2_mask.py` on `patch_diffs/` |
| `accumulate-quarterly` | pipeline | **not built** — OR each quarter onto the prior |
| `cog` | pipeline | `sam2_build_cog.py` per run |
| `persist-masks` | pipeline | `sam2_persistence.py` per band group, then mosaic |
| `stage` | pipeline | rename and lay out the two trees, write sidecars |
| `publish` | pipeline | push to `gs://amw-dev/` and source.coop |
| `manifest` | pipeline | regenerate `MANIFEST.yaml` |

Stages are **idempotent** — skip what exists — and **verify their own output**,
because the failure modes here are silent. Observed in one session: `gsutil cp -I`
reporting success while copying 2 of 15,752 files; `sam2_mask.py` defaulting to
2023 imagery when dates are omitted; shell globs exceeding `ARG_MAX` and failing
mid-list. None announced itself; all are catchable by counting outputs.

**For human-run stages the pipeline emits commands rather than running them.**
That is what keeps error-prone parameters — dates, cache directories, the andes
`_0.2_` naming, the boundary argument `geo_filter.py` requires — derived from the
period list instead of typed.

## Ordering

Detections must be complete before either persistence stage: `persistence.py`
reads every period. The quarterly mask prompts come *from* it, so the chain is

```
inference → concat → filter → postprocess → persist-detections
          → mask-annual, mask-quarterly → cog → persist-masks → stage → publish
```

`mask-quarterly` cannot start before `persist-detections`, since `patch_diffs/`
is its input.

## The refresh cases

**Quarterly.** New period only for inference and SAM2; both persistence layers
recompute in full. No replacement: the quarter is added to the provisional edge.

**Yearly.** Year `Y` becomes confirmable once `Y+1` exists, so the refresh
*replaces* rather than adds:

- Detections: `Y`'s provisional quarters give way to confirmed `Y`. Stateless
  recompute handles this — but note the published quarterly provisionals are
  **replaced, not promoted**; 45.5% of them can never confirm, since they come
  from different imagery. See persistence-planning, "The provisional edge is
  replaced, not confirmed".
- Masks: `Y`'s annual mask **supersedes** the accumulated quarterly estimate
  within `Y`. Not an OR — that double-counts expansion. This is the step that
  does not exist yet.

## Build order

1. `accumulate-quarterly` — OR each quarter's diff mask onto the prior mask.
2. The yearly **supersede** step for masks.
3. `stage` — the rename/layout layer and the sidecar writers.
4. `manifest` — regenerate from the staged trees rather than by hand.
5. `pipeline.py` — the driver tying the above together, with `--print` for the
   human-run stages.
6. Rewrite `core/README.md` from this spec once it runs.
