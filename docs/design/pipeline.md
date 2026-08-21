# Pipeline: from inference to published product

**Recorded 2026-08-18. Built 2026-08-19.** How the whole product runs end to end,
by someone who did not build it. `core/README.md` is the operator's version of this;
this document is why it is shaped that way.

`scripts/pipeline.py` is the driver. Every scripted stage has been exercised, the
staged trees have been built end to end, and the argument handling and command
construction are verified — but **the full chain has never been driven from
`inference` through `publish` in one pass.** That is the next quarterly refresh.

## Constraints that shape the design

- **Everything runs on the VM.** Transfers to a laptop are slow, so the pipeline
  produces its outputs on the VM and pushes once, at the end, to
  `gs://amw-published` (the store of record), which is mirrored server-side to
  `gs://amw-dev/published/`. Local review pulls from there. The published bucket
  holds **one model vintage at a time**: on a model change its contents move
  elsewhere wholesale rather than accumulating versions in the paths.
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

**Two kinds of configuration.** `scripts/pipeline_config.py` holds what a human
sets — `MODEL`, `ALL_CURRENT_PERIODS`, `SUBREGIONS` and the thresholds — because
those are the contract *between* stages: they appear in filenames, so every stage
must agree. `Config` dataclasses hold what belongs to one module. Threshold-named
directories are derived via `postprocess_tag()` rather than written out, so a name
and the parameters behind it cannot disagree.

**Period vocabulary is a leaf module.** `core/periods.py` holds `Period`,
`QUARTER_SPANS` and `encode_period`, with no third-party imports. It used to live in `persistence.py`, which meant `pipeline.py --list`
loaded geopandas and pyproj to print stage names, and put the vocabulary that
*produces* every inference date downstream of the module that *consumes*
inference output. `date_span` is the single source of every calendar date the
pipeline emits.

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
  amazon_basin_detections.geojson
  amazon_basin_mining_scar_masks.tif
  postprocessed/
    amazon_basin_2018_t0.43_t-iso0.75.geojson   ... 2025, Q125 ... Q226
    config.txt
  raw_detections/                  # flat, no andes nesting
    amazon_basin_2018_t0.4.geojson ...
    andes_supplemental_2018_t0.2.geojson ...
    config.txt

data/staging_gs/                   # internal: the above, plus
  postprocessed_t0.55_d5_3km_t-iso0.8/
  cumulative/                      # patch detections per period end
    amazon_basin_cumulative_2018-2018.geojson ... 2018-Q226
    patch_diffs/                   # SAM2 prompt set for quarters
  cumulative_dissolved/            # display product; front end converts to pmtiles
    diffs/
  mining_scar_masks/               # copied from data/outputs/sam2/
    <run>/                         # tile-wise *-msk.tif, *-logits.tif, mask_config.txt
    persistence_masks/             # per-band onset rasters
    config.txt                     # SAM2 checkpoint and weights common to every run
```

**Published data is patches, not dissolved polygons.** The dissolved layers are not
mirrored to Source Cooperative; the front end pulls them from `gs://amw-published`
and converts to pmtiles.

**Per-band onset rasters are kept**, not just the master COG: 25 MB against
61 minutes to rebuild, and they are the intermediate the master is mosaicked
from, so a single band can be redone without redoing 23.

## Stages

| stage | run by | tool |
| --- | --- | --- |
| `review-config` | **human** | prints `ALL_CURRENT_PERIODS` — stage 0, because the period list is the one thing a human must set |
| `inference` | **human** | `inference_pipeline.py`, per subregion × period |
| `concat` | pipeline | `scripts/concatenate.py` — dedupes seam duplicates |
| `filter` | pipeline | `scripts/geo_filter.py` — clips andes to its boundary |
| `postprocess` | pipeline | `postprocess.py` at t0.43 and t0.55 |
| `persist-detections` | pipeline | `persistence.py --dissolve` |
| `mask-annual` | **human** | `sam2_mask.py` on postprocessed t0.43 |
| `mask-quarterly` | **human** | `sam2_mask.py` on `patch_diffs/` |
| `cog` | pipeline | `sam2_build_cog.py`, on the run directories the given periods name |
| `persist-masks` | pipeline | `sam2_persistence.py` per band group, then mosaic |
| `stage` | pipeline | `scripts/stage_outputs.py` — lay out both trees, sidecars, READMEs |
| `manifest` | pipeline | **stub** — prints a reminder; `MANIFEST.yaml` is still hand-maintained |
| `publish` | **human** | emits the push to `gs://amw-published`, the mirror, and source.coop |

**`accumulate-quarterly` was dropped, not deferred.** The plan was to OR each
quarter's diff mask onto the prior mask to build a `*_full` raster per quarter. The
onset raster removes the need: a pixel stores the period mining was first confirmed
there, so any cumulative is the threshold `0 < onset <= code` and no accumulation
step exists to get wrong. `core/sam2_combine_masks.py` still implements the OR merge
but nothing calls it.

Stages are **idempotent** — skip what exists — and **verify their own output**,
because the failure modes here are silent. Observed in one session: `gsutil cp -I`
reporting success while copying 2 of 15,752 files; `sam2_mask.py` defaulting to
2023 imagery when dates are omitted; shell globs exceeding `ARG_MAX` and failing
mid-list. None announced itself; all are catchable by counting outputs.

**For human-run stages the pipeline emits commands rather than running them.**
That is what keeps error-prone parameters — dates, cache directories, the andes
`_0.2_` naming, the boundary and `--outpath` arguments `geo_filter.py` requires —
derived from the period list instead of typed. There is no flag for it: printing is
the only thing the driver does with those stages, so asking for it would be asking
the operator to restate what the stage already is. `--dry-run` is the one flag that
means "change nothing", and it applies to the scripted stages.

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
  within `Y`. Not an OR — that double-counts expansion. Promotion works, since
  `published_periods` publishes an annual as soon as the rule can resolve it.
  **Retirement does not:** the superseded quarters keep publishing alongside it, and
  nothing deletes a layer that stops being staged. First bites Q1 2027; tracked as
  "Retiring superseded quarterly layers" in persistence-planning.

## What was built, and what is left

The build order this document opened with, as it stands:

1. ~~`accumulate-quarterly`~~ — dropped; the onset raster removes the need.
2. **The yearly supersede for masks — half done.** Promotion works, retirement does
   not. See "The refresh cases" above.
3. `stage` — done, `scripts/stage_outputs.py`, including both READMEs rendered from
   templates in `scripts/templates/`.
4. `manifest` — **still a stub.** It prints a reminder; `MANIFEST.yaml` is
   hand-maintained.
5. `pipeline.py` — done.
6. `core/README.md` — rewritten.

Two things worth knowing before the next run:

- **The full chain has never run in one pass.** Scripted stages have been exercised
  individually and the staged trees built end to end, but always with the human
  stages performed out of band. The first true test is the next quarterly refresh.
- **A quarterly update means editing `ALL_CURRENT_PERIODS` in
  `scripts/pipeline_config.py`,**
  then naming the new period with `--periods`. The flag is required and its values
  must be members of that list, because `persist-detections`, `persist-masks` and
  `stage` read the list rather than the flag -- they recompute from the whole
  history, and a one-period stack has no witnesses. `pipeline.py` refuses an
  unlisted period rather than half-running it, and stage 0 exists to put the list
  in front of the operator first.
