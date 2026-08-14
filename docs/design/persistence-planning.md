# Stabilizing cumulative estimates with a temporal persistence check

**Recorded:** 2026-08-07 (design discussion; no implementation yet.)  
**Updated:** 2026-08-07 — prototype run on real data. Adds the **nested rule** (early confirmation via `n=2`, provably non-withdrawing), measured results for masks and detections, test-set metrics, and the **pipeline ordering** decision for SAM2. Supersedes the flat rejection of early resolution below.

## Background: two halves of the pipeline

Our product has two components. **Detections** are the polygon layer identifying *where* mining is present. **SAM2 rasters** are the segmentation masks that measure *how much* area each detected site covers. They behave differently over time and currently have different problems.

## Problem 1: the rasters fluctuate — this is what our partner sees

SAM2 masks are recomputed from scratch at every period. We do this deliberately: it lets us capture expansion within previously detected sites, and we avoid merging masks across periods because mask generosity varies with mosaic quality, so a simple merge would systematically overestimate area. The cost is that reported area moves up *and down* between periods purely from imagery conditions — cloud cover, season, compositing artifacts — which contradicts our description of the layer as cumulative.

## Problem 2: the detections don't fluctuate, but their precision decays

Detections are accumulated as a union — once a site is detected it stays in the record — so the detection layer is already monotonic and does not fluctuate. Its problem is different: each period contributes a fresh crop of false positives while true sites saturate, so precision degrades as the archive lengthens. We currently patch this by applying a stricter confidence threshold to the detections that feed the cumulative product (`t0.55/t-iso0.8` rather than the single-period `t0.43/t-iso0.75`). That works, but it is an ad hoc correction: it has to be re-tightened as the record grows, and it costs us sensitivity to genuine mine expansion.

## Proposed change

Require temporal corroboration before a location enters the cumulative record: a detection or mask pixel must be confirmed in at least **2 of 3 consecutive annual periods** (`k=2`, `n=3`). Once confirmed, it stays permanently.

This works because real mine scars are permanent and recur every year, while cloud and mosaic artifacts appear once and vanish. Requiring two occurrences removes most transient errors while leaving genuine features essentially untouched.

### The nesting property, and the nested rule

**`k=2/n=2` confirmations are a strict subset of `k=2/n=3` confirmations.** If a location is detected at `Y` and `Y+1`, it necessarily has 2 detections inside `[Y, Y+2]`, so it satisfies 2-of-3. The converse fails: the gap pattern (detected, missed, detected) passes `n=3` but not `n=2`. Verified exhaustively over all 256 possible 8-year detection sequences: zero violations.

Two consequences:

1. **Early confirmation is safe.** Apply `n=2` only where the 3-window is not yet complete, and it can *only ever add* locations, never remove them. The confirmed layer stays monotonic under reprocessing as well as under time. Operationally this is just `min(onset_n2, onset_n3)` — both are deterministic functions of the full stack, so there is no per-detection confirmation state to track across runs. This is what replaces the rejected "resolve early" idea.
2. **`n=2` now, `n=3` later is a safe migration.** Switching from the shorter to the longer window is purely additive, so choosing `n=2` as an interim regulator does not lock us out of `n=3` when mask accuracy improves.

Under the nested rule the provisional edge shrinks from two years to one: with annual data through 2025, 2024 is confirmable via the `n=2` path and only 2025 remains fully provisional.

## Why we propose to apply this to both halves

**For the rasters**, it is the direct fix: it makes reported area monotonic by construction — the total can never fall — without the systematic overestimate that a naive merge would introduce.

**For the detections**, it is not fixing a fluctuation, since there isn't one. It replaces our ad hoc stricter cutoff with a principled use of real temporal evidence. Instead of demanding higher confidence from a single look, we ask whether the site is actually still there the following year — which is the genuine signal, and should let us relax the confidence threshold back toward the value we tuned on single-period data while *improving* precision. We expect a net gain in both precision and sensitivity to expansion, **but this needs to be measured before we commit.**

**For both together**, consonance matters: the two halves should share one definition of when a site enters the cumulative record. Otherwise the polygon layer and the area layer admit sites on different rules, and the first-detection dates below would disagree between them.

**But apply them as a hierarchy, not in parallel.** Running the same rule independently on two spatial supports lets them disagree — a mask pixel can confirm at 2020 while its detection confirms at 2021, if one had a gap year the other did not. Resolve by making **detections authoritative for *when*, masks authoritative for *how much***: confirmed area at year `Y` = persistence-confirmed mask pixels lying within detections confirmed by `Y`. The polygon layer sets onset dates, the mask layer measures extent within them, and the first-detection-year layer has a single unambiguous source.

## Pipeline ordering: where persistence sits relative to SAM2

SAM2 runs prompted by detections, which raises the question of whether persistence should filter the detections *before* segmentation. It should not.

**Run SAM2 on the full, unfiltered per-period `t0.43` detection set.** Persistence then applies downstream as a pure selection over masks that already exist.

- **No rerun is ever needed.** Under every rule, onset at `Y` requires a detection at `Y`, so any location that can ever confirm was already in year `Y`'s raw detection set and already has a mask. Later confirmation *promotes* an existing mask; later rejection *discards* one. Neither needs new inference.
- **It keeps the factorization clean.** Everything inside a period stays period-local and all temporal logic lives in one place. Filtering before SAM2 would inject temporal state into the per-period stage, so the contents of a given year's SAM2 pass would depend on which later years happen to exist — reprocessing the same year at different times would give different inputs.
- **SAM2's input must be the loose set**, not the tightened one. If SAM2 ran on `t0.55` while persistence ran on `t0.43`, loose-only locations could confirm with no mask available.
- **Cost:** roughly 20% more segmentation than survives — the locations persistence eventually rejects. Acceptable, and the price of never needing a rerun.

### Reconciling the provisional edge: filter, never rerun

The published provisional years use the tightened `t0.55` set while SAM2 ran on `t0.43`. Reconcile by **filtering the existing masks**, not by re-running SAM2 on the `t0.55` subset.

The reason is correctness, not cost: **SAM2's output depends on its prompt set.** A rerun on a different detection subset changes which polygons clip into each tile, hence the box prompts, hence the segmentation — so a `t0.55` rerun does *not* yield the `t0.43` mask restricted to `t0.55` locations. If provisional years used rerun masks and confirmed years used `t0.43` masks, a location's area would change at the moment it confirms, for no reason connected to imagery. That reintroduces exactly the unexplainable fluctuation this design exists to remove, at the most visible point in the series. Computing masks once makes each location's mask immutable: confirmation changes only *whether* it is included, never *what it says*.

Verified: the `t0.55/iso0.8` detections are a strict subset of `t0.43/iso0.75` (2024: 130,646 of 162,400, zero orphans; 2025: 120,534 of 159,893, zero orphans). So filtering cannot lose a provisional detection.

**Do not filter by clipping to patch footprints** — SAM2 routinely segments scars extending beyond the 48 px patch that prompted them, so clipping shaves real extent. Use **connected-component attribution**: label contiguous mask regions and keep a region whole if *any* retained detection intersects it. Best done per-tile before mosaicking (tiles are 256 px, so labelling is trivial; a 5.4 G-pixel mosaic would need windowed labelling with cross-window merges). Caveat: a scar spanning two tiles could be retained in one and dropped in its neighbour, leaving a seam — unlikely given detection density over real scars, but if observed, move the attribution to after mosaicking within each UTM/lat group.

### A provisional threshold on the mask side (`t_prov,mask`)

The detection-side `t_prov` has a mask-side counterpart: for provisional years, re-derive the binary mask from the logits at a **stricter logit threshold**, so the provisional mask approximates what the persistence-confirmed mask will look like once corroboration arrives. Same principle as everywhere else here — substitute instantaneous evidence for the temporal evidence that does not exist yet.

**Logits never need recomputing; masks do.** The logits are the durable artifact and the binary mask is a cheap re-derivation from them, so no SAM2 or Earth Engine rerun is involved. But it is *not* a bare re-threshold — see the next point.

**Implementation trap in the current code.** In `SAM2_Masker.predict` the two products are not derived from the same array:

```
log_odds = best_logits + prior                       # SAM2 logit resolution
upsampled = self.upsample_logits(log_odds, tile_shape)  # bilinear + Gaussian smooth
mask      = (upsampled > 0)                          # <- mask thresholds THIS
save_tile(mask, product_type="mask")
save_tile(log_odds, product_type="logits")           # <- but THIS is saved
```

The saved `-logits.tif` is `log_odds` — before the bilinear upsample to tile resolution and before the Gaussian smoothing (`smoothing_sigma`). Re-thresholding it directly yields a coarser, unsmoothed mask that will not agree with the production mask even at threshold 0. Any re-derivation must **replay upsample + smooth, then threshold**. Both steps are deterministic and cheap. (Alternatively, save the upsampled/smoothed logits so re-thresholding is a true one-liner — worth considering, at the cost of larger logit rasters.)

**Re-derive per tile, not on the logits mosaic.** Gaussian smoothing and bilinear interpolation do not commute with mosaicking, so thresholding the merged logits mosaic is not equivalent to re-deriving per-tile masks and then merging. (The max-reduce/OR equivalence noted elsewhere holds only for a *uniform threshold applied to already-comparable rasters*, which the pre-upsample logits are not.) Correct order: per-tile upsample → smooth → threshold at `t_prov,mask` → mosaic with the union rule.

**Current baseline:** the production mask thresholds log-odds at **0** (probability 0.5), so `t_prov,mask > 0`.

**Calibration.** New labelled data would let us validate the chosen value, but a first estimate needs no labels at all, because for 2018–2023 we know *both* sides already: the single-year mask at any threshold is re-derivable from stored logits, and the persistence-confirmed mask is known. So sweep the threshold and pick the value at which the single-year mask best reproduces the increment that persistence eventually confirmed for that onset year:

- **Target quantity:** `confirmed(Y) − confirmed(Y−1)`, i.e. the area attributable to onset year `Y`, versus the single-year mask at `t_prov,mask` restricted to locations not already in `confirmed(Y−1)`.
- **Objective:** per-pixel agreement (IoU or F1), *not* area equality — a threshold can hit the right total in the wrong places. Use total-area match only as a secondary check.
- **Starting bracket:** `n3` admits 77–81% of each year's new OR area on the UTM 21 band, so `t_prov,mask` should shed roughly 20% of what a threshold-0 mask would newly add. That is where to begin the sweep.

So the labelling effort is for *validation*, and does not block a first working value.

### Net pipeline shape

1. Detect → postprocess at **`t0.43`** (per-period, no temporal logic)
2. SAM2 on the full `t0.43` set → per-tile masks, computed **once**
3. Select detections: persistence-confirmed, or `t0.55` for the provisional edge
4. Attribute masks to selected detections by connected component
5. Re-derive masks from stored logits: threshold at 0 for confirmed years, at `t_prov,mask` for provisional years (replaying upsample + smooth per tile)
6. Mosaic → persistence-filter mask pixels → cumulative layer

Steps 1–2 are period-local and never recomputed. All temporal logic is in 3–5 and is pure selection over fixed inputs.

## Measured results (prototype, 2026-08-07)

### Masks — UTM 21, lat band [-8, 0], annual 2018–2025

Areas in hectares, cos-latitude weighted. Outputs in `data/outputs/sam2/persistence-tests/` and `persistence-tests2-2/`.

| Year | Existing (per-year) | OR | n3 (2,3) | n2 (2,2) | nested |
| --- | --- | --- | --- | --- | --- |
| 2018 | 93,840 | 93,840 | 84,249 | 80,056 | 84,249 |
| 2019 | 109,528 | 123,313 | 107,902 | 101,309 | 107,902 |
| 2020 | 126,080 | 151,524 | 129,652 | 123,728 | 129,652 |
| 2021 | 147,516 | 184,489 | 155,238 | 148,499 | 155,238 |
| 2022 | 160,092 | 212,903 | 177,027 | 169,137 | 177,027 |
| 2023 | 163,175 | 233,628 | 193,862 | 186,046 | 193,862 |
| 2024 | 174,830 | 254,870 | — | 198,957 | **205,285** |
| 2025 | 165,610 | 268,084 | — | — | — |

- **The fluctuation is real:** the existing series drops **−5.3% (−9,220 ha) from 2024 to 2025**, the only decrease. Both cumulative regimes are monotonic.
- **The ratchet is confirmed:** the OR-vs-n3 gap widens with archive length, −10.2% (2018) → −17.0% (2023), flattening toward ~20% because the per-increment rejection rate is stable (n3 admits 77–81% of each year's new OR area).
- **Pure `n2` costs ~4–4.5% permanently** relative to `n3` — the gap-pattern locations, never recovered.
- **Nested reproduces `n3` exactly** for 2018–2023 and extends to 2024, cutting the provisional share of the eventual total from ~12% to ~7%.
- **A weak 2025 mosaic suppressed the `n2` gain.** The `n2` path admitted only 53.8% of the 2024 OR increment against the ~78% historical rate; roughly **5,100 ha** of genuine 2024 area should confirm when 2026 lands and `n3` can use it as an alternative second witness. Under a wholesale switch to `n2` that area would be lost permanently rather than deferred — the main argument for nesting over switching.

### Detections — basin-wide, annual 2018–2025

Cumulative patch counts. Outputs in `data/outputs/48px_v4.10b-.../persistence_t0.43_d5_3km_t-iso0.75/`.

**Corrected 2026-08-14.** The original table was computed with a centroid join key rounded to
6 decimals. The 2024 GeoJSONs — alone among all years — are written at 6-decimal coordinate
precision (every other year is full float64), so their centroids, formed by averaging two
already-rounded corners, do not land on the same 6-dp key as the other years. Roughly 40% of
2024 patches therefore registered as never-before-seen locations, and the inflation carried
forward into 2025. The table below uses a **5-decimal key**, which is robust to this (the
half-patch grid step is 0.00217°, so 5 dp resolves the grid ~217×) and reproduces the 6-dp
figures exactly for 2018–2023. See "Operational gotchas" for the write-side fix.

| Year | OR t0.43 | OR t0.55 (current) | n3 | n2 | nested | n3 vs current |
| --- | --- | --- | --- | --- | --- | --- |
| 2018 | 117,731 | 95,097 | 103,991 | 99,316 | 103,991 | +9.4% |
| 2019 | 146,836 | 120,377 | 125,548 | 118,629 | 125,548 | +4.3% |
| 2020 | 169,421 | 140,052 | 142,633 | 137,253 | 142,633 | +1.8% |
| 2021 | 194,174 | 161,435 | 162,814 | 156,749 | 162,814 | +0.9% |
| 2022 | 216,771 | 180,806 | 181,537 | 175,336 | 181,537 | +0.4% |
| 2023 | 238,635 | 200,175 | 198,397 | 191,908 | 198,397 | −0.9% |
| 2024 | 263,225 | 220,957 | — | 200,890 | 206,686 | — |
| 2025 | 309,713 | 256,156 | — | — | — | — |

- **Persistence at loose thresholds converges on threshold-tightening**: `n3` declines smoothly
  from +9.4% above the current product at 2018 to −0.9% at 2023, tracking it within ±1% from
  2021 onward. Two unrelated mechanisms landing in the same place is good evidence the tightened
  thresholds were doing roughly the right amount of work and that persistence is a principled
  substitute.
- **The trend is the N-dependence**: a fixed threshold cannot track FP accrual that grows with
  archive length. At 2018 the tightened threshold *over*-corrects (`n3` is +9.4% above it)
  because there is no accrual yet to correct. The correction shrinks year on year and crosses
  over at 2023.
- **There is no 2024 anomaly.** The 2024 OR increment is +24,590 patches, squarely inside the
  21,000–29,000/yr norm, and the same is true at the raw-detection level (+27,495 at t ≥ 0.43).
  The apparent +82,245 was entirely the join-key artifact described above.
- **The 2025 increment is genuinely elevated**, but modestly: +35,199 on the current t0.55
  product against a 2019–2024 mean of 20,977, so **1.68×**, not double. It is larger on the loose
  t0.43 column (1.92×) than the stringent one, which is the direction expected if part of the
  excess is low-confidence detections scattering into new locations rather than new mining. This
  is the kind of excess persistence should absorb once 2026 annual lands.
- **The test-set metrics below predate this correction** and were computed on layers built with
  the 6-dp key. The two OR rows are likely unaffected at chip level (the spurious extra entries
  are geometrically coincident duplicates), but `n3_2023`, `n2_2024`, and `n3_provis_2025` need
  recomputation — `n3` at 2023 gains ~3,400 patches once the key is fixed.

### Test-set metrics

`core/persistence_evaluation.ipynb`, protocol matching `model_evaluation.ipynb` §3 (chip positive iff it intersects any patch). Pooled val+test2+test3, 3,879 chips, 723 positive.

| Layer | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `or_t055_2025` (current) | 718 | 37 | 5 | 3119 | 0.9510 | 0.9931 | 0.9716 |
| `or_t043_2025` (loose, no persistence) | 718 | 82 | 5 | 3074 | 0.8975 | 0.9931 | 0.9429 |
| `n3_2023` (confirmed only) | 709 | 28 | 14 | 3128 | 0.9620 | 0.9806 | 0.9712 |
| `n3_provis_2025` (proposed) | 718 | 33 | 5 | 3123 | 0.9561 | 0.9931 | **0.9742** |
| `n2_2024` | 709 | 27 | 14 | 3129 | 0.9633 | 0.9806 | 0.9719 |

- **The one result with real signal:** persistence recovers the precision that loose thresholds give up at **zero recall cost** — 82 FP → 33 FP with TP unchanged at 718. A 49-chip difference, well outside noise. This is the direct test of persistence as a substitute for threshold-tightening.
- **Everything else is within noise.** `n3_provis_2025` beats the current product on F1, but by **4 false positives**. The honest conclusion is *equivalence* to the current product, plus monotonicity and freedom from N-drift — not superiority.
- **What this protocol cannot show:** N-drift (it evaluates a single 2025 snapshot, so the strongest argument for persistence is structurally invisible), and dating errors (a chip hits if *any* patch intersects it, regardless of onset year).

## Fixing the raster grid

**Recorded 2026-08-14.** Every pixel-wise temporal rule needs pixel *(i,j)* to mean the same
ground location in every period. It currently does not. This section diagnoses why and specifies
the fix. It is a prerequisite for the raster half of persistence, and it lands with the SAM2
rerun that persistence-on-detections requires anyway.

### Diagnosis: three things float at once

1. **Per-tile resolution is back-derived, never declared.** `GEEDataExtractor.affine_from_tile`
   (`core/gee.py`) builds each tile's transform as `from_bounds(*tile.geometry.bounds, width,
   height)` — degrees-per-pixel is `tile_span_deg / n_pixels`. Earth Engine rasterizes at 10 m
   scale and returns an integer pixel count, and metres-per-degree of longitude varies with
   latitude, so every tile lands on a slightly different deg/px. Nothing ever reconciles them.

2. **The mosaic averages those resolutions.** `build_mask_union_vrt` / `build_logit_max_vrt` call
   `gdalbuildvrt -separate` with no `-tr` and no `-resolution`. GDAL's default is
   `-resolution average`, so the output step is the *mean over that year's tile set*. A different
   set of detected tiles each year means a different mean each year. The VRT extent is likewise
   the union of whatever tiles happened to be present, so the origin is inherited from the
   year's westernmost/northernmost tile.

3. **`gdalwarp` never re-anchors.** It runs with no `-tr`/`-te`/`-tap`. Source and destination
   CRS are both EPSG:4326, so the transform is effectively identity and
   `GDALSuggestedWarpOutput` simply reproduces the VRT geotransform. The comment "snap to a
   consistent grid" at `sam2_build_cog.py:191` describes an intent the command does not
   implement.

**EPSG:4326 is the background condition, not the cause.** In a geographic CRS there is no such
thing as "a 10 m pixel" — some nominal degree size has to be chosen by fiat, and because the code
never chooses one explicitly it gets chosen implicitly and differently every run. The same bug is
possible in UTM; conversely 4326 is perfectly stable once a nominal resolution is declared.

### Magnitude

Annual masks, UTM 21 / lat[−8,0]. Resolution spread 2018→2025 is 0.0263%
(9.0531470e-05 → 9.0555260e-05 deg):

| Year | drift from resolution alone, across raster width | origin shift | total offset of pixel *(i,j)* |
| --- | --- | --- | --- |
| 2019 | +36.7 m (3.6 px) | 0 | 3.6 px |
| 2023 | +144.7 m (14.4 px) | −4.85 km | **468 px** |
| 2024 | +153.6 m (15.3 px) | −4.85 km | **467 px** |
| 2025 | +163.8 m (16.3 px) | −4.85 km | **466 px** |

The origin jump between 2019 and 2023 (a tile appearing ~4.85 km further west) dominates, but
even with a shared origin the resolution drift alone reaches ~16 px at the far edge. Either is
fatal for `k`-of-`n`. This is also why existing per-year area figures are not strictly
comparable — they are computed on slightly different pixel areas.

### A fourth misalignment: masks and logits differ within a single run

Measured on the 2025 per-band COGs: the logits raster is coarser than its own mask raster by
**exactly 35/32 = 1.09375**, in every band checked (utm17/18/20/22). The ratio being exact and
band-invariant makes this a per-tile property — the mask tile and the logit tile cover identical
bounds at different pixel counts (560 vs 512) — not an averaging artifact.

So re-thresholding a stored `-logits.tif` cannot reproduce its own mask even at threshold 0, for
*two* independent reasons: the missing upsample + smooth already noted above, and this resolution
mismatch. Making logits the gridded primary and the mask a pure threshold of it resolves both.

### Rasters are already being resampled

Worth stating plainly, because the fix is often mistaken for introducing resampling: tiles are
resampled **today**, silently. Any tile whose native resolution differs from the VRT's averaged
resolution is resampled on read, with **nearest**, plus a sub-pixel origin shift wherever the
tile origin is not an integer number of VRT pixels from the VRT origin.

Under the averaged grid roughly half the tiles are *decimated* (target coarser than source), and
nearest-decimating a binary raster can drop thin features outright. The fixed grid changes the
count of resampling steps not at all; it makes the target deterministic, and — because `R` is
chosen finer than every observed source resolution — turns decimation into mild oversampling.

### The fixed grid

- **Nominal resolution `R = 0.00009°`** (~10.02 m in latitude). Finer than every observed source
  resolution (masks span 9.0404e-05 … 9.0952e-05), so regridding never discards detail. Round
  enough to recognise in a `gdalinfo` dump.
- **Global lattice.** Every raster origin must be an integer multiple of `R` from (0, 0). All
  UTM-zone × lat-band groups then share one lattice and abut exactly, so neighbouring bands
  mosaic without resampling and a `gdalbuildvrt` over them is a zero-copy global view.
- **Fixed extent per group**, derived from the UTM-zone × lat-band definition (not from the
  observed tiles) and snapped outward to multiples of `R`. Zone 21 × lat[−8,0] becomes a fixed
  66,667 × 88,889 raster every year, ~10% more pixels than today's 62,976 × 85,916; the added
  margin is empty blocks that ZSTD reduces to near nothing.
- Pass `-tr R R` plus an explicit snapped `-te` to **both** `gdalbuildvrt` and `gdalwarp` —
  pinning the VRT too avoids a second resample. Compute the snapped extent in Python rather than
  relying on `-tap`, so the result does not depend on GDAL version behaviour. `-r near` for
  masks, `-r bilinear` for logits.

Pixel *(i,j)* then denotes the same ground location in every period and every band, permanently
and under reprocessing, and the ad-hoc regrid the prototype needed before every temporal
operation disappears.

### Logits as the durable artifact

Persistence needs stored logits going forward, not just masks. Three decisions:

- **Include the spatial prior.** `soft_spatial_prior` is a deterministic function of the t0.43
  detection set, which this design has already frozen as SAM2's permanent input. Excluding it
  would mean carrying every tile's detection set forever just to reconstruct `log_odds`.
- **Upsample before storing.** This is what puts logits on the fixed grid and makes them
  mosaickable. Cost is modest: the 35/32 ratio above means ~1.196× more pixels, not the ~5× a
  256×256 SAM2 low-res output would have implied.
- **Do not apply the Gaussian smoothing before storing** — but **keep applying it per tile** when
  deriving masks. Storing unsmoothed keeps `smoothing_sigma` tunable without re-running SAM2;
  it does *not* imply moving the smoothing downstream of mosaicking.

  **Measured 2026-08-14, and it reverses an earlier proposal in this document.** The suggestion
  was to switch from `smooth → threshold → OR per tile` to `max-mosaic → smooth → threshold`,
  on the theory that smoothing after mosaicking sees true neighbours rather than
  `gaussian_filter`'s reflected tile edges. Tested on 13 real SAM2 logit tiles split into
  overlapping halves, scored against the seamless reference (smooth the whole field, then
  threshold):

  | overlap | IoU current | IoU proposed | area err current | area err proposed |
  | --- | --- | --- | --- | --- |
  | 4 px | 0.9595 | 0.9590 | 0.00% | 0.22% |
  | 12 px | 0.9616 | 0.9585 | 0.27% | 0.96% |
  | 32 px | 0.9581 | 0.9511 | 0.31% | 1.82% |

  (Overlap perturbed by σ=2.0 in logit units. At σ=5.0 the proposed ordering inflates area by up
  to 4.47%.) The proposed ordering *is* exact when overlapping tiles agree pixel-for-pixel
  (IoU 1.0000 vs 0.9990), but they do not agree — overlapping tiles come from separate SAM2 runs
  on different image context. **Max-reduce on raw logits is biased upward**, since the max of two
  noisy fields exceeds either mean, and the smoother then spreads that inflated max across the
  seam. Taking the max at the *binary* level, after thresholding, bounds the error instead.

  So the original specification stands: **re-derive per tile — replay upsample + smooth, threshold,
  then mosaic with the union rule.** The logits mosaic produced by `build_cog` is for inspection
  and analysis, not a substrate for mask re-derivation.

  Caveat: the test tiles available had **no genuine overlaps**, so tile disagreement was simulated
  with additive noise. Confirm on real overlapping tiles when they are to hand; the conclusion is
  unlikely to flip, since the upward bias of max-reduce is structural rather than noise-model
  dependent.

- **Confirmed on real tiles: thresholding stored logits does not reproduce the stored mask.**
  Mean IoU 0.84 across 8 tiles (pixel agreement 99.77%, but that is dominated by background).
  For these tiles logits and mask share a resolution, so upsampling is a no-op and the gap is
  *entirely* the Gaussian smoothing. This is the "implementation trap" above, measured: any
  re-derivation must replay the smoothing, and `smoothing_sigma` is not a cosmetic parameter.

**Clamp and quantize.** Measured distribution of stored `log_odds` (utm20 lat[−8,0], 2025,
436k sampled finite pixels): range **−862 to +11**, median **−126**, with **96% of pixels outside
±8** and **91% outside ±16**. The enormous negative tail is the prior, whose penalty
`−(dist_outside / prior_sigma)²` grows quadratically without bound — it is deterministic geometry
carrying no decision-relevant information, and storing it at float32 precision is why the logits
COGs run ~28× the size of their masks (178 MB across 13 bands at current resolution).

Clamping to [−16, 16] saturates ~91% of pixels to a constant that compresses to nearly nothing,
while preserving full fidelity across the entire decision-relevant band (±8 log-odds is
probability 0.00034 to 0.99966 — far wider than any `t_prov,mask` we would calibrate). Store
**int8 at scale 0.125** over that clamped range, with int16 at scale 0.001 as the conservative
fallback. Acceptance test: the quantized logits must reproduce the float32 mask **exactly** at
threshold 0.

**Layout: chunked by UTM zone × lat band, on the global lattice.** The fixed lattice largely
dissolves the chunked-vs-global question, since aligned chunks give a global VRT view for free.
Keep **masks global** (~46 MB basin-wide, and that is the artifact consumers want) and **logits
chunked** (they are ~2 orders of magnitude larger, and the bands are already the unit of
parallelism for the temporal work).

### Measured on real per-tile rasters (2026-08-14)

Verified against 16 real `*-msk.tif` / `*-logits.tif` tiles in
`data/outputs/sam2/test_region_48px_v0.X-.../`, which show a **324 ppm
resolution spread among themselves** — the per-tile drift, confirmed directly
rather than inferred from the mosaics.

| output | size (px) | resolution | MB | sec |
| --- | --- | --- | --- | --- |
| fixed band extent, mask | 68,446 × 90,667 | 0.00009 | 1.86 | 172 |
| fixed band extent, half the tiles | 68,446 × 90,667 | 0.00009 | 1.86 | 141 |
| snapped union extent, mask | 2,825 × 2,832 | 0.00009 | 0.01 | 0.6 |
| snapped union extent, logits | 2,825 × 2,832 | 0.00009 | 3.83 | 1.4 |

Confirmed: different tile subsets yield a byte-identical grid; all outputs sit
on the global lattice at exactly `GRID_RES`; the union extent is an *integral*
pixel offset from the band extent (dx=37,559, dy=−30,849); uncovered ground
reads nodata (2), not 0; and the mask OR / logit max reductions survive.

**Cost note.** The fixed band extent writes a 6.2 Gpx raster regardless of tile
count, so it runs ~170 s/band against ~0.6 s for the tight union. The output is
tiny either way (nodata compresses away). Because the cost is set by output
size rather than input count, the *relative* overhead shrinks at production
tile counts. Both modes are correct — lattice alignment is what temporal rules
need — so `--extent_mode {band,union}` selects between identical-grids
convenience and write speed. Default `band`.

**Area is preserved, not pixel counts.** `GRID_RES` is finer than the sources,
so nearest resampling raises pixel counts by the areal scale factor (~0.6% for
these tiles: 29,710 → 30,109). Ground area is preserved to ~0.7%, the residual
being edge rounding. Well inside the segmentation-area error bar, but worth
knowing before comparing pixel counts across grid regimes.

### Tasks

- [x] Add grid constants (`R`, lattice anchor, band-extent derivation) to `sam2_build_cog.py`.
- [x] Pin `-tr`/`-te` on both `gdalbuildvrt` and `gdalwarp`; drop reliance on `-resolution average`. Also pinned `-r` and `-srcnodata`/`-vrtnodata`, the latter because a fixed extent creates large uncovered regions that must read as nodata rather than 0 — otherwise the mask OR reduces unobserved ground to "observed, not mining".
- [ ] Emit logits on the mask grid (upsampled, prior included, unsmoothed).
- [x] ~~Move Gaussian smoothing downstream of mosaicking~~ — **rejected on measurement**
      2026-08-14; see above. Keep `smooth → threshold → OR` per tile.
- [ ] Clamp + quantize logits; verify exact mask reproduction at threshold 0.
- [ ] Confirm the smoothing-order result on genuinely overlapping tiles when available.
- [ ] Regenerate 2018–2025 masks and logits on the fixed grid as part of the persistence rerun.
- [ ] Confirm the temporal code path needs no regrid step once inputs are aligned.

## Deliverables

- **A first-detection-year layer**: for each location, the year mining was confirmed to have begun. Every cumulative figure we report derives from it and cannot decrease. This matches the convention used by Hansen Global Forest Change, so it will be familiar to technical partners.
- **A confirmed/provisional split**: the two most recent years and all quarterly updates are labeled provisional and may be revised; confirmed history is never revised. Quarterly remains an early-warning layer, with the annual mosaic serving as confirmation.

## Cost and risk

Roughly one to two weeks, dominated by calibration rather than engineering. It requires no re-running of SAM2 or Earth Engine — every per-period output from 2018 through 2025 already exists on disk under both threshold regimes, so this is post-processing over data we already have.

**How we will evaluate it.** We build the new cumulative detection product under the persistence rule and compare it directly against the existing cumulative dataset — same periods, same code path, one rule changed — and assemble full-basin outputs under both regimes for visual inspection. Our test sets contain too few false positives to estimate recurrence rates directly, so the paired comparison and visual review are the practical evidence rather than a standalone diagnostic.

**The main known limitation** is that persistent landscape false positives (sandbars, exposed rock, roads) recur every period and will pass any persistence check. They set a constant precision floor rather than causing drift over time, so this proposal does not address them; reducing them is a training-data workstream.

## Design details

Notes from the design discussion that are not part of the summary above but bear on implementation:

- **Confirmation runs forward, not backward.** Scars are permanent, so the window is anchored at the candidate onset year `Y` and spans `[Y, Y+n-1]`. A trailing window would require the mine to have existed before it started.
- **Confirmation and dating are separate.** Confirm with `k`-of-`n`; the onset year is the first clear *that passes its own confirmation window*. This avoids backdating a real 2022 onset to a spurious 2019 cloud artifact.
- **Confirm only on a complete window** — *superseded by the nested rule above.* Originally: an onset year `Y` is evaluated once all `n` periods in `[Y, Y+n-1]` exist, with no partial resolution, since general early resolution would require tracking per-detection confirmation state across runs. That objection does not apply to the nested rule, which gets the same timeliness gain from a stateless recomputation (`min(onset_n2, onset_n3)`) whose result can only grow. Under plain `n=3` the provisional edge is the two most recent years (2024 and 2025 with data through 2025); under the nested rule it is one (2025 only).
- **`k=2` over `k=3`.** Unanimity would systematically drop sites in cloud-wrecked years, and would also drop short-lived operations that partially heal by `Y+2`.
- **Don't run persistence on quarterly.** Cloud loss is seasonal, not random, so a fixed `k`-of-`n` over quarters preferentially confirms dry-season-visible pixels — a bias, not noise. Quarterly stays provisional under a stricter instantaneous threshold `t_prov`; annual confirms.
- **`t_prov` needs separate calibration per cadence.** A value tuned on annual mosaics will not transfer to quarterly. We have paired 2025 quarterly and 2025 annual coverage, which is the natural calibration experiment — retain the quarterly outputs after they are superseded.
- **Compute isolation distance per-period.** `kth_neighbor_km_on_catalog` is already run per-period (confirmed: every file in `postprocessed_*` is a single period, and `cumulative_*` sits downstream), so there is no N-dependence today. Keep it that way, so `k`-of-`n` is the only cross-period operation in the system.
- **Validate by onset, not by blob.** Sampling locations with onset year `Y` and checking imagery at `Y-1` vs `Y` is well-posed against our time-stamped labels; asking "is this cumulative polygon a mine?" is not. This dissolves the timestamp/cumulative mismatch that makes our current cumulative metrics unreliable.
- **Use cumulative P/R as a paired A/B only.** Label bias affects the old and new regimes roughly equally and largely cancels in the difference, but corrupts absolute levels. Report the delta, not the level.
- **What this does not fix.** Persistent landscape false positives (sandbars, exposed rock, roads) recur every period and pass any persistence check. They set a constant precision floor rather than causing drift, and need hard negatives in training — a separate workstream.

## Operational gotchas found while prototyping

- **The annual mask COGs are not co-registered.** Each year has its own extent *and its own resolution*. Root cause, magnitude, and the fix are now in "Fixing the raster grid" above; until that lands, **any pixel-wise temporal rule must resample to a common grid first** (union extent, finest resolution, `-tap`, nearest). It also means existing per-year area figures are computed on slightly different pixel areas.
- **Detection patches are on a near-stable grid, but the join still needs rounding — at 5 decimals, not 6.** Two separate effects:
  - The underlying patch coordinates are *not* bit-identical across years; exact float matching starts breaking down from 2020 (the 2018–2025 union inflates from 309,713 to 508,316 under exact matching). Some rounding is mandatory regardless of file format.
  - **The 2024 GeoJSONs are written at 6-decimal coordinate precision; every other year is full float64.** Centroids formed by averaging two already-rounded corners then miss a 6-dp key: 2024's overlap with 2023 collapses from 126,138 patches at 5 dp to 73,743 at 6 dp and 4,763 at 7 dp. This is what produced the phantom 2024 detection anomaly.

  Use **5 decimals** — the half-patch grid step is 0.00217°, so 5 dp resolves the grid ~217× while absorbing both effects; 4 dp begins to collide (3 collisions in 2018 alone). Better still, snap centroids to the patch lattice rather than trusting file coordinates. Separately, pin `COORDINATE_PRECISION` explicitly on every GeoJSON write so this cannot recur — there was no explicit setting anywhere in `core/` or `scripts/`, leaving it to float with the GDAL/pyogrio version on whichever VM ran the job. A few hundred duplicate centroids per year also exist (overlapping source tiles); dedupe keeping the highest confidence, or a location can cast two votes for its own persistence.
- **Exploit nodata when processing the mask rasters.** Only 3.6% (2018) to 6.1% (2025) of the band's bounding box carries data. Probing occupancy once at 1/32 resolution (which hits the COG overviews, ~0.1 s/year) and skipping empty blocks cuts the per-pixel work by ~68%. Net end-to-end gain is only ~2×, though: reading uniform nodata blocks was already cheap, and the `gdal_translate -of COG` writes are a fixed cost the skip cannot touch. Note also that in pass 1 skipped blocks need *no* write (GeoTIFF initialises to 0 = "never detected"), but in pass 2 they *must* be written, since the correct value there is nodata (2) — skipping that write silently mislabels every unobserved pixel as "observed, not mining".
- **The training patch sets are symlink farms.** `data/training_patches2026-05-04T09:47` symlinks most of its content into earlier directories. `Path.glob` follows these correctly, but `find` without `-L` reports the splits as nearly empty — which will make a complete eval set look unusable.
- **Per-year cumulative rasters are redundant.** All of them are thresholds of the `first_year_*` raster, so emitting the first-year layer alone and filtering (`first_year <= Y`) is cheaper to produce and easier to review — one layer styled by onset year rather than eight to flip between.

## Implementation checklist

Prototype phase (done, but as throwaway scripts — none of this is in the repo):

- [x] **Detections:** `k`-of-`n` admission over per-period `postprocessed_t0.43_d5_3km_t-iso0.75`, for `n2` and `n3`; outputs + `detection_counts.csv` in `persistence_t0.43_d5_3km_t-iso0.75/`.
- [x] **Rasters:** `k`-of-`n` over the UTM 21 lat[-8,0] annual masks; `first_year_*` and per-year cumulative COGs in `persistence-tests/` and `persistence-tests2-2/`.
- [x] **Provisional edge:** `cumulative_n3_provis_{2024,2025}` using `t0.55` as `t_prov`, with a `status` field (195,011 confirmed + 65,055 provisional 2024 + 36,835 provisional 2025).
- [x] **Evaluate (paired A/B):** `core/persistence_evaluation.ipynb` — verified by execution.
- [x] **Nesting property** verified exhaustively over all 256 8-year sequences.

Implementation phase (to do):

- [ ] **Port the prototype logic into the repo.** The prototype scripts were scratch-only and are gone; they need rewriting as proper modules, parameterised by UTM zone / lat band (the raster side was hardcoded to `utm21_lat_-8_0`) and by rule.
- [ ] **Fix `build_cog` grid alignment** so annual mask COGs share a grid by construction, removing the need to regrid before every temporal operation. Full specification and task list in "Fixing the raster grid" above.
- [x] **Pin `COORDINATE_PRECISION` on all GeoJSON writes** — done 2026-08-14 at **9 decimals**. Applied in `core/inference_engine.py` (the raw-detection write, where the 2024 vintage originated), `core/postprocess.py` (patch and dissolved outputs), and `scripts/{concatenate,geo_filter,dissolve}.py`. Confirmed cause: inference years were run on different VMs with different GDAL/pyogrio versions, so the unset option floated.

  **Why 9 and not 6.** 9 dp is ~0.1 mm, absurd against a 10 m pixel — but the binding constraint is not ground resolution, it is join stability against the existing full-precision archive, which we are not rewriting. Because the join key is a centroid derived as `(minx + maxx) / 2`, rounding the corners at write time perturbs the centroid by up to half the write quantum, and a fraction of patches then land on the far side of a 5-dp bin edge. Measured loss when a newly written year is joined at 5 dp against a full-precision year:

  | new-file precision | match loss @5-dp join | @4-dp join |
  | --- | --- | --- |
  | 6 dp | **−4.68%** | −0.41% |
  | 7 dp | −0.53% | −0.05% |
  | 8 dp | −0.06% | −0.00% |
  | **9 dp** | **−0.01%** | 0.00% |

  9 dp still saves ~17% on disk versus full float64. This corrects an earlier claim in this document that writer precision and join precision are independent — that holds only when *both* sides share a precision. A mixed archive is exactly the case where it fails, and a mixed archive is what we have.

  **Keep the 5-dp join rule permanently.** It is what makes the already-6-dp 2024 vintage usable at all (126,138 overlap with 2023 at 5 dp, versus 73,743 at 6 dp). Once the archive is rewritten, or the join snaps centroids to the patch lattice rather than rounding them, the writer can drop to 6 dp and the whole fragility disappears.

- [x] **Ground-area conversion for jurisdiction reporting** — `core/raster_area.py`, added 2026-08-14. EPSG:4326 pixels are not constant-area: across the basin (+10 to −20°) they vary **4.5%**, from 99.70 m² at the equator to 93.84 m² at −20°. Area depends only on the row, so the correction is a one-dimensional per-row lookup. Uses the exact WGS84 ellipsoidal cell-area formula (verified: whole-ellipsoid total reproduces 5.100656e14 m² to 0.0 ppm), not a spherical approximation. Provides `pixel_area_m2_by_row`, `area_ha`, `zonal_area_ha`, and a CLI.

  Sanity check on UTM21 lat[−8,0] 2023: 163,248.9 ha exact, against 163,175 ha in the mask table above — 0.045% apart, so the prototype's cos-weighting was already sound and no published figure moves. The danger is not the basin total but *per-jurisdiction* totals, where a single global pixel-area constant would bias southern jurisdictions against northern ones by up to ~6%.

  **Never cache a pixel-area constant.** Pre-fix rasters carry per-period resolutions differing by ~0.03% and the fixed grid uses 0.00009°; a naive 10 m × 10 m assumption is already −0.29% off on the band tested. Always derive from the raster's own transform, which everything in that module does.
- [ ] **Rasters at scale:** run the remaining UTM/lat bands.
- [ ] **Connected-component mask attribution** to the selected detection set (per-tile, pre-mosaic).
- [ ] **Calibrate `t_prov,annual`** (detections) by matching precision against the persistence-confirmed layer on 2018–2022. (2024/25 currently use `t0.55` as a stand-in, uncalibrated.)
- [ ] **Calibrate `t_prov,mask`** (logits) by the label-free sweep over 2018–2023 described above; gather labelled data to validate the result, not to find it.
- [ ] **Decide whether to save upsampled/smoothed logits** rather than raw `log_odds`, so provisional masks are a true re-threshold. Either way, fix the re-derivation path to replay upsample + smooth per tile before thresholding.
- [ ] **Calibrate `t_prov,quarterly`** from the paired 2025 quarterly vs 2025 annual comparison.
- [x] **Investigate the 2024 detection anomaly** — closed 2026-08-14. There is no anomaly in the data; it was a 6-dp join-key artifact against 2024's reduced-precision GeoJSONs. Corrected tables above. Residual real finding: the 2025 increment is 1.68× the historical mean on the t0.55 product, worth revisiting once 2026 annual lands.
- [ ] **Recompute the test-set metrics** on 5-dp-keyed layers; the published table predates the correction.
- [ ] Confirmed/provisional split plumbed through to published outputs.
- [ ] Decide `n2` vs nested for production. Interim call: `n2` as an additional regulator while masks run over-generous, migrating to `n3`/nested later — safe because the switch is purely additive.
