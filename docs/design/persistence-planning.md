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

### This changes what SAM2 is prompted with — and is untested

**Recorded 2026-08-15.** To date masks have been derived from the *cumulative*
detections, not per-period ones. Step 2 above is therefore a real change of
practice, not a restatement, and it has not been validated.

The argument for it: estimate each period's mask from what is visible in that
period, then accumulate — rather than accumulate first and ask SAM2 to segment
ground where a scar may have healed, or is shrouded by cloud in that period's
mosaic. It is also simpler, because every accumulation step then lives in
post-processing and can be revised without re-running SAM2. The cost is storing
two sets of masks, period and cumulative.

Two things to check before trusting a basin-wide comparison against the old
series:

- **SAM2's output depends on its prompt set.** Box prompts come from whichever
  polygons clip into each tile, so running on per-period `t0.43` rather than
  cumulative `t0.55` changes the masks *over shared ground too*. The new series
  will not be "the old masks plus more", and a naive area diff against the old
  product will mix this effect with the intended one.
- **Roughly 20% more segmentation than survives**, since persistence later
  discards some of what was masked. Accepted: it is the price of never needing
  a rerun.

### Quarterly masks: segment the diff, not the period

Quarterly mosaics are badly cloud-affected. The detector was trained with enough
cloud remnants and data holes to reject corrupted regions gracefully; **SAM2 has
not yet seen enough of this in fine-tuning**, so segmenting a full quarter would
ask it to work exactly where it is weakest. The existing mitigation — segment
only the quarterly *differential* and OR-merge onto the prior mask — remains
correct, and the annual change above does not disturb its rationale.

What changes is the reference the diff is taken against:

- **Was:** cumulative `t0.55` now − cumulative `t0.55` previous.
- **Becomes:** the quarter's period `t0.43` detections − the accumulated selected
  set through the previous period.

Measured on the current archive, that increment is **3–8% of the period
detection set** (1,779–7,786 locations against 30,000–120,000 per quarter), so
SAM2's exposure to a cloud-wrecked mosaic drops 12–30×. Retain the existing
`≤ 11 ha` drop on dissolved diff polygons: that threshold is half a patch, so it
removes shards below the resolution at which a detection means anything.

**The resulting asymmetry is deliberate.** Annual periods get a fresh full
segmentation; quarters get increment-only. That is not an inconsistency awaiting
tidy-up — we re-segment a year because the annual mosaic can see the whole scar,
and refuse to re-segment a quarter because it cannot. Anyone "fixing" quarters to
match the annual path would reintroduce the cloud problem.

**Year-boundary reconciliation.** When annual year `Y` lands, its mask
*supersedes* the accumulated quarterly estimate within `Y` rather than adding to
it — the mask-side analogue of provisional → confirmed. If the accumulation code
gets this wrong, quarterly shards persist alongside the annual mask that replaced
them and expansion is double-counted.

#### Connected-component attribution on shards

Attribution still applies to quarters, but behaves differently, and two things
are easy to get wrong.

- **It is vacuous unless the mask is computed on a looser set than the one
  selected.** Segment the `t0.43` diff and select at `t0.55`; selecting the same
  set that was segmented retains every component by construction.
- **Attribute before the OR-merge, never after.** A shard is spatially contiguous
  with the scar it extends, so once merged onto the prior mask, labelling yields
  one enormous component that any single detection retains. The merge destroys
  the structure attribution depends on. This is a hard ordering constraint.

The character of the operation also changes. Attribution exists because SAM2
segments past the patch that prompted it, so clipping would shave real extent —
but when the component *is* the increment, that permissiveness has little to bite
on and the rule behaves closer to a hard filter. **The failure mode reverses**:
rather than over-retaining, the risk becomes dropping genuine expansion whose
prompting detection fell between `t0.43` and `t0.55`. Measure this once real diff
masks exist.

**Structural caveat for reporting.** Quarterly masks only ever add, so quarterly
area is monotone by construction. That is right for a cumulative product, but it
means quarterly area change is a *lower bound* on expansion rather than a
measurement, and it structurally cannot show the recession the annual series can.

**The real fix** is the one already applied to the detector: fine-tune SAM2 on
cloud-corrupted chips. Then quarters could be segmented like annual periods and
the asymmetry would disappear. Until then this is load-bearing design rather than
a temporary hack, and should be documented as such.

## Attributing masks to confirmed detections

**Recorded 2026-08-15.** Persistence rejects detections; the mask has to follow.
Plain connected-component attribution — keep a component whole if *any* retained
detection intersects it — is too permissive to do that job, because mask area
whose own detections are revised out survives wherever it is contiguous with a
component that has a confirmed detection somewhere in it.

### The leak, measured

Connected components of the 2023 mask over a 204 Mpx window on the Tapajós
(the densest mining in the basin), 8-connectivity:

| | |
| --- | --- |
| components | 2,636 |
| **largest component** | **463,572 px = 4,636 ha, 7.6% of window mask area** |
| top 10 components | 28.1% of area |
| top 100 | 63.4% of area |
| median component | 253 px (2.5 ha) |
| components < 100 px | 35% of components, 0.5% of area |

So a single surviving detection anywhere in the largest blob would retain
4,636 ha; ten would retain 28% of the region. In dense mining, rejection would
have almost no purchase. (These masks are cumulative-`t0.55`-derived and so more
connected than the per-period masks will be — treat the blob sizes as an upper
bound. Attribution also runs per period, before accumulation, which bounds
component size further.)

Two things partly mitigate and one does not. Pixel-level `k`-of-`n` runs on the
mask independently, so *transient* mask area is removed whether or not its
detections confirmed. The residual leak is area that recurs across periods, is
contiguous with confirmed scar, and never has a confirming detection of its own.
Persistent landscape false positives (sandbars, exposed rock) pass both checks —
a training-data problem, as noted elsewhere.

### Resolve a contradiction in this document

The hierarchy section says confirmed area is "persistence-confirmed mask pixels
**lying within** detections confirmed by `Y`", which reads as clipping. The
pipeline-ordering section says explicitly **not** to clip, and to use
connected-component attribution. Those are different rules with materially
different answers. The rule below supersedes both.

### The prior already defines how far a mask may extend

`soft_spatial_prior` is zero inside the detection footprint and applies
`penalty = -(dist_outside / prior_sigma)²` outside, with `prior_sigma = 12` px.
A pixel survives threshold 0 only where `best_logits > (dist / 12)²`, so the
maximum extent beyond a detection is `12·√(max logit)` — a hard geometric cap
imposed by the pipeline itself, not a tuning choice.

Measured on real stored logits (max log-odds 11.33, which equals max
`best_logits` since the prior is zero inside detections):

| logit | max extent |
| --- | --- |
| theoretical max, 11.33 | 40.4 px = **404 m** |
| p99 of positive logits, 10.08 | 38.1 px = 381 m |
| median positive logit, 3.69 | 23.0 px = 230 m |

And where mask pixels actually sit, relative to the detections that prompted
them (same Tapajós window):

| | |
| --- | --- |
| **inside a detection footprint** | **93.9% of mask pixels** |
| p99 distance beyond | 11.0 px = 111 m |
| p99.9 | 18.4 px = 185 m |
| beyond the 40.4 px cap | 0.000% |

(The few pixels reading further are window-edge artifacts — detections just
outside the query bbox were not loaded. The comparison is against *dissolved
cumulative* detection polygons, which are more generous than individual patch
footprints, so 93.9% is an upper bound on what clipping would preserve.)

### Rule

**Bounded geodesic growth from retained detections, capped at the prior-implied
extent (≈40 px / 404 m), measured through the mask.** This is principled rather
than tuned — the cap is the distance beyond which the pipeline's own prior makes
a mask pixel impossible — and near-lossless, since the p99.9 halo is 185 m, less
than half the cap. It bounds the blob problem exactly: a retained detection can
only pull in mask it could plausibly have generated itself.

If measurement shows the cap still leaks, the fallback is a **nearest-seed
partition**: assign every mask pixel to its nearest prompting detection within
the component and keep pixels whose seed was retained, so a rejected detection
loses its own neighbourhood and nothing else. Costlier (distance transform with
indices, ideally geodesic within-mask) and less obviously principled, but
strictly tighter.

Do not clip to patch footprints, and do not use unbounded connected components.

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

## Confirmation recipes A–D

**Recorded 2026-08-16.** The confirmation rule has two free axes — how long the
window runs, and whether quarterly periods may corroborate — giving four
recipes. `core/persistence.py` implements all of them as configuration.

| | window `[Y, Y+1]` (one following year) | window `[Y, Y+2]` (two following years) |
| --- | --- | --- |
| **annuals only** | **A** — n=2 | **C** — the classic k=2 of n=3 |
| **annuals + quarters** | **B** | **D** |

`D+` denotes D with `early_confirm`: evaluated against whatever periods exist
rather than waiting for the window to close. Safe by construction, since the
witness set only grows toward a fixed endpoint, so confirmations accumulate and
are never withdrawn.

### Cumulative detections under each recipe

Locations with confirmed onset ≤ T, against the published t0.55 union:

| through | published t0.55 | A | B | C | D |
| --- | --- | --- | --- | --- | --- |
| 2018 | 95,085 | 99,302 (+4.4%) | 99,302 | 103,977 (+9.4%) | 103,977 |
| 2019 | 120,364 | 118,615 (−1.5%) | 118,615 | 125,535 (+4.3%) | 125,535 |
| 2020 | 140,038 | 137,237 (−2.0%) | 137,237 | 142,618 (+1.8%) | 142,618 |
| 2021 | 161,405 | 156,734 (−2.9%) | 156,734 | 162,800 (+0.9%) | 162,800 |
| 2022 | 180,776 | 175,321 (−3.0%) | 175,321 | 181,592 (+0.5%) | 181,592 |
| 2023 | 200,148 | 192,761 (−3.7%) | 192,761 | 198,794 (−0.7%) | 199,388 (−0.4%) |
| 2024 | 215,484 | 202,173 (−6.2%) | 205,575 (−4.6%) | 207,468\* (−3.7%) | 211,596\* (−1.8%) |

**A ≡ B and C ≡ D through 2022**, exactly — quarterly data begins in 2025, so
there are no extra witnesses to add for earlier onsets. The four recipes are
really two until 2023. `*` marks lower bounds: both window-3 columns need 2026
annual to close 2024.

`C`/`D` track the published series closely (+9.4% at 2018 decaying to −0.7% at
2023); `A`/`B` sit 2–4% below throughout, the cost of the shorter window.

### Test-set metrics for A–D

Same protocol and pooled split as above. Each layer is confirmed onsets plus the
t0.55 provisional layer for annual periods whose window has not closed;
`[confirmed only]` drops the provisional part. Quarters serve as witnesses in
B/D but are not themselves added to the layer, so the comparison isolates the
rule.

| layer | patches | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `or_t055_2025` (current) | 250,398 | 718 | 37 | 5 | 3119 | 0.9510 | 0.9931 | 0.9716 |
| `or_t043_2025` (loose) | 302,520 | 718 | 82 | 5 | 3074 | 0.8975 | 0.9931 | 0.9429 |
| **A** window2 annual | 236,125 | 718 | **32** | 5 | 3124 | 0.9573 | 0.9931 | **0.9749** |
| **B** window2 + quarters | 239,527 | 718 | 34 | 5 | 3122 | 0.9548 | 0.9931 | 0.9736 |
| **C** window3 annual | 245,569 | 718 | 33 | 5 | 3123 | 0.9561 | 0.9931 | 0.9742 |
| **D** window3 + quarters | 246,163 | 718 | 35 | 5 | 3121 | 0.9535 | 0.9931 | 0.9729 |
| **D+** window3 + quarters, early | 251,274 | 718 | 37 | 5 | 3119 | 0.9510 | 0.9931 | 0.9716 |
| A [confirmed only] | 202,173 | 715 | 30 | 8 | 3126 | 0.9597 | 0.9889 | 0.9741 |
| B [confirmed only] | 205,575 | 715 | 33 | 8 | 3123 | 0.9559 | 0.9889 | 0.9721 |
| C [confirmed only] | 198,794 | 711 | **29** | 12 | 3127 | 0.9608 | 0.9834 | 0.9720 |
| D [confirmed only] | 199,388 | 711 | 31 | 12 | 3125 | 0.9582 | 0.9834 | 0.9706 |

- **Every recipe holds recall at 718/723 and beats the current product on false
  positives** (32–37 against 37), so they differ only in precision, and only by a
  handful of chips. The earlier conclusion stands: this protocol shows
  *equivalence*, not superiority.
- **The 82 → 32 FP result is the one with real signal**, and it survives for all
  four recipes. Persistence recovers what loose thresholds give up, at zero
  recall cost, regardless of which window or witness set is chosen.
- **Quarters cost about 2 FP with no recall gain**, consistently: A→B is 32→34
  and C→D is 33→35. Two chips is noise on its own, but the direction is the same
  in two independent pairs and it agrees with the confidence evidence that
  quarter-only corroborations are weaker. Treat it as a weak prior against
  quarters, not a finding.
- **`D+` scores identically to the current product but is not the same layer.**
  The confusion matrices match cell for cell, yet the layers share only
  Jaccard 0.891 — 14,934 patches unique to `D+`, 14,058 to `or_t055` — and even
  their false positives differ, sharing 31 of 37. Two different layers landing
  on the same matrix is coincidence at chip resolution, not equivalence.

  What `D+` does show is worth more. It is **strictly additive over C** (their
  intersection is all of C), adding 5,705 patches at **median confidence 0.497** —
  detections below the provisional threshold, rescued on a single witness. Those
  additions gain **zero true positives** and cost **4 false positives**. So the
  case against maximal early confirmation is not that the layer degenerates
  toward the unfiltered union; it is that the marginal detections early
  confirmation admits are the weakest available and buy no recall.
- **The provisional layer is carrying real recall.** Confirmed-only C has 12 FN
  against 5 for the full layer, so provisional detections supply 7 true positives
  the confirmed core does not yet have.
- **This protocol cannot separate these recipes.** A 5-chip spread across 3,879
  chips is not a basis for choosing. The decision should rest on the structural
  properties — when a period becomes final, and whether confirmations can be
  withdrawn — and on visual review of the differing detections.

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

**Measured, and clamping alone does the work — no quantization needed.** Saturating at ±16
collapses the tail to a constant that ZSTD removes, entirely in float32, with no change of dtype
or nodata handling. On 16 real logit tiles:

| clamp | predictor 1 | predictor 3 | vs unclamped |
| --- | --- | --- | --- |
| none | 3,657 KB | 2,852 KB | 100% |
| ±32 | 1,033 KB | 929 KB | 33% |
| **±16** | 588 KB | **541 KB** | **19%** |
| ±8 | 303 KB | 301 KB | 11% |

**±16 is chosen to be lossless, not merely adequate.** Clamping happens *before* the smoothing is
replayed, so too tight a limit perturbs the smoothed field near the decision boundary. Verified on
real tiles, including a genuine 2× upsample in the path: at ±16 the re-derived mask is
**bit-identical** to the unclamped production mask on every tile (0 differing pixels); at ±8 it is
not (23 px, IoU 0.9987); at ±4 it fails outright (IoU 0.9768). ±16 log-odds is a probability of
1 − 1.1e-7, far outside any threshold a `t_prov,mask` sweep would explore.

Implemented as `MaskConfig.logit_clamp`. **Quantization to int8/int16 is deferred**: it would add
maybe another 1.4× on top of the 5.3× clamping already provides, but requires reworking the nodata
sentinel (`LOGIT_NODATA` is `nan`, which has no integer equivalent) through `save_tile`,
`build_logit_max_vrt`'s max-reduce pixel function, and `build_cog`. Not worth bundling with a
change that is currently provably lossless.

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
- [x] Clamp logits at ±16 (`MaskConfig.logit_clamp`) — 5.3× smaller in float32, re-derived mask
      bit-identical to production on all 16 real test tiles. Quantization to int8/int16 deferred;
      see above for why it is a separate change.
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
- **Per-year cumulative layers are derivationally redundant, but still worth writing.** All of them are thresholds of the `first_year_*` layer (`first_year <= Y`), so the first-year layer is the thing to store and the one that must be correct. The claim originally made here — that it is also *easier to review* — is wrong, and was corrected in practice: filtering by attribute in QGIS is awkward, and flipping layer visibility on and off is the natural way to compare snapshots. Emit the first-year layer as the product, and materialise the per-year files as review artifacts.

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
- [ ] **Connected-component mask attribution** to the selected detection set. Now that the fixed grid puts every tile and band on one lattice, prefer labelling *after* mosaicking — windowed connected components with cross-window merging, or a flood-fill propagated from rasterized detection seeds — which removes the tile-seam failure the per-tile approach accepts. For quarters, attribute on the per-period diff mask before the OR-merge; see "Quarterly masks" above.
- [ ] **Compare the four attribution rules on real per-period masks** — clip to footprint, bounded geodesic growth at several `N` (including the prior-implied ≈40 px), nearest-seed partition, and unbounded connected components — reporting retained area under each on one band. If bounded growth and unbounded CC agree to ~1%, the cap is free and settles it; if they diverge, the choice is load-bearing. Existing masks were derived from cumulative `t0.55`, so they characterise component structure but their retention rates will not transfer.
- [ ] **Confirm the prior-implied cap on per-period masks.** The ≈40 px figure comes from `12·√(max logit)` with max log-odds 11.33; re-derive it from the new logits, since a different prompt set may change the achievable maximum.
- [ ] **Validate per-period masks against the old cumulative-derived series** on a few paired tiles, isolating the prompt-set effect from the intended change.
- [ ] **Calibrate `t_prov,annual`** (detections) by matching precision against the persistence-confirmed layer on 2018–2022. (2024/25 currently use `t0.55` as a stand-in, uncalibrated.)
- [ ] **Calibrate `t_prov,mask`** (logits) by the label-free sweep over 2018–2023 described above; gather labelled data to validate the result, not to find it.
- [ ] **Decide whether to save upsampled/smoothed logits** rather than raw `log_odds`, so provisional masks are a true re-threshold. Either way, fix the re-derivation path to replay upsample + smooth per tile before thresholding.
- [ ] **Calibrate `t_prov,quarterly`** from the paired 2025 quarterly vs 2025 annual comparison.
- [x] **Investigate the 2024 detection anomaly** — closed 2026-08-14. There is no anomaly in the data; it was a 6-dp join-key artifact against 2024's reduced-precision GeoJSONs. Corrected tables above. Residual real finding: the 2025 increment is 1.68× the historical mean on the t0.55 product, worth revisiting once 2026 annual lands.
- [ ] **Recompute the test-set metrics** on 5-dp-keyed layers; the published table predates the correction.
- [ ] Confirmed/provisional split plumbed through to published outputs.
- [ ] Decide `n2` vs nested for production. Interim call: `n2` as an additional regulator while masks run over-generous, migrating to `n3`/nested later — safe because the switch is purely additive.
