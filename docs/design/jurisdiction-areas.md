# Jurisdiction areas: what was wrong, what is fixed, what remains

**Recorded 2026-08-28.** The trigger was a small thing: for 2026 Q2 the nine
country areas agreed with the basin total to three significant figures and
disagreed in the fourth. Chasing it turned up four independent faults, of which
that was the second largest.

Two are fixed in code on `area_revisions`; two are measured and documented here
but not fixed. None of the four is visible in the published data, which is the
point of writing it down: each produced plausible numbers and clean logs.

| | effect on Q2 2026 | status |
| --- | --- | --- |
| UTM zone off-by-one | every area ~1% high | fixed |
| country-code filter discarding border fragments | countries 3,284 ha short of the basin | fixed |
| gaps in subnational coverage | 1,021 ha reaches no jurisdiction at all | measured, not fixed |
| overlapping subnational polygons | 2,875 ha counted twice | fixed |

## The measurements this rests on

Every claim below was checked against the raster directly, without going through
the vector pipeline, using `raster_area.py`. That module exists precisely so the
vector path has an independent counterpart: it counts onset pixels in the mask
COG and weights them by exact WGS84 ellipsoidal cell area, sharing no code and no
assumptions with `preprocess_mining_areas.py`. For Q2 2026 cumulative it gives
**1,150,148.52 ha** basin-wide. Any vector-side number that disagrees materially
with that is wrong, and that is how both faults were localised.

The mask is `amazon_basin_mining_scar_masks.tif` — `uint16`, onset-coded (years
as the year, quarters as `year*10+quarter`), so a cumulative through any period
is `0 < onset <= code`. Verified identical (`md5 7bb52092…`) across
`data/staging_gs/`, `data/staging_source-coop/`, the `persistence_masks/`
`_first_year` copy, and a fresh source.coop download. The inputs were never in
question.

## Fixed: every published area is ~1% too high

`calculate_area_using_utm` measured each geometry in a per-zone UTM projection,
selecting the zone by longitude band from `zone_min`. `zone_min` was `32718`
(UTM 18S) while the first band, starting at `lon_min = -84`, is UTM 17S. So every
geometry was measured one zone east of its own — 3–9° off the central meridian
instead of 0–3° — and transverse Mercator scale grows with that distance.

Ed fixed the constant on 2026-08-14 in `c982371`, but it sat on an unmerged
branch until PR #31 landed 2026-08-26. The publish ran 2026-08-23. **The code on
main is correct; the published data is not.**

The signature is distinctive: inflation depends on where a country's mining sits
inside its UTM band, so it varies by country while staying near-constant over
time. Predicted from projection geometry alone versus measured:

| | predicted | observed |
| --- | --- | --- |
| Suriname | 0.31% | 0.311% |
| Brazil | 1.00% | 1.010% |
| Peru | 1.43% | 1.449% |
| Guyana | 1.48% | 1.517% |

Suriname is the tell — its mining sits unusually far east within its band, so
shifting a zone east lands it closer to the wrong meridian than most, and it
barely inflates. Reconstructing with `zone_min = 32718` reproduces every
published figure to the cent.

Q2 2026 basin: published **1,164,264.90 ha**, correct **≈1,151,982 ha**.

### What was done instead of just fixing the constant

`3783c9e` replaces per-zone UTM with a single global equal-area CRS
(`EPSG:6933`). Correctly zoned, UTM was accurate to a thousandth of a percent, so
this is about fragility, not precision. The zone construction had three
independent ways to fail:

- the off-by-one above;
- `.cx` selects by *bounding box* and later iterations overwrite earlier ones, so
  a polygon straddling a zone boundary silently took whichever zone came last;
- a fragment split at a border could land in a different zone than the polygon it
  was cut from, so the same geometry measured differently depending on what it
  was intersected against.

None can exist without zones. Verified against pyproj geodesic areas: 0.02 ha
apart over 1.15M ha of mining fragments, and 50 ha over the 8.2-billion-ha basin
polygon — equal-area holds at both ends of the scale range. Also ~3× faster,
being vectorized rather than row-wise.

A residual ~0.027% remains between any vector method and direct pixel counting.
That is the vectorization edge effect — tracing pixel boundaries and measuring
the polygon, versus summing exact cell areas — not a projection artifact. It will
not go away by changing CRS.

## Fixed: countries could not sum to the basin

Still present. Mining fragments are country-coded **twice**, from two
independently sourced layers:

1. `intersect_and_calculate_areas(mining_gdf, admin_areas_gdf, ...)` overlays
   mining with the **subnational** layer (partner sources), and the result is
   prefixed, so each fragment carries `admin_country_code`.
2. Each dataset pass then overlays again with its own layer — for
   `national_admin`, GADM level 0 — giving a second `country_code`.

`ignore_if_outside_country` keeps a row only if the two agree, **or if
`country_code` is null**. AMAZ has no country code, so it takes the null branch
and keeps everything; countries drop what the two layers disagree about.

The layers disagree over an **814,109 ha** strip along international borders —
different sources, independently simplified at the same 0.001° tolerance.
Fragments there are dropped entirely, not reassigned: the neighbouring country's
polygon doesn't contain them, so no row exists to receive them.

Measured: mining inside the disagreement strip **3,282.59 ha**; pipeline deficit
**3,283.99 ha**. The same thing. Mining is 3× over-represented in the strip
relative to its area share, because riverine mining sits on river borders.

Ruled out along the way, each by measurement: the boundary coverage gap (12 ha,
not 3,284), the UTM zone heuristic (−0.001% on real geometry), the vectorizer
(exact to 0.0009%), and stale inputs (a fresh run reproduces the deficit).

## How it was fixed

Implemented on `area_revisions` in `preprocess_mining_areas.py`.

**Split by country first.** `split_by_national_boundaries` overlays mining with the
national country polygons before anything else, stamping `country_code_auth` and a
`mining_fragment_id`. Without this a fragment bounded only by the subnational layer
can straddle a national border, and there is no single country to attribute it
against — a gap easily papered over with `representative_point()`, which would be
guessing.

**Prefer the domestic polygon.** `prefer_domestic_rows` keeps the rows whose area of
interest lies in the fragment's own country, and where none does keeps its largest
row. Applied at stage 1 and again per dataset. This resolves most border cases by
actual containment rather than by proximity.

**Repair what containment cannot resolve.** `repair_admin_country_codes` handles the
remainder — fragments with no domestic subnational coverage at all — by re-assigning
country and region from the nearest subnational area *within* the authoritative
country. `ignore_if_outside_country` then passes for every row and stays in place as
a guard rather than a filter; it is deliberately still `True` for all four datasets.

Result: the country total equals the basin total exactly.

### The safeguards are load-bearing

Partners resupply boundaries on their own cycle, so incoming mismatches recur and
may be far worse than today's ~111 m sliver. These are not nice-to-have:

- **The nearest-join is bounded** at `REPAIR_MAX_DISTANCE_M`. Unbounded, a badly
  divergent delivery would silently attribute mining to the wrong region.
- **Fragments that cannot be attributed are dropped, but capped.** Past
  `UNATTRIBUTABLE_SHARE_LIMIT` of the area needing repair, the run stops rather than
  dropping quietly.
- **The repaired volume is logged every run.** Repairing the disagreement conceals
  it, so this is the only signal of layer drift. Baselines: strip **814,109 ha**,
  mining in strip **3,282.59 ha** for one period.
- **The invariant is asserted**, not warned: countries must tie to the basin within
  50 ha in every period, and no fragment may remain mis-attributed after the repair.
- **Outputs are quarantined** before a run starts. The checks fire before any
  jurisdiction output is written, so without this a dead run leaves the previous
  files looking current and `upload_data_to_s3.py` would publish them.

Still worth adding at ingest: assert the jurisdiction id set against the previous
publish. The Indigenous territories layer silently drifted 35 territories between
January and August, and one delivery renamed a territory and split another while
leaving the totals identical to the cent — no area-based check would have seen it.

### Part 3 — dropped, not deferred

Clipping the subnational layer to the national layer would make the disagreement
impossible. It is not worth it:

- It means republishing partner geometry reshaped by our own heuristic, on every
  ingest. Anyone comparing our layer against the partner's source sees a
  discrepancy we introduced.
- Partners resupply on their own cycle, so it would re-run every delivery, each
  time dragging in tile regeneration and a frontend deploy (see below).
- Clip *alone* is worse than the status quo: the strip is covered by one layer's
  polygon only, so clipping it back leaves the strip covered by nothing, and stage
  1 would drop those fragments entirely. It would need clip **and** fill to match
  what the runtime repair does.

The repair keeps the same heuristic confined to derived numbers, never to published
boundaries.

## Measured, not fixed: mining that reaches no jurisdiction

Stage 1 clips mining to the subnational calculator layer, so that layer's coverage
*defines* the basin total — anything outside it is absent from every jurisdiction,
the countries and `AMAZ` included.

The layers do not quite reach their countries' national outlines. Brazil's set,
built from Legal Amazon municipalities, leaves **655,454 ha** of the country
uncovered across 8,086 separate gaps; basin-wide **1,021 ha** of Q2 2026 mining
falls in such a gap and is lost entirely. Most gaps are edge mismatch rather than
missing municipalities — the wide ones are on the coast and in the far north-east.

This is not academic. **63.71 ha of mining inside the Yanomami territory is lost
this way — 0.84% of its 7,610.77 ha.** The UTM inflation was masking it: the
published figure read *higher* than the true masked extent despite the loss. With
the projection fixed, the territory will read ~1% lower and that 0.84% shortfall is
no longer offset.

A QGIS layer of the Brazilian gaps, with `area_ha` and `mining_pieces` per polygon,
can be regenerated by differencing the national polygon against the union of that
country's subnational polygons.

## Measured, not fixed: mining missing from the subnational breakdown

Separately, **529.87 ha** survives stage 1 but has no *display*-layer polygon of its
own country covering it, so it is absent from the subnational breakdown — though
still counted for its country and for the basin. Measured per country it comes to
520.72 ha, 98% of the gap, and 96% of that is three countries: Venezuela 277.81,
Brazil 130.08, Guyana 112.09. Five countries lose nothing.

Guyana is instructive: it uses the *same file* for calculator and display, so
anything of its own that survives stage 1 under a foreign polygon has nothing
domestic to land in. Colombia and Suriname share that property.

These fragments have no correct domestic region to go to. Counting them under a
neighbouring country's region, which is what happened before, was worse.

## Fixed: the same ground counted twice

Subnational polygons from neighbouring countries overlap along borders, so one
fragment could match two of them and have its area counted once for each. For Q2
2026 that inflated the basin total by **2,875.49 ha (+0.16%)** — the pipeline
reported 1,151,991.07 ha against a true in-basin extent of 1,150,136.73 ha.

**96% of it is cross-country**; within-country overlap is only 110.64 ha. So
`prefer_domestic_rows` at stage 1 removes nearly all of it. Restricting to
same-country coverage *without* the fallback would have been worse than the
disease: 1,538.89 ha sits in a country covered only by a neighbour's polygons —
French Guiana alone has 247 ha, 1% of its total — and would have gone from
double-counted to absent.

It also cut how much rests on guesswork. Resolving border cases by actual
containment left only 350 fragments across all 13 periods needing the
nearest-neighbour repair, down from 2,041 — an 88% reduction in the population
attributed by proximity rather than by coverage.

## Where the numbers landed

Q2 2026 cumulative, against a true in-basin extent of **1,150,136.73 ha**:

| | published | after |
| --- | --- | --- |
| AMAZ (basin) | 1,164,264.90 | 1,149,140.05 |
| sum of 9 countries | 1,160,966.92 | 1,149,140.05 |
| deficit | −3,297.98 | −0.00 |
| subnational_admin | 1,164,484.81 | 1,148,706.12 |
| indigenous_territories | 188,049.50 | 186,053.81 |
| protected_areas | 314,898.29 | 311,752.65 |

The basin moved from **+1,854.34 ha (0.16% high)** to **−996.68 ha (0.087% low)** —
the error roughly halved and changed character, from inflation to a known shortfall.
Its components: −1,021.15 ha lost to coverage gaps, +110.64 ha of within-country
overlap that de-duplicating by country does not catch, less a few hectares of
unattributable drops. That accounts for about −915 of the −997.

**Roughly 80 ha (0.007%) is unaccounted for.** Most likely the orphan fallback
keeping only a fragment's largest row, which discards the remainder when a fragment
is genuinely split across two foreign polygons. Recorded rather than explained
away.

## Why the layers disagree at all

The national layer is GADM 4.1 level 0. The subnational layers are nine separate
national sources at inconsistent administrative levels — Venezuela at parish level,
Colombia at department level, Brazil from Legal Amazon municipalities — with a
second, coarser set for display. Both are simplified at tolerance **0.001**, but
independently, from different vertex sets, so identical tolerances still produce
boundaries that do not coincide.

Simplification is the smaller half. At 0.001 deg (~111 m) each layer's boundary can
move by that much, so along 15,511 km of internal borders it could open at most
~345,000 ha of disagreement. The observed strip is **814,109 ha**. At least 58% of
it — realistically much more — is a genuine difference between GADM and the national
agencies, two institutions rendering the same border differently. That is the
argument for repairing at runtime rather than trying to reconcile the sources.

## Which layer does what

`ADMIN_AREAS_GEOJSON` (the calculator set, 1,407 polygons) is used **once**, at
stage 1, and every dataset is built from that overlay. A hole in it removes mining
from everything. It also supplies `regionId` for the Mining Calculator.

`SUBNATIONAL_ADMIN_GEOJSON` (the display set, 95 polygons) is only one of the four
jurisdiction datasets. A hole in it costs the subnational breakdown alone.

## The shape this should take: attribute on the raster

Worth stating plainly, because it reframes most of the above: **the fragments this
pipeline works so hard to attribute are created by the pipeline itself.**

`convert_rasters_to_vector.py` traces contiguous runs of mining pixels into
polygons, so a scar straddling a border becomes one polygon crossing it. Stage 1
then intersects those polygons with jurisdiction polygons, and that intersection is
what splits them. The raster has no fragments: every pixel is in exactly one place.
Measured, the difference is stark — the raster route gives a country-vs-basin
deficit of **±12 ha**, where the vector route gave **3,284 ha**.

So the country-code filter discarding border pieces, the runtime repair, the
prefer-domestic rule and orphan handling are all artifacts of vectorising before
attributing: they exist to reassemble fragments we made. A pixel cannot be split, so
none of them arises in a zonal formulation. Neither does the ~0.027% vectorisation
edge effect, nor the 12 minutes and 1.6 GB of intermediates the vector route costs
per run.

**Double-counting is different and does not go away.** Overlapping polygons still
force a choice about which zone a pixel belongs to. Burning zone labels into a raster
resolves it silently — the last polygon burned wins — and per-zone masking reproduces
the vector double-count exactly. What changes is that the choice becomes a visible
line of code instead of an emergent property of `gpd.overlay`. Measured in the
prototype: protected areas self-overlap by 5.58%, and label-burning put the PA total
2.7% *below* the vector route purely through how that overlap was resolved.

The right answer also differs by layer, which nothing in the pipeline currently
distinguishes. Two countries' subnational polygons overlapping across a border is
spurious and should be de-duplicated. An Indigenous territory overlapping a protected
area is a real co-designation and should count in both totals. A raster formulation
would need telling which is which, just as the vector one does.

**Nothing that ships as geometry depends on the vectorised mask.** Tracing the
consumers, `mining_gdf` feeds only area calculation, the illegality overlay and the
jurisdiction intersections — all of which yield *numbers per zone*. The display
polygons and tiles come from `MINING_DIFFERENCES_FILES`, a different product
entirely.

What would need re-expressing as zonal cross-tabulations:

- **`locations`** — totals per (jurisdiction x subnational region).
- **`illegality_areas`** — totals per (jurisdiction x illegality class); illegality
  is a per-pixel property in the first place.

Both are the same machinery: burn each zone layer into an integer label array per
raster block and accumulate with `bincount`. A prototype covering all four
jurisdiction datasets in one pass over the COG runs in about five minutes, against
the ~20 the vector route takes after a 12-minute vectorisation.

Two things it would force into the open, which is an argument for it rather than
against:

- **A pixel straddling a border goes wholly to one jurisdiction** rather than being
  split proportionally. That is what makes totals tie by construction, and for a
  10 m pixel it is the more honest treatment.
- **Overlapping zones need a rule per layer**, as above — spurious cross-border
  overlap de-duplicated, genuine co-designation counted in both.

It would not fix the coverage gaps — those are a property of the boundary layers.
But it would make them *visible*: pixels matching no zone can simply be counted, so
the 1,021 ha currently lost at stage 1 becomes a reported number instead of a silent
subtraction.

This supersedes a good part of what is on `area_revisions`. That is the honest
assessment: January's plan — raster as the underlying data, no splitting — was
right, and the pipeline adopted only half of it, vectorising the raster and then
doing vector overlays that reintroduce every geometry problem the raster avoids.

## Republishing: what it actually costs

The corrected numbers only reach anyone through a republish, and the frontend
makes that heavier than it looks. `src/constants/map.ts` hardcodes
`DATA_UPDATED_AT`, and **both** the data and tiles URLs derive from it:

```ts
const DATA_BASE_URL  = `${NEXT_PUBLIC_DATA_URL}/${DATA_UPDATED_AT}`;
const TILES_BASE_URL = `${NEXT_PUBLIC_TILES_URL}/amw/${DATA_UPDATED_AT}`;
```

So a new date folder needs a frontend PR and deploy, **and** the tiles must exist
at the new prefix or the map renders nothing — even though these changes don't
alter tile geometry. `upload_tiles_to_s3.py` cannot be skipped when the date
moves.

Where the numbers on screen come from, since it isn't obvious: `AreaSummary`
reads `intersected_area_ha_cumulative` from `*_yearly.json`, and the illegality
breakdown and `locations` come from `*_impacts_unfiltered_dict.json` — all JSON
off the CDN. Tiles supply geometry, `id`, and tooltip labels. The only tile-borne
area value is a boolean filter (`mining_affected_area_ha > 0`) deciding whether a
polygon renders at all, which nothing here crosses.

## Boundary layers are no longer in the repo

`3f2d205` and `a4eb8dd` untrack everything under `data/boundaries/*/out/` and
`data/outputs/website/`, leaving the files on disk. The repo's
`indigenous_territories.geojson` had drifted 35 territories behind the layer that
produced the published data; a run against it silently dropped three Ecuadorian
territories and shifted the IT totals. Committed copies of partner data go stale
and nothing announces it — and while they were tracked, any `git checkout -- data`
would restore the stale version over a freshly synced one.

The stable reference boundaries outside those folders (`Amazon_ACA.geojson`, the
basin outlines) stay in the repo: they define the study area rather than
mirroring a partner's data.

Consequence: a fresh clone can't run the pipeline until someone syncs. That is
the intended trade — an empty checkout that fails loudly beats a stale one that
quietly produces wrong numbers.
