# Jurisdiction areas: two deficits, and the plan for the remaining one

**Recorded 2026-08-28.** Why the country areas in the published data don't sum to
the basin total, and why every published area is about 1% too high. Two
independent faults, found while chasing the first. One is fixed in code; the
other has a plan here.

The trigger was a small thing: for 2026 Q2 the nine country areas agree with the
basin total to three significant figures and disagree in the fourth. That turned
out to be the smaller of two problems.

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

## Deficit B: every published area is ~1% too high

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

## Deficit A: countries can't sum to the basin

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

## The plan

### Part 1 — make country attribution authoritative

`ignore_if_outside_country: False` for the `national_admin` dataset only
(`preprocess_mining_areas.py`, in `datasets_to_process`). The national polygon
*is* the country; checking it against the subnational layer's opinion can only
discard fragments. Keep it `True` for IT and PA, where a territory straddling a
border genuinely shouldn't be credited to the wrong country.

Tested: deficit −3,283.99 ha → +0.36 ha.

Part 1 is a stopgap. Part 2 subsumes it — once the codes agree the filter stops
dropping anything on its own, and it stays in place as a real guard.

### Part 2 — repair the attribution instead of discarding it

1. **Make the country unambiguous.** Stage 1 splits mining by the subnational
   layer only, so a fragment can straddle a national border and have no single
   true country. Add an overlay against the national polygons in stage 1. This is
   what makes the repair well-defined — don't skip it and reach for
   `representative_point()`.
2. **Detect** rows where `admin_country_code` disagrees with the authoritative
   code.
3. **Repair** them: re-assign `admin_country`, `admin_country_code`,
   `admin_id_field`, `admin_name_field` by `sjoin_nearest` against `admin_areas`
   restricted to the authoritative country.

Note this is upstream of all four dataset passes, so IT, PA and subnational
totals move too — their filters also stop dropping. Those are the area types
where the Mining Calculator is enabled (`hideMiningCalculator` disables it for
countries), so size the change before shipping.

### Part 3 — dropped, not deferred

Clipping the subnational layer to the national layer would make the disagreement
impossible. It is not worth it:

- It means republishing partner geometry reshaped by our own heuristic, on every
  ingest. Anyone comparing our layer against the partner's source sees a
  discrepancy we introduced.
- Partners resupply on their own cycle, so it would re-run every delivery, each
  time dragging in tile regeneration and a frontend deploy (see below).
- Clip *alone* is worse than the status quo: the strip is covered by one layer's
  polygon only (subnational self-overlap is 0.04%), so clipping it back leaves
  the strip covered by nothing, and stage 1 would drop those fragments entirely.
  It would need clip **and** fill to match what Part 2 does at runtime.

Part 2 keeps the same heuristic confined to derived numbers, never to published
boundaries.

### Part 4 — make it announce itself

With Part 3 dropped, these are load-bearing, not nice-to-have:

- **Bound the nearest-join.** Repair only within a distance threshold; beyond it,
  raise with count and hectares. Today's strip is ~111 m wide so "nearest" is
  safe; a future delivery could diverge by kilometres and an unbounded join would
  silently attribute mining to the wrong region.
- **Log repaired volume every run** — fragment count and hectares. Baseline
  **3,284 ha**. It is the only visibility into layer drift, since the repair's
  job is to absorb it.
- **Check at ingest.** When partners deliver, measure the strip and the mining
  inside it before running. Baselines: strip **814,109 ha**, mining in strip
  **3,282.59 ha**. Also assert the jurisdiction id set against the previous
  publish. A jump is a conversation with the partner, not something to repair
  quietly.
- **Assert the invariant.** After `calculate_mining_area_timeseries`, per period:
  `|sum(countries) - AMAZ| < 50 ha`, and area entering the filter equals area
  leaving it for `national_admin`. Raise, don't warn.

That last point is the lesson of this whole investigation. Both faults produced
plausible numbers and clean logs for months. Nothing short of an assertion would
have caught either.

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
