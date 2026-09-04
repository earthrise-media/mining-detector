<!-- Static; upload to the bucket root by hand after editing:
     gcloud storage cp scripts/templates/amw_archived_README.md gs://amw-archived/README.md -->
# gs://amw-archived — Amazon Mining Watch imagery archive

**Archived September 2026.** Sentinel-2 mosaics over the Amazon
basin, generated during the summer 2026 model runs and kept for a future model
rebuild or a full data rerun.

This is a **cache of a reproducible computation, not a data store of record.**
Nothing here is unique — Sentinel-2 L1C is public and permanent, and the recipe
is in git. What it represents is roughly **two months of Earth Engine Partner
Tier credits**, which is the thing that is hard to get again. 

**Read this before you touch the data.** The bucket is ARCHIVE storage class:
retrieval runs about $0.05/GB, so reading all ~15 TB costs on the order of
$750, and every object carries a 365-day minimum storage duration. Do not
browse it casually or sync it to a laptop. Storage itself is ~$220/yr.

## Layout

One directory per period. The period is in the directory name — annual
(`2018`…`2025`) or quarterly (`Q125` onward) — so it is not repeated here.
Fourteen periods in all, eight annual and six quarterly, roughly 1.1 TB and
275k tiles each. Basin coverage is the same in every one.

Within a period, files are flat, one GeoTIFF per tile:

    {collection}_{tile.key}_{start}_{end}.tif
    S2L1C_552:12:10.0:17:27:-115_2018-01-01_2018-12-31.tif

`tile.key` is a **Descartes Labs `DLTile` key** (`descarteslabs.geo.DLTile`),
which reads `tilesize:pad:resolution:zone:ti:tj`:

| | |
| --- | --- |
| `552` | tilesize — the valid interior, in pixels |
| `12` | pad — pixels added on *every* side, so rasters are **576×576** (552 + 2×12) |
| `10.0` | resolution, m/pixel |
| `17` | UTM zone of the tile grid |
| `27:-115` | tile indices within that zone |

The pad is not margin for context. It is what keeps the **48 px inference
patches, cut at stride 24 (half a patch width), in one unbroken lattice across
tile boundaries.** Two properties make that work: `552` is a whole number of
strides (23 × 24), and the pad is half a stride. So the outermost patch in a
tile overhangs its interior by exactly 12 px — it can only be built because the
pad is there — and its centre sits exactly one stride from the first centre in
the neighbouring tile. No seam, no gap, no double coverage.

Change `tilesize` or `pad` without re-checking that arithmetic and inference
will still run, but the patch grid will quietly break at every tile edge.

## Two things that will bite you

**The rasters are EPSG:4326, even though the tile grid is UTM.** The DLTile key
names a UTM zone, but `write_tile` warps to WGS84 lat/lon on the way out, so
pixels are nominally-10 m rather than exactly 10 m square on the ground.

**Band order is not standard Sentinel-2 order — `B8A` precedes `B8`.** All 13
L1C bands, in this order:

    B1 B2 B3 B4 B5 B6 B7 B8A B8 B9 B10 B11 B12

Indexing by position without checking this silently swaps NIR bands. uint16,
deflate-compressed, internally tiled, no nodata value.

## How they were made

Earth Engine, project `earthindex`, via `GEE_Data_Extractor` in `core/gee.py`:

- `COPERNICUS/S2_HARMONIZED` — **Level-1C, top-of-atmosphere.** Not L2A surface
  reflectance; the models were trained on TOA and expect it.
- Cloud-masked against `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`, keeping
  pixels with `cs_cdf >= 0.6`.
- Reduced to a per-period **median** composite, then clipped and scaled to each
  tile at 576×576.

Parameters live in `DataConfig` (`core/gee.py`) and periods in
`core/periods.py`. Regenerating requires only those plus an EE project with
quota.

## Provenance

    repo    https://github.com/earthrise-media/mining-detector
    commit  8ae81c978a9e9e3d733d71a8b419adcfffc767f2

Pin to that commit before trusting the parameters above — `DataConfig` defaults
have changed before and will again.

## Custodianship

This bucket has no automated lifecycle and no process keeping it alive. It costs
about $220/yr and is worth roughly two months of unrepeatable credits. If the
project changes hands, hand this over explicitly — an unowned line item is what
gets deleted in a cleanup, and it will be deleted precisely when someone finally
needs it.
