# Dense embedding cache & inference (cls + patch tokens)

**Recorded:** 2026-04-02 (design session; implementation draft in repo follows this note.)

## Context

- **Foundation model:** ViT-style geo-FM at **224×224** → one **CLS** token and a **spatial grid** of patch tokens (e.g. 14×14 for patch16 on 224).
- **Inference tiling:** Plan to tile at **448×448**, yielding **four** non-overlapping **224×224** windows (`q0`–`q3`). Each window gets a full `embed_dense` forward (cls + all patch tokens).
- **Probe:** Trained on **cls ∥ spatial_patch** for a *selected* patch (training notebook); at inference we want the probe on **every** patch cell within each 224 window, so we need **all** tokens cached, not only CLS.

## Agreed layout: two Parquet files per parent tile

For each **parent** footprint (e.g. one **DLTile** / same `tile.key` used in filenames today):

1. **CLS + geometry** — one row per 224×224 window (typically **4** rows for a 448 parent).
   - **Geometry** is the geographic footprint of that **224 window** (the natural link for the CLS token).
   - Columns include **`parent_key`**, **`quadrant`** (`0`–`3`), **`window_id`** (`{parent_key}_q{quadrant}`), **`cls0`…`cls{D-1}`**, **`geometry`** (EPSG:4326).

2. **Patch tokens** — one row per ViT patch cell (e.g. **4 × 14 × 14 = 784** rows per parent).
   - **No geometry** in this file; patch footprints are **recomputed later** from window bounds + `(patch_row, patch_col)` + fixed 224 / patch pixel size.
   - Columns include **`parent_key`**, **`quadrant`**, **`window_id`**, **`patch_row`**, **`patch_col`**, **`spatial0`…`spatial{D-1}`**.

## Indexing & keys

- **`parent_key`:** Derived from the existing **DLTile `tile.key`** (same convention as current embedding cache filenames), so artifacts line up with current naming.
- **`window_id`:** `{parent_key}_q{quadrant}` with `quadrant ∈ {0,1,2,3}` for the 448→4×224 layout (generalizes if `N≠4` as long as rows are consistent).

## Probe / training consumption

- **Join** CLS to patches on **`window_id`** (or `parent_key` + `quadrant`) to build **concat(cls, spatial)** per patch row, matching the training feature layout.
- **Separate roles** in two tables avoids repeating WKT on hundreds of patch rows and keeps CLS semantics tied to **window geometry**.

## Scale & format

- Row counts per parent remain modest (**~788** cls + patch rows in the 4×(196+1) case); **long Parquet** is acceptable.
- Use **float32** for embedding columns; stable sort keys (`quadrant`, `patch_row`, `patch_col`) for reproducible reads.

## Cache filenames (implementation)

- Alongside legacy single-file `embed()` caches, dense caches use a distinct suffix pair under `embeddings_cache_dir`:
  - `"{collection}_{tile.key}_{start}_{end}_embed_dense_cls.parquet"`
  - `"{collection}_{tile.key}_{start}_{end}_embed_dense_patch.parquet"`

### Code (draft)

- `gee/dense_embedding_cache.py` — path helpers, `save_dense_embedding_parquets` / `load_dense_embedding_parquets`, and `merge_cls_patch_for_probe` (builds a single `(N, 2·Dim)` feature matrix + index `DataFrame` for probe inference).
- `gee/gee.py` — `InferenceEngine._make_dense_embedding_cache_paths`, and `embed_dense` loads/saves the pair when `tile` is set and `embeddings_cache_dir` is configured.

## Operational notes

- **Atomic writes:** interrupted runs may leave one file without the other; callers should treat **both** present as a valid cache hit (optional manifest / temp+rename can be added later).
- **Versioning:** cache keys should remain tied to **collection, date range, tile key, FM weights** (implicit via path layout and operator discipline).
- **Legacy path:** `embed()` + single `_embeddings.parquet` unchanged; `embed_dense()` reads/writes the **pair** only.

## Future work (not in initial draft)

- Wire **`produce_tile_input` / bulk inference** to load dense pair, run probe on all patch rows, aggregate scores to raster or GeoJSON.
- Optional **manifest** (`done.json`) per parent after successful write.
