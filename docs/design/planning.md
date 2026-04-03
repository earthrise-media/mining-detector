# Dense embedding cache & inference (cls + patch tokens)

**Recorded:** 2026-04-02 (design session; implementation draft in repo follows this note.)  
**Updated:** 2026-04-02 — inference wiring (448→4×224→patch geometries into existing `predict_on_tile_embeddings` path) and pooling notes.  
**Updated:** 2026-04-02 — `InferenceConfig.embedding_strategy`: either `cls_only` or `dense`; cache and FM loader/embed path are **not** mixed (see below).

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

- `gee/dense_embedding_cache.py` — path helpers, `save_dense_embedding_parquets` / `load_dense_embedding_parquets`, `merge_cls_patch_for_probe`, `build_patch_cell_geometries`.
- `gee/gee.py` — `InferenceConfig.embedding_strategy` (`cls_only` | `dense`); FM loader selection; `split_parent_pixels_to_embed_windows`; `produce_tile_input` / `predict_on_tile_pixels` / `predict_on_tile` wired per strategy; `embed` vs `embed_dense` guarded by strategy; dense cache paths on `embed_dense` when caching enabled.

## Operational notes

- **Atomic writes:** interrupted runs may leave one file without the other; callers should treat **both** present as a valid cache hit (optional manifest / temp+rename can be added later).
- **Versioning:** cache keys should remain tied to **collection, date range, tile key, FM weights** (implicit via path layout and operator discipline).
- **Legacy path:** `embed()` + single `_embeddings.parquet` unchanged; `embed_dense()` reads/writes the **pair** only.

## Inference wiring (dense paradigm) — design

**Goal:** For each **parent tile** (target footprint **448×448** px at the GEE extraction resolution, or any `N` windows of **224×224**), either **restore** dense embeddings from the Parquet pair or **compute** them from pixels, then run the **existing** Keras probe + `_preds_to_gdf` path with **one row per ViT patch cell** (e.g. **4 × 14 × 14 = 784** geometries and feature vectors).

### Why this fits the current pipeline

Today the consumer already supports `mode == "embeddings"`: it calls `predict_on_tile_embeddings(embeddings, chip_geoms, tile)` where `embeddings` is `(N, F)` and `chip_geoms` has `N` footprints. The dense paradigm only changes **how N and F are produced**:

1. Build **`features`** of shape `(N_patches, 2·Dim)` via `merge_cls_patch_for_probe` (or equivalent in-process), where `N_patches = N_windows · H · W` (e.g. 4·14·14).
2. Build **`chip_geoms`** as a `GeoDataFrame` with **one polygon (or point) per patch cell**, CRS EPSG:4326, **same row order** as `features`.
3. Pass those into **`predict_on_tile_embeddings`** unchanged. Output GeoJSON becomes **dense positives** at patch scale unless/until we add pooling (below).

No change is required to the **TF model** interface: it still sees a single 2D batch. The **geometry** column semantics change from “one chip per 48×48 stride” to “one cell per ViT patch,” which is intentional for this paradigm.

### Cache vs compute

| Step | Behavior |
|------|----------|
| **Cache hit** | Load `*_embed_dense_cls.parquet` + `*_embed_dense_patch.parquet`, run `merge_cls_patch_for_probe`, attach **patch geometries** (see below), enqueue `mode="embeddings"` with `(features, patch_geoms)`. |
| **Cache miss** | `get_tile_data(tile)` → raster shaped for the **parent** (448×448 when `tilesize`/padding match that contract). **Split** into **N** crops of **224×224** (typically **4** quadrants in pixel space). Build **224 window geometries** (four sub-footprints of the parent bounds in EPSG:4326 — same logic as mapping chip pixels to lon/lat today). Call **`embed_dense`** on the stack of crops + window geoms; optionally **write** the dense Parquet pair; then merge + patch geometries as on cache hit. |

**Order of attempts in `produce_tile_input`:**

- **`embedding_strategy == "dense"`:** (1) dense Parquet **pair** on disk → (2) raw pixels → split parent into 224 windows → `embed_dense` → optional cache write → patch geometries → `predict_on_tile_embeddings`.
- **`embedding_strategy == "cls_only"`:** (1) legacy **`*_embeddings.parquet`** → (2) raw pixels → `cut_chips` / `embed` / existing path.

There is **no** cross-format fallback: the strategy picks **one** cache type and **one** FM forward (`embed` vs `embed_dense`).

### Patch geometry construction

- **Inputs:** For each 224 window, an axis-aligned **polygon** in EPSG:4326 (from `chip_geoms` / cls parquet), and integers **`H, W`** (e.g. 14×14), **`model_input_px`** = 224.
- **Method (simple, matches “recompute later”):** Interpolate along the window’s min/max lon and lat to form a **regular geographic grid** of `H×W` smaller rectangles (each covers one ViT patch in lon/lat space). This is the usual “divide the window bbox into a grid” approach; it aligns with **bicubic-resampled** pixels feeding the ViT at first order, not a full geodesic equal-area projection.
- **Output:** `GeoDataFrame` with `H·W` rows per window, concatenated in **stable order** (`quadrant`, `patch_row`, `patch_col`) to match `merge_cls_patch_for_probe`.
- **Optional metadata columns** on the output GDF (for pooling / debugging): `parent_key`, `window_id`, `patch_row`, `patch_col`, `quadrant` — can be carried alongside `geometry` into GeoJSON if useful (may require schema tolerance in downstream tools).

### Configuration & tile contract

- **`InferenceConfig.embedding_strategy`** (implemented):
  - **`cls_only`** (default): frozen full-model checkpoint via `_load_embed_model_frozen`, **`embed()`** + legacy **`*_embeddings.parquet`** cache only. **`embed_dense()`** must not be used in this mode (raises).
  - **`dense`**: ViT with `get_intermediate_layers` via `_load_embed_model`, **`embed_dense()`** + **`*_embed_dense_{cls,patch}.parquet`** pair only. **`embed()`** raises in this mode.
- **Cache selection:** `produce_tile_input` consults **only** the cache format matching `embedding_strategy` — **no** fallback from dense files to legacy parquet or vice versa. Operators choose strategy and directory layout accordingly.
- **Parent size:** Dense inference expects the parent raster to split evenly into **`embed_model_chip_size`×`embed_model_chip_size`** windows (default **224**). A **2×2** grid (**448×448** parent) is the primary case; other counts (e.g. 1×1 or 3×3) work if height/width are integer multiples of 224.
- **`geo_chip_size` / `embed_model_chip_size`:** For **`dense`**, set **`geo_chip_size == embed_model_chip_size == 224`** so each 224 crop is not rescaled before the FM. **`cls_only`** keeps the existing **48** (or other) chip + bicubic resize behavior.
- **SAM2 / masking:** Same caveat as today: **`mode == "embeddings"`** skips inline SAM2 on pixels; dense cached runs need a **separate masking pass** if masks are required.

### Future: pooling & aggregation (design only)

Downstream we may not want **784** independent positives per parent tile.

- **Pre-probe pooling (representation):** Pool **patch tokens** (or their logits) inside each 224 window — e.g. attention, max/mean over 14×14 — to produce **one vector per window** or **one per 448 tile**, then a smaller `N` for `predict_on_tile_embeddings`. Changes training target unless the probe is retrained on pooled features.
- **Post-probe pooling (decisions):** Keep per-patch scores but **aggregate** for map products — e.g. max or percentile over 14×14 within a window, majority vote, or “any patch above τ” for a single footprint per 224 window. GeoJSON can store **patch polygons** today and add a **derived layer** after aggregation later.
- **Implementation hook:** Keep geometry + score pipeline **patch-native** first; add an optional **`aggregation`** parameter (`none` | `per_window` | `per_parent`) in a later iteration so bulk export can slim outputs without rewriting the FM.

---

## Implementation checklist

- [x] **`InferenceConfig.embedding_strategy`:** `cls_only` | `dense`; selects FM loader (`_load_embed_model_frozen` vs `_load_embed_model`), `embed` vs `embed_dense`, and cache format.
- [x] **`produce_tile_input`:** strategy-specific cache only; dense path loads pair → `merge_cls_patch_for_probe` + `build_patch_cell_geometries` → `mode="embeddings"`.
- [x] **`predict_on_tile_pixels`:** dense branch splits parent raster into `k_h×k_w` windows of size `embed_model_chip_size` (default 224), `embed_dense`, then probe on patch rows.
- [x] **`dense_embedding_cache.build_patch_cell_geometries`:** lon/lat grid per 224 window, row order aligned with `merge_cls_patch_for_probe`.
- [ ] Tests or **smoke notebook** on one tile: cache round-trip → many patch preds → GeoJSON sanity.
- Optional **manifest** (`done.json`) per parent after successful Parquet write.
