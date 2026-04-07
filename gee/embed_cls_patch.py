"""
Training-time export of ViT cls + one selected spatial patch per viewport.

Streams on-disk GeoTIFF super-chips, batches jittered views through
:class:`gee.InferenceEngine.embed_dense`, and builds a GeoDataFrame suitable
for Parquet (e.g. probe training). Inference tile caches use
:mod:`dense_embedding_cache` separately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point


@dataclass
class DenseEmbedConfig:
    """Hyperparameters for cls+patch training embedding export."""

    viewport: int = 224
    patch_px: int = 16
    patch_ct: int = 14
    embedding_batch_size: int = 16
    embed_model_name: str = "ssl4eo_vit_s16"
    embed_model_path: str = (
        "../models/SSL4EO/pretrained/ssl4eo_vit_small_patch16_weights.pth"
    )
    train_n_views: int = 8
    eval_n_views: int = 8
    quantized: bool = False
    progress_every: int = 500

    def __post_init__(self) -> None:
        if self.viewport != self.patch_ct * self.patch_px:
            raise ValueError(
                "viewport must equal patch_ct * patch_px (ViT geometry + jitter grid)"
            )


def make_embedding_engine(
    cfg: DenseEmbedConfig,
    *,
    gee_dummy_start_date: str = "2000-01-01",
    gee_dummy_end_date: str = "2000-01-02",
) -> Any:
    """Build an ``InferenceEngine`` for local ViT embedding only.

    No Keras ``model_path`` is required; only ``embed_dense`` is used.

    Dummy dates satisfy the constructor; GEE extraction is not used on this path.
    """
    import gee

    inference_config = gee.InferenceConfig(
        embedding_strategy="cls_patch",
        embedding_batch_size=cfg.embedding_batch_size,
        embed_model_name=cfg.embed_model_name,
        embed_model_path=cfg.embed_model_path,
        embed_model_chip_size=cfg.viewport,
        geo_chip_size=cfg.viewport,
    )
    return gee.InferenceEngine(
        gee_dummy_start_date,
        gee_dummy_end_date,
        gee.DataConfig(),
        inference_config,
    )


def feature_column_names_cls_patch(output_dim: int) -> List[str]:
    """Column names for :func:`cls_patch_feature_batch` row layout.

    Order matches the concatenated vector **CLS token then patch token** (each length
    ``output_dim``, e.g. 384 + 384 = 768 for ViT-S/16): ``cls0``…``cls{output_dim-1}``,
    then ``spatial0``…``spatial{output_dim-1}``. Same convention as
    :func:`dense_embedding_cache.merge_cls_patch_for_probe` and
    :func:`model_library.MLP_with_targeted_dropout` (first half = CLS).
    """
    return [f"cls{i}" for i in range(output_dim)] + [
        f"spatial{i}" for i in range(output_dim)
    ]


def raster_center_point(raster_src: Any) -> Any:
    """Bounds midpoint as a Point; training GeoTIFFs are EPSG:4326 (repo convention)."""
    bounds = raster_src.bounds
    return Point((bounds.left + bounds.right) / 2.0, (bounds.bottom + bounds.top) / 2.0)


def load_chip(
    path: Path, bands_to_use: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, Any]:
    """Load one GeoTIFF chip as float32 HWC and its EPSG:4326 label-center geometry."""
    path = Path(path)
    with rasterio.open(path) as raster_src:
        chip_center = raster_center_point(raster_src)
        chip_array = raster_src.read()
        if bands_to_use is not None:
            chip_array = chip_array[np.array(bands_to_use, dtype=np.intp), :, :]
        chip_array = np.moveaxis(chip_array, 0, -1)
        chip_array = chip_array.astype(np.float32) / 10000.0
    return chip_array, chip_center


def load_dataset(
    data_dir: Union[str, Path],
    splits: Optional[Sequence[str]] = None,
    source_files: Optional[Iterable[Union[str, Path]]] = None,
    bands_to_use: Optional[Sequence[int]] = None,
) -> Generator[
    Tuple[np.ndarray, int, str, str, Any],
    None,
    None,
]:
    """
    Generator: yield one on-disk chip at a time to avoid holding the full dataset in RAM.

    Layout: data_dir / <source_stem> / {split} / {0|1} / *.tif

    Yields
    ------
    chip_hwc : np.ndarray
        Image (H, W, C), float32.
    label : int
        0 or 1 from parent folder name.
    split : str
        Split folder name (e.g. train, val).
    chip_path_posix : str
        Path to the .tif.
    chip_center : shapely Point
        Chip center in EPSG:4326.

    source_files: optional iterable of source folder names or paths; stems must match
        directories under data_dir. None = all top-level source dirs.

    If nothing matches, the generator yields no items (see collect_cls_patch_embedding_table warning).
    """
    root = Path(data_dir)
    if isinstance(splits, str):
        raise TypeError(
            "splits must be a list/tuple of split folder names (e.g. ['train']), "
            "not a single str — iterating a str splits per-character and yields almost no files."
        )
    allowed_source_stems = (
        {Path(s).stem for s in source_files} if source_files is not None else None
    )

    def iter_source_dirs():
        if allowed_source_stems is None:
            for child in sorted(p for p in root.iterdir() if p.is_dir()):
                yield child.name, child
        else:
            for source_stem in sorted(allowed_source_stems):
                source_dir = root / source_stem
                if source_dir.is_dir():
                    yield source_stem, source_dir

    if splits is None:
        split_names_found = set()
        for source_stem, source_dir in iter_source_dirs():
            for tif_path in source_dir.glob("**/[01]/*.tif"):
                if tif_path.parent.name not in {"0", "1"}:
                    continue
                split_names_found.add(tif_path.parent.parent.name)
        use_splits = sorted(split_names_found)
    else:
        use_splits = list(splits)

    for split_name in use_splits:
        saw_any_tif = False
        for source_stem, source_dir in iter_source_dirs():
            for class_subdir, class_label in (("0", 0), ("1", 1)):
                for tif_path in sorted(source_dir.glob(f"{split_name}/{class_subdir}/*.tif")):
                    saw_any_tif = True
                    chip_hwc, chip_center = load_chip(tif_path, bands_to_use)
                    chip_path_posix = tif_path.as_posix()
                    yield chip_hwc, class_label, split_name, chip_path_posix, chip_center
        if not saw_any_tif:
            print(f"No files found for split={split_name!r}; skipping.")


def quantize(
    embeddings: np.ndarray, lower_bound: float = -5, upper_bound: float = 5
) -> np.ndarray:
    clipped = np.clip(embeddings, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    scaled = normalized * 255
    return scaled.astype(np.uint8)


def hash_jitter(
    path_key: str, view_index: int = 0, *, patch_ct: int
) -> Tuple[int, int]:
    """
    Reproducible pseudo-random ViT patch grid indices (row, col) in [0, patch_ct).

    Hashes ``{path_key}|view{view_index}``. The full string (including
    ``view_index``) is mixed into the entire MD5 digest, so every output bit
    depends on both. Row/col use the first and last 8 hex digits (first and last
    32 bits of the digest) as two well-separated chunks.
    """
    key = f"{path_key}|view{view_index}"
    hash_hex = hashlib.md5(key.encode()).hexdigest()
    grid_row = int(hash_hex[:8], 16) % patch_ct
    grid_col = int(hash_hex[-8:], 16) % patch_ct
    return grid_row, grid_col


def get_seeded_random_jitter(
    path_key: str, view_index: int, *, patch_ct: int
) -> Tuple[int, int]:
    """
    Reproducible (row, col) patch indices for this path and view_index.

    Uses an isolated ``np.random.RandomState`` per call (no global RNG pollution).
    """
    seed_str = f"{path_key}_{view_index}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    grid_col = rng.randint(0, patch_ct)
    grid_row = rng.randint(0, patch_ct)
    return grid_row, grid_col


def extract_jittered_viewport(
    super_chip: np.ndarray,
    patch_loc: Tuple[int, int],
    *,
    viewport: int,
    patch_px: int,
) -> np.ndarray:
    """Crop a viewport×viewport window so the label center falls in ViT patch (row, col).

    ``patch_loc`` is (row_i, col_j) in the patch_ct×patch_ct grid (0-based), matching ``embed_dense`` spatial layout.
    ``super_chip`` is (nrows, ncols, bands).
    """
    row_i, col_j = patch_loc
    nrows, ncols = super_chip.shape[0], super_chip.shape[1]
    row_center = nrows // 2
    col_center = ncols // 2
    half_patch = patch_px // 2
    start_row = row_center + half_patch - (row_i + 1) * patch_px
    start_col = col_center + half_patch - (col_j + 1) * patch_px

    if (
        start_row < 0
        or start_col < 0
        or start_row + viewport > nrows
        or start_col + viewport > ncols
    ):
        raise ValueError(
            f"Viewport out of bounds: super_chip {nrows}x{ncols}, "
            f"start=({start_row},{start_col}), patch_loc={patch_loc!r}"
        )

    viewport_hwc = super_chip[
        start_row : start_row + viewport, start_col : start_col + viewport
    ]
    if viewport_hwc.shape[0] != viewport or viewport_hwc.shape[1] != viewport:
        raise RuntimeError(
            f"Expected ({viewport},{viewport}) crop, got {viewport_hwc.shape[:2]}"
        )
    return viewport_hwc


def iter_viewports_for_chip(
    super_chip: np.ndarray,
    path_key: str,
    *,
    n_views: int,
    viewport: int,
    patch_px: int,
    patch_ct: int,
) -> Generator[Tuple[int, np.ndarray, Tuple[int, int]], None, None]:
    """Yield (view_index, viewport_hwc, patch_loc) for one on-disk super-chip.

    Patch (row, col) comes from :func:`hash_jitter` — reproducible from
    ``path_key`` and ``view_index`` (use a stable chip path string).
    """
    for view_index in range(n_views):
        patch_loc = hash_jitter(path_key, view_index=view_index, patch_ct=patch_ct)
        viewport_hwc = extract_jittered_viewport(
            super_chip, patch_loc, viewport=viewport, patch_px=patch_px
        )
        yield view_index, viewport_hwc, patch_loc


def extract_target_patches(
    spatial_emb: np.ndarray, patch_locs: np.ndarray, *, patch_ct: int
) -> np.ndarray:
    """
    spatial_emb: (B, patch_ct, patch_ct, Dim)
    patch_locs: (B, 2) with (row_i, col_j) per batch row
    """
    grid_height, grid_width = spatial_emb.shape[1], spatial_emb.shape[2]
    if grid_height != patch_ct or grid_width != patch_ct:
        raise ValueError(
            f"spatial_emb grid {grid_height}x{grid_width} expected {patch_ct}x{patch_ct} "
            "(set patch_ct=... to match model)"
        )
    patch_locs = np.asarray(patch_locs, dtype=np.intp)
    batch_indices = np.arange(len(spatial_emb), dtype=np.intp)
    rows = patch_locs[:, 0]
    cols = patch_locs[:, 1]
    return spatial_emb[batch_indices, rows, cols, :]


def cls_patch_feature_batch(
    engine: Any,
    viewports_bhwc: np.ndarray,
    chip_geom: Any,
    patch_locs_batch: np.ndarray,
    *,
    feature_col_names: Sequence[str],
    quantized: bool = False,
    patch_ct: int,
) -> np.ndarray:
    """Run ``embed_dense`` once for all jittered viewports of one super-chip.

    Parameters
    ----------
    viewports_bhwc : array (B, H, W, C)
    chip_geom : shapely geometry (repeated B times for the engine API)
    patch_locs_batch : array (B, 2) int — ViT patch grid indices per view

    Returns
    -------
    (B, D) float (or uint8 if ``quantized``)
        Each row is **concat(class token, selected patch token)** along the feature
        axis, with ``D = 2 * dim`` and ``feature_col_names`` from
        :func:`feature_column_names_cls_patch` (``cls*`` then ``spatial*``).
    """
    viewports_batch = np.asarray(viewports_bhwc, dtype=np.float32)
    if viewports_batch.ndim != 4:
        raise ValueError(f"expected viewports (B,H,W,C), got shape {viewports_batch.shape}")
    num_views = viewports_batch.shape[0]
    patch_locs_arr = np.asarray(patch_locs_batch, dtype=np.intp)
    if patch_locs_arr.shape != (num_views, 2):
        raise ValueError(
            f"patch_locs_batch shape {patch_locs_arr.shape}, expected ({num_views}, 2)"
        )

    chip_geoms_geodataframe = gpd.GeoDataFrame(
        geometry=[chip_geom] * num_views, crs="EPSG:4326"
    )
    emb_dict = engine.embed_dense(viewports_batch, chip_geoms_geodataframe, tile=None)
    spatial = emb_dict["spatial"]
    if spatial.shape[1] != patch_ct or spatial.shape[2] != patch_ct:
        raise ValueError(
            f"embed_dense spatial shape {spatial.shape[1:3]} != patch_ct {patch_ct}x{patch_ct}"
        )
    patch_token_embeddings = extract_target_patches(
        spatial, patch_locs_arr, patch_ct=patch_ct
    )
    cls_and_patch = np.concatenate([emb_dict["cls"], patch_token_embeddings], axis=1)
    expected_width = len(feature_col_names)
    if cls_and_patch.shape[1] != expected_width:
        raise ValueError(
            f"Embedding width {cls_and_patch.shape[1]} != number of columns {expected_width}"
        )
    return quantize(cls_and_patch) if quantized else cls_and_patch


def _resolve_feature_col_names(
    engine: Any, feature_col_names: Optional[Sequence[str]]
) -> List[str]:
    if feature_col_names is not None:
        return list(feature_col_names)
    output_dim = engine.embed_model.norm.normalized_shape[0]
    return feature_column_names_cls_patch(output_dim)


def collect_cls_patch_embedding_table(
    cfg: DenseEmbedConfig,
    data_dir: Union[str, Path],
    engine: Any,
    *,
    feature_col_names: Optional[Sequence[str]] = None,
    splits: Optional[Sequence[str]] = None,
    source_files: Optional[Iterable[Union[str, Path]]] = None,
    bands_to_use: Optional[Sequence[int]] = None,
) -> gpd.GeoDataFrame:
    """Stream super-chips from disk; batch all views per chip through the ViT once.

    Hyperparameters (views, viewport geometry, quantization, progress cadence) come
    from ``cfg``. Train split uses ``cfg.train_n_views``; other splits use
    ``cfg.eval_n_views``. Viewports use :func:`hash_jitter`.

    ``cfg.embedding_batch_size`` (via :class:`gee.InferenceConfig`) should be
    >= ``cfg.train_n_views`` so the engine does not split one chip's views across
    GPU sub-batches.

    If ``feature_col_names`` is None, names are inferred from ``engine`` (cls* then spatial*).
    """
    names = _resolve_feature_col_names(engine, feature_col_names)
    output_rows: List[dict] = []
    chip_stream = load_dataset(
        data_dir,
        splits=splits,
        source_files=source_files,
        bands_to_use=bands_to_use,
    )
    for chip_index, (
        super_chip,
        label,
        split_name,
        chip_path_posix,
        chip_center_geom,
    ) in enumerate(chip_stream):
        if cfg.progress_every and (chip_index + 1) % cfg.progress_every == 0:
            print(f"Embedded {chip_index + 1} chips (latest split={split_name!r})...")
        n_views = (
            cfg.train_n_views if split_name == "train" else cfg.eval_n_views
        )
        viewport_rasters: List[np.ndarray] = []
        view_indices: List[int] = []
        jitter_patch_locations: List[Tuple[int, int]] = []
        for view_index, viewport_hwc, patch_loc in iter_viewports_for_chip(
            super_chip,
            chip_path_posix,
            n_views=n_views,
            viewport=cfg.viewport,
            patch_px=cfg.patch_px,
            patch_ct=cfg.patch_ct,
        ):
            viewport_rasters.append(viewport_hwc)
            view_indices.append(view_index)
            jitter_patch_locations.append(patch_loc)
        if not viewport_rasters:
            continue
        viewports_batch = np.stack(viewport_rasters, axis=0)
        patch_locs_batch = np.array(jitter_patch_locations, dtype=np.intp)
        cls_spatial_features_batch = cls_patch_feature_batch(
            engine,
            viewports_batch,
            chip_center_geom,
            patch_locs_batch,
            feature_col_names=names,
            quantized=cfg.quantized,
            patch_ct=cfg.patch_ct,
        )
        for view_row_index, view_index in enumerate(view_indices):
            row_dict = {
                "label": label,
                "split": split_name,
                "path": chip_path_posix,
                "view": view_index,
                "patch": jitter_patch_locations[view_row_index],
                "geometry": chip_center_geom,
            }
            for column_name, value in zip(
                names, cls_spatial_features_batch[view_row_index]
            ):
                row_dict[column_name] = value
            output_rows.append(row_dict)

    if not output_rows:
        allowed_repr = (
            None if source_files is None else {Path(s).stem for s in source_files}
        )
        print(
            f"Warning: no .tif chips embedded under {Path(data_dir).resolve()!r} "
            f"splits={splits!r} (source filter={allowed_repr!r})."
        )
        return gpd.GeoDataFrame(
            {
                "label": pd.Series([], dtype=np.int64),
                "split": pd.Series([], dtype=object),
                "path": pd.Series([], dtype=object),
                "view": pd.Series([], dtype=np.int64),
                "patch": pd.Series([], dtype=object),
                **{c: pd.Series([], dtype=np.float32) for c in names},
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )

    return gpd.GeoDataFrame(output_rows, crs="EPSG:4326")
