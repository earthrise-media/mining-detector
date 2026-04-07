"""
Parquet cache for :meth:`inference_engine.InferenceEngine.embed_dense` outputs.

Two files per parent tile (see ``docs/design/planning.md``):
  - *_embed_dense_cls.parquet   — one row per 224 window, cls + geometry
  - *_embed_dense_patch.parquet — one row per ViT patch cell, spatial only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


class DenseCachePaths(NamedTuple):
    cls_parquet: Path
    patch_parquet: Path


def make_dense_cache_paths(
    emb_cache_dir: Path,
    collection: str,
    tile_key: str,
    start_date: str,
    end_date: str,
) -> DenseCachePaths:
    """Return paths for the cls / patch pair (distinct from legacy ``*_embeddings.parquet``)."""
    safe = emb_cache_dir
    stem = f"{collection}_{tile_key}_{start_date}_{end_date}"
    return DenseCachePaths(
        cls_parquet=safe / f"{stem}_embed_dense_cls.parquet",
        patch_parquet=safe / f"{stem}_embed_dense_patch.parquet",
    )


def cls_column_names(dim: int) -> List[str]:
    return [f"cls{i}" for i in range(dim)]


def spatial_column_names(dim: int) -> List[str]:
    return [f"spatial{i}" for i in range(dim)]


def save_dense_embedding_parquets(
    paths: DenseCachePaths,
    cls_tokens: np.ndarray,
    spatial: np.ndarray,
    chip_geoms: gpd.GeoDataFrame,
    parent_key: str,
) -> None:
    """
    Write cls + patch Parquet pair.

    Parameters
    ----------
    cls_tokens
        (N, Dim) float32/float64.
    spatial
        (N, H, W, Dim) patch grid per window.
    chip_geoms
        N rows aligned with cls_tokens / spatial; geometry = 224 window footprint.
    parent_key
        Usually ``tile.key`` for the parent tile (e.g. DLTile id).
    """
    if cls_tokens.ndim != 2:
        raise ValueError(f"cls_tokens must be (N, Dim), got {cls_tokens.shape}")
    if spatial.ndim != 4:
        raise ValueError(f"spatial must be (N, H, W, Dim), got {spatial.shape}")
    n, dim = cls_tokens.shape
    n_sp, h, w, dim_s = spatial.shape
    if n != n_sp or dim != dim_s:
        raise ValueError(
            f"cls {cls_tokens.shape} incompatible with spatial {spatial.shape}"
        )
    if len(chip_geoms) != n:
        raise ValueError(
            f"chip_geoms length {len(chip_geoms)} != N={n}"
        )

    cls_names = cls_column_names(dim)
    spat_names = spatial_column_names(dim)

    cls_rows: List[Dict[str, Any]] = []
    for i in range(n):
        window_id = f"{parent_key}_q{i}"
        row: Dict[str, Any] = {
            "parent_key": parent_key,
            "quadrant": i,
            "window_id": window_id,
            "geometry": chip_geoms.geometry.iloc[i],
        }
        for j, name in enumerate(cls_names):
            row[name] = float(cls_tokens[i, j])
        cls_rows.append(row)

    cls_gdf = gpd.GeoDataFrame(cls_rows, crs=chip_geoms.crs or "EPSG:4326")
    cls_gdf.to_parquet(paths.cls_parquet, index=False)

    patch_rows: List[Dict[str, Any]] = []
    for i in range(n):
        window_id = f"{parent_key}_q{i}"
        for r in range(h):
            for c in range(w):
                pr: Dict[str, Any] = {
                    "parent_key": parent_key,
                    "quadrant": i,
                    "window_id": window_id,
                    "patch_row": r,
                    "patch_col": c,
                }
                for j, name in enumerate(spat_names):
                    pr[name] = float(spatial[i, r, c, j])
                patch_rows.append(pr)

    pd.DataFrame(patch_rows).to_parquet(paths.patch_parquet, index=False)


def load_dense_embedding_parquets(
    paths: DenseCachePaths,
) -> Optional[Dict[str, Any]]:
    """
    Load cls + spatial tensors if both Parquet files exist.

    Returns
    -------
    dict with keys ``cls`` (N, Dim), ``spatial`` (N, H, W, Dim),
    ``chip_geoms`` (GeoDataFrame, N rows), ``parent_key``, ``window_ids`` (list),
    or ``None`` if either file is missing.
    """
    if not paths.cls_parquet.is_file() or not paths.patch_parquet.is_file():
        return None

    cls_gdf = gpd.read_parquet(paths.cls_parquet)
    if not isinstance(cls_gdf, gpd.GeoDataFrame):
        cls_gdf = gpd.GeoDataFrame(cls_gdf, geometry="geometry")

    cls_cols = [c for c in cls_gdf.columns if c.startswith("cls") and c[3:].isdigit()]
    if not cls_cols:
        raise ValueError("No cls0, cls1, ... columns in cls parquet")
    cls_cols.sort(key=lambda x: int(x[3:]))
    dim = len(cls_cols)

    cls_gdf = cls_gdf.sort_values("quadrant", kind="mergesort").reset_index(drop=True)
    n = len(cls_gdf)
    cls_arr = cls_gdf[cls_cols].to_numpy(dtype=np.float32)

    patch_df = pd.read_parquet(paths.patch_parquet)
    patch_df = patch_df.sort_values(
        ["quadrant", "patch_row", "patch_col"], kind="mergesort"
    ).reset_index(drop=True)

    pk_cls = set(cls_gdf["parent_key"].unique())
    pk_patch = set(patch_df["parent_key"].unique())
    if pk_cls != pk_patch or len(pk_cls) != 1:
        raise ValueError(
            f"parent_key mismatch between cls {pk_cls} and patch {pk_patch}"
        )

    spat_cols = [
        c for c in patch_df.columns if c.startswith("spatial") and c[7:].isdigit()
    ]
    if not spat_cols:
        raise ValueError("No spatial0, spatial1, ... columns in patch parquet")
    spat_cols.sort(key=lambda x: int(x[7:]))
    if len(spat_cols) != dim:
        raise ValueError(
            f"cls dim {dim} != spatial dim {len(spat_cols)}"
        )

    h = int(patch_df["patch_row"].max()) + 1
    w = int(patch_df["patch_col"].max()) + 1
    if h * w * n != len(patch_df):
        raise ValueError(
            f"patch rows {len(patch_df)} != n*h*w ({n}*{h}*{w})"
        )

    spatial_arr = np.empty((n, h, w, dim), dtype=np.float32)
    for q in range(n):
        sub = patch_df.loc[patch_df["quadrant"] == q, spat_cols]
        if len(sub) != h * w:
            raise ValueError(f"quadrant {q}: expected {h*w} rows, got {len(sub)}")
        spatial_arr[q] = sub.to_numpy(dtype=np.float32).reshape(h, w, dim)

    parent_key = str(cls_gdf["parent_key"].iloc[0])
    window_ids = cls_gdf["window_id"].tolist()

    chip_geoms = cls_gdf[["geometry"]].copy()
    if cls_gdf.crs is not None:
        chip_geoms = chip_geoms.set_crs(cls_gdf.crs)

    return {
        "cls": cls_arr,
        "spatial": spatial_arr,
        "chip_geoms": chip_geoms,
        "parent_key": parent_key,
        "window_ids": window_ids,
    }


def build_patch_cell_geometries(
    window_geoms: gpd.GeoDataFrame,
    grid_side: int,
) -> gpd.GeoDataFrame:
    """
    Subdivide each 224 (FM input) window polygon into a ``grid_side×grid_side``
    geographic grid. Row order matches :func:`merge_cls_patch_for_probe`:
    window 0, patch (0,0)…(0,W-1), (1,0)…, then window 1, …
    """
    if grid_side < 1:
        raise ValueError("grid_side must be >= 1")
    n_win = len(window_geoms)
    if n_win == 0:
        return gpd.GeoDataFrame(geometry=[], crs=window_geoms.crs or "EPSG:4326")

    crs = window_geoms.crs or "EPSG:4326"
    polys: List[Any] = []
    for qi in range(n_win):
        geom = window_geoms.geometry.iloc[qi]
        west, south, east, north = geom.bounds
        dx = (east - west) / grid_side
        dy = (north - south) / grid_side
        for pr in range(grid_side):
            for pc in range(grid_side):
                left = west + pc * dx
                right = west + (pc + 1) * dx
                maxy = north - pr * dy
                miny = north - (pr + 1) * dy
                polys.append(box(left, miny, right, maxy))

    return gpd.GeoDataFrame(geometry=polys, crs=crs)


def merge_cls_patch_for_probe(
    loaded: Dict[str, Any],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build feature matrix (len, 2*Dim) = concat(cls, spatial) per patch row + index frame.

    Returns
    -------
    features
        (n_patches_total, 2 * Dim) float32
    index_df
        Columns parent_key, quadrant, window_id, patch_row, patch_col (and optional
        geometry left to caller if they join cls geoms per window).
    """
    cls_arr = loaded["cls"]
    spatial = loaded["spatial"]
    n, h, w, dim = spatial.shape
    rows_feat: List[np.ndarray] = []
    rows_meta: List[Dict[str, Any]] = []

    for q in range(n):
        c = cls_arr[q]
        for r in range(h):
            for cc in range(w):
                s = spatial[q, r, cc]
                rows_feat.append(np.concatenate([c, s]).astype(np.float32, copy=False))
                rows_meta.append(
                    {
                        "parent_key": loaded["parent_key"],
                        "quadrant": q,
                        "window_id": loaded["window_ids"][q],
                        "patch_row": r,
                        "patch_col": cc,
                    }
                )

    feat = np.stack(rows_feat, axis=0)
    meta = pd.DataFrame(rows_meta)
    return feat, meta
