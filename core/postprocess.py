#!/usr/bin/env python3
"""
postprocess.py — dual-threshold filtering of patch detections by spatial isolation.

Applies a main confidence threshold (t_main), computes each patch's distance to its
k-th nearest neighbor among surviving patches, then requires a stricter threshold
(t_iso) for isolated patches (distance > D km).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors

KTH_NEIGHBOR_FIELD = "kth_neighbor_km"
DEFAULT_CONFIDENCE_FIELD = "confidence"
DISSOLVE_CRS = "EPSG:4326"  # buffer_deg is in decimal degrees (~1 m at equator for 1e-5)


def centroids_xy_m(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Patch centroids as (n, 2) coordinates in meters (projected CRS)."""
    g = gdf.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    try:
        g = g.to_crs(g.estimate_utm_crs())
    except Exception:
        g = g.to_crs("EPSG:3857")
    c = g.geometry.centroid
    return np.column_stack([c.x.to_numpy(dtype=np.float64), c.y.to_numpy(dtype=np.float64)])


def kth_neighbor_km_on_catalog(coords_m: np.ndarray, k: int) -> np.ndarray:
    """Distance (km) from each point to its k-th nearest *other* point in coords_m."""
    n = len(coords_m)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= k:
        return out
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
    nn.fit(coords_m)
    dists_m, _ = nn.kneighbors(coords_m)
    out[:] = dists_m[:, k] / 1000.0
    return out


def keep_dual_threshold(
    confidence: np.ndarray,
    kth_neighbor_km: np.ndarray,
    *,
    t_main: float,
    t_iso: float,
    isolation_km: float,
) -> np.ndarray:
    """
    Return boolean mask of patches to keep.

    Positive if confidence >= t_main; isolated patches (kth_neighbor_km > isolation_km,
    or non-finite distance) also require confidence >= t_iso.
    """
    is_isolated = np.isfinite(kth_neighbor_km) & (kth_neighbor_km > isolation_km)
    if np.any(~np.isfinite(kth_neighbor_km)):
        is_isolated = is_isolated | ~np.isfinite(kth_neighbor_km)
    above_main = confidence >= t_main
    above_iso = confidence >= t_iso
    return np.where(is_isolated, above_main & above_iso, above_main)


def _fmt_param(value: float) -> str:
    return f"{value:g}"


def default_outpath(
    inpath: Path,
    *,
    t_main: float,
    k: int,
    isolation_km: float,
    t_iso: float,
) -> Path:
    """Output path: <stem>_t{t_main}_d{k}_{D}km_t-iso{t_iso}.geojson (notebook convention)."""
    stem = inpath.stem
    tag = (
        f"_t{_fmt_param(t_main)}_d{k}_{_fmt_param(isolation_km)}km"
        f"_t-iso{_fmt_param(t_iso)}"
    )
    return inpath.parent / f"{stem}{tag}.geojson"


def dissolved_outpath(patches_outpath: Path) -> Path:
    """Patch output path with -dissolved before the extension."""
    return patches_outpath.with_name(f"{patches_outpath.stem}-dissolved{patches_outpath.suffix}")


def dissolve_patches(
    gdf: gpd.GeoDataFrame,
    buffer_deg: float = 0.00001,
    conf_field: str = DEFAULT_CONFIDENCE_FIELD,
) -> gpd.GeoDataFrame:
    """
    Buffered dissolve of detection patches; aggregates confidence as mean.

    Reprojects to EPSG:4326 before buffering: ``buffer_deg`` is in decimal degrees
    (repo convention for inference GeoJSON).
    """
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(DISSOLVE_CRS)
    else:
        gdf = gdf.to_crs(DISSOLVE_CRS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dissolved_geom = gdf.buffer(buffer_deg, join_style=2).union_all()
        dissolved = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf.crs)
        dissolved = dissolved.explode(index_parts=False).reset_index(drop=True)
        dissolved.geometry = dissolved.buffer(-buffer_deg, join_style=2)

    joined = gpd.sjoin(gdf, dissolved, how="inner", predicate="intersects")
    mean_conf = joined.groupby("index_right")[conf_field].mean()
    dissolved[conf_field] = mean_conf.reindex(dissolved.index)

    dissolved.set_crs(gdf.crs, inplace=True)
    print(f"Dissolved {len(gdf)} patches to {len(dissolved)} polygons.")
    return dissolved


def dual_threshold_filter(
    gdf: gpd.GeoDataFrame,
    *,
    t_main: float,
    t_iso: float,
    k: int,
    isolation_km: float,
    confidence_field: str = DEFAULT_CONFIDENCE_FIELD,
) -> gpd.GeoDataFrame:
    """
    Filter detections with the dual-threshold rule.

    1. Keep patches with confidence >= t_main (neighbor catalog).
    2. Compute k-th nearest-neighbor distance on that catalog.
    3. Drop isolated catalog patches (distance > isolation_km) below t_iso.
    """
    if confidence_field not in gdf.columns:
        raise KeyError(
            f"Input needs {confidence_field!r}; columns: {list(gdf.columns)}"
        )

    confidence = gdf[confidence_field].to_numpy(dtype=np.float64)
    at_main = gdf.loc[confidence >= t_main].copy()
    if len(at_main) == 0:
        out = at_main.copy()
        out[KTH_NEIGHBOR_FIELD] = np.array([], dtype=np.float64)
        return out

    coords_m = centroids_xy_m(at_main)
    at_main[KTH_NEIGHBOR_FIELD] = kth_neighbor_km_on_catalog(coords_m, k=k)

    keep = keep_dual_threshold(
        at_main[confidence_field].to_numpy(dtype=np.float64),
        at_main[KTH_NEIGHBOR_FIELD].to_numpy(dtype=np.float64),
        t_main=t_main,
        t_iso=t_iso,
        isolation_km=isolation_km,
    )
    return at_main.loc[keep].copy()


def main(args: argparse.Namespace) -> None:
    inpath = Path(args.geojson_path)
    gdf = gpd.read_file(inpath)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    n_in = len(gdf)
    filtered = dual_threshold_filter(
        gdf,
        t_main=args.t_main,
        t_iso=args.t_iso,
        k=args.k,
        isolation_km=args.D,
        confidence_field=args.confidence_field,
    )

    outpath = Path(args.outpath) if args.outpath else default_outpath(
        inpath,
        t_main=args.t_main,
        k=args.k,
        isolation_km=args.D,
        t_iso=args.t_iso,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_file(outpath, driver="GeoJSON", index=False)

    n_at_main = int((gdf[args.confidence_field] >= args.t_main).sum())
    n_iso = int(
        (filtered[KTH_NEIGHBOR_FIELD] > args.D).sum()
        if len(filtered) and KTH_NEIGHBOR_FIELD in filtered.columns
        else 0
    )
    msg = (
        f"Read {n_in} patches from {inpath.name}; "
        f"{n_at_main} at t_main={args.t_main:g}; "
        f"wrote {len(filtered)} ({n_iso} isolated with d > {args.D:g} km) "
        f"to {outpath}"
    )
    if args.dissolve:
        dissolved_path = dissolved_outpath(outpath)
        dissolved = dissolve_patches(
            filtered, conf_field=args.confidence_field
        )
        dissolved.to_file(dissolved_path, driver="GeoJSON", index=False)
        msg += f"; dissolved -> {dissolved_path}"
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Dual-threshold post-processing: filter patch detections by confidence "
            "and spatial isolation (k-th nearest neighbor distance)."
        ),
    )
    parser.add_argument(
        "geojson_path",
        help="Input GeoJSON with patch detections and a confidence field.",
    )
    parser.add_argument(
        "--outpath",
        default=None,
        help=(
            "Output GeoJSON path. Default: input stem + "
            "_t{t_main}_d{k}_{D}km_t{t_iso}.geojson"
        ),
    )
    parser.add_argument(
        "--t-main",
        dest="t_main",
        type=float,
        default=0.43,
        help="Main confidence threshold (default: 0.43).",
    )
    parser.add_argument(
        "--t-iso",
        dest="t_iso",
        type=float,
        default=0.75,
        help="Stricter threshold for isolated patches (default: 0.75).",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=5,
        help="Neighbor rank for isolation distance, e.g. 5 = 5th NN (default: 5).",
    )
    parser.add_argument(
        "-D",
        "--D",
        type=float,
        default=3.0,
        help="Isolation cutoff in km; patches with k-th NN distance > D use t_iso.",
    )
    parser.add_argument(
        "--confidence-field",
        default=DEFAULT_CONFIDENCE_FIELD,
        help=f"Confidence column name (default: {DEFAULT_CONFIDENCE_FIELD}).",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help=(
            "Also write a buffered-dissolve polygon layer as "
            "<patch-outstem>-dissolved.geojson"
        ),
    )

    main(parser.parse_args())
