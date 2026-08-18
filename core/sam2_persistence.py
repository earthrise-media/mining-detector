#!/usr/bin/env python3
"""
Cumulative mine-scar masks with temporal persistence.

Produces one **onset raster** per UTM/lat band group: each pixel holds the year
its mining was confirmed, or 0. Every cumulative derives from it by
``onset != 0 & onset <= Y``, the raster analogue of the detection layer's
first-detection product, and it cannot decrease by construction.

Three things combine, and the order matters:

1. **Pixel-level persistence.** A pixel's mask onset is the first year it is
   masked *and* corroborated within the window -- the same ``k``-of-``n`` rule the
   detections use (recipe A: k=2, window=2), so the two halves admit a site on one
   definition. Only years whose window has closed can be onsets.

2. **Attribution to confirmed detections.** Detections are authoritative for
   *when*, masks for *how much*. Mask area is kept only where a confirmed
   detection could plausibly have generated it: bounded geodesic growth through
   the mask from confirmed detections, capped at the prior-implied extent.

   The cap is what makes rejection bite. Keeping whole connected components
   instead lets one surviving detection retain a 4,636 ha blob (7.6% of mask area
   in a Tapajós window); clipping to detection footprints instead discards the
   14.4% of per-period mask area that legitimately sits outside one. The cap is
   the pipeline's own bound: with ``penalty = -(dist/prior_sigma)^2``, a pixel
   cannot clear threshold beyond ``prior_sigma*sqrt(max_logit)``. Measured on 2023
   that is 42.7 px against a furthest observed mask pixel of 33.1 px, so it clips
   no real scar.

3. **Monotonicity.** Both inputs to a year -- persistent pixels and confirmed
   detections -- only grow with Y, so the admitted set only grows, and a pixel's
   onset is the first year it is admitted.

Years are stacked by windowed reads at integer lattice offsets. Per-year band
mosaics have different extents (each is the snapped union of that year's tiles)
but share the global grid, so no resampling is involved.

Usage:
    python sam2_persistence.py --years 2018 2019 2020 2021 2022 2023 2024 2025 \
        --group utm21_lat_-8_0
"""
from __future__ import annotations

import argparse
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import scipy.ndimage as ndi
from rasterio.windows import Window
from skimage.graph import MCP_Geometric

try:
    from .persistence import PersistenceConfig
    from .sam2_logits import MASK_NODATA
except ImportError:
    from persistence import PersistenceConfig
    from sam2_logits import MASK_NODATA

REPO = Path(__file__).resolve().parent.parent
GRID_RES = 0.00009
PRIOR_SIGMA = 12.0
LOGIT_CLAMP = 16.0

#: No confirmed mining. Unobserved ground reads the same: for a cumulative
#: product "not confirmed" is the answer either way, and a year's observed
#: footprint is recoverable from that year's own mask.
NO_ONSET = 0

ATTRIBUTION_RULES = ("bounded_growth", "nearest_seed", "components", "clip")


@dataclass
class MaskPersistenceConfig:
    """The rule, and how mask area is attributed to confirmed detections."""

    #: Inherited so the mask and detection halves cannot drift apart.
    k: int = PersistenceConfig.k
    window: int = PersistenceConfig.window

    #: How mask area is tied to confirmed detections. ``bounded_growth`` is the
    #: chosen rule; the others exist to be measured against it.
    attribution: str = "bounded_growth"

    #: Growth cap in pixels. ``None`` derives ``prior_sigma*sqrt(max logit)``
    #: from the data, falling back to the clamp-implied ceiling.
    cap_px: Optional[float] = None

    #: Processing window. Attribution is spatial, so windows are read with a halo
    #: of ``cap_px`` and only their interior is written.
    window_px: int = 2048

    prior_sigma: float = PRIOR_SIGMA

    def fallback_cap_px(self) -> float:
        """Hard ceiling from the logit clamp, independent of any data."""
        return self.prior_sigma * math.sqrt(LOGIT_CLAMP)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def band_group_paths(sam2_root: Path, years: Sequence[int], group: str,
                     pattern: str = "Amazon_ACA*_{y}-01-01_{y}-12-31_t0.43*"
                     ) -> Dict[int, Path]:
    """Per-year mask mosaic for one UTM/lat band group."""
    out = {}
    for year in years:
        hits = glob.glob(str(sam2_root / pattern.format(y=year) / "cog_outputs"
                             / f"mining_mask_*_{group}_epsg4326.tif"))
        if len(hits) > 1:
            raise ValueError(f"{year} {group}: {len(hits)} candidates")
        if hits:
            out[year] = Path(hits[0])
    return out


def union_grid(paths: Sequence[Path]) -> Tuple[rasterio.Affine, int, int]:
    """Common grid covering every input, exploiting the shared lattice.

    Origins are integer multiples of GRID_RES, so the union is an integer pixel
    box and every input aligns to it without resampling.
    """
    i0 = j0 = math.inf
    i1 = j1 = -math.inf
    for path in paths:
        with rasterio.open(path) as ds:
            ci = round(ds.transform.c / GRID_RES)
            fj = round(ds.transform.f / GRID_RES)
            i0, j0 = min(i0, ci), min(j0, fj - ds.height)
            i1, j1 = max(i1, ci + ds.width), max(j1, fj)
    width, height = int(i1 - i0), int(j1 - j0)
    transform = rasterio.Affine(GRID_RES, 0, i0 * GRID_RES,
                                0, -GRID_RES, j1 * GRID_RES)
    return transform, width, height


def derive_cap_px(paths: Sequence[Path], config: MaskPersistenceConfig,
                  sample: int = 40) -> float:
    """``prior_sigma*sqrt(max logit)``, from the logits beside the masks."""
    best = -math.inf
    for path in paths:
        tile_dir = path.parent.parent
        logits = sorted(glob.glob(str(tile_dir / "*-logits.tif")))
        if not logits:
            continue
        step = max(1, len(logits) // sample)
        for lp in logits[::step][:sample]:
            with rasterio.open(lp) as ds:
                arr = ds.read(1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                best = max(best, float(finite.max()))
    if not math.isfinite(best) or best <= 0:
        return config.fallback_cap_px()
    return config.prior_sigma * math.sqrt(best)


def read_on_grid(path: Path, transform: rasterio.Affine, window: Window
                 ) -> np.ndarray:
    """Read ``window`` of the union grid from ``path``, padding where absent."""
    with rasterio.open(path) as ds:
        di = round((transform.c - ds.transform.c) / GRID_RES)
        dj = round((ds.transform.f - transform.f) / GRID_RES)
        src = Window(window.col_off + di, window.row_off + dj,
                     window.width, window.height)
        return ds.read(1, window=src, boundless=True, fill_value=MASK_NODATA)


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

def mask_onset_index(stack: np.ndarray, config: MaskPersistenceConfig
                     ) -> np.ndarray:
    """First year index at which each pixel's mask is corroborated.

    ``stack`` is (n_years, h, w) boolean. Returns int16 indices, -1 where never.
    Only years whose window has closed are candidates, matching
    ``persistence.resolvable_periods``.
    """
    n = len(stack)
    onset = np.full(stack.shape[1:], -1, dtype=np.int16)
    counts = stack.astype(np.uint8)
    for i in range(n - config.window + 1):
        corroborated = counts[i:i + config.window].sum(axis=0) >= config.k
        take = stack[i] & corroborated & (onset < 0)
        onset[take] = i
    return onset


def attribute(mask: np.ndarray, seeds: np.ndarray, config: MaskPersistenceConfig,
              cap_px: float) -> np.ndarray:
    """Mask area a confirmed detection could plausibly have generated."""
    if not mask.any() or not seeds.any():
        return np.zeros_like(mask)

    rule = config.attribution
    if rule == "clip":
        return mask & seeds
    if rule == "components":
        labels, _ = ndi.label(mask, np.ones((3, 3)))
        keep = np.unique(labels[seeds & mask])
        return np.isin(labels, keep[keep > 0])
    if rule == "bounded_growth":
        # Geodesic distance *through the mask* from any seed. MCP_Geometric gives
        # true euclidean path length, so the cap is comparable to the analytic
        # prior bound; iterative dilation would give chebyshev and over-reach
        # diagonally by up to sqrt(2).
        costs = np.where(mask, 1.0, np.inf)
        starts = np.argwhere(seeds & mask)
        if not len(starts):
            return np.zeros_like(mask)
        mcp = MCP_Geometric(costs, fully_connected=True)
        dist, _ = mcp.find_costs(starts.tolist())
        return mask & np.isfinite(dist) & (dist <= cap_px)
    if rule == "nearest_seed":
        # Every mask pixel belongs to its nearest seed; a rejected detection
        # loses its own neighbourhood and nothing else. Seeds here are already
        # the retained set, so this reduces to a plain distance cap within the
        # component -- the strictly tighter fallback.
        dist = ndi.distance_transform_edt(~(seeds & mask))
        labels, _ = ndi.label(mask, np.ones((3, 3)))
        keep = np.unique(labels[seeds & mask])
        return mask & np.isin(labels, keep[keep > 0]) & (dist <= cap_px)
    raise ValueError(f"unknown attribution rule {rule!r}")


def confirmed_detection_mask(dets: gpd.GeoDataFrame, sindex, through_year: int,
                             transform: rasterio.Affine, window: Window
                             ) -> np.ndarray:
    """Rasterize detections confirmed at or before ``through_year``."""
    win_transform = rasterio.windows.transform(window, transform)
    bounds = rasterio.windows.bounds(window, transform)
    hits = list(sindex.intersection(bounds))
    shape = (int(window.height), int(window.width))
    if not hits:
        return np.zeros(shape, dtype=bool)
    sub = dets.iloc[hits]
    sub = sub[(sub["status"] == "confirmed") & (sub["onset_year"] <= through_year)]
    if sub.empty:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        ((g, 1) for g in sub.geometry if g is not None and not g.is_empty),
        out_shape=shape, transform=win_transform, fill=0, dtype="uint8"
    ).astype(bool)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_onset_raster(mask_paths: Dict[int, Path], detections: Path,
                       out_path: Path, config: MaskPersistenceConfig,
                       cap_px: Optional[float] = None) -> dict:
    """Write the onset-year raster for one band group.

    Windows are read with a ``cap_px`` halo and only their interior is written:
    attribution grows through the mask, so a pixel near a window edge must see
    the seeds and mask just outside it or growth would be truncated at an
    arbitrary boundary.
    """
    years = sorted(mask_paths)
    paths = [mask_paths[y] for y in years]
    transform, width, height = union_grid(paths)
    if cap_px is None:
        cap_px = config.cap_px or derive_cap_px(paths, config)
    halo = int(math.ceil(cap_px)) + 2

    resolvable = years[:len(years) - config.window + 1]
    print(f"  grid {width:,} x {height:,}  cap {cap_px:.1f} px  halo {halo} px")
    print(f"  onset candidates: {resolvable[0]}-{resolvable[-1]} "
          f"(window {config.window} needs {config.window - 1} following year(s))")

    dets = gpd.read_file(detections, columns=["onset_year", "status"])
    sindex = dets.sindex

    profile = dict(driver="GTiff", height=height, width=width, count=1,
                   dtype="uint16", crs="EPSG:4326", transform=transform,
                   nodata=NO_ONSET, tiled=True, blockxsize=512, blockysize=512,
                   compress="zstd", BIGTIFF="IF_SAFER")

    step = config.window_px
    n_win = math.ceil(width / step) * math.ceil(height / step)
    stats = dict(windows=n_win, nonempty=0, onset_px=0, cap_px=cap_px)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        done = 0
        for row in range(0, height, step):
            for col in range(0, width, step):
                done += 1
                w = min(step, width - col)
                h = min(step, height - row)
                # padded read window, clipped to the grid
                pc, pr = max(0, col - halo), max(0, row - halo)
                pw = min(width, col + w + halo) - pc
                ph = min(height, row + h + halo) - pr
                padded = Window(pc, pr, pw, ph)

                stack = np.stack([read_on_grid(p, transform, padded) == 1
                                  for p in paths])
                if not stack.any():
                    continue

                mask_onset = mask_onset_index(stack, config)
                onset = np.zeros((ph, pw), dtype=np.uint16)
                for idx, year in enumerate(resolvable):
                    persistent = (mask_onset >= 0) & (mask_onset <= idx)
                    if not persistent.any():
                        continue
                    seeds = confirmed_detection_mask(
                        dets, sindex, year, transform, padded)
                    admitted = attribute(persistent, seeds, config, cap_px)
                    fresh = admitted & (onset == NO_ONSET)
                    onset[fresh] = year

                interior = onset[row - pr:row - pr + h, col - pc:col - pc + w]
                if interior.any():
                    stats["nonempty"] += 1
                    stats["onset_px"] += int((interior != NO_ONSET).sum())
                    dst.write(interior, 1, window=Window(col, row, w, h))
                if done % 200 == 0:
                    print(f"    {done:,}/{n_win:,} windows, "
                          f"{stats['onset_px']:,} px confirmed", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = MaskPersistenceConfig()
    ap.add_argument("--sam2_root", type=Path,
                    default=REPO / "data/outputs/sam2")
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(range(2018, 2026)))
    ap.add_argument("--group", required=True,
                    help="band group tag, e.g. utm21_lat_-8_0")
    ap.add_argument("--detections", type=Path, default=(
        REPO / "data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/cumulative"
        / "Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble"
          "_detections_first_year.geojson"))
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--attribution", choices=ATTRIBUTION_RULES,
                    default=defaults.attribution)
    ap.add_argument("--cap-px", dest="cap_px", type=float, default=None)
    ap.add_argument("--window-px", dest="window_px", type=int,
                    default=defaults.window_px)
    ap.add_argument("--k", type=int, default=defaults.k)
    ap.add_argument("--window", type=int, default=defaults.window)
    args = ap.parse_args()

    config = MaskPersistenceConfig(
        k=args.k, window=args.window, attribution=args.attribution,
        cap_px=args.cap_px, window_px=args.window_px)

    paths = band_group_paths(args.sam2_root, args.years, args.group)
    missing = sorted(set(args.years) - set(paths))
    print(f"{args.group}: {len(paths)} of {len(args.years)} years"
          + (f"  (missing {missing})" if missing else ""))
    if len(paths) < config.window:
        raise SystemExit(f"need at least {config.window} years")

    outdir = args.outdir or (REPO / "data/outputs/sam2/persistence_masks")
    out = outdir / f"mask_onset_{args.group}_{config.attribution}.tif"
    stats = build_onset_raster(paths, args.detections, out, config)
    print(f"\n  {stats['onset_px']:,} confirmed px in "
          f"{stats['nonempty']:,}/{stats['windows']:,} windows -> {out}")


if __name__ == "__main__":
    main()
