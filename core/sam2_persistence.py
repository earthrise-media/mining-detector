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
    from .persistence import PersistenceConfig, Period
    from .sam2_logits import MASK_NODATA
except ImportError:
    from persistence import PersistenceConfig, Period
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

#: Runs contributing mask to a band group. The Andes supplemental is a second
#: pass at a lower raw threshold over ground *inside* Amazon_ACA, so its masks
#: are additional coverage of the same band groups, not separate regions --
#: omitting it left utm18 lat[-16,-8] 86% below the published product.
RUN_PATTERNS = ("Amazon_ACA*_{y}-01-01_{y}-12-31_t0.43*",
                "andes_supplemental*_{y}-01-01_{y}-12-31")


def band_group_paths(sam2_root: Path, years: Sequence[int], group: str,
                     patterns: Sequence[str] = RUN_PATTERNS
                     ) -> Dict[int, List[Path]]:
    """Per-year mask mosaics for one UTM/lat band group, across runs.

    A year maps to *every* run's mosaic for that group; the driver unions them.
    Detections are deduplicated where the runs overlap, but masks need no
    equivalent step -- the union is idempotent, so a pixel masked by both runs is
    simply masked.
    """
    out: Dict[int, List[Path]] = {}
    for year in years:
        found: List[Path] = []
        for pattern in patterns:
            for hit in glob.glob(str(sam2_root / pattern.format(y=year)
                                     / "cog_outputs"
                                     / f"mining_mask_*_{group}_epsg4326.tif")):
                found.append(Path(hit))
        if found:
            out[year] = sorted(found)
    return out


def encode_period(tag: str) -> int:
    """Sortable uint16 code for a period tag: ``2024`` -> 2024, ``Q125`` -> 20251.

    Chronological under plain comparison (2024 < 20251 < 20252 < 20261), so any
    cumulative is ``0 < onset <= code``. Years and quarters share one raster,
    which is what makes the year-boundary supersede structural: quarters are only
    emitted for years the rule cannot yet resolve, so recomputing after the next
    annual lands replaces them with the confirmed year automatically.
    """
    period = Period.parse(tag)
    return period.year if period.is_annual else period.year * 10 + period.quarter


def quarter_group_paths(sam2_root: Path, quarters: Sequence[str], group: str
                        ) -> Dict[str, List[Path]]:
    """Per-quarter diff-mask mosaics for one band group.

    Quarterly SAM2 runs segment the *increment* (patch_diffs), not the period, so
    these cover only locations new to that quarter.
    """
    out: Dict[str, List[Path]] = {}
    for tag in quarters:
        span = Period.parse(tag).date_span
        hits = [Path(h) for h in glob.glob(str(
            sam2_root / f"*growth_{tag}" / "cog_outputs"
            / f"mining_mask_{span}_{group}_epsg4326.tif"))]
        if hits:
            out[tag] = sorted(hits)
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


def selected_detection_mask(dets: gpd.GeoDataFrame, sindex, through_code: int,
                            transform: rasterio.Affine, window: Window,
                            statuses: Tuple[str, ...] = ("confirmed",)
                            ) -> np.ndarray:
    """Rasterize detections selected at or before ``through_code``.

    ``statuses`` is ``("confirmed",)`` for annual onsets and includes
    ``"provisional"`` at the quarterly edge, where nothing is confirmed yet.
    """
    win_transform = rasterio.windows.transform(window, transform)
    bounds = rasterio.windows.bounds(window, transform)
    hits = list(sindex.intersection(bounds))
    shape = (int(window.height), int(window.width))
    if not hits:
        return np.zeros(shape, dtype=bool)
    sub = dets.iloc[hits]
    sub = sub[sub["status"].isin(statuses) & (sub["_code"] <= through_code)]
    if sub.empty:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        ((g, 1) for g in sub.geometry if g is not None and not g.is_empty),
        out_shape=shape, transform=win_transform, fill=0, dtype="uint8"
    ).astype(bool)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_onset_raster(mask_paths: Dict[int, List[Path]], detections: Path,
                       out_path: Path, config: MaskPersistenceConfig,
                       cap_px: Optional[float] = None,
                       quarter_paths: Optional[Dict[str, List[Path]]] = None
                       ) -> dict:
    """Write the onset-year raster for one band group.

    Windows are read with a ``cap_px`` halo and only their interior is written:
    attribution grows through the mask, so a pixel near a window edge must see
    the seeds and mask just outside it or growth would be truncated at an
    arbitrary boundary.
    """
    years = sorted(mask_paths)
    quarter_paths = quarter_paths or {}
    quarters = sorted(quarter_paths, key=encode_period)
    flat = ([p for y in years for p in mask_paths[y]]
            + [p for q in quarters for p in quarter_paths[q]])
    transform, width, height = union_grid(flat)
    if cap_px is None:
        cap_px = config.cap_px or derive_cap_px(flat, config)
    halo = int(math.ceil(cap_px)) + 2

    resolvable = years[:len(years) - config.window + 1]
    print(f"  grid {width:,} x {height:,}  cap {cap_px:.1f} px  halo {halo} px")
    print(f"  onset candidates: {resolvable[0]}-{resolvable[-1]} "
          f"(window {config.window} needs {config.window - 1} following year(s))")

    dets = gpd.read_file(detections, columns=["onset", "onset_year", "status"])
    dets["_code"] = [encode_period(v) for v in dets["onset"]]
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

                stack = np.stack([
                    np.logical_or.reduce([read_on_grid(p, transform, padded) == 1
                                          for p in mask_paths[y]])
                    for y in years])

                # Do NOT skip the window when the annual stack is empty: a
                # location first segmented in a quarter has no annual mask
                # anywhere in its window, and an early-out here dropped exactly
                # the newest mining. Found on Q226 detections near
                # 8.106S 55.740W, where 4,221 quarterly pixels sat in a window
                # with zero annual coverage.
                onset = np.zeros((ph, pw), dtype=np.uint16)
                if stack.any():
                    mask_onset = mask_onset_index(stack, config)
                    for idx, year in enumerate(resolvable):
                        persistent = (mask_onset >= 0) & (mask_onset <= idx)
                        if not persistent.any():
                            continue
                        seeds = selected_detection_mask(
                            dets, sindex, year, transform, padded)
                        admitted = attribute(persistent, seeds, config, cap_px)
                        fresh = admitted & (onset == NO_ONSET)
                        onset[fresh] = year
                elif not quarters:
                    continue        # nothing annual, and no quarterly edge to add

                # Quarterly edge: no corroboration is possible yet, so a
                # quarter's diff mask is admitted wherever a provisional
                # detection of that quarter can reach it. Applied after the
                # annual years so a confirmed onset always wins -- which is also
                # why re-running after the next annual supersedes these.
                for tag in quarters:
                    code = encode_period(tag)
                    diff = np.logical_or.reduce(
                        [read_on_grid(p, transform, padded) == 1
                         for p in quarter_paths[tag]])
                    if not diff.any():
                        continue
                    seeds = selected_detection_mask(
                        dets, sindex, code, transform, padded,
                        statuses=("confirmed", "provisional"))
                    admitted = attribute(diff, seeds, config, cap_px)
                    fresh = admitted & (onset == NO_ONSET)
                    onset[fresh] = code

                interior = onset[row - pr:row - pr + h, col - pc:col - pc + w]
                if interior.any():
                    stats["nonempty"] += 1
                    stats["onset_px"] += int((interior != NO_ONSET).sum())
                    dst.write(interior, 1, window=Window(col, row, w, h))
                if done % 200 == 0:
                    print(f"    {done:,}/{n_win:,} windows, "
                          f"{stats['onset_px']:,} px confirmed", flush=True)
    return stats


def mosaic_onset_rasters(rasters: Sequence[Path], out_path: Path,
                         blocksize: int = 512) -> dict:
    """Merge the per-band onset rasters into one COG.

    Band groups partition the basin -- a tile belongs to exactly one, by its
    centre -- so the parts are disjoint and a plain mosaic is exact. No derived
    band, no pixel function: the expensive union happened upstream.
    """
    import subprocess
    import tempfile

    rasters = [Path(r) for r in rasters]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onset_mosaic_") as tmp:
        vrt = Path(tmp) / "onset.vrt"
        subprocess.run(["gdalbuildvrt", "-srcnodata", str(NO_ONSET),
                        "-vrtnodata", str(NO_ONSET), str(vrt)]
                       + [str(r) for r in rasters], check=True,
                       capture_output=True, text=True)
        subprocess.run(["gdal_translate", str(vrt), str(out_path), "-of", "COG",
                        "-co", "COMPRESS=ZSTD", "-co", f"BLOCKSIZE={blocksize}",
                        "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS",
                        "-a_nodata", str(NO_ONSET)],
                       check=True, capture_output=True, text=True)
    with rasterio.open(out_path) as ds:
        return dict(width=ds.width, height=ds.height, parts=len(rasters))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = MaskPersistenceConfig()
    ap.add_argument("--sam2_root", type=Path,
                    default=REPO / "data/outputs/sam2")
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(range(2018, 2026)))
    ap.add_argument("--quarters", nargs="*", default=[],
                    help="Quarterly diff runs to append at the provisional edge, "
                         "e.g. --quarters Q125 Q225")
    ap.add_argument("--mosaic", type=Path, default=None,
                    help="Merge existing per-band onset rasters into this COG "
                         "and exit; --group is then ignored")
    ap.add_argument("--group", default=None,
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

    if args.mosaic:
        outdir = args.outdir or (REPO / "data/outputs/sam2/persistence_masks")
        parts = sorted(Path(outdir).glob("mask_onset_*_bounded_growth.tif"))
        if not parts:
            raise SystemExit(f"no per-band onset rasters in {outdir}")
        info = mosaic_onset_rasters(parts, args.mosaic)
        print(f"{info['parts']} band rasters -> {info['width']:,}x{info['height']:,}"
              f"  {args.mosaic}")
        return
    if not args.group:
        raise SystemExit("--group is required (or use --mosaic)")

    config = MaskPersistenceConfig(
        k=args.k, window=args.window, attribution=args.attribution,
        cap_px=args.cap_px, window_px=args.window_px)

    paths = band_group_paths(args.sam2_root, args.years, args.group)
    missing = sorted(set(args.years) - set(paths))
    n_runs = sum(len(v) for v in paths.values())
    print(f"{args.group}: {len(paths)} of {len(args.years)} years, "
          f"{n_runs} run-mosaics"
          + (f"  (missing {missing})" if missing else ""))
    if len(paths) < config.window:
        raise SystemExit(f"need at least {config.window} years")

    outdir = args.outdir or (REPO / "data/outputs/sam2/persistence_masks")
    out = outdir / f"mask_onset_{args.group}_{config.attribution}.tif"
    qpaths = quarter_group_paths(args.sam2_root, args.quarters, args.group)
    if args.quarters:
        missing = [q for q in args.quarters if q not in qpaths]
        print(f"  quarterly edge: {len(qpaths)} of {len(args.quarters)} runs"
              + (f"  (missing {missing})" if missing else ""))
    stats = build_onset_raster(paths, args.detections, out, config,
                               quarter_paths=qpaths)
    print(f"\n  {stats['onset_px']:,} confirmed px in "
          f"{stats['nonempty']:,}/{stats['windows']:,} windows -> {out}")


if __name__ == "__main__":
    main()
