#!/usr/bin/env python3
"""
Migrate August-2026 unsmoothed ``-logits.tif`` tiles to the smoothed convention.

The August 2026 vintage stored ``clip(upsampled_log_odds, +/-16)`` without the
Gaussian regularization, so thresholding it directly gave IoU ~0.84 instead of
the mask -- a trap that looks like it works. Current code stores
``smooth(clip(...))``, so ``stored > 0`` *is* the mask.

No SAM2 re-run is needed: smoothing is a forward operation, so ``smooth(stored)``
is exactly the field production thresholds.

For each tile directory:

1. move ``*-logits.tif`` to a sibling ``<dir>-logits-unsmoothed/``
2. write the smoothed version back at the original path
3. **verify** ``(smoothed > 0) == (mask == 1)`` against the tile's own
   ``-msk.tif``, which must hold bit-exactly -- that is what the +/-16 clamp was
   chosen to guarantee. Any mismatch names the tile rather than leaving the pass
   to be trusted.

The originals are kept, not deleted; remove the ``*-logits-unsmoothed/`` folders
once you are satisfied.

Usage:
    python scripts/convert_logits_to_smoothed.py DIR [DIR ...] [--dry-run]
    python scripts/convert_logits_to_smoothed.py --all-sam2
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "core"))

from sam2_logits import (DEFAULT_SMOOTHING_SIGMA, MASK_NODATA, mask_from_logits,
                         smooth_logits)

SAM2_ROOT = REPO / "data/outputs/sam2"
SUFFIX = "-logits-unsmoothed"


def convert_dir(tile_dir: Path, sigma: float, dry_run: bool) -> dict:
    tile_dir = Path(tile_dir)
    logits = sorted(glob.glob(str(tile_dir / "*-logits.tif")))
    stash = tile_dir.parent / f"{tile_dir.name}{SUFFIX}"
    stats = dict(directory=tile_dir.name, tiles=len(logits), converted=0,
                 verified=0, mismatched=0, no_mask=0, bad=[])
    if not logits:
        return stats
    if stash.exists() and any(stash.iterdir()):
        stats["bad"].append(f"{stash.name} already populated; refusing")
        return stats
    if dry_run:
        return stats

    # Refuse the pre-July-2026 vintage. Those logits are raw log_odds on a
    # coarser grid (35/32) than their own mask, so smoothing them produces
    # neither the old field nor the new one -- they need an upsample as well,
    # which cannot be reproduced outside the original torch path. Detect by
    # co-registration rather than by date, since the folder name proves nothing.
    probe = Path(logits[0])
    probe_mask = probe.with_name(probe.name.replace("-logits.tif", "-msk.tif"))
    if probe_mask.exists():
        with rasterio.open(probe) as a, rasterio.open(probe_mask) as b:
            if (a.width, a.height) != (b.width, b.height):
                stats["bad"].append(
                    f"logits {a.width}x{a.height} vs mask {b.width}x{b.height}"
                    " -- pre-upsample vintage, skipping")
                return stats

    stash.mkdir(parents=True, exist_ok=True)
    for src in logits:
        src = Path(src)
        moved = stash / src.name
        shutil.move(str(src), str(moved))

        with rasterio.open(moved) as ds:
            profile = ds.profile
            unsmoothed = ds.read(1)
        smoothed = smooth_logits(unsmoothed, sigma).astype("float32")
        with rasterio.open(src, "w", **profile) as dst:
            dst.write(smoothed, 1)
        stats["converted"] += 1

        # acceptance test: thresholding the new file must reproduce the mask
        mask_path = src.with_name(src.name.replace("-logits.tif", "-msk.tif"))
        if not mask_path.exists():
            stats["no_mask"] += 1
            continue
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)
        derived = mask_from_logits(smoothed, as_uint8=True)
        if np.array_equal(derived == 1, mask == 1):
            stats["verified"] += 1
        else:
            stats["mismatched"] += 1
            if len(stats["bad"]) < 5:
                n = int((( derived == 1) != (mask == 1)).sum())
                stats["bad"].append(f"{src.name}: {n} px differ")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--all-sam2", action="store_true",
                    help=f"every tile directory under {SAM2_ROOT}")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SMOOTHING_SIGMA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = list(args.dirs)
    if args.all_sam2:
        targets += [p for p in sorted(SAM2_ROOT.iterdir())
                    if p.is_dir() and not p.name.endswith(SUFFIX)
                    and glob.glob(str(p / "*-logits.tif"))]
    if not targets:
        raise SystemExit("nothing to do; pass directories or --all-sam2")

    print(f"sigma {args.sigma}{'  (dry run)' if args.dry_run else ''}\n")
    print(f"{'tiles':>7} {'conv':>6} {'verified':>9} {'MISMATCH':>9} "
          f"{'no msk':>7}  directory")
    total = dict(tiles=0, converted=0, verified=0, mismatched=0, no_mask=0)
    for d in targets:
        s = convert_dir(d, args.sigma, args.dry_run)
        for k in total:
            total[k] += s[k]
        print(f"{s['tiles']:>7,} {s['converted']:>6,} {s['verified']:>9,} "
              f"{s['mismatched']:>9,} {s['no_mask']:>7,}  {s['directory'][:52]}")
        for msg in s["bad"]:
            print(f"          ! {msg}")
    print(f"\n{total['tiles']:,} tiles, {total['converted']:,} converted, "
          f"{total['verified']:,} verified, {total['mismatched']:,} mismatched, "
          f"{total['no_mask']:,} without a mask")
    if total["mismatched"]:
        raise SystemExit("MISMATCHES: thresholding the converted logits does not "
                         "reproduce the mask; do not delete the originals")


if __name__ == "__main__":
    main()
