#!/usr/bin/env python3
"""
Write ``mask_config.txt`` beside existing SAM2 tile outputs, post hoc.

New runs emit this from ``MaskConfig.write_config()``. Runs that predate it need
one written after the fact, because without it the smoothed and unsmoothed logit
vintages are indistinguishable -- same dtype, grid and value range -- and
thresholding the wrong one fails silently.

The vintage is **determined from the artifacts, not assumed from the commit**:

* co-registration -- logits on the same grid as their mask, or the coarser
  pre-upsample vintage
* clamp -- the observed maximum absolute finite value
* smoothing -- whether ``stored > 0`` reproduces the tile's own mask, or whether
  replaying a Gaussian at ``sigma`` is needed first

Fields that cannot be recovered from a raster (the prior sigma, the SAM2 weights)
are written as declared defaults and labelled as such, so a reader can tell
measurement from assumption.

Usage:
    python scripts/write_mask_config.py DIR [DIR ...]
    python scripts/write_mask_config.py --all-sam2
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "core"))

from sam2_logits import DEFAULT_SMOOTHING_SIGMA, smooth_logits

SAM2_ROOT = REPO / "data/outputs/sam2"
SAMPLE = 12          # tiles probed per directory
AGREE_PX = 8         # per-tile pixel disagreement still counted as "reproduces"


def probe(tile_dir: Path, sigma: float, sample: int = SAMPLE) -> dict:
    """Determine the logits vintage of one tile directory."""
    logits = sorted(glob.glob(str(tile_dir / "*-logits.tif")))
    out = dict(n_logits=len(logits), vintage="unknown", clamp=None,
               sigma=None, checked=0, direct_ok=0, replay_ok=0)
    if not logits:
        return out

    step = max(1, len(logits) // sample)
    clamps = []
    for path in logits[::step][:sample]:
        path = Path(path)
        mask_path = path.with_name(path.name.replace("-logits.tif", "-msk.tif"))
        if not mask_path.exists():
            continue
        with rasterio.open(path) as a, rasterio.open(mask_path) as b:
            if (a.width, a.height) != (b.width, b.height):
                out["vintage"] = "pre-upsample (raw log_odds, coarser than mask)"
                return out
            lg, mask = a.read(1), b.read(1)
        finite = lg[np.isfinite(lg)]
        if finite.size:
            clamps.append(float(np.abs(finite).max()))
        truth = mask == 1
        out["checked"] += 1
        if int(((lg > 0) != truth).sum()) <= AGREE_PX:
            out["direct_ok"] += 1
        elif int(((smooth_logits(lg, sigma) > 0) != truth).sum()) <= AGREE_PX:
            out["replay_ok"] += 1

    if clamps:
        out["clamp"] = round(max(clamps), 2)
    if out["checked"]:
        if out["direct_ok"] == out["checked"]:
            out["vintage"], out["sigma"] = "smoothed", sigma
        elif out["replay_ok"] == out["checked"]:
            out["vintage"], out["sigma"] = "unsmoothed", sigma
        else:
            out["vintage"] = (f"inconsistent ({out['direct_ok']} direct, "
                              f"{out['replay_ok']} replay, of {out['checked']})")
    return out


def write_config(tile_dir: Path, info: dict, name: str = "mask_config.txt") -> Path:
    stored = {
        "smoothed": "smooth(clip(upsampled_log_odds, +/-clamp))",
        "unsmoothed": "clip(upsampled_log_odds, +/-clamp)   # threshold needs a smoothing replay",
    }.get(info["vintage"], info["vintage"])

    lines = [
        "# Written post hoc by scripts/write_mask_config.py.",
        "# 'measured' fields were determined from the rasters themselves;",
        "# 'declared' fields cannot be recovered from a raster.",
        "",
        f"logits_stored = {stored}          # measured",
        f"smoothing_sigma = {info['sigma']}          # measured",
        f"logit_clamp = {info['clamp']}          # measured (observed maximum)",
        f"prior_sigma = 12.0          # declared (MaskConfig default)",
        f"tiles = {info['n_logits']}",
        f"verified_on = {info['checked']} sampled tiles",
    ]
    path = tile_dir / name
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--all-sam2", action="store_true")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SMOOTHING_SIGMA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = list(args.dirs)
    if args.all_sam2:
        targets += [p for p in sorted(SAM2_ROOT.iterdir())
                    if p.is_dir() and not p.name.endswith("-logits-unsmoothed")
                    and glob.glob(str(p / "*-logits.tif"))]
    if not targets:
        raise SystemExit("nothing to do; pass directories or --all-sam2")

    print(f"{'tiles':>7} {'probed':>7} {'clamp':>7} {'sigma':>6}  "
          f"{'vintage':<12} directory")
    for d in targets:
        info = probe(Path(d), args.sigma)
        print(f"{info['n_logits']:>7,} {info['checked']:>7} "
              f"{str(info['clamp']):>7} {str(info['sigma']):>6}  "
              f"{info['vintage'][:12]:<12} {Path(d).name[:46]}")
        if not args.dry_run and info["vintage"] in ("smoothed", "unsmoothed"):
            write_config(Path(d), info)
        elif not args.dry_run:
            print(f"        ! not writing a config for vintage "
                  f"{info['vintage']!r}")


if __name__ == "__main__":
    main()
