#!/usr/bin/env python3
"""
Deriving mine-scar masks from saved SAM2 logits.

READ THIS BEFORE THRESHOLDING A ``-logits.tif``.

The saved logits are **not** the field the production mask thresholds. They are
saved deliberately unsmoothed, so that ``smoothing_sigma`` stays retunable
without re-running SAM2. The production mask is::

    mask = smooth(upsampled_log_odds) > 0

while the saved artifact is ``clip(upsampled_log_odds, ±16)`` with no smoothing.
Thresholding it directly therefore does **not** reproduce the mask, even at
threshold 0 -- measured IoU ~0.84 against the real product. That is expected,
not a bug.

To derive a mask correctly, replay the smoothing first::

    from sam2_logits import mask_from_logits
    mask = mask_from_logits(logits_array)            # reproduces the product
    mask = mask_from_logits(logits_array, threshold=1.5)   # a stricter t_prov

Two rules that are easy to get wrong:

1. **Replay per tile, before mosaicking.** Do not threshold a logits mosaic.
   Gaussian smoothing does not commute with the max-reduce used to merge
   overlapping tiles: max-reduce on raw logits is biased upward (the max of two
   noisy fields exceeds either mean) and smoothing then spreads that inflated
   max across the seam. Measured, this inflates area by up to 1.8% at a 12 px
   overlap and 4.5% under larger tile disagreement. The correct order is
   per-tile smooth -> threshold -> mosaic with the union (OR) rule. The logits
   mosaic written by ``sam2_build_cog.py`` is for inspection and analysis, not
   a substrate for masks.

2. **Sigma must match the run that produced the logits.** It is recorded in
   ``MaskConfig.smoothing_sigma``; the default here is the production value.
   Changing it is legitimate (that is why logits are stored unsmoothed) but it
   is a different product, not a bug fix.

The ±16 clamp on the stored logits is lossless with respect to the mask: it was
chosen so that the re-derived mask is bit-identical to the unclamped one after
the smoothing replay. See ``MaskConfig.logit_clamp``.

Background and measurements: docs/design/persistence-planning.md.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

#: Gaussian regularization applied to upsampled logits before thresholding.
#: Single source of truth; ``MaskConfig.smoothing_sigma`` defaults to this.
DEFAULT_SMOOTHING_SIGMA = 2.5

#: Saturation limit (log-odds) for persisted logits. Lossless w.r.t. the mask.
DEFAULT_LOGIT_CLAMP = 16.0

#: Log-odds cutoff for the production mask (probability 0.5).
PRODUCTION_THRESHOLD = 0.0

#: uint8 sentinel for "not observed" in mask rasters.
MASK_NODATA = 2


def smooth_logits(logits: np.ndarray,
                  smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA
                  ) -> np.ndarray:
    """Replay the Gaussian regularization on saved (unsmoothed) logits.

    NaN marks unobserved ground. ``ndi.gaussian_filter`` would smear NaN across
    the whole neighbourhood, so NaNs are held out and the result renormalized
    by the smoothed validity mask -- equivalent to smoothing over observed
    pixels only.
    """
    if not smoothing_sigma:
        return np.asarray(logits, dtype=np.float64)

    values = np.asarray(logits, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.all():
        return ndi.gaussian_filter(values, sigma=smoothing_sigma)

    filled = np.where(valid, values, 0.0)
    num = ndi.gaussian_filter(filled, sigma=smoothing_sigma)
    den = ndi.gaussian_filter(valid.astype(np.float64), sigma=smoothing_sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.where(valid, out, np.nan)


def mask_from_logits(logits: np.ndarray,
                     threshold: float = PRODUCTION_THRESHOLD,
                     smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
                     as_uint8: bool = False) -> np.ndarray:
    """Mask from saved logits, replaying the smoothing first.

    With default arguments this reproduces the production mask exactly. Raise
    ``threshold`` above 0 for a stricter provisional mask (``t_prov,mask``).

    Args:
        logits: saved (unsmoothed) log-odds for a **single tile**. NaN = unobserved.
        threshold: log-odds cutoff. 0 is the production operating point.
        smoothing_sigma: must match the run that produced ``logits``.
        as_uint8: return 0/1/``MASK_NODATA`` uint8 instead of a boolean array.
            Only this form can represent "not observed".

    Returns:
        Boolean mask, or uint8 with :data:`MASK_NODATA` where input was NaN.
    """
    smoothed = smooth_logits(logits, smoothing_sigma)
    mask = smoothed > threshold

    if not as_uint8:
        return mask

    out = mask.astype(np.uint8)
    out[~np.isfinite(smoothed)] = MASK_NODATA
    return out


def mosaic_masks(masks) -> np.ndarray:
    """Union-merge per-tile uint8 masks that already share a grid.

    1 wins over 0, and nodata only survives where every input is nodata. This
    is the correct merge for masks; do not substitute a max-reduce over logits
    followed by a single threshold (see the module docstring).
    """
    stack = np.stack([np.asarray(m, dtype=np.uint8) for m in masks])
    nodata = stack == MASK_NODATA
    return np.where(nodata.all(axis=0), MASK_NODATA,
                    np.where((stack == 1).any(axis=0), 1, 0)).astype(np.uint8)
