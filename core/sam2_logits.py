#!/usr/bin/env python3
"""
Deriving mine-scar masks from saved SAM2 logits.

READ THIS BEFORE THRESHOLDING A ``-logits.tif``.

The saved logits **are** the field the production mask thresholds::

    stored = smooth(clip(upsampled_log_odds, ±16))
    mask   = stored > 0

So thresholding is the whole operation, and a stricter cutoff is a true
re-threshold::

    from sam2_logits import mask_from_logits
    mask = mask_from_logits(logits_array)                   # the product mask
    mask = mask_from_logits(logits_array, threshold=1.5)    # a stricter t_prov

Because thresholding now commutes with the max-reduce used to merge overlapping
tiles -- ``max(a,b) > t`` is identically ``(a>t) or (b>t)`` -- a logits *mosaic*
may be thresholded directly. That was not true of the unsmoothed vintage and is
the main reason for the change.

The ±16 clamp is lossless with respect to the mask: verified bit-identical to
the unclamped mask on every real tile tested. See ``MaskConfig.logit_clamp``.

``smoothing_sigma`` is fixed at 2.5 and baked in. Changing it now requires
re-running SAM2, which is deliberate: measured over 750 tiles, sigma across 0-5
moves basin mask area only ~2%, and an IoU scan against hand annotations found
the optimum at 1-1.5 worth +0.015 mean IoU while *worsening* the area ratio. The
two criteria conflict and both effects are small, so the flexibility did not pay
for a stored field that silently mis-thresholds.

Earlier vintages
----------------
Two, both superseded:

**August 2026 -- unsmoothed.** ``clip(upsampled_log_odds, ±16)``, co-registered
with its mask but not smoothed, so ``stored > 0`` gives IoU ~0.84 rather than the
mask. Pass ``smoothing_sigma=2.5`` to :func:`mask_from_logits` to read one, and
mind that per-tile smoothing must precede mosaicking for these: max-reduce on
unsmoothed logits is biased upward and smoothing across that seam inflates area
(0.17% measured on a real overlapping pair, up to 1.8% synthetic at 12 px).
``scripts/convert_logits_to_smoothed.py`` migrates them.

**Through July 2026 -- raw ``log_odds``.** Prior included but neither upsampled
nor smoothed, so not co-registered with their own masks: the logits raster is
coarser by exactly 35/32 in every UTM/lat band. Not rescuable here -- they need
an upsample as well, and would still not match bit-for-bit because the bilinear
upsample cannot be reproduced outside the original torch path. Use them as
diagnostic rasters, not as a substrate for masks.

Which vintage a file belongs to is recorded in the ``*_config.txt`` written
beside the tiles, not in the raster itself: the clamp and the spatial prior are
equally irreversible, so run-level is the right granularity.

Background and measurements: docs/design/persistence-planning.md.
"""
from __future__ import annotations

from typing import Optional

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
                     smoothing_sigma: Optional[float] = None,
                     as_uint8: bool = False) -> np.ndarray:
    """Mask from saved logits: a plain threshold.

    Current logits are stored **smoothed**, so thresholding is the whole
    operation and it reproduces the production mask at ``threshold=0``. Raise
    the threshold for a stricter provisional mask (``t_prov,mask``).

    Args:
        logits: saved (smoothed) log-odds for a single tile. NaN = unobserved.
        threshold: log-odds cutoff. 0 is the production operating point.
        smoothing_sigma: replay smoothing before thresholding. Only for the
            **unsmoothed** August 2026 vintage; leave ``None`` for current files.
            Applying it to a smoothed file smooths twice.
        as_uint8: return 0/1/``MASK_NODATA`` uint8 instead of a boolean array.
            Only this form can represent "not observed".

    Returns:
        Boolean mask, or uint8 with :data:`MASK_NODATA` where input was NaN.
    """
    field = logits if smoothing_sigma is None else smooth_logits(
        logits, smoothing_sigma)
    mask = field > threshold

    if not as_uint8:
        return mask

    out = mask.astype(np.uint8)
    out[~np.isfinite(field)] = MASK_NODATA
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
