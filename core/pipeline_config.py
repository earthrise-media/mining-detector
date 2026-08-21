"""What a human sets before running the pipeline.

Every value here is shared across stages and across modules, which is why it is
not in a `Config` dataclass: those belong to a single module, and these are the
contract *between* modules.

Two kinds of parameter, and only the first kind lives here:

**Contract parameters** appear in filenames. Every stage finds its input by
constructing a path that embeds them, so all stages must agree -- a threshold is
schema, not a tuning knob. Changing one here changes every derived path at once,
which is the point; changing one on an emitted command line does not, and the
next stage then looks for a file that was never written.

**Behaviour parameters** change what a stage computes but not where anything
lands: `pad`, `tilesize` and `clear_threshold` in `core/gee.py::DataConfig`,
`prior_sigma` and `smoothing_sigma` in `MaskConfig`. Edit the dataclass default.
Note that the outputs will be named identically to any produced before the
change, and nothing in the published provenance records it.

To explore a parameter rather than change the product, bypass the pipeline: call
the underlying script with `--outdir` pointing somewhere separate.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# the product
# --------------------------------------------------------------------------

#: Detection model. Paths derive from it, so this is the only place it appears.
MODEL = "48px_v4.10b-18d-20g-21a-22bc-ensemble"

#: Every period the product covers. **Add a period here before running it.**
#: `persist-detections`, `persist-masks` and `stage` read this rather than
#: `--periods`, because they recompute from the whole history; a period absent
#: from the list is invisible to them, so `pipeline.py` refuses one.
ALL_CURRENT_PERIODS: List[str] = [
    "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    "Q125", "Q225", "Q325", "Q425",
    "Q126", "Q226",
]

#: The basin is inferred as six subregions, split at lon -66/-56 and lat -5.
#: Whole-basin runs proved unreliable, so this is the normal path and every
#: full-basin pass concatenates. Names follow
#: data/boundaries/Amazon_ACA/Amazon_ACA_{n}.geojson.
SUBREGIONS = [1, 2, 3, 4, 5, 6]

# --------------------------------------------------------------------------
# thresholds: the filename contract
# --------------------------------------------------------------------------

#: Raw confidence floor for inference, basin and Andes supplemental.
RAW_THRESHOLD = 0.4
ANDES_THRESHOLD = 0.2

#: The same two as they appear in filenames. Not derived: the archive writes the
#: basin at two decimals and the supplemental at one, and the files are named
#: that way on the buckets. Formatting cannot reconcile them.
RAW_TAG = "0.40"
ANDES_TAG = "0.2"

#: Dual-threshold postprocess: (t_main, t_iso). The loose set is the recommended
#: single-period product and what SAM2 is prompted from; the stringent set stands
#: in for corroboration at the provisional edge.
LOOSE = (0.43, 0.75)
STRINGENT = (0.55, 0.8)

#: Isolation rule shared by both: k-th nearest neighbour within this distance.
NEIGHBOURS = 5
ISOLATION_KM = 3.0


def postprocess_tag(t_main: float, t_iso: float,
                    k: int = NEIGHBOURS, isolation_km: float = ISOLATION_KM) -> str:
    """The `t0.43_d5_3km_t-iso0.75` fragment naming a postprocessed product.

    Derived rather than written out, so the directory name and the parameters
    that produced it cannot disagree. Matches `persistence.detection_path`.
    """
    return f"t{t_main:g}_d{k}_{isolation_km:g}km_t-iso{t_iso:g}"


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

CORE = REPO / "core"
SCRIPTS = REPO / "scripts"
BASE = REPO / "data/outputs" / MODEL
SAM2 = REPO / "data/outputs/sam2"
GS = REPO / "data/staging_gs"
SOURCE_COOP = REPO / "data/staging_source-coop"
