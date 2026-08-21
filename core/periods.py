"""Inference periods: the vocabulary, and the list the product currently covers.

A leaf module on purpose -- no third-party imports. Everything downstream needs
to know what dates a period spans, and asking that question should not load the
geospatial stack. It previously lived in ``persistence.py``, which meant
``pipeline.py --list`` imported geopandas, numpy and pyproj to print stage names.

`date_span` is the single source of every calendar date in the pipeline. Emitted
inference commands, SAM2 date arguments and detection filenames all derive from
it, so the quarter boundaries below are defined once and nowhere else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

QUARTER_SPANS = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}
QUARTER_TAG_RE = re.compile(r"^Q([1-4])(\d{2})$")


# --------------------------------------------------------------------------
# THE ONE THING A HUMAN SETS
# --------------------------------------------------------------------------

#: Every period the published product covers. **Edit this when adding a period,
#: before running anything.**
#:
#: This is the only variable the pipeline needs a human to set -- every other
#: default is correct as it stands. Three stages read it rather than the
#: ``--periods`` argument, because they recompute from the whole history rather
#: than from one period: ``persist-detections``, ``persist-masks`` and ``stage``.
#: A period absent from this list is therefore invisible to them and cannot
#: reach the product, which is why ``pipeline.py`` refuses a ``--periods`` value
#: that is not a member.
#:
#: Quarters exist only for years the corroboration rule cannot yet resolve. When
#: a year becomes resolvable its quarters stay listed -- see "Retiring superseded
#: quarterly layers" in docs/design/persistence-planning.md.
ALL_CURRENT_PERIODS: List[str] = [
    "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    "Q125", "Q225", "Q325", "Q425",
    "Q126", "Q226",
]


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Period:
    """One inference period: a calendar year, or a quarter within one.

    Ordering is chronological by end date, with the annual period sorting after
    the quarters it contains -- an annual mosaic is only assembled once its year
    is complete.
    """
    sort_key: Tuple[int, int, int] = field(init=False, repr=False)
    year: int
    quarter: Optional[int] = None

    def __post_init__(self):
        end_month = 12 if self.quarter is None else self.quarter * 3
        object.__setattr__(
            self, "sort_key",
            (self.year, end_month, 1 if self.quarter is None else 0))

    @property
    def is_annual(self) -> bool:
        return self.quarter is None

    @property
    def tag(self) -> str:
        return str(self.year) if self.is_annual else f"Q{self.quarter}{self.year % 100}"

    @property
    def date_span(self) -> str:
        """The ``{start}_{end}`` fragment used in detection filenames."""
        if self.is_annual:
            return f"{self.year}-01-01_{self.year}-12-31"
        start, end = QUARTER_SPANS[self.quarter]
        return f"{self.year}-{start}_{self.year}-{end}"

    @classmethod
    def parse(cls, tag: str) -> "Period":
        tag = str(tag).strip()
        m = QUARTER_TAG_RE.match(tag.upper())
        if m:
            quarter, yy = int(m.group(1)), int(m.group(2))
            return cls(year=2000 + yy, quarter=quarter)
        if re.fullmatch(r"\d{4}", tag):
            return cls(year=int(tag))
        raise ValueError(
            f"Unrecognised period {tag!r}; expected a year (2024) or a quarter tag (Q125)")


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
