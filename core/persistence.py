#!/usr/bin/env python3
"""
Temporal persistence: which detections have earned a place in the cumulative record.

A detection enters the confirmed record only when corroborated by later
observation. Real mine scars are permanent and recur; cloud artefacts and
mosaic noise appear once and vanish. Requiring ``k`` detections within a window
of ``n`` consecutive periods removes most transient error while leaving genuine
features essentially untouched, and replaces the ad-hoc threshold-tightening
that cumulative products previously relied on.

Two properties make this safe to re-run at any time:

* **Anchored forward.** The window runs ``[Y, Y+n-1]``, never backwards -- a
  mine cannot have existed before it started. Onset is the first period that
  passes *its own* window, so a real 2022 onset is never backdated to a
  spurious 2019 cloud artefact.
* **Stateless.** Onset is a pure function of the full period stack. Nothing is
  carried between runs, so reprocessing cannot drift.

The rule is configurable because the choice is not settled; see
docs/design/persistence-planning.md. The axes are the window length and whether
quarterly periods count as witnesses:

===========  ==========================  ==============================
             ``window=2`` (one year on)  ``window=3`` (two years on)
===========  ==========================  ==============================
annual only  A                           C  (the classic k=2 of n=3)
+ quarters   B                           D
===========  ==========================  ==============================

Periods too recent to resolve stay **provisional**, carrying the stricter
instantaneous threshold ``t_prov`` in place of temporal evidence they cannot
yet have. Provisional detections may be withdrawn; confirmed ones are not.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from .postprocess import PostprocessConfig, centroid_key
except ImportError:
    from postprocess import PostprocessConfig, centroid_key

LocKey = Tuple[float, float]

QUARTER_SPANS = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}
QUARTER_TAG_RE = re.compile(r"^Q([1-4])(\d{2})$")


# --------------------------------------------------------------------------
# periods
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


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class PersistenceConfig:
    """The confirmation rule, and how the provisional edge is represented."""

    #: Detections required within the window, counting the onset period itself.
    #: k=3 would drop sites in cloud-wrecked years and short-lived operations
    #: that partly heal; k=2 is the working value.
    k: int = 2

    #: Window length in *years*, including the onset year. window=3 with k=2 is
    #: the classic rule: seen in year Y and again within the following two.
    window: int = 3

    #: Whether quarterly periods inside the window count as corroborating
    #: witnesses. Quarterly cloud loss is seasonal rather than random, so
    #: quarters bias towards dry-season-visible ground; with k=2 a single clear
    #: look suffices, which blunts but does not remove that bias.
    use_quarterly_witnesses: bool = False

    #: Confirm using whatever periods exist, without waiting for the window to
    #: close. Safe: the witness set only grows towards a fixed endpoint, so
    #: confirmations accumulate monotonically and are never withdrawn -- but the
    #: answer keeps changing until the window closes.
    early_confirm: bool = False

    #: Also evaluate every shorter window and take the earliest onset. Shorter
    #: windows yield a strict subset of longer ones, so this only ever confirms
    #: sooner, never differently. Off by default.
    nested: bool = False

    #: Instantaneous threshold standing in for temporal evidence on periods too
    #: recent to resolve. Count-matched against eventual confirmation, 0.50-0.53
    #: fits 2018-2023 better than the 0.55 inherited from the old cumulative.
    t_prov: float = 0.55

    #: Join key precision for "same location, different period".
    centroid_decimals: int = PostprocessConfig.centroid_decimals

    def witness_periods(self, onset: Period, available: Sequence[Period]) -> List[Period]:
        """Periods that may corroborate an onset at ``onset``."""
        last_year = onset.year + self.window - 1
        return [p for p in available
                if onset.year <= p.year <= last_year
                and (p.is_annual or self.use_quarterly_witnesses)]

    def window_is_closed(self, onset: Period, available: Sequence[Period]) -> bool:
        """True once every annual period the window needs is in hand."""
        needed = {onset.year + d for d in range(self.window)}
        have = {p.year for p in available if p.is_annual}
        return needed <= have


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def detection_path(base: Path, period: Period, *, region_stem: str,
                   t_main: float, t_iso: float, k: int = 5,
                   isolation_km: float = 3.0) -> Path:
    """Path to a postprocessed single-period detection file."""
    tag = f"t{t_main:g}_d{k}_{isolation_km:g}km_t-iso{t_iso:g}"
    folder = base / f"postprocessed_{tag}"
    return folder / f"{region_stem}_{period.date_span}_{tag}.geojson"


def load_period(path: Path, decimals: int) -> gpd.GeoDataFrame:
    """Read one period's detections, keyed by rounded patch centroid."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    kx, ky = centroid_key(gdf, decimals=decimals)
    gdf = gdf.assign(_kx=kx, _ky=ky)
    return gdf


def period_keys(gdf: gpd.GeoDataFrame) -> Set[LocKey]:
    return set(zip(gdf["_kx"], gdf["_ky"]))


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

def compute_onsets(
    detected: Dict[Period, Set[LocKey]],
    config: PersistenceConfig,
) -> Dict[LocKey, Tuple[Period, Period, int]]:
    """Confirmed onsets.

    Returns ``{location: (onset_period, confirmed_in, n_witnesses)}``, where
    ``confirmed_in`` is the earliest period at which the count reached ``k`` --
    the observation that settled it.

    Only annual periods are candidate onsets. A quarter is corroboration, not a
    starting point: quarterly mosaics are too cloud-affected to date an onset
    against, and dating from them would make onset depend on which quarters
    happened to be clear.
    """
    available = sorted(detected)
    windows = range(2, config.window + 1) if config.nested else (config.window,)

    best: Dict[LocKey, Tuple[Period, Period, int]] = {}
    for span in windows:
        sub = PersistenceConfig(**{**config.__dict__, "window": span})
        for onset in [p for p in available if p.is_annual]:
            if not config.early_confirm and not sub.window_is_closed(onset, available):
                continue
            witnesses = sub.witness_periods(onset, available)
            for loc in detected[onset]:
                seen = [p for p in witnesses if loc in detected[p]]
                if len(seen) < config.k:
                    continue
                settled = sorted(seen)[config.k - 1]
                prior = best.get(loc)
                if prior is None or onset.sort_key < prior[0].sort_key:
                    best[loc] = (onset, settled, len(seen))
    return best


def resolvable_periods(available: Sequence[Period],
                       config: PersistenceConfig) -> List[Period]:
    """Annual periods whose confirmation window can be evaluated."""
    return [p for p in available
            if p.is_annual
            and (config.early_confirm or config.window_is_closed(p, available))]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_first_detection_layer(
    frames: Dict[Period, gpd.GeoDataFrame],
    provisional: Dict[Period, gpd.GeoDataFrame],
    config: PersistenceConfig,
) -> gpd.GeoDataFrame:
    """One row per location: when mining was confirmed to have begun there.

    Every cumulative figure derives from this layer by filtering on ``onset``,
    which is why the per-period cumulatives are redundant with it. Confirmed
    rows never move; provisional rows may be withdrawn or promoted.
    """
    detected = {p: period_keys(g) for p, g in frames.items()}
    onsets = compute_onsets(detected, config)

    rows, seen = [], set()
    for period in sorted(frames):
        if period not in {o[0] for o in onsets.values()}:
            continue
        gdf = frames[period]
        keys = list(zip(gdf["_kx"], gdf["_ky"]))
        take = [i for i, key in enumerate(keys)
                if key not in seen and onsets.get(key, (None,))[0] == period]
        if not take:
            continue
        sub = gdf.iloc[take].copy()
        meta = [onsets[k] for k in (keys[i] for i in take)]
        sub["onset"] = period.tag
        sub["onset_year"] = period.year
        sub["status"] = "confirmed"
        sub["confirmed_in"] = [m[1].tag for m in meta]
        sub["n_witness"] = [m[2] for m in meta]
        rows.append(sub)
        seen.update(keys[i] for i in take)

    for period in sorted(provisional):
        gdf = provisional[period]
        keys = list(zip(gdf["_kx"], gdf["_ky"]))
        take = [i for i, key in enumerate(keys) if key not in seen]
        if not take:
            continue
        sub = gdf.iloc[take].copy()
        sub["onset"] = period.tag
        sub["onset_year"] = period.year
        sub["status"] = "provisional"
        sub["confirmed_in"] = None
        sub["n_witness"] = 1
        rows.append(sub)
        seen.update(keys[i] for i in take)

    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    out = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True),
                           crs=rows[0].crs)
    return out.drop(columns=["_kx", "_ky"])


def cumulative_through(layer: gpd.GeoDataFrame, period: Period) -> gpd.GeoDataFrame:
    """Everything with an onset at or before ``period``."""
    order = {t: Period.parse(t).sort_key for t in layer["onset"].unique()}
    keep = layer["onset"].map(order) <= period.sort_key
    return layer.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    config = PersistenceConfig(
        k=args.k, window=args.window,
        use_quarterly_witnesses=(args.witnesses == "all"),
        early_confirm=args.early_confirm, nested=args.nested,
        t_prov=args.t_prov)

    base = Path(args.base)
    periods = [Period.parse(t) for t in list(args.years) + list(args.quarters)]
    periods.sort()

    frames, provisional = {}, {}
    resolvable = set(resolvable_periods(periods, config))
    for p in periods:
        path = detection_path(base, p, region_stem=args.region_stem,
                              t_main=args.t_main, t_iso=args.t_iso)
        if not path.is_file():
            raise SystemExit(f"missing detections for {p.tag}: {path}")
        frames[p] = load_period(path, config.centroid_decimals)
        if p not in resolvable:
            prov_path = detection_path(base, p, region_stem=args.region_stem,
                                       t_main=config.t_prov, t_iso=args.t_prov_iso)
            if prov_path.is_file():
                provisional[p] = load_period(prov_path, config.centroid_decimals)
            else:
                print(f"  ! no t_prov file for {p.tag}, skipping provisional layer")

    print(f"{len(frames)} periods, {len(resolvable)} resolvable under "
          f"k={config.k} window={config.window} witnesses={args.witnesses}")

    layer = build_first_detection_layer(frames, provisional, config)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "detections_first_year.geojson"
    layer.to_file(dest, driver="GeoJSON", index=False,
                  COORDINATE_PRECISION=PostprocessConfig.coordinate_precision)

    counts = layer.groupby(["onset", "status"]).size().reset_index(name="n")
    counts["_o"] = counts["onset"].map(lambda t: Period.parse(t).sort_key)
    for _, r in counts.sort_values("_o").iterrows():
        print(f"  {r['onset']:>6}  {r['status']:<12} {r['n']:>8,}")
    confirmed = int((layer["status"] == "confirmed").sum())
    print(f"\n  {confirmed:,} confirmed + {len(layer) - confirmed:,} provisional "
          f"= {len(layer):,} locations -> {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = PersistenceConfig()
    parser.add_argument("--base", default="../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble",
                        help="Model output directory holding postprocessed_*/ folders")
    parser.add_argument("--region_stem",
                        default="Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40")
    parser.add_argument("--years", nargs="*", default=[],
                        help="Annual periods, e.g. --years 2018 2019 2020")
    parser.add_argument("--quarters", nargs="*", default=[],
                        help="Quarterly periods, e.g. --quarters Q125 Q225")
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--window", type=int, default=defaults.window,
                        help="Window length in years, including the onset year")
    parser.add_argument("--witnesses", choices=("annual", "all"), default="annual",
                        help="'all' lets quarterly periods corroborate")
    parser.add_argument("--early-confirm", action="store_true",
                        help="Confirm without waiting for the window to close")
    parser.add_argument("--nested", action="store_true",
                        help="Also evaluate shorter windows; take the earliest onset")
    parser.add_argument("--t-main", dest="t_main", type=float,
                        default=PostprocessConfig.t_main)
    parser.add_argument("--t-iso", dest="t_iso", type=float,
                        default=PostprocessConfig.t_iso)
    parser.add_argument("--t-prov", dest="t_prov", type=float, default=defaults.t_prov)
    parser.add_argument("--t-prov-iso", dest="t_prov_iso", type=float, default=0.8)
    parser.add_argument("--outdir", default="../data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble/persistence")
    main(parser.parse_args())
