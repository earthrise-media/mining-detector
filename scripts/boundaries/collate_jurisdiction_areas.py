# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.2",
# ]
# ///
"""
Collate the per-jurisdiction yearly mining summaries into a single CSV.

Pulls every `*_yearly.json` jurisdiction timeseries from the AMW media CDN,
joins the identity metadata (`country`, `name`, `bbox`, ...) from the matching
`*_impacts_unfiltered_dict.json` files, and writes one flat CSV meant to be
opened in a spreadsheet.

The publish folder (`DATA_DATE`) is resolved automatically: the CDN bucket
does not allow listing, so we HEAD one sentinel file per candidate date,
walking back from today to `DATA_UPDATED_AT` in constants.py (which is the
oldest folder we know to be published, since that is what the upload scripts
push to). The newest folder that answers wins. Pass --data-date to pin it.

Usage:
    uv run scripts/boundaries/collate_jurisdiction_areas.py
    uv run scripts/boundaries/collate_jurisdiction_areas.py --data-date 20260724
    uv run scripts/boundaries/collate_jurisdiction_areas.py --out /tmp/areas.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from constants import DATA_UPDATED_AT

BASE = "https://media-amw.earthgenome.org"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately undated: the AMW website links straight at this path, so the
# name has to survive a data update. Versioning is the repo's job.
OUT_NAME = "mined_areas_by_jurisdiction.csv"

# Probed once per candidate date to decide whether that folder was published.
SENTINEL = "data/boundaries/national_admin/out/national_admin_yearly.json"

# Backstop on the date probe, so a long-stale DATA_UPDATED_AT can't turn into
# thousands of HEAD requests.
MAX_PROBE_DAYS = 400

JURISDICTIONS = [
    {
        "type": "national_admin",
        "yearly": "data/boundaries/national_admin/out/national_admin_yearly.json",
        "meta": "data/boundaries/national_admin/out/national_admin_impacts_unfiltered_dict.json",
    },
    {
        "type": "subnational_admin",
        "yearly": "data/boundaries/subnational_admin/out/admin_areas_display_yearly.json",
        "meta": "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered_dict.json",
    },
    {
        "type": "indigenous_territories",
        "yearly": "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_yearly.json",
        "meta": "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered_dict.json",
    },
    {
        "type": "protected_areas",
        "yearly": "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_yearly.json",
        "meta": "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered_dict.json",
    },
]

COLUMNS = [
    "id",
    "type",
    "country",
    "country_code",
    "name",
    "status",
    "admin_year",
    "intersected_area_ha",
    "intersected_area_ha_cumulative",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
]

AREA_COLUMNS = ["intersected_area_ha", "intersected_area_ha_cumulative"]

# Row order, broadest first: the basin-wide roll-up, then countries, then the
# finer jurisdictions in the order declared above. A reader scrolling from the
# top meets the headline numbers before the 3,000-row long tail.
TYPE_ORDER = [spec["type"] for spec in JURISDICTIONS]
AMAZON_ID = "AMAZ"  # the whole-basin row, sorted ahead of the countries


def resolve_data_date(explicit: str | None) -> str:
    """Return the newest published CDN folder name (YYYYMMDD)."""
    if explicit:
        if not exists(explicit):
            sys.exit(f"No data at {BASE}/{explicit}/{SENTINEL}")
        return explicit

    floor = date(
        int(DATA_UPDATED_AT[:4]), int(DATA_UPDATED_AT[4:6]), int(DATA_UPDATED_AT[6:8])
    )
    today = date.today()
    oldest = max(floor, today - timedelta(days=MAX_PROBE_DAYS))
    if oldest > floor:
        print(
            f"warning: only probing back {MAX_PROBE_DAYS} days; "
            f"constants.DATA_UPDATED_AT={DATA_UPDATED_AT} is older than that"
        )

    print(f"Probing {BASE} for the newest publish folder (back to {oldest:%Y%m%d})...")
    day = today
    while day >= oldest:
        stamp = f"{day:%Y%m%d}"
        if exists(stamp):
            found = (today - day).days
            print(f"  found {stamp} ({found} day(s) back, {found + 1} probes)")
            return stamp
        day -= timedelta(days=1)

    sys.exit(
        f"No publish folder found between {oldest:%Y%m%d} and {today:%Y%m%d}. "
        f"Pass --data-date to pin one."
    )


def exists(data_date: str) -> bool:
    req = Request(f"{BASE}/{data_date}/{SENTINEL}", method="HEAD")
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.status < 400
    except (URLError, OSError):
        return False


def fetch_json(data_date: str, rel: str):
    url = f"{BASE}/{data_date}/{rel}"
    print(f"GET {url}")
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def meta_frame(records: list[dict], jurisdiction_type: str) -> pd.DataFrame:
    """Keep the identity fields; drop the calculator/illegality nests."""
    rows = []
    for r in records:
        bbox = r.get("bbox") or [None] * 4  # [minx, miny, maxx, maxy]
        rows.append(
            {
                "id": r["id"],
                "type": jurisdiction_type,
                "country": r.get("country"),
                "country_code": r.get("country_code"),
                # national_admin has no name_field; its display name is the country
                "name": r.get("name_field") or r.get("country"),
                "status": r.get("status_field"),
                "bbox_minx": bbox[0],
                "bbox_miny": bbox[1],
                "bbox_maxx": bbox[2],
                "bbox_maxy": bbox[3],
            }
        )
    return pd.DataFrame(rows)


def collate(data_date: str) -> pd.DataFrame:
    frames = []
    for spec in JURISDICTIONS:
        yearly = pd.DataFrame(fetch_json(data_date, spec["yearly"]))
        meta = meta_frame(fetch_json(data_date, spec["meta"]), spec["type"])
        merged = yearly.merge(meta, on="id", how="left", validate="many_to_one")
        if missing := int(merged["type"].isna().sum()):
            print(f"  warning: {missing} yearly rows with no metadata")
        merged["type"] = merged["type"].fillna(spec["type"])
        print(f"  {spec['type']}: {len(yearly)} yearly rows, {len(meta)} jurisdictions")
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    df = df.assign(
        _type=pd.Categorical(df["type"], categories=TYPE_ORDER, ordered=True),
        _amazon=df["id"].ne(AMAZON_ID),  # False sorts first
    )
    return df.sort_values(
        ["_type", "_amazon", "country", "name", "admin_year"], na_position="last"
    ).drop(columns=["_type", "_amazon"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-date",
        help="CDN publish folder, YYYYMMDD (default: newest one found on the CDN)",
    )
    ap.add_argument("--out", type=Path, help="output CSV path")
    ap.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="round hectare columns to this many decimals (default: 2; "
        "0.01 ha is far below one pixel). Use a big number to keep full precision.",
    )
    args = ap.parse_args()

    data_date = resolve_data_date(args.data_date)
    df = collate(data_date)

    df[AREA_COLUMNS] = df[AREA_COLUMNS].round(args.decimals)

    out = args.out or (REPO_ROOT / "data" / "boundaries" / OUT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(
        f"\ndata date {data_date} | {len(df):,} rows | "
        f"{df['id'].nunique():,} jurisdictions | "
        f"types={sorted(df['type'].dropna().unique())}"
    )
    print(f"rows with no bbox: {int(df['bbox_minx'].isna().sum())}")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
