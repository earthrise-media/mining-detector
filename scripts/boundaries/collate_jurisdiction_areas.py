# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.2",
# ]
# ///
"""
Collate the per-jurisdiction yearly mining summaries into CSVs.

Pulls every `*_yearly.json` jurisdiction timeseries from the AMW media CDN,
joins the identity metadata (`country`, `name`, `bbox`, ...) from the matching
`*_impacts_unfiltered_dict.json` files, and writes two flat CSVs:

  mined_areas_by_jurisdiction.csv          one row per jurisdiction per year
  illegality_analysis_by_jurisdiction.csv  one row per jurisdiction (for ILLEGALITY_DATA_UPDATED_AT)

The second file is separate because `illegality_areas` describes the latest period only.

The publish folder (`DATA_DATE`) is resolved automatically: the CDN bucket
does not allow listing, so we HEAD one sentinel file per candidate date,
walking back from today to `DATA_UPDATED_AT` in constants.py (which is the
oldest folder we know to be published, since that is what the upload scripts
push to). The newest folder that answers wins. Pass --data-date to pin it.

Usage:
    uv run scripts/boundaries/collate_jurisdiction_areas.py
    uv run scripts/boundaries/collate_jurisdiction_areas.py --data-date 20260724
    uv run scripts/boundaries/collate_jurisdiction_areas.py --mining-out /tmp/mining.csv --illegality-out /tmp/illegality.csv
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
from constants import DATA_UPDATED_AT, ENTIRE_AMAZON_ID, ILLEGALITY_DATA_UPDATED_AT

BASE = "https://media-amw.earthgenome.org"
REPO_ROOT = Path(__file__).resolve().parents[2]

# data/public is served straight out of the repo — the AMW website links at
# this path — so neither the folder nor the filenames may change, and the names
# carry no date. Versioning is the repo's job. See data/public/README.md.
MINING_OUT_PATH = REPO_ROOT / "data" / "public" / "mined_areas_by_jurisdiction.csv"
ILLEGALITY_OUT_PATH = (
    REPO_ROOT / "data" / "public" / "illegality_analysis_by_jurisdiction.csv"
)

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

# The illegality bands of `illegality_areas`, keyed by `admin_illegality_max`
# and ordered as a reader expects to meet them (worst first).
ILLEGALITY_LEVELS = [
    (4, "very_high"),
    (3, "high"),
    (2, "medium"),
    (1, "low"),
]

ILLEGALITY_COLUMNS = [
    f"illegality_{label}_affected_area_{suffix}"
    for _, label in ILLEGALITY_LEVELS
    for suffix in ("ha", "pct")
]

# The part of the cumulative mined area that no illegality band accounts for,
# i.e. the mined area we have no illegality measurement for. Derived in
# illegality_table(), so it is not part of ILLEGALITY_COLUMNS.
ILLEGALITY_NA_COLUMN = "illegality_na_affected_area_ha"

# Bands can sum to slightly more than the cumulative total through float noise
# alone; only an overshoot beyond this is worth a warning.
ILLEGALITY_NA_TOLERANCE_HA = 0.01

MINING_COLUMNS = [
    # Leads the row so a copy that has drifted away from this repo still says
    # which publish it came from. Reprocessing has restated past years before
    # (2023 moved 45% between the 2026-07-24 and 2026-08-22 publishes), and a
    # detached CSV otherwise gives a reader no way to tell which vintage it is.
    "date_published",
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

ILLEGALITY_TABLE_COLUMNS = [
    "date_published",
    "id",
    "type",
    "country",
    "country_code",
    "name",
    "status",
    "admin_year",
    "intersected_area_ha_cumulative",
    *ILLEGALITY_COLUMNS,
    ILLEGALITY_NA_COLUMN,
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
]

AREA_COLUMNS = ["intersected_area_ha", "intersected_area_ha_cumulative"]

# Rounded with --decimals like the other hectare columns. The matching `_pct`
# columns are left alone: the default of 2 would flatten a share like 0.047.
ILLEGALITY_AREA_COLUMNS = [
    f"illegality_{label}_affected_area_ha" for _, label in ILLEGALITY_LEVELS
]

# Row order, broadest first: the basin-wide roll-up, then countries, then the
# finer jurisdictions in the order declared above. A reader scrolling from the
# top meets the headline numbers before the 3,000-row long tail.
TYPE_ORDER = [spec["type"] for spec in JURISDICTIONS]


def as_folder(data_date: str) -> str:
    """Accept either 20260822 or 2026-08-22; CDN folders use the former.

    The CSV publishes the dashed form, so a reader can paste the date they see
    in the file straight back into --data-date.
    """
    folder = data_date.replace("-", "")
    if not (len(folder) == 8 and folder.isdigit()):
        sys.exit(f"--data-date should be YYYYMMDD or YYYY-MM-DD, got {data_date!r}")
    return folder


def as_published(data_date: str) -> str:
    """20260822 -> 2026-08-22, for the human reading the CSV."""
    return f"{data_date[:4]}-{data_date[4:6]}-{data_date[6:8]}"


def resolve_data_date(explicit: str | None) -> str:
    """Return the newest published CDN folder name (YYYYMMDD)."""
    if explicit:
        folder = as_folder(explicit)
        if not exists(folder):
            sys.exit(f"No data at {BASE}/{folder}/{SENTINEL}")
        return folder

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


def illegality_fields(record: dict) -> dict:
    """Flatten `illegality_areas` into an area and a share per band.

    A jurisdiction with no `illegality_areas` at all is left blank instead, 
    since that is an absence of illegality data rather than an absence of mining.
    """
    areas = record.get("illegality_areas") or []
    by_level = {}
    for entry in areas:
        level = entry.get("admin_illegality_max")
        if level is None:
            continue
        # Bands outside 1-4 would be a schema change upstream; ignore them
        # rather than inventing a column for them here.
        by_level[int(level)] = entry

    fields = {}
    for level, label in ILLEGALITY_LEVELS:
        entry = by_level.get(level)
        absent = 0.0 if areas else None
        fields[f"illegality_{label}_affected_area_ha"] = (
            entry.get("mining_affected_area") if entry else absent
        )
        fields[f"illegality_{label}_affected_area_pct"] = (
            entry.get("mining_affected_area_pct") if entry else absent
        )
    return fields


def meta_frame(records: list[dict], jurisdiction_type: str) -> pd.DataFrame:
    """Keep the identity fields and the illegality split; drop the calculator nests."""
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
                **illegality_fields(r),
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

    df = pd.concat(frames, ignore_index=True).assign(
        date_published=as_published(data_date)
    )
    # A band that is blank everywhere in a file leaves an object column, which
    # rounds badly and writes "None" into the CSV where a reader wants a gap.
    df[ILLEGALITY_COLUMNS] = df[ILLEGALITY_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.assign(
        _type=pd.Categorical(df["type"], categories=TYPE_ORDER, ordered=True),
        _amazon=df["id"].ne(ENTIRE_AMAZON_ID),  # False sorts first
    )
    return (
        df.sort_values(
            ["_type", "_amazon", "country", "name", "admin_year"], na_position="last"
        )
        .drop(columns=["_type", "_amazon"])
        .reset_index(drop=True)
    )


def mining_areas_table(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df[MINING_COLUMNS].copy()
    out[AREA_COLUMNS] = out[AREA_COLUMNS].round(decimals)
    return out


def illegality_table(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """One row per jurisdiction for ILLEGALITY_DATA_UPDATED_AT.

    Carries the cumulative mined area at that year, its split across the
    illegality bands, and `illegality_na_affected_area_ha`: the remainder with no
    illegality measurement. A jurisdiction with no split at all has its whole
    cumulative area in the remainder.
    """
    current = df[df["admin_year"].eq(ILLEGALITY_DATA_UPDATED_AT)]
    if dropped := df["id"].nunique() - current["id"].nunique():
        print(
            f"  warning: {dropped} jurisdictions have no data for "
            f"{ILLEGALITY_DATA_UPDATED_AT} and are omitted"
        )

    if dupes := int(current.duplicated(["type", "id"]).sum()):
        print(
            f"  warning: {dupes} jurisdictions have more than one row for "
            f"{ILLEGALITY_DATA_UPDATED_AT}"
        )

    # A boolean mask keeps the row order set in collate().
    out = current.copy()

    # Round before deriving the remainder, so the hectare columns of each row
    # add up exactly to the cumulative total as written in the CSV.
    hectares = ["intersected_area_ha_cumulative", *ILLEGALITY_AREA_COLUMNS]
    out[hectares] = out[hectares].round(decimals)

    # sum() skips blank bands, so a row with no split sums to 0 and its whole
    # cumulative area counts as unmeasured.
    remainder = out["intersected_area_ha_cumulative"] - out[
        ILLEGALITY_AREA_COLUMNS
    ].sum(axis=1)
    if overshoot := int(remainder.lt(-ILLEGALITY_NA_TOLERANCE_HA).sum()):
        print(
            f"  warning: {overshoot} jurisdictions have illegality bands summing "
            f"to more than their cumulative mined area; {ILLEGALITY_NA_COLUMN} "
            f"is negative there"
        )
    # Re-round to clear float noise; adding 0.0 turns -0.0 into 0.0 so the CSV
    # doesn't show "-0.0".
    out[ILLEGALITY_NA_COLUMN] = remainder.round(decimals) + 0.0

    return out.loc[:, ILLEGALITY_TABLE_COLUMNS]


def write(df: pd.DataFrame, out: Path, label: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {label}: {out} ({out.stat().st_size:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-date",
        help="CDN publish folder, YYYYMMDD or YYYY-MM-DD "
        "(default: newest one found on the CDN)",
    )
    ap.add_argument("--mining-out", type=Path, help="output CSV path for the yearly areas")
    ap.add_argument(
        "--illegality-out",
        type=Path,
        help="output CSV path for the illegality analysis",
    )
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

    mining_areas = mining_areas_table(df, args.decimals)
    illegality = illegality_table(df, args.decimals)

    print(
        f"\ndata date {data_date} | {len(mining_areas):,} rows | "
        f"{mining_areas['id'].nunique():,} jurisdictions | "
        f"types={sorted(mining_areas['type'].dropna().unique())}"
    )
    print(f"rows with no bbox: {int(mining_areas['bbox_minx'].isna().sum())}")
    print(
        f"illegality: {len(illegality):,} rows | "
        f"years {illegality['admin_year'].min()}-{illegality['admin_year'].max()} | "
        f"no split: {int(illegality['illegality_very_high_affected_area_ha'].isna().sum())}"
    )

    write(mining_areas, args.mining_out or MINING_OUT_PATH, "mining areas")
    write(illegality, args.illegality_out or ILLEGALITY_OUT_PATH, "illegality")
    return 0


if __name__ == "__main__":
    sys.exit(main())
