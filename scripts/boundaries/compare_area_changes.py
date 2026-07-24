# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx>=0.27",
#     "pandas>=2.2",
# ]
# ///
"""
Compare earthgenome AMW boundary JSON files between two date folders.

Two shapes are handled:

  *_impacts_unfiltered_dict.json
      Array of records keyed by `id`, with a scalar total area column
      (e.g. `mining_affected_area_ha`) plus nested `locations` /
      `illegality_areas`.  We compare the top-level scalar area column(s)
      per `id`.

  *_yearly.json
      Array of {id, admin_year, intersected_area_ha,
      intersected_area_ha_cumulative}.  We compare the area column(s)
      per (id, admin_year).

Area columns are auto-detected (fields ending in `_ha`, or named
`affectedArea` / `mining_affected_area*`) so the comparison still works if
columns were renamed or added between the two dates.

Usage:
    uv run scripts/boundaries/compare_area_changes.py
    uv run scripts/boundaries/compare_area_changes.py --new-date 20260722 --old-date 20260331 --tol 0.5 --out-dir ./scripts/boundaries/diffs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

BASE = "https://media-amw.earthgenome.org"

# Paths (relative to the date folder) of every file to compare.
REL_PATHS = [
    "data/boundaries/national_admin/out/national_admin_impacts_unfiltered_dict.json",
    "data/boundaries/national_admin/out/national_admin_yearly.json",
    "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered_dict.json",
    "data/boundaries/subnational_admin/out/admin_areas_display_yearly.json",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered_dict.json",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_yearly.json",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered_dict.json",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_yearly.json",
]

# Field names we treat as "area columns" even if they don't end in _ha.
AREA_NAME_HINTS = ("affectedarea", "mining_affected_area")

# Descriptive (non-numeric) fields to carry into the diff CSVs when present.
# These label each row (id) so the diffs are readable without a lookup.
DESC_FIELDS = ("country", "name_field", "status_field")


def url_for(date: str, rel: str) -> str:
    return f"{BASE}/{date}/{rel}"


def fetch_json(client: httpx.Client, url: str):
    r = client.get(url, timeout=60.0)
    r.raise_for_status()
    return r.json()


def is_area_col(col: str) -> bool:
    c = col.lower()
    return c.endswith("_ha") or any(h in c for h in AREA_NAME_HINTS)


def to_frame(records) -> pd.DataFrame:
    """Flatten the top level of a list-of-dicts into a DataFrame.

    Nested list columns (locations, illegality_areas) and other non-scalar
    values are dropped for the scalar comparison; they can be huge and vary
    row-to-row, so we compare the top-level area totals instead.
    """
    if not isinstance(records, list):
        raise ValueError("Expected a top-level JSON array")
    df = pd.json_normalize(records, max_level=0)
    # Drop columns whose values are lists/dicts (nested structures).
    scalar_cols = [
        c for c in df.columns
        if not df[c].map(lambda v: isinstance(v, (list, dict))).any()
    ]
    return df[scalar_cols]


def key_cols_for(df: pd.DataFrame) -> list[str]:
    """Pick the columns that identify a row."""
    candidates = [c for c in ("id", "gid_0", "admin_year") if c in df.columns]
    # Prefer id (+ admin_year for yearly files).
    keys = []
    if "id" in candidates:
        keys.append("id")
    elif "gid_0" in candidates:
        keys.append("gid_0")
    if "admin_year" in candidates:
        keys.append("admin_year")
    if not keys:
        # Fall back to the first column.
        keys = [df.columns[0]]
    return keys


def compare_file(rel: str, old_records, new_records, tol: float):
    """Return (schema_report_str, diff_dataframe_or_None)."""
    old_df = to_frame(old_records)
    new_df = to_frame(new_records)

    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)

    old_area = {c for c in old_df.columns if is_area_col(c)}
    new_area = {c for c in new_df.columns if is_area_col(c)}

    lines = []
    lines.append(f"  rows:         old={len(old_df)}  new={len(new_df)}")
    lines.append(f"  columns old:  {sorted(old_cols)}")
    lines.append(f"  columns new:  {sorted(new_cols)}")

    added = new_cols - old_cols
    removed = old_cols - new_cols
    if added:
        lines.append(f"  + added cols:   {sorted(added)}")
    if removed:
        lines.append(f"  - removed cols: {sorted(removed)}")

    lines.append(f"  area cols old: {sorted(old_area)}")
    lines.append(f"  area cols new: {sorted(new_area)}")
    if old_area != new_area:
        lines.append("  ** AREA COLUMNS CHANGED between the two dates **")

    # Build the key. Use whichever key columns both frames share.
    old_keys = key_cols_for(old_df)
    new_keys = key_cols_for(new_df)
    keys = [k for k in old_keys if k in new_keys] or old_keys
    lines.append(f"  join keys:    {keys}")

    # Area columns present in BOTH -> direct numeric diff on shared name.
    shared_area = sorted(old_area & new_area)
    # Area columns unique to each side (renamed / new) -> reported side-by-side.
    old_only_area = sorted(old_area - new_area)
    new_only_area = sorted(new_area - old_area)

    # Descriptive label fields (country / name_field / type_field) present on
    # each side but NOT already a join key. Carried through so the CSV rows are
    # self-describing; suffixed __old/__new since they can differ across dates.
    old_desc = [c for c in DESC_FIELDS if c in old_df.columns and c not in keys]
    new_desc = [c for c in DESC_FIELDS if c in new_df.columns and c not in keys]
    if old_desc or new_desc:
        lines.append(f"  desc fields:  old={old_desc}  new={new_desc}")

    old_sub = old_df[keys + old_desc + sorted(old_area)].copy()
    new_sub = new_df[keys + new_desc + sorted(new_area)].copy()

    merged = old_sub.merge(
        new_sub, on=keys, how="outer",
        suffixes=("__old", "__new"), indicator=True,
    )

    # For any descriptive field that exists on BOTH sides, collapse to one
    # column when old == new (compared only over rows present on both sides),
    # else keep both to surface relabelings. For outer-only rows, coalesce so
    # the single column still carries whichever side has a value.
    both_rows = merged["_merge"] == "both"
    for f in sorted(set(old_desc) & set(new_desc)):
        o, n = merged[f"{f}__old"], merged[f"{f}__new"]
        same_on_both = o[both_rows].fillna("").eq(n[both_rows].fillna("")).all()
        if same_on_both:
            merged[f] = o.combine_first(n)
            merged.drop(columns=[f"{f}__old", f"{f}__new"], inplace=True)

    # Compute per-shared-column deltas.
    for col in shared_area:
        o = pd.to_numeric(merged.get(f"{col}__old"), errors="coerce")
        n = pd.to_numeric(merged.get(f"{col}__new"), errors="coerce")
        merged[f"{col}__delta"] = n - o
        merged[f"{col}__pct"] = (n - o) / o.replace(0, pd.NA) * 100

    # Keep rows that are new/removed OR that changed beyond tolerance on any
    # shared area column. A row counts as changed if EITHER:
    #   - the numeric delta exceeds tol, OR
    #   - the value appeared/disappeared (null -> number or number -> null),
    #     which yields a NaN delta and would otherwise be missed.
    changed_mask = merged["_merge"] != "both"
    for col in shared_area:
        o = pd.to_numeric(merged.get(f"{col}__old"), errors="coerce")
        n = pd.to_numeric(merged.get(f"{col}__new"), errors="coerce")
        delta = merged[f"{col}__delta"]
        beyond_tol = delta.abs() > tol
        appeared_or_dropped = o.isna() ^ n.isna()  # exactly one side null
        changed_mask = changed_mask | beyond_tol.fillna(False) | appeared_or_dropped

    diff = merged[changed_mask].copy()

    # Column order: keys, descriptive labels, then everything else.
    desc_out = [
        c for c in diff.columns
        if any(c == f or c.startswith(f"{f}__") for f in DESC_FIELDS)
    ]
    lead = keys + desc_out
    diff = diff[lead + [c for c in diff.columns if c not in lead]]

    lines.append(
        f"  changed/added/removed rows (tol={tol}): {len(diff)} of {len(merged)}"
    )
    if old_only_area or new_only_area:
        lines.append(
            f"  note: unmatched area cols kept for manual review — "
            f"old_only={old_only_area}  new_only={new_only_area}"
        )

    return "\n".join(lines), diff if len(diff) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-date", default="20260720")
    ap.add_argument("--old-date", default="20260331")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="abs area delta below this is treated as unchanged")
    ap.add_argument("--out-dir", default="scripts/boundaries/diffs",
                    help="directory to write per-file diff CSVs into")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing NEW={args.new_date}  vs  OLD={args.old_date}\n")

    with httpx.Client(follow_redirects=True) as client:
        for rel in REL_PATHS:
            name = rel.split("/")[-1]
            print(f"=== {name} ===")
            try:
                new_records = fetch_json(client, url_for(args.new_date, rel))
            except Exception as e:  # noqa: BLE001
                print(f"  !! failed to fetch NEW: {e}\n")
                continue
            try:
                old_records = fetch_json(client, url_for(args.old_date, rel))
            except Exception as e:  # noqa: BLE001
                print(f"  !! failed to fetch OLD: {e}\n")
                continue

            try:
                report, diff = compare_file(rel, old_records, new_records, args.tol)
            except Exception as e:  # noqa: BLE001
                print(f"  !! comparison error: {e}\n")
                continue

            print(report)
            if diff is not None:
                csv_path = out_dir / (name.replace(".json", "") + "__diff.csv")
                diff.to_csv(csv_path, index=False)
                print(f"  -> wrote {len(diff)} diff rows to {csv_path}")
            else:
                print("  -> no differences beyond tolerance")
            print()

    print(f"Done. Diff CSVs (if any) are in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
