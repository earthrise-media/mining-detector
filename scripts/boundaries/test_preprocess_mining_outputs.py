"""
Data-quality tests for the outputs of preprocess_mining_areas.py.

Run them after preprocess_mining_areas.py and before uploading anything:
- `_yearly.json`: one row per area and period, with cumulative and per-period area
- `_impacts_unfiltered_dict.json`: one record per area, with totals, the illegality
  breakdown and the Mining Calculator locations for every period
- `_impacts_unfiltered.geojson` / `_impacts.geojson`: the same records with geometry
- `_impacts_yearly.csv`: the yearly rows joined to the area attributes
- `mining_<period>_simplified.geojson` / `mining_combined_full.geojson`

Most checks tie the files to each other rather than only checking each file on its
own: the yearly timeseries, the totals, the illegality breakdown and the locations all
come from the same per-period snapshots, so they must agree up to rounding.

Test groups, in the order they appear:
- presence and freshness: every output exists and was written after its inputs
- `_yearly.json`: schema, one row per area and period, increments vs cumulative values,
  cumulative values not shrinking, areas vs the Entire Amazon and country totals
- `_impacts_unfiltered_dict.json`: ids, schema, totals vs the timeseries, the illegality
  breakdown and the Mining Calculator locations
- `.geojson`: top-level ids, same values as the dict, filtered = areas with mining
- `_impacts_yearly.csv`: same rows and values as the yearly json
- geometry sanity: mined area vs polygon area, bbox vs geometry
- mining simplified files: one non-empty file per period, combined file covers them all
- regression against the published data (opt-in, see below)

Strictness: structural problems (duplicate ids, missing periods, bad schema) fail.
Numeric equalities allow for the script rounding each group to 2 decimals before
summing. Small cumulative dips are reported as warnings and only fail past
DIP_FAIL_HA / DIP_FAIL_FRAC.

Reporting: the console shows a short message per failure and warning, with a few rows
of the areas involved (id, name, type, country). Every finding, with the test's
description and the complete tables, goes to a Markdown report, by default
`logs/preprocess_outputs_report.md` (AMW_REPORT_PATH to change it). The console hooks
live in conftest.py next to this file.

Missing output files make the tests that need them skip; `test_outputs_exist` is the
one that fails for them.

Optional comparison against the previously published data on the CDN:
    AMW_COMPARE_PUBLISHED=latest      newest publish folder older than DATA_UPDATED_AT
    AMW_COMPARE_PUBLISHED=20261005    a specific folder
    AMW_PUBLISHED_BASE=<url>          somewhere other than the production CDN
The folder used, and its newest period, are named in the console and the report.

To check outputs somewhere other than this repo checkout, set AMW_OUTPUTS_ROOT to a
folder with the same `data/...` layout.
"""

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/test_preprocess_mining_outputs.py`
# (extra arguments are passed on to pytest, e.g. `-k yearly -x`).

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
#     "numpy",
#     "pandas",
#     "pytest",
#     "shapely",
#     "tabulate",
# ]
# ///

import functools
import inspect
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import shape

# scripts/boundaries is a package, so pytest doesn't put it on sys.path by itself,
# and the pipeline scripts import constants as a top-level module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import (  # noqa: E402
    COMBINED_MINING_FILE,
    DATA_UPDATED_AT,
    ENTIRE_AMAZON_ID,
    ILLEGALITY_DATA_UPDATED_AT,
    MINING_SIMPLIFIED_FILES,
    MINING_YEARS_QUARTERS,
)

ROOT = Path(os.environ.get("AMW_OUTPUTS_ROOT") or Path(__file__).resolve().parents[2])
REPORT_PATH = Path(
    os.environ.get("AMW_REPORT_PATH") or ROOT / "logs" / "preprocess_outputs_report.md"
)


# Mirrors datasets_to_process in preprocess_mining_areas.py. Not imported from there,
# because that module pulls in rasterio and the whole processing stack.
@dataclass(frozen=True)
class Dataset:
    name: str
    file: str
    has_locations: bool
    # how areas of this dataset are described in reports
    label: str

    def output(self, suffix):
        return self.file.replace(".geojson", suffix)

    @property
    def yearly(self):
        return self.output("_yearly.json")

    @property
    def dict_json(self):
        return self.output("_impacts_unfiltered_dict.json")

    @property
    def unfiltered_geojson(self):
        return self.output("_impacts_unfiltered.geojson")

    @property
    def filtered_geojson(self):
        return self.output("_impacts.geojson")

    @property
    def yearly_csv(self):
        return self.output("_impacts_yearly.csv")


NATIONAL = Dataset(
    "national_admin",
    "data/boundaries/national_admin/out/national_admin.geojson",
    has_locations=False,
    label="National",
)
SUBNATIONAL = Dataset(
    "subnational_admin",
    "data/boundaries/subnational_admin/out/admin_areas_display.geojson",
    has_locations=True,
    label="Subnational",
)
INDIGENOUS = Dataset(
    "indigenous_territories",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories.geojson",
    has_locations=True,
    label="Indigenous territory",
)
PROTECTED = Dataset(
    "protected_areas",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas.geojson",
    has_locations=True,
    label="Protected area",
)
DATASETS = [NATIONAL, SUBNATIONAL, INDIGENOUS, PROTECTED]
WITH_LOCATIONS = [d for d in DATASETS if d.has_locations]

# Same order as JURISDICTION_OUTPUT_SUFFIXES in preprocess_mining_areas.py
OUTPUT_SUFFIXES = (
    "_yearly.json",
    "_impacts.geojson",
    "_impacts_unfiltered.geojson",
    "_impacts_unfiltered_dict.json",
    "_impacts_yearly.csv",
)

YEARLY_COLUMNS = ["id", "admin_year", "intersected_area_ha", "intersected_area_ha_cumulative"]

# The script rounds every group (illegality category, location) to 2 decimals before
# they are summed, so a sum of n such values can be off by up to n * 0.005, plus a
# little for float noise.
ROUNDING_TOL_HA = 0.005
FLOAT_TOL_HA = 0.01

# A cumulative value going down between periods is reported as a warning while it is
# small, and fails past the larger of these two. Snapshots are built as "everything
# detected up to period i", so they shouldn't shrink at all; small dips can still come
# from border re-attribution.
DIP_FAIL_HA = 1.0
DIP_FAIL_FRAC = 0.001
# Dips below this are float noise from summing the same fragments in another order,
# and aren't reported at all
DIP_NOISE_HA = 0.01

# Same tolerance assert_country_totals_tie uses in preprocess_mining_areas.py
COUNTRY_TIE_TOLERANCE_HA = 50.0

# Subnational areas summed per country should give the national total for that
# country. Summing to more means double-counting, which fails past this share.
COVERAGE_FAIL_FRAC = 0.01

# Mined area may not exceed the area's own polygon area (measured in EPSG:6933)
POLYGON_AREA_SLACK_FRAC = 0.01
POLYGON_AREA_SLACK_HA = 1.0

# admin_illegality_max values that may appear in illegality_areas; -1 marks periods
# without illegality data and must have been filtered out
ILLEGALITY_CATEGORIES = {0, 1, 2, 3, 4}

# Country codes as the Mining Calculator API expects them (SR -> SU, GY -> GU)
CALCULATOR_COUNTRY_CODES = {"BO", "BR", "CO", "EC", "GF", "GU", "PE", "SU", "VE"}

EQUAL_AREA_CRS = "EPSG:6933"

# Regression against the published data. AMW_PUBLISHED_BASE points it elsewhere, e.g.
# at a local copy served with `python -m http.server`.
PUBLISHED_BASE = os.environ.get("AMW_PUBLISHED_BASE", "https://media-amw.earthgenome.org")
PUBLISHED_SENTINEL = NATIONAL.yearly
MAX_PROBE_DAYS = 400
REGRESSION_FAIL_FRAC = 0.02
REGRESSION_WARN_FRAC = 0.10
REGRESSION_MIN_HA = 10.0


# ---------------------------------------------------------------------------
# reporting: findings, compact console tables and the Markdown report
# ---------------------------------------------------------------------------

# Rows of a finding's table shown in the console; the report has all of them
CONSOLE_FAILURE_ROWS = 10
CONSOLE_WARNING_ROWS = 5

# Text columns are cut to this many characters in the console, to keep tables on one
# line per row; the report has them in full
CONSOLE_TEXT_WIDTH = {"name": 25, "type": 20}
CONSOLE_DEFAULT_TEXT_WIDTH = 40

# Shorter, consistent column headers for every table in the console and the report
SHORT_NAMES = {
    "admin_year": "period",
    "intersected_area_ha": "added_ha",
    "intersected_area_ha_cumulative": "cumulative_ha",
    "intersected_area_ha_cumulative_published": "published_ha",
    "previous_cumulative": "previous_ha",
    "mining_affected_area_ha": "total_ha",
    "expected_increment": "expected_added_ha",
    "amazon_cumulative": "amazon_ha",
    "yearly_latest": "yearly_latest_ha",
    "yearly_at_cutoff": "yearly_at_cutoff_ha",
    "yearly_cumulative": "cumulative_ha",
    "illegality_sum": "bands_sum_ha",
    "locations_sum": "locations_sum_ha",
    "change_frac": "change",
    "affectedArea": "affected_ha",
}
# Shown as a percentage rather than hectares
PERCENT_COLUMNS = {"change_frac"}


class DataQualityError(AssertionError):
    """A data-quality check failed. The message is the compact console report.

    conftest.py shows this message instead of pytest's usual traceback.
    """


@dataclass
class Finding:
    nodeid: str
    level: str  # "failure" or "warning"
    title: str
    summary: str
    table: pd.DataFrame | None = None
    lines: list = field(default_factory=list)


# Everything the report needs, filled in while the tests run
REPORT = {"findings": [], "published": None}


def current_test():
    # PYTEST_CURRENT_TEST is "<nodeid> (<phase>)"
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0]


def _record(level, title, summary, table, lines, sort_by):
    if table is not None and sort_by is not None and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False)
    finding = Finding(current_test(), level, title, summary, table, list(lines or []))
    REPORT["findings"].append(finding)
    return finding


def warn(title, summary, table=None, lines=None, sort_by=None):
    """Record a warning: something looks off, but not by enough to fail the run."""
    _record("warning", title, summary, table, lines, sort_by)


def check(ok, title, summary, table=None, lines=None, sort_by=None):
    """Fail the test with a compact report unless `ok`.

    `table` holds the offending rows (shown in part in the console, in full in the
    report), `lines` a list of offending items, and `sort_by` the column to put the
    largest offenders first by.
    """
    if ok:
        return
    finding = _record("failure", title, summary, table, lines, sort_by)
    raise DataQualityError(console_text(finding, CONSOLE_FAILURE_ROWS))


def _format_value(column, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.1%}" if column in PERCENT_COLUMNS else f"{value:,.2f}"
    return str(value).replace("\n", " ")


def _formatted_columns(df):
    """(header, cell strings, right-aligned) for each column of a frame."""
    columns = []
    for col in df.columns:
        cells = [_format_value(col, v) for v in df[col].tolist()]
        numeric = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        columns.append((SHORT_NAMES.get(col, col), cells, numeric))
    return columns


def terminal_width():
    # conftest.py sets COLUMNS from the real terminal before output is captured
    return shutil.get_terminal_size((120, 40)).columns


def format_table(df, max_rows=None, width=None, indent="    ", truncate=True):
    """A plain-text table that fits `width`, with at most `max_rows` rows.

    Text is cut to CONSOLE_TEXT_WIDTH. If the table is still too wide, columns are left
    out, with a note that the report has them: first country and type, then columns
    from the right, but never id or the last column, which is the size of the problem.
    """
    total = len(df)
    shown = df.head(max_rows) if max_rows else df
    columns = []
    for header, cells, numeric in _formatted_columns(shown):
        limit = CONSOLE_TEXT_WIDTH.get(header, CONSOLE_DEFAULT_TEXT_WIDTH)
        if truncate and not numeric:
            cells = [c if len(c) <= limit else c[: limit - 1] + "…" for c in cells]
        col_width = max([len(header), *map(len, cells)])
        columns.append((header, cells, numeric, col_width))

    def too_wide():
        return sum(c[3] for c in columns) + 2 * (len(columns) - 1) > width - len(indent)

    hidden = 0
    if width is not None:
        headers = [c[0] for c in columns]
        drop_order = [h for h in ("country", "type") if h in headers]
        drop_order += [h for h in reversed(headers[1:-1]) if h not in drop_order]
        for header in drop_order:
            if not too_wide():
                break
            columns = [c for c in columns if c[0] != header]
            hidden += 1

    def row(values):
        parts = [
            v.rjust(w) if numeric else v.ljust(w)
            for v, (_, _, numeric, w) in zip(values, columns, strict=True)
        ]
        return indent + "  ".join(parts).rstrip()

    lines = [row([c[0] for c in columns])]
    lines += [row(values) for values in zip(*(c[1] for c in columns), strict=True)]
    notes = []
    if max_rows and total > max_rows:
        notes.append(f"{total - max_rows} more row(s)")
    if hidden:
        notes.append(f"{hidden} more column(s)")
    if notes:
        lines.append(f"{indent}… {' and '.join(notes)} in the report")
    return "\n".join(lines)


def console_text(finding, max_rows):
    """A finding as a few short lines for the console."""
    lines = [f"{finding.title}: {finding.summary}"]
    if finding.table is not None and len(finding.table):
        lines.append(format_table(finding.table, max_rows, width=terminal_width()))
    width = terminal_width() - 4
    for item in finding.lines[:max_rows]:
        item = str(item)
        if len(item) > width:
            # cut from the middle: the area comes first and the problem last
            head = width * 2 // 5
            item = item[:head] + "…" + item[len(item) - (width - head - 1) :]
        lines.append(f"    {item}")
    if len(finding.lines) > max_rows:
        lines.append(f"    … {len(finding.lines) - max_rows} more in the report")
    return "\n".join(lines)


def markdown_table(df):
    """The complete table in Markdown, falling back to plain text without tabulate."""
    columns = _formatted_columns(df)
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return "```\n" + format_table(df, indent="", truncate=False) + "\n```"
    # built positionally, so two columns sharing a short header don't collapse into one
    data = pd.DataFrame(
        list(zip(*(cells for _, cells, _ in columns), strict=True)) if len(df) else [],
        columns=[header for header, _, _ in columns],
    ).map(lambda c: c.replace("|", "\\|"))
    return data.to_markdown(
        index=False,
        disable_numparse=True,
        colalign=["right" if numeric else "left" for _, _, numeric in columns],
    )


def nodeid_label(nodeid):
    return nodeid.split("::", 1)[-1]


def published_description():
    published = REPORT["published"]
    if not published:
        return "not compared (set AMW_COMPARE_PUBLISHED=latest or YYYYMMDD)"
    return (
        f"{published['folder']} (newest period {published['newest_period']}), "
        f"from AMW_COMPARE_PUBLISHED={published['requested']} at {PUBLISHED_BASE}"
    )


def console_summary():
    """Lines conftest.py prints at the end of the run: warnings and the comparison used."""
    lines = []
    for finding in (f for f in REPORT["findings"] if f.level == "warning"):
        lines.append(f"WARNING {nodeid_label(finding.nodeid)}")
        lines.append("  " + console_text(finding, CONSOLE_WARNING_ROWS).replace("\n", "\n  "))
        lines.append("")
    if REPORT["published"]:
        lines.append(f"Compared with published data: {published_description()}")
    return lines


def write_report(items, outcomes):
    """Write the Markdown report for the tests that ran, and return its path.

    `items` are pytest's collected tests and `outcomes` maps their nodeids to
    passed/failed/skipped; conftest.py calls this at the end of the session.
    """
    docs = {item.nodeid: inspect.cleandoc(item.function.__doc__ or "") for item in items}
    findings = REPORT["findings"]
    failures = [f for f in findings if f.level == "failure"]
    warnings_ = [f for f in findings if f.level == "warning"]
    warning_counts = Counter(f.nodeid for f in warnings_)
    counts = Counter(outcomes.get(item.nodeid, "not run") for item in items)

    out = [
        "# Data-quality report: preprocess_mining_areas.py outputs",
        "",
        f"- Run: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Outputs: `{ROOT}`",
        f"- DATA_UPDATED_AT: {DATA_UPDATED_AT}",
        f"- Periods: {MINING_YEARS_QUARTERS[0]} to {MINING_YEARS_QUARTERS[-1]} "
        f"({len(MINING_YEARS_QUARTERS)} periods)",
        f"- Illegality cutoff: {ILLEGALITY_DATA_UPDATED_AT}",
        f"- Published comparison: {published_description()}",
        "",
        "## Summary",
        "",
        f"**{counts['failed']} failed, {counts['passed']} passed, {counts['skipped']} skipped, "
        f"{len(warnings_)} warning(s)**",
        "",
    ]
    # one line per finding up front, details below, the full checklist at the end
    for icon, group in [("❌", failures), ("⚠️", warnings_)]:
        out += [f"- {icon} {f.title}: {f.summary} (`{nodeid_label(f.nodeid)}`)" for f in group]
    out += [""]

    for heading, icon, group in [("Failures", "❌", failures), ("Warnings", "⚠️", warnings_)]:
        if not group:
            continue
        out += [f"## {heading}", ""]
        for finding in group:
            out += [f"### {icon} {finding.title}", "", f"`{nodeid_label(finding.nodeid)}`", ""]
            doc = docs.get(finding.nodeid)
            if doc:
                out += [f"> {line}".rstrip() for line in doc.splitlines()] + [""]
            out += [f"**{finding.summary}**", ""]
            if finding.table is not None and len(finding.table):
                out += [markdown_table(finding.table), ""]
            if finding.lines:
                out += [f"- {line}" for line in finding.lines] + [""]

    checklist = pd.DataFrame(
        [
            {
                "check": item.originalname,
                "parameters": getattr(getattr(item, "callspec", None), "id", ""),
                "outcome": outcomes.get(item.nodeid, "not run")
                + (
                    f", {warning_counts[item.nodeid]} warning(s)"
                    if warning_counts[item.nodeid]
                    else ""
                ),
                "what it checks": docs[item.nodeid].split("\n", 1)[0],
            }
            for item in items
        ],
        columns=["check", "parameters", "outcome", "what it checks"],
    )
    out += ["## All checks", "", markdown_table(checklist), ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    return REPORT_PATH


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rounding_tol(n_terms):
    return ROUNDING_TOL_HA * n_terms + FLOAT_TOL_HA


def cleanup_country_code(country_code):
    # same mapping as prepare_for_mining_calculator_and_save
    return {"SR": "SU", "GY": "GU"}.get(country_code, country_code)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_dips(offenders, title, amount_col, base_col):
    """Warn about small violations, fail on large ones.

    `amount_col` is how far each row is past the limit (positive), `base_col` the
    value it is measured against, for the relative threshold.
    """
    if offenders.empty:
        return
    limit = np.maximum(DIP_FAIL_HA, DIP_FAIL_FRAC * offenders[base_col].abs())
    hard = offenders[offenders[amount_col] > limit]
    soft = offenders[offenders[amount_col] <= limit]
    threshold = f"max({DIP_FAIL_HA} ha, {DIP_FAIL_FRAC:.1%})"
    if not soft.empty:
        warn(title, f"{len(soft)} small case(s), within {threshold}", soft, sort_by=amount_col)
    check(hard.empty, title, f"{len(hard)} case(s) over {threshold}", hard, sort_by=amount_col)


def path(rel):
    return ROOT / rel


def require(rel):
    p = path(rel)
    if not p.exists():
        stale = p.with_name(p.name + ".stale")
        if stale.exists():
            pytest.skip(
                f"{rel} is missing but {stale.name} exists: the last "
                "preprocess_mining_areas.py run died before writing it"
            )
        pytest.skip(f"{rel} is missing")
    return p


# ---------------------------------------------------------------------------
# loaders, cached so every file is parsed once per session
# ---------------------------------------------------------------------------


@functools.cache
def _load_json(rel):
    with open(path(rel), encoding="utf-8") as f:
        return json.load(f)


def load_json(rel):
    require(rel)
    return _load_json(rel)


def yearly(ds):
    return pd.DataFrame(load_json(ds.yearly))


def yearly_or_empty(ds):
    """The yearly frame, or an empty one when the dataset had no mining at all."""
    df = yearly(ds)
    return df if len(df) else pd.DataFrame(columns=YEARLY_COLUMNS)


def dict_records(ds):
    return load_json(ds.dict_json)


def geojson_records(ds, filtered=False):
    data = load_json(ds.filtered_geojson if filtered else ds.unfiltered_geojson)
    return [feature["properties"] for feature in data["features"]]


RECORD_SOURCES = {
    "dict": dict_records,
    "unfiltered_geojson": lambda ds: geojson_records(ds, filtered=False),
    "filtered_geojson": lambda ds: geojson_records(ds, filtered=True),
}


def by_id(records):
    return {r["id"]: r for r in records}


def last_cumulative(ds):
    """Each id's cumulative area at the newest period."""
    df = yearly_or_empty(ds)
    if df.empty:
        return pd.Series(dtype=float)
    latest = df["admin_year"].max()
    return df[df["admin_year"] == latest].set_index("id")["intersected_area_ha_cumulative"]


def cumulative_at(ds, year):
    df = yearly_or_empty(ds)
    return df[df["admin_year"] == year].set_index("id")["intersected_area_ha_cumulative"]


def illegality_cutoff_year():
    """The newest period that carries illegality data."""
    candidates = [y for y in MINING_YEARS_QUARTERS if y <= ILLEGALITY_DATA_UPDATED_AT]
    check(
        candidates, "No illegality period", f"No period at or before {ILLEGALITY_DATA_UPDATED_AT}"
    )
    return max(candidates)


def national_cumulative_by_country():
    """National yearly rows keyed by (country_code, admin_year)."""
    df = yearly(NATIONAL)
    codes = {r["id"]: r.get("country_code") for r in dict_records(NATIONAL)}
    df = df[df["id"] != ENTIRE_AMAZON_ID].copy()
    df["country_code"] = df["id"].map(codes)
    return df.set_index(["country_code", "admin_year"])["intersected_area_ha_cumulative"]


# ---------------------------------------------------------------------------
# describing areas in reports
# ---------------------------------------------------------------------------


def describe_record(ds, record):
    """Name, type and country of one area record (from the dict or the published data).

    The name is `name_field`, or `country` for the national dataset, whose rows are
    countries plus the Entire Amazon. The type is the area's own `type_field` where the
    source has one (e.g. a protected area category), else the dataset's label.
    """
    return {
        "name": record.get("name_field") or record.get("country"),
        "type": record.get("type_field") or ds.label,
        "country": record.get("country_code"),
    }


@functools.cache
def area_info(ds):
    """Name, type and country of every area of a dataset, indexed by id.

    Read from the dict file. Empty when that file is missing, so a test that only
    needs the yearly file still runs and just reports bare ids.
    """
    columns = ["name", "type", "country"]
    if not path(ds.dict_json).exists():
        return pd.DataFrame(columns=columns, index=pd.Index([], name="id"))
    rows = [{"id": r["id"], **describe_record(ds, r)} for r in _load_json(ds.dict_json)]
    df = pd.DataFrame(rows, columns=["id", *columns])
    return df.drop_duplicates("id").set_index("id")


def with_area_info(ds, df):
    """Insert the name, type and country of each row's area right after its id column."""
    if "id" not in df.columns or df.empty:
        return df
    info = area_info(ds)
    df = df.copy()
    position = df.columns.get_loc("id") + 1
    for col in reversed(info.columns):
        df.insert(position, col, df["id"].map(info[col]))
    return df


@functools.cache
def _labels(ds):
    # plain dict, since describe() runs once per record and pandas lookups are slow
    return {
        area_id: f"{area_id} '{row.name}' ({row.type})"
        for area_id, row in zip(area_info(ds).index, area_info(ds).itertuples(), strict=True)
    }


def describe(ds, area_id):
    """`<id> '<name>' (<type>)`, for reports that list areas one per line."""
    return _labels(ds).get(area_id, str(area_id))


ds_param = pytest.mark.parametrize("ds", DATASETS, ids=lambda d: d.name)
ds_locations_param = pytest.mark.parametrize("ds", WITH_LOCATIONS, ids=lambda d: d.name)
source_param = pytest.mark.parametrize("source", list(RECORD_SOURCES))


# ---------------------------------------------------------------------------
# presence and freshness
# ---------------------------------------------------------------------------


def test_outputs_exist():
    """Every output the website and the upload script need is on disk.

    Covers the five per-dataset outputs of all four datasets, the simplified mining
    file of every period and the combined mining file. preprocess_mining_areas.py
    moves the previous run's outputs aside as `.stale` before it starts, so a file
    with only a `.stale` copy means the last run died before writing it.
    """
    expected = [ds.output(suffix) for ds in DATASETS for suffix in OUTPUT_SUFFIXES]
    expected += list(MINING_SIMPLIFIED_FILES) + [COMBINED_MINING_FILE]
    missing = []
    for rel in expected:
        p = path(rel)
        if not p.exists():
            stale = p.with_name(p.name + ".stale")
            note = " (only a .stale copy: the last run died)" if stale.exists() else ""
            missing.append(rel + note)
    check(not missing, "Missing outputs", f"{len(missing)} file(s) missing", lines=missing)


@ds_param
def test_outputs_newer_than_source_boundaries(ds):
    """The outputs were written after the boundary file they were computed from.

    Warning only. An output older than its boundary file suggests the boundaries were
    re-standardized after the last preprocessing run, so the numbers may be stale.
    Syncing from S3 can leave modification times that mean nothing, hence no failure.
    """
    source = require(ds.file)
    older = [
        ds.output(suffix)
        for suffix in OUTPUT_SUFFIXES
        if path(ds.output(suffix)).exists()
        and path(ds.output(suffix)).stat().st_mtime < source.stat().st_mtime
    ]
    if older:
        warn(
            "Outputs older than their boundary file",
            f"{ds.file} changed after these were written, so they may be stale",
            lines=older,
        )


# ---------------------------------------------------------------------------
# _yearly.json
# ---------------------------------------------------------------------------


@ds_param
def test_yearly_schema(ds):
    """The yearly file has the expected columns, with no missing or negative values.

    Columns are exactly id, admin_year, intersected_area_ha (area added in the period)
    and intersected_area_ha_cumulative (total area at the period), as written by
    calculate_mining_area_timeseries. Ids are strings, periods are ints, and areas are
    finite and >= 0.
    """
    df = yearly(ds)
    check(len(df) > 0, "Empty yearly file", f"{ds.yearly} has no rows")
    check(
        list(df.columns) == YEARLY_COLUMNS,
        "Unexpected columns",
        f"expected {YEARLY_COLUMNS}, got {list(df.columns)}",
    )
    nulls = df[df[YEARLY_COLUMNS].isna().any(axis=1)]
    check(nulls.empty, "Missing values", f"{len(nulls)} row(s)", with_area_info(ds, nulls))
    not_str = df[~df["id"].map(lambda v: isinstance(v, str))]
    check(not_str.empty, "Ids that aren't strings", f"{len(not_str)} row(s)", not_str)
    check(
        pd.api.types.is_integer_dtype(df["admin_year"]),
        "Periods that aren't ints",
        f"admin_year is {df['admin_year'].dtype}",
    )
    areas = df[["intersected_area_ha", "intersected_area_ha_cumulative"]].to_numpy(dtype=float)
    non_finite = df[~np.isfinite(areas).all(axis=1)]
    check(non_finite.empty, "Non-finite areas", f"{len(non_finite)} row(s)", non_finite)
    negative = df[(areas < 0).any(axis=1)]
    check(negative.empty, "Negative areas", f"{len(negative)} row(s)", with_area_info(ds, negative))


@ds_param
def test_yearly_id_and_year_unique(ds):
    """Each area has at most one row per period.

    A duplicated (id, admin_year) would make the website and the collated CSVs count
    that area twice for that period.
    """
    df = yearly(ds)
    dupes = df[df.duplicated(["id", "admin_year"], keep=False)].sort_values(["id", "admin_year"])
    check(
        dupes.empty,
        "Duplicated area and period",
        f"{len(dupes)} rows share an (id, admin_year)",
        with_area_info(ds, dupes),
    )


@ds_param
def test_yearly_has_every_period_for_every_id(ds):
    """Every area has a row for every period in MINING_YEARS_QUARTERS.

    calculate_mining_area_timeseries builds the full grid of ids x periods, carrying
    values forward where an area has no mining in a period. A missing period means a
    mining raster was not loaded, or MINING_YEARS_QUARTERS in constants.py disagrees
    with the data.
    """
    df = yearly(ds)
    years = set(df["admin_year"].unique())
    expected = set(MINING_YEARS_QUARTERS)
    check(
        years == expected,
        "Periods differ from MINING_YEARS_QUARTERS",
        f"missing {sorted(expected - years)}, unexpected {sorted(years - expected)}",
    )
    counts = df.groupby("id")["admin_year"].nunique().rename("periods").reset_index()
    incomplete = counts[counts["periods"] != len(expected)]
    check(
        incomplete.empty,
        "Areas missing periods",
        f"{len(incomplete)} area(s) without all {len(expected)} periods",
        with_area_info(ds, incomplete),
    )


@ds_param
def test_yearly_increment_not_above_cumulative(ds):
    """The area added in a period is never more than the total area at that period."""
    df = yearly(ds)
    over = df[df["intersected_area_ha"] > df["intersected_area_ha_cumulative"] + 1e-6]
    check(
        over.empty,
        "Added area above the cumulative area",
        f"{len(over)} row(s) where intersected_area_ha > intersected_area_ha_cumulative",
        with_area_info(ds, over),
    )


@ds_param
def test_yearly_increments_reconcile_with_cumulative(ds):
    """The area added in a period is the cumulative minus the previous period's.

    That is how calculate_mining_area_timeseries derives it: the first period's
    increment is its own total, and later increments are the difference with the
    previous period, clipped at 0 when the snapshot shrank. A mismatch means the
    cumulative sum isn't working across periods (e.g. rows out of order, or an
    increment computed across two different areas).
    """
    df = yearly(ds).sort_values(["id", "admin_year"]).reset_index(drop=True)
    previous = df.groupby("id")["intersected_area_ha_cumulative"].shift()
    expected = (df["intersected_area_ha_cumulative"] - previous).clip(lower=0)
    # the first period has no previous snapshot, so its increment is its own total
    expected = expected.fillna(df["intersected_area_ha_cumulative"])
    df["expected_increment"] = expected
    df["difference_ha"] = (df["intersected_area_ha"] - expected).abs()
    wrong = df[df["difference_ha"] > 1e-6]
    check(
        wrong.empty,
        "Added areas don't match the cumulative values",
        f"{len(wrong)} row(s)",
        with_area_info(ds, wrong),
        sort_by="difference_ha",
    )


@ds_param
def test_yearly_cumulative_does_not_decrease(ds):
    """An area's cumulative mined area never goes down from one period to the next.

    The snapshots are built as "everything detected up to period i"
    (convert_rasters_to_vector.py), so they can only grow. Dips under DIP_NOISE_HA
    are float noise and ignored. Small dips are warnings, usually border
    re-attribution moving a sliver between areas; dips over max(DIP_FAIL_HA,
    DIP_FAIL_FRAC) fail, and point at a broken raster period or attribution.
    """
    df = yearly(ds).sort_values(["id", "admin_year"]).reset_index(drop=True)
    df["previous_cumulative"] = df.groupby("id")["intersected_area_ha_cumulative"].shift()
    df["dip_ha"] = df["previous_cumulative"] - df["intersected_area_ha_cumulative"]
    dips = df[df["dip_ha"] > DIP_NOISE_HA][
        ["id", "admin_year", "previous_cumulative", "intersected_area_ha_cumulative", "dip_ha"]
    ]
    check_dips(
        with_area_info(ds, dips),
        "Cumulative area decreasing between periods",
        "dip_ha",
        "previous_cumulative",
    )


@ds_param
def test_yearly_not_above_entire_amazon(ds):
    """No area has more mining than the Entire Amazon (AMAZ) in the same period.

    AMAZ, in the national yearly file, covers the whole basin, so every area of every
    dataset lies inside it.
    """
    df = yearly(ds)
    amazon = yearly(NATIONAL)
    amazon = amazon[amazon["id"] == ENTIRE_AMAZON_ID].set_index("admin_year")[
        "intersected_area_ha_cumulative"
    ]
    check(len(amazon), f"No {ENTIRE_AMAZON_ID} rows", f"none in {NATIONAL.yearly}")
    df = df.assign(amazon_cumulative=df["admin_year"].map(amazon))
    df["excess_ha"] = df["intersected_area_ha_cumulative"] - df["amazon_cumulative"]
    over = df[df["excess_ha"] > FLOAT_TOL_HA][
        ["id", "admin_year", "intersected_area_ha_cumulative", "amazon_cumulative", "excess_ha"]
    ]
    check(
        over.empty,
        f"Areas larger than {ENTIRE_AMAZON_ID}",
        f"{len(over)} row(s)",
        with_area_info(ds, over),
        sort_by="excess_ha",
    )


def test_national_countries_tie_to_entire_amazon():
    """The countries add up to the Entire Amazon (AMAZ) in every period, within 50 ha.

    Re-checks assert_country_totals_tie from preprocess_mining_areas.py on the file as
    written. A break means mining was dropped or counted twice along a border.
    """
    df = yearly(NATIONAL)
    is_amazon = df["id"] == ENTIRE_AMAZON_ID
    amazon = df[is_amazon].groupby("admin_year")["intersected_area_ha_cumulative"].sum()
    countries = df[~is_amazon].groupby("admin_year")["intersected_area_ha_cumulative"].sum()
    tie = pd.DataFrame({"amazon_ha": amazon, "countries_ha": countries}).fillna(0)
    tie["difference_ha"] = tie["countries_ha"] - tie["amazon_ha"]
    broken = tie[tie["difference_ha"].abs() > COUNTRY_TIE_TOLERANCE_HA].reset_index()
    check(
        broken.empty,
        f"Countries don't add up to {ENTIRE_AMAZON_ID}",
        f"{len(broken)} period(s) off by more than {COUNTRY_TIE_TOLERANCE_HA} ha",
        broken,
    )


def test_subnational_sums_match_national_per_country():
    """A country's subnational regions add up to that country's national total.

    Both are restricted to mining in the country itself (prefer_domestic_rows), so
    they should match. More than the national total fails past COVERAGE_FAIL_FRAC: some
    ground is counted under two regions. Less only warns: it means some mining in the
    country falls outside every display region of that country (e.g. the display layer
    doesn't reach the national border), so it is missing from the subnational numbers.
    """
    df = yearly(SUBNATIONAL)
    codes = {r["id"]: r.get("country_code") for r in dict_records(SUBNATIONAL)}
    df = df.assign(country_code=df["id"].map(codes))
    unknown = df[df["country_code"].isna()]["id"].unique()
    check(
        len(unknown) == 0,
        "Subnational areas without a country",
        f"{len(unknown)} area(s) without a country_code",
        lines=[describe(SUBNATIONAL, i) for i in unknown],
    )

    sums = df.groupby(["country_code", "admin_year"])["intersected_area_ha_cumulative"].sum()
    national = national_cumulative_by_country()
    cmp = pd.DataFrame({"subnational_ha": sums, "national_ha": national}).fillna(0)
    cmp["difference_ha"] = cmp["subnational_ha"] - cmp["national_ha"]
    cmp = cmp.reset_index()
    abs_difference = cmp["difference_ha"].abs()
    noticeable = cmp[abs_difference > np.maximum(1.0, 0.001 * cmp["national_ha"])]

    excess = noticeable[
        noticeable["difference_ha"] > COVERAGE_FAIL_FRAC * noticeable["national_ha"]
    ]
    check(
        excess.empty,
        "Subnational areas add up to more than the country",
        f"{len(excess)} country-period(s) over the national total by more than "
        f"{COVERAGE_FAIL_FRAC:.0%}: some ground is counted under two regions",
        excess,
        sort_by="difference_ha",
    )
    if not noticeable.empty:
        warn(
            "Subnational areas differ from the national totals",
            f"{len(noticeable)} country-period(s); a shortfall means mining outside every "
            "display region of its country",
            noticeable.assign(abs_difference=noticeable["difference_ha"].abs())
            .sort_values("abs_difference", ascending=False)
            .drop(columns="abs_difference"),
        )


# ---------------------------------------------------------------------------
# _impacts_unfiltered_dict.json
# ---------------------------------------------------------------------------


@ds_param
@source_param
def test_ids_unique(ds, source):
    """Every area id appears once, in the dict and in both geojsons.

    save_to_geojson asserts this before writing the geojsons; the dict is written
    separately, so it is checked here too. Duplicates would make the website pick one
    of the records arbitrarily.
    """
    records = RECORD_SOURCES[source](ds)
    ids = pd.Series([r.get("id") for r in records])
    check(ids.notna().all(), "Records without an id", f"{int(ids.isna().sum())} record(s)")
    dupes = sorted(ids[ids.duplicated(keep=False)].unique())
    check(
        not dupes,
        "Duplicated ids",
        f"{len(dupes)} id(s) appear more than once in the {source}",
        lines=[describe(ds, i) for i in dupes],
    )


@ds_param
def test_dict_ids_match_source_boundaries(ds):
    """The dict has exactly one record per area of the source boundary file.

    The script left-merges its results onto the boundary file, so no area should be
    added or dropped. A duplicated id in the results would add rows here.
    """
    source = require(ds.file)
    source_ids = set(gpd.read_file(source, columns=["id"], ignore_geometry=True)["id"])
    output_ids = {r["id"] for r in dict_records(ds)}
    check(
        len(dict_records(ds)) == len(source_ids),
        "Record count differs from the boundary file",
        f"{len(dict_records(ds))} records for {len(source_ids)} source areas",
    )
    check(
        output_ids == source_ids,
        "Ids differ from the boundary file",
        f"{len(source_ids - output_ids)} missing, {len(output_ids - source_ids)} unexpected",
        lines=[f"missing: {i}" for i in sorted(source_ids - output_ids)]
        + [f"unexpected: {describe(ds, i)}" for i in sorted(output_ids - source_ids)],
    )


@ds_param
def test_dict_schema(ds):
    """Each dict record has the fields the website reads, with sensible values.

    - mining_affected_area_ha: null (no mining) or a finite number >= 0
    - illegality_areas: a list when there is mining, null when there isn't
    - locations_per_year: present for every dataset except national, a dict per year
    - bbox: four finite numbers, [minx, miny, maxx, maxy] in lon/lat, min <= max
    """
    problems = []
    for r in dict_records(ds):
        label = describe(ds, r.get("id"))
        area = r.get("mining_affected_area_ha")
        if area is not None and not (is_number(area) and area >= 0):
            problems.append(f"{label}: mining_affected_area_ha={area!r}")

        illegality = r.get("illegality_areas")
        if area is None and illegality is not None:
            problems.append(f"{label}: illegality_areas set without a mined area")
        if area is not None and not isinstance(illegality, list):
            problems.append(f"{label}: illegality_areas={illegality!r}, expected a list")

        if ds.has_locations:
            if "locations_per_year" not in r:
                problems.append(f"{label}: no locations_per_year")
            elif area is not None and not isinstance(r["locations_per_year"], dict):
                problems.append(f"{label}: locations_per_year={r['locations_per_year']!r}")
        elif "locations_per_year" in r:
            problems.append(f"{label}: locations_per_year should not be in {ds.name}")

        bbox = r.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(map(is_number, bbox))):
            problems.append(f"{label}: bbox={bbox!r}")
        else:
            minx, miny, maxx, maxy = bbox
            if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
                problems.append(f"{label}: bbox out of range {bbox}")
    check(not problems, "Malformed dict records", f"{len(problems)} problem(s)", lines=problems)


@ds_param
def test_dict_ids_with_mining_match_yearly(ds):
    """The areas with a mined total in the dict are exactly the areas in the yearly file.

    Both come from the same summary: an area with mining in any period gets yearly rows
    and a total, an area without gets neither (a null total in the dict).
    """
    areas = {r["id"]: r.get("mining_affected_area_ha") for r in dict_records(ds)}
    yearly_ids = set(yearly_or_empty(ds)["id"])
    missing_from_dict = sorted(yearly_ids - set(areas))
    check(
        not missing_from_dict,
        "Areas in the yearly file but not in the dict",
        f"{len(missing_from_dict)} id(s)",
        lines=missing_from_dict,
    )
    no_yearly = sorted(i for i, a in areas.items() if a is not None and i not in yearly_ids)
    check(
        not no_yearly,
        "Areas with a mined total but no yearly rows",
        f"{len(no_yearly)} area(s)",
        lines=[describe(ds, i) for i in no_yearly],
    )
    no_area = sorted(i for i in yearly_ids if areas[i] is None)
    check(
        not no_area,
        "Areas with yearly rows but no mined total",
        f"{len(no_area)} area(s)",
        lines=[describe(ds, i) for i in no_area],
    )


@ds_param
def test_dict_total_matches_latest_yearly_cumulative(ds):
    """An area's mining_affected_area_ha equals its cumulative area at the newest period.

    summarize_latest_snapshot takes each area's newest snapshot, which is also the last
    value of its timeseries (carried forward when the area has no rows in the newest
    period). They differ only by the rounding to 2 decimals.
    """
    latest = last_cumulative(ds)
    rows = [
        (r["id"], r["mining_affected_area_ha"], latest[r["id"]])
        for r in dict_records(ds)
        if r.get("mining_affected_area_ha") is not None and r["id"] in latest.index
    ]
    df = pd.DataFrame(rows, columns=["id", "mining_affected_area_ha", "yearly_latest"])
    df["difference_ha"] = (df["mining_affected_area_ha"] - df["yearly_latest"]).abs()
    wrong = df[df["difference_ha"] > rounding_tol(1)]
    check(
        wrong.empty,
        "Total differs from the last yearly cumulative",
        f"{len(wrong)} area(s) where mining_affected_area_ha isn't the newest cumulative",
        with_area_info(ds, wrong),
        sort_by="difference_ha",
    )


def illegality_sums(records):
    rows = []
    for r in records:
        bands = r.get("illegality_areas") or []
        rows.append(
            {
                "id": r["id"],
                "mining_affected_area_ha": r.get("mining_affected_area_ha") or 0.0,
                "illegality_sum": sum(b["mining_affected_area"] for b in bands),
                "bands": len(bands),
            }
        )
    return pd.DataFrame(rows, columns=["id", "mining_affected_area_ha", "illegality_sum", "bands"])


def latest_location_sums(records):
    rows = []
    for r in records:
        per_year = r.get("locations_per_year") or {}
        if not per_year:
            continue
        latest = max(per_year, key=int)
        rows.append(
            {
                "id": r["id"],
                "period": latest,
                "mining_affected_area_ha": r.get("mining_affected_area_ha") or 0.0,
                "locations_sum": sum(loc["affectedArea"] for loc in per_year[latest]),
                "locations": len(per_year[latest]),
            }
        )
    return pd.DataFrame(
        rows, columns=["id", "period", "mining_affected_area_ha", "locations_sum", "locations"]
    )


@ds_param
@source_param
def test_illegality_sum_not_above_total(ds, source):
    """The illegality bands of an area don't add up to more than its mined total.

    Checked in the dict and in both geojsons. The breakdown is from the illegality
    cutoff period (ILLEGALITY_DATA_UPDATED_AT) and the total from the newest one, so a
    snapshot that shrank after the cutoff can make the bands exceed the total a
    little: that is a warning, larger excesses fail. See
    test_illegality_sum_matches_yearly_at_cutoff for the exact comparison.
    """
    df = illegality_sums(RECORD_SOURCES[source](ds))
    df["excess_ha"] = (
        df["illegality_sum"] - df["mining_affected_area_ha"] - rounding_tol(df["bands"] + 1)
    )
    check_dips(
        with_area_info(ds, df[df["excess_ha"] > 0]),
        f"Illegality bands above the total ({source})",
        "excess_ha",
        "mining_affected_area_ha",
    )


@ds_locations_param
@source_param
def test_latest_locations_sum_not_above_total(ds, source):
    """The newest period's Mining Calculator locations don't add up to more than the total.

    Checked in the dict and in both geojsons. Both come from the same newest snapshot,
    so the locations can only fall short of the total by the slivers of 0 ha that
    prepare_for_mining_calculator_and_save skips.
    """
    df = latest_location_sums(RECORD_SOURCES[source](ds))
    df["excess_ha"] = (
        df["locations_sum"] - df["mining_affected_area_ha"] - rounding_tol(df["locations"] + 1)
    )
    over = df[df["excess_ha"] > 0]
    check(
        over.empty,
        f"Latest locations above the total ({source})",
        f"{len(over)} area(s)",
        with_area_info(ds, over),
        sort_by="excess_ha",
    )


@ds_param
def test_illegality_categories_valid(ds):
    """Each illegality band has a known category, once per area, and valid numbers.

    admin_illegality_max must be one of ILLEGALITY_CATEGORIES (0 = outside every
    illegality zone). -1 marks periods after ILLEGALITY_DATA_UPDATED_AT and must have
    been filtered out of the breakdown. Areas are >= 0 and percentages are numbers.
    """
    problems = []
    for r in dict_records(ds):
        label = describe(ds, r["id"])
        bands = r.get("illegality_areas") or []
        categories = [b.get("admin_illegality_max") for b in bands]
        if len(set(categories)) != len(categories):
            problems.append(f"{label}: duplicated categories {categories}")
        for b in bands:
            category = b.get("admin_illegality_max")
            if not is_number(category) or category not in ILLEGALITY_CATEGORIES:
                problems.append(f"{label}: admin_illegality_max={category!r}")
            area = b.get("mining_affected_area")
            if not is_number(area) or area < 0:
                problems.append(f"{label}: mining_affected_area={area!r}")
            if not is_number(b.get("mining_affected_area_pct")):
                problems.append(
                    f"{label}: mining_affected_area_pct={b.get('mining_affected_area_pct')!r}"
                )
    check(not problems, "Invalid illegality bands", f"{len(problems)} problem(s)", lines=problems)


@ds_param
def test_illegality_sum_matches_yearly_at_cutoff(ds):
    """An area's illegality bands add up to its cumulative area at the illegality cutoff.

    The breakdown covers the whole snapshot at the newest period up to
    ILLEGALITY_DATA_UPDATED_AT, category 0 (outside every zone) included, so it must
    equal the yearly cumulative at that period, up to rounding. Areas with mining only
    after the cutoff have no bands and a cumulative of 0 at the cutoff. A mismatch
    means part of the mining was dropped from, or doubled in, the breakdown.
    """
    cutoff = illegality_cutoff_year()
    at_cutoff = cumulative_at(ds, cutoff)
    df = illegality_sums(dict_records(ds))
    df["yearly_at_cutoff"] = df["id"].map(at_cutoff).fillna(0.0)
    df["difference_ha"] = (df["illegality_sum"] - df["yearly_at_cutoff"]).abs()
    wrong = df[df["difference_ha"] > rounding_tol(df["bands"])]
    check(
        wrong.empty,
        f"Illegality bands don't add up to the cumulative at {cutoff}",
        f"{len(wrong)} area(s)",
        with_area_info(ds, wrong),
        sort_by="difference_ha",
    )


@ds_param
def test_illegality_percentages(ds):
    """Each band's mining_affected_area_pct is its share of the bands' total, summing to 1.

    The script divides each band's area by the sum of the bands and rounds to 3
    decimals, so each share may be off by 0.0005 and their sum by 0.0005 per band.
    """
    problems = []
    for r in dict_records(ds):
        bands = r.get("illegality_areas") or []
        total = sum(b["mining_affected_area"] for b in bands)
        if not bands or total <= 0:
            continue
        label = describe(ds, r["id"])
        pct_sum = sum(b["mining_affected_area_pct"] for b in bands)
        if abs(pct_sum - 1) > 0.0005 * len(bands) + 1e-9:
            problems.append(f"{label}: percentages sum to {pct_sum}")
        for b in bands:
            expected = b["mining_affected_area"] / total
            if abs(b["mining_affected_area_pct"] - expected) > 0.0005 + 1e-9:
                problems.append(
                    f"{label}: category {b['admin_illegality_max']} pct "
                    f"{b['mining_affected_area_pct']} vs {expected:.4f}"
                )
    check(
        not problems, "Wrong illegality percentages", f"{len(problems)} problem(s)", lines=problems
    )


@ds_locations_param
def test_locations_schema(ds):
    """Every Mining Calculator location is well formed.

    - year keys are periods of MINING_YEARS_QUARTERS, as strings
    - each location has exactly country, regionId and affectedArea
    - country is a code the Mining Calculator API knows (SR and GY mapped to SU, GU)
    - regionId is an int
    - affectedArea is a finite number >= 0
    - a (country, regionId) appears once per year, since group_and_sum_locations merges them

    An affectedArea of exactly 0 is a warning, not a failure: it is a sliver under
    0.005 ha that prepare_for_mining_calculator_and_save kept (its area was > 0) and
    group_and_sum_locations then rounded to 0.
    """
    valid_years = {str(y) for y in MINING_YEARS_QUARTERS}
    problems = []
    zero_area = []
    for r in dict_records(ds):
        label = describe(ds, r["id"])
        per_year = r.get("locations_per_year") or {}
        for year, locations in per_year.items():
            if year not in valid_years:
                problems.append(f"{label}: unexpected year key {year!r}")
            if not isinstance(locations, list):
                problems.append(f"{label} {year}: locations={locations!r}")
                continue
            seen = set()
            for loc in locations:
                if set(loc) != {"country", "regionId", "affectedArea"}:
                    problems.append(f"{label} {year}: keys {sorted(loc)}")
                    continue
                if loc["country"] not in CALCULATOR_COUNTRY_CODES:
                    problems.append(f"{label} {year}: country {loc['country']!r}")
                if not isinstance(loc["regionId"], int) or isinstance(loc["regionId"], bool):
                    problems.append(f"{label} {year}: regionId {loc['regionId']!r}")
                if not is_number(loc["affectedArea"]) or loc["affectedArea"] < 0:
                    problems.append(f"{label} {year}: affectedArea {loc['affectedArea']!r}")
                elif loc["affectedArea"] == 0:
                    zero_area.append((r["id"], year, loc["country"], loc["regionId"]))
                key = (loc["country"], loc["regionId"])
                if key in seen:
                    problems.append(f"{label} {year}: duplicated location {key}")
                seen.add(key)
    if zero_area:
        df = pd.DataFrame(zero_area, columns=["id", "period", "location_country", "regionId"])
        warn(
            "Locations with affectedArea 0",
            f"{len(df)} location(s), slivers under 0.005 ha rounded away",
            with_area_info(ds, df),
        )
    check(not problems, "Malformed locations", f"{len(problems)} problem(s)", lines=problems)


@ds_locations_param
def test_locations_sum_matches_yearly_every_period(ds):
    """For every period, an area's locations add up to its cumulative area at that period.

    prepare_locations_per_year builds each period's locations from that period's own
    snapshot, which is also what the yearly cumulative is, so they agree up to the
    rounding of each location to 2 decimals.

    Also warns about periods where an area has a cumulative area but no locations: the
    area had no mining in that period's snapshot and the yearly value was carried
    forward from an earlier period.
    """
    df = yearly_or_empty(ds)
    cumulative = df.set_index(["id", "admin_year"])["intersected_area_ha_cumulative"]
    rows = []
    for r in dict_records(ds):
        for year, locations in (r.get("locations_per_year") or {}).items():
            rows.append(
                {
                    "id": r["id"],
                    "admin_year": int(year),
                    "locations_sum": sum(loc["affectedArea"] for loc in locations),
                    "locations": len(locations),
                }
            )
    loc = pd.DataFrame(rows, columns=["id", "admin_year", "locations_sum", "locations"])
    loc["yearly_cumulative"] = [
        cumulative.get((i, y), np.nan) for i, y in zip(loc["id"], loc["admin_year"], strict=True)
    ]
    no_yearly = loc[loc["yearly_cumulative"].isna()]
    check(
        no_yearly.empty,
        "Locations for periods with no yearly row",
        f"{len(no_yearly)} (area, period)",
        with_area_info(ds, no_yearly),
    )
    loc["difference_ha"] = (loc["locations_sum"] - loc["yearly_cumulative"]).abs()
    wrong = loc[loc["difference_ha"] > rounding_tol(loc["locations"])]
    check(
        wrong.empty,
        "Locations don't add up to the yearly cumulative",
        f"{len(wrong)} (area, period)",
        with_area_info(ds, wrong),
        sort_by="difference_ha",
    )

    have = set(zip(loc["id"], loc["admin_year"], strict=True))
    no_locations = np.array(
        [(i, y) not in have for i, y in zip(df["id"], df["admin_year"], strict=True)], dtype=bool
    )
    carried = df[(df["intersected_area_ha_cumulative"] > rounding_tol(1)) & no_locations]
    if not carried.empty:
        warn(
            "Periods with a cumulative area but no locations",
            f"{len(carried)} (area, period): the value was carried forward from an earlier period",
            with_area_info(ds, carried[["id", "admin_year", "intersected_area_ha_cumulative"]]),
            sort_by="intersected_area_ha_cumulative",
        )


def test_subnational_locations_inside_their_region():
    """A subnational area's locations are calculator regions inside that area.

    The script keeps only calculator regions whose id starts with the display region's
    id (ignore_if_outside_of_state), in the same country. regionId is the calculator id
    with the country code removed and cast to int, so leading zeros are stripped from
    both sides before comparing. A failure means mining was attributed to a region in
    another state or country.
    """
    problems = []
    for r in dict_records(SUBNATIONAL):
        label = describe(SUBNATIONAL, r["id"])
        country = r.get("country_code")
        prefix = str(r.get("id_field")).replace(str(country), "").lstrip("0")
        for year, locations in (r.get("locations_per_year") or {}).items():
            for loc in locations:
                if loc["country"] != cleanup_country_code(country):
                    problems.append(f"{label} {year}: country {loc['country']} vs {country}")
                elif not str(loc["regionId"]).lstrip("0").startswith(prefix):
                    problems.append(
                        f"{label} {year}: regionId {loc['regionId']} outside {r.get('id_field')}"
                    )
    check(
        not problems,
        "Locations outside their subnational region",
        f"{len(problems)} location(s)",
        lines=problems,
    )


@pytest.mark.parametrize("ds", [INDIGENOUS, PROTECTED], ids=lambda d: d.name)
def test_locations_in_the_areas_own_country(ds):
    """An indigenous territory's or protected area's locations are in its own country.

    Warning only, for the newest period. prefer_domestic_rows keeps a fragment under a
    foreign area when no area of its own country covers it, so an area that crosses a
    border (or whose boundary overshoots it) can legitimately hold locations in the
    neighbouring country. Large amounts are worth a look at the boundaries.
    """
    foreign = []
    for r in dict_records(ds):
        own = cleanup_country_code(r.get("country_code"))
        latest = max(r.get("locations_per_year") or {"0": []}, key=int)
        for loc in (r.get("locations_per_year") or {}).get(latest, []):
            if loc["country"] != own:
                foreign.append((r["id"], loc["country"], loc["regionId"], loc["affectedArea"]))
    if foreign:
        df = pd.DataFrame(foreign, columns=["id", "location_country", "regionId", "affectedArea"])
        warn(
            "Latest locations in another country",
            f"{len(df)} location(s), {df['affectedArea'].sum():,.2f} ha in all",
            with_area_info(ds, df),
            sort_by="affectedArea",
        )


# ---------------------------------------------------------------------------
# .geojson
# ---------------------------------------------------------------------------

COMPUTED_PROPERTIES = ["mining_affected_area_ha", "illegality_areas", "locations_per_year"]


def same_value(a, b, tol=1e-6):
    """Deep equality that tolerates float noise between the two JSON writers."""
    if is_number(a) and is_number(b):
        return abs(a - b) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y, tol) for x, y in zip(a, b, strict=True))
    return a == b


@ds_param
@pytest.mark.parametrize("filtered", [False, True], ids=["unfiltered", "filtered"])
def test_geojson_top_level_ids(ds, filtered):
    """Every geojson feature has a top-level id equal to its properties.id.

    save_to_geojson copies the id to the top level so the map can address features by
    it (e.g. for feature state). A missing or different id breaks highlighting and
    lookups on the website.
    """
    data = load_json(ds.filtered_geojson if filtered else ds.unfiltered_geojson)
    bad = [
        f"{describe(ds, f['properties'].get('id'))}: top-level id {f.get('id')!r}"
        for f in data["features"]
        if f.get("id") is None or f.get("id") != f["properties"].get("id")
    ]
    check(
        not bad,
        "Features with a missing or different top-level id",
        f"{len(bad)} feature(s)",
        lines=bad,
    )


@ds_param
def test_unfiltered_geojson_matches_dict(ds):
    """The unfiltered geojson holds the same areas and values as the dict.

    Both are written from the same merged frame, the dict with a bbox instead of the
    geometry. Same ids, same property names, and the same mined total, illegality
    breakdown and locations, so checks on one file hold for the other.
    """
    records = by_id(dict_records(ds))
    problems = []
    for props in geojson_records(ds):
        label = describe(ds, props["id"])
        r = records.get(props["id"])
        if r is None:
            problems.append(f"{label}: not in the dict")
            continue
        if set(props) != set(r) - {"bbox"}:
            problems.append(
                f"{label}: properties differ {sorted(set(props) ^ (set(r) - {'bbox'}))}"
            )
        for key in COMPUTED_PROPERTIES:
            if key in props and not same_value(props[key], r.get(key)):
                problems.append(f"{label}: {key} differs")
    if len(geojson_records(ds)) != len(records):
        problems.append(f"{len(geojson_records(ds))} features vs {len(records)} dict records")
    check(
        not problems,
        "Unfiltered geojson differs from the dict",
        f"{len(problems)} problem(s)",
        lines=problems,
    )


@ds_param
def test_filtered_geojson_is_areas_with_mining(ds):
    """`_impacts.geojson` holds exactly the areas with mining, unchanged.

    It is the unfiltered geojson keeping only features with mining_affected_area_ha
    > 0, so the two must agree on those features' values.
    """
    unfiltered = by_id(geojson_records(ds))
    filtered = by_id(geojson_records(ds, filtered=True))
    expected = {i for i, p in unfiltered.items() if (p.get("mining_affected_area_ha") or 0) > 0}
    missing = sorted(expected - set(filtered))
    unexpected = sorted(set(filtered) - expected)
    check(
        not missing and not unexpected,
        "Filtered geojson isn't the areas with mining",
        f"{len(missing)} area(s) with mining missing, {len(unexpected)} unexpected",
        lines=[f"missing: {describe(ds, i)}" for i in missing]
        + [f"unexpected: {describe(ds, i)}" for i in unexpected],
    )
    differs = [
        describe(ds, i)
        for i, p in filtered.items()
        if i in unfiltered
        and not all(same_value(p.get(k), unfiltered[i].get(k)) for k in COMPUTED_PROPERTIES)
    ]
    check(
        not differs,
        "Filtered features differ from unfiltered ones",
        f"{len(differs)} feature(s)",
        lines=differs,
    )


# ---------------------------------------------------------------------------
# _impacts_yearly.csv
# ---------------------------------------------------------------------------


@ds_param
def test_yearly_csv_matches_yearly_json(ds):
    """The reference CSV has the same rows and values as the yearly json.

    It is the area attributes left-merged with the yearly rows, with the areas renamed
    to mining_affected_area_ha / mining_affected_area_ha_cumulative and the nested
    columns dropped. Areas without mining appear once, with no period.
    """
    csv = pd.read_csv(require(ds.yearly_csv), dtype={"id": str}, low_memory=False)
    required = ["id", "admin_year", "mining_affected_area_ha", "mining_affected_area_ha_cumulative"]
    missing = [c for c in required if c not in csv.columns]
    check(not missing, "Columns missing from the CSV", f"{missing} not in {ds.yearly_csv}")
    leftover = [
        c for c in ["locations_per_year", "illegality_areas", "geometry"] if c in csv.columns
    ]
    check(not leftover, "Columns that should have been dropped", f"{leftover} in {ds.yearly_csv}")

    df = yearly_or_empty(ds)
    with_year = csv[csv["admin_year"].notna()].copy()
    with_year["admin_year"] = with_year["admin_year"].astype(int)
    ids_without_yearly = {r["id"] for r in dict_records(ds)} - set(df["id"])
    check(
        len(with_year) == len(df),
        "CSV row count differs from the yearly json",
        f"{len(with_year)} rows with a period vs {len(df)} yearly rows",
    )
    check(
        csv["admin_year"].isna().sum() == len(ids_without_yearly),
        "Wrong number of areas without mining in the CSV",
        f"{int(csv['admin_year'].isna().sum())} rows without a period, expected "
        f"{len(ids_without_yearly)}",
    )

    csv_values = with_year[required].rename(
        columns={
            "mining_affected_area_ha": "csv_added_ha",
            "mining_affected_area_ha_cumulative": "csv_cumulative_ha",
        }
    )
    merged = csv_values.merge(df, on=["id", "admin_year"], how="outer", indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    check(
        unmatched.empty,
        "CSV rows not in the yearly json",
        f"{len(unmatched)} row(s)",
        with_area_info(ds, unmatched.drop(columns="_merge")),
    )
    merged["difference_ha"] = np.maximum(
        (merged["csv_added_ha"] - merged["intersected_area_ha"]).abs(),
        (merged["csv_cumulative_ha"] - merged["intersected_area_ha_cumulative"]).abs(),
    )
    wrong = merged[merged["difference_ha"] > 1e-6].drop(columns="_merge")
    check(
        wrong.empty,
        "CSV values differ from the yearly json",
        f"{len(wrong)} row(s)",
        with_area_info(ds, wrong),
        sort_by="difference_ha",
    )


# ---------------------------------------------------------------------------
# geometry sanity
# ---------------------------------------------------------------------------


@functools.cache
def _geometries(rel):
    data = _load_json(rel)
    ids = [f["properties"]["id"] for f in data["features"]]
    geoms = [shape(f["geometry"]) if f.get("geometry") else None for f in data["features"]]
    # GeoJSON written by geopandas is in lon/lat
    return gpd.GeoSeries(geoms, index=ids, crs="EPSG:4326")


def geometries(ds):
    require(ds.unfiltered_geojson)
    return _geometries(ds.unfiltered_geojson)


@ds_param
def test_mined_area_not_above_polygon_area(ds):
    """No area has more mining than its own surface.

    Mining is intersected with each area's polygon before being summed, so the mined
    total can't exceed the polygon's area (measured in EPSG:6933, with 1% + 1 ha of
    slack for the simplified geometries). A failure points at a unit error or at
    mining attributed to the wrong area.
    """
    geoms = geometries(ds)
    polygon_ha = geoms.to_crs(EQUAL_AREA_CRS).area / 1e4
    df = pd.DataFrame(
        [(r["id"], r.get("mining_affected_area_ha")) for r in dict_records(ds)],
        columns=["id", "mining_affected_area_ha"],
    ).dropna()
    df["polygon_area_ha"] = df["id"].map(polygon_ha)
    df["excess_ha"] = df["mining_affected_area_ha"] - (
        df["polygon_area_ha"] * (1 + POLYGON_AREA_SLACK_FRAC) + POLYGON_AREA_SLACK_HA
    )
    over = df[df["excess_ha"] > 0]
    check(
        over.empty,
        "Mined area larger than the area itself",
        f"{len(over)} area(s)",
        with_area_info(ds, over),
        sort_by="excess_ha",
    )


@ds_param
def test_dict_bbox_matches_geometry(ds):
    """The dict's bbox is the bounds of the area's geometry in the geojson.

    The website zooms to an area using this bbox, so a wrong one sends the map to the
    wrong place.
    """
    geoms = geometries(ds)
    problems = []
    for r in dict_records(ds):
        geom = geoms.get(r["id"])
        if geom is None or geom.is_empty:
            continue
        if not np.allclose(r["bbox"], geom.bounds, atol=1e-7):
            problems.append(
                f"{describe(ds, r['id'])}: bbox {r['bbox']} vs bounds {list(geom.bounds)}"
            )
    check(
        not problems, "bbox differs from the geometry", f"{len(problems)} area(s)", lines=problems
    )


# ---------------------------------------------------------------------------
# mining simplified files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "year,rel", list(zip(MINING_YEARS_QUARTERS, MINING_SIMPLIFIED_FILES, strict=True)), ids=str
)
def test_mining_simplified_file(year, rel):
    """Each period's simplified mining file, shown on the map, is non-empty and labelled.

    Every feature's year is the file's period. Empty or invalid geometries left by
    the simplification are reported as a warning.
    """
    gdf = gpd.read_file(require(rel))
    check(len(gdf) > 0, "Empty mining file", f"{rel} has no features")
    check("year" in gdf.columns, "No year column", f"{rel} has no year column")
    check(
        (gdf["year"] == year).all(),
        "Wrong years in a mining file",
        f"{rel} holds years {sorted(gdf['year'].unique())}, expected only {year}",
    )
    geoms = gdf.geometry.to_numpy()
    empty = int((shapely.is_missing(geoms) | shapely.is_empty(geoms)).sum())
    invalid = int((~shapely.is_valid(geoms) & ~shapely.is_missing(geoms)).sum())
    if empty or invalid:
        warn(
            "Empty or invalid geometries",
            f"{rel}: {empty} empty and {invalid} invalid out of {len(gdf)}",
        )


def test_mining_combined_file():
    """The combined mining file covers every period, with year stored as an int.

    The website filters this file by year, so a missing period or a year stored as text
    would hide mining.
    """
    df = gpd.read_file(require(COMBINED_MINING_FILE), columns=["year"], ignore_geometry=True)
    check(len(df) > 0, "Empty combined mining file", f"{COMBINED_MINING_FILE} has no features")
    check(
        pd.api.types.is_integer_dtype(df["year"]),
        "Year isn't an int",
        f"year is {df['year'].dtype} in {COMBINED_MINING_FILE}",
    )
    years = set(df["year"].unique())
    expected = set(MINING_YEARS_QUARTERS)
    check(
        years == expected,
        "Combined mining file periods differ from MINING_YEARS_QUARTERS",
        f"missing {sorted(expected - years)}, unexpected {sorted(years - expected)}",
    )


# ---------------------------------------------------------------------------
# regression against the published data (opt-in)
# ---------------------------------------------------------------------------


def _published_exists(folder):
    req = Request(f"{PUBLISHED_BASE}/{folder}/{PUBLISHED_SENTINEL}", method="HEAD")
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.status < 400
    except (URLError, OSError):
        return False


@functools.cache
def _resolve_published_folder(requested):
    if requested != "latest":
        folder = requested.replace("-", "")
        if not _published_exists(folder):
            raise RuntimeError(f"No published data at {PUBLISHED_BASE}/{folder}")
        return folder
    # the newest folder older than this run, so a run that was already uploaded
    # isn't compared with itself
    day = date(int(DATA_UPDATED_AT[:4]), int(DATA_UPDATED_AT[4:6]), int(DATA_UPDATED_AT[6:8]))
    for _ in range(MAX_PROBE_DAYS):
        day -= timedelta(days=1)
        if _published_exists(f"{day:%Y%m%d}"):
            return f"{day:%Y%m%d}"
    raise RuntimeError(f"No published folder in the {MAX_PROBE_DAYS} days before {DATA_UPDATED_AT}")


@functools.cache
def _fetch_published(folder, rel):
    with urlopen(f"{PUBLISHED_BASE}/{folder}/{rel}", timeout=120) as resp:
        return json.loads(resp.read())


def published(rel):
    """Fetch a published file, and record which folder and newest period are compared."""
    requested = os.environ.get("AMW_COMPARE_PUBLISHED")
    if not requested:
        pytest.skip("Set AMW_COMPARE_PUBLISHED=latest|YYYYMMDD to compare with published data")
    folder = _resolve_published_folder(requested)
    if not REPORT["published"]:
        national = _fetch_published(folder, NATIONAL.yearly)
        REPORT["published"] = {
            "requested": requested,
            "folder": folder,
            "newest_period": max((r["admin_year"] for r in national), default=None),
        }
    return folder, _fetch_published(folder, rel)


@ds_param
def test_regression_no_areas_removed(ds):
    """Areas that had mining in the published data still exist.

    Warning only: boundary updates can legitimately rename or drop areas, but an area
    disappearing unexpectedly breaks links to it on the website. The names come from
    the published records, since these areas are no longer in the current outputs.
    """
    folder, old = published(ds.dict_json)
    new_ids = {r["id"] for r in dict_records(ds)}
    removed = [
        {
            "id": r["id"],
            **describe_record(ds, r),
            "mining_affected_area_ha": r.get("mining_affected_area_ha"),
        }
        for r in old
        if (r.get("mining_affected_area_ha") or 0) > 0 and r["id"] not in new_ids
    ]
    if removed:
        warn(
            f"Areas with mining in {folder} are gone",
            f"{len(removed)} area(s) no longer in the outputs",
            pd.DataFrame(removed),
            sort_by="mining_affected_area_ha",
        )


@ds_param
def test_regression_past_periods_stable(ds):
    """Past periods' cumulative areas haven't moved much since the published data.

    New detections should only add a new period; the history should stay put unless
    the model or the boundaries changed. For the national dataset (countries and the
    Entire Amazon), moves over REGRESSION_FAIL_FRAC fail. For every dataset, areas that
    moved over REGRESSION_WARN_FRAC and REGRESSION_MIN_HA are warned about.
    """
    folder, old = published(ds.yearly)
    old = pd.DataFrame(old)
    new = yearly(ds)
    cmp = new.merge(old, on=["id", "admin_year"], suffixes=("", "_published"))
    cmp["change_ha"] = (
        cmp["intersected_area_ha_cumulative"] - cmp["intersected_area_ha_cumulative_published"]
    )
    abs_change = cmp["change_ha"].abs()
    cmp["change_frac"] = abs_change / cmp["intersected_area_ha_cumulative_published"].clip(
        lower=1e-9
    )
    cols = [
        "id",
        "admin_year",
        "intersected_area_ha_cumulative_published",
        "intersected_area_ha_cumulative",
        "change_ha",
        "change_frac",
    ]
    moved = with_area_info(ds, cmp[abs_change > REGRESSION_MIN_HA][cols])
    moved = moved.assign(_size=moved["change_ha"].abs()).sort_values("_size", ascending=False)
    moved = moved.drop(columns="_size")

    if ds is NATIONAL:
        big = moved[moved["change_frac"] > REGRESSION_FAIL_FRAC]
        check(
            big.empty,
            f"National totals for past periods moved since {folder}",
            f"{len(big)} (area, period) moved over {REGRESSION_FAIL_FRAC:.0%}",
            big,
        )
    noticeable = moved[moved["change_frac"] > REGRESSION_WARN_FRAC]
    if not noticeable.empty:
        warn(
            f"Past periods moved since {folder}",
            f"{noticeable['id'].nunique()} area(s) moved over {REGRESSION_WARN_FRAC:.0%} "
            f"and {REGRESSION_MIN_HA} ha",
            noticeable,
        )


def test_regression_latest_total_not_lower():
    """The newest Entire Amazon total is not lower than the newest published one.

    Each data update adds detections, so the basin-wide total at the newest period
    should be at least the previous publish's. Small drops warn, larger ones fail.
    """
    folder, old = published(NATIONAL.yearly)
    old = pd.DataFrame(old)
    new = yearly(NATIONAL)

    def latest_amazon(df):
        amazon = df[df["id"] == ENTIRE_AMAZON_ID]
        row = amazon[amazon["admin_year"] == amazon["admin_year"].max()].iloc[0]
        return int(row["admin_year"]), float(row["intersected_area_ha_cumulative"])

    old_year, old_total = latest_amazon(old)
    new_year, new_total = latest_amazon(new)
    drop = pd.DataFrame(
        [
            {
                "published_period": old_year,
                "published_ha": old_total,
                "current_period": new_year,
                "current_ha": new_total,
                "drop_ha": old_total - new_total,
            }
        ]
    )
    check_dips(
        drop[drop["drop_ha"] > 0],
        f"Newest {ENTIRE_AMAZON_ID} total lower than in {folder}",
        "drop_ha",
        "published_ha",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
