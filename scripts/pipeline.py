#!/usr/bin/env python3
"""
Driver for the detection -> mask -> published-product pipeline.

Runs on the VM. Stages are idempotent (skip what exists) and verify their own
output, because the failure modes here are silent: in one week we saw
``gsutil cp -I`` report success while copying 2 of 15,752 files, ``sam2_mask.py``
default to 2023 imagery when dates were omitted, and shell globs fail mid-list at
ARG_MAX. None announced itself; all are caught by counting outputs.

Some stages are **run by a human**, not by this tool -- detection and SAM2
inference take hours across parallel tmux sessions on several VMs. For those the
driver *emits* commands with every parameter derived from the period list, so
dates, cache directories and boundary arguments are never typed. ``publish``
is emitted rather than executed for a different reason: it is outward-facing, and
the rasters want reviewing first.

    python scripts/pipeline.py --list
    python scripts/pipeline.py inference --periods 2026-Q3
    python scripts/pipeline.py postprocess persist-detections
    python scripts/pipeline.py publish

Design and layout: docs/design/pipeline.md
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "core"))

from persistence import Period

MODEL = "48px_v4.10b-18d-20g-21a-22bc-ensemble"
BASE = REPO / "data/outputs" / MODEL
SAM2 = REPO / "data/outputs/sam2"
CORE = REPO / "core"
SCRIPTS = REPO / "scripts"

DEFAULT_PERIODS = [str(y) for y in range(2018, 2026)] + [
    "Q125", "Q225", "Q325", "Q425", "Q126", "Q226"]
SUBREGIONS = [1, 2, 3, 4, 5, 6]

HUMAN = {"inference", "mask-annual", "mask-quarterly", "publish"}
ORDER = ["inference", "concat", "filter", "postprocess", "persist-detections",
         "mask-annual", "mask-quarterly", "cog", "persist-masks", "stage",
         "manifest", "publish"]


def dates(tag: str):
    return Period.parse(tag).date_span.split("_")


def cache_dir(tag: str) -> str:
    """Image cache naming from the operator's runbook; year-granular."""
    return f"/mnt/tempdisk/amw_image_cache{Period.parse(tag).year}_552-12/"


# --------------------------------------------------------------------------
# emitted commands (human-run)
# --------------------------------------------------------------------------

def cmds_inference(periods: Sequence[str]) -> List[str]:
    out = []
    for tag in periods:
        s, e = dates(tag)
        for r in SUBREGIONS:
            out.append(
                f"python inference_pipeline.py --model ../models/{MODEL}.h5 "
                f"--region_path ../data/boundaries/Amazon_ACA/Amazon_ACA_{r}.geojson "
                f"--start_date {s} --end_date {e} "
                f"--image_cache_dir {cache_dir(tag)} --pred_threshold 0.4 --tries 3")
        out.append(
            f"python inference_pipeline.py --model ../models/{MODEL}.h5 "
            f"--region_path ../data/boundaries/andes_supplemental.geojson "
            f"--start_date {s} --end_date {e} "
            f"--image_cache_dir {cache_dir(tag)} --pred_threshold 0.2 --tries 3")
    return out


def cmds_mask_annual(periods: Sequence[str]) -> List[str]:
    out = []
    for tag in (t for t in periods if Period.parse(t).is_annual):
        s, e = dates(tag)
        pp = (f"../data/outputs/{MODEL}/postprocessed_t0.43_d5_3km_t-iso0.75/"
              f"Amazon_ACA_{MODEL}_0.40_{s}_{e}_t0.43_d5_3km_t-iso0.75.geojson")
        andes = (f"../data/outputs/{MODEL}/raw_detections/andes_supplemental/"
                 f"andes_supplemental_{MODEL}_0.2_{s}_{e}.geojson")
        out.append(f"python sam2_mask.py {pp} --start_date {s} --end_date {e} "
                   f"--cog --image_cache_dir {cache_dir(tag)}")
        out.append(f"python sam2_mask.py {andes} --start_date {s} --end_date {e} "
                   f"--cog --image_cache_dir {cache_dir(tag)}")
    return out


def cmds_mask_quarterly(periods: Sequence[str]) -> List[str]:
    out = []
    for tag in (t for t in periods if not Period.parse(t).is_annual):
        s, e = dates(tag)
        diff = (f"../data/outputs/{MODEL}/cumulative/patch_diffs/"
                f"Amazon_ACA_{MODEL}_growth_{tag}.geojson")
        out.append(f"python sam2_mask.py {diff} --start_date {s} --end_date {e} "
                   f"--cog --image_cache_dir {cache_dir(tag)}")
    return out


def cmds_publish(_periods: Sequence[str]) -> List[str]:
    """Emitted, never executed: outward-facing, and the rasters want review."""
    # amw-published is the store of record and is versioned; amw-dev/published is
    # its backup, synced bucket-to-bucket so nothing round-trips through a laptop
    # -- every silent transfer failure we have hit was a local<->bucket sync.
    record = "gs://amw-published"
    backup = "gs://amw-dev/published"
    coop = "s3://earthgenome/amazon-mining-watch"
    return [
        "# review the rasters before running any of this",
        "",
        "# 1. store of record",
        f"gsutil -m rsync -r data/staging_gs/ {record}/",
        "# counts must match: cp -I has reported success while copying 2 of",
        "# 15,752 files, and an unquoted glob exceeds ARG_MAX mid-list",
        f"gsutil ls '{record}/**' | wc -l",
        "find data/staging_gs -type f | wc -l",
        "",
        "# 2. backup, server-side (no egress, no local round trip)",
        f"gsutil -m rsync -r -d {record}/ {backup}/",
        "",
        "# 3. public subset.",
        "#",
        "#    FIRST: Source Cooperative issues temporary, AWS-compatible",
        "#    credentials -- there is no long-lived key to configure. Log in at",
        "#    https://source.coop, open the earthgenome/amazon-mining-watch",
        "#    repository, and copy the credential block it shows into this shell:",
        "#",
        "#      export AWS_ACCESS_KEY_ID=...",
        "#      export AWS_SECRET_ACCESS_KEY=...",
        "#      export AWS_SESSION_TOKEN=...",
        "#      export AWS_DEFAULT_REGION=\"us-east-1\"",
        "#      export AWS_ENDPOINT_URL=\"https://data.source.coop\"",
        "#",
        "#    All five matter. AWS_ENDPOINT_URL is what points the CLI at Source",
        "#    Cooperative -- without it the commands below address real AWS S3",
        "#    instead, and either fail on credentials or reach a bucket that is",
        "#    not ours.",
        "#",
        "#    The first three expire, and 1.4 GB is long enough to outlive them.",
        "#    If the upload dies with an auth error, re-copy the block and re-run:",
        "#    sync compares size and mtime, so it resumes instead of restarting.",
        "#",
        "#    sync, not cp --recursive, for that reason and because a quarterly",
        "#    refresh changes only a handful of files. No --delete: anything",
        "#    removed from the staging tree stays on the bucket until deleted by",
        "#    hand, which is what keeps archived/ -- present on the bucket, absent",
        "#    from staging -- from being swept away.",
        f"aws s3 sync data/staging_source-coop/ {coop}/",
        "# archived/ lives on the bucket but not in staging, so exclude it or",
        "#  the two counts can never agree and the check gets ignored",
        f"aws s3 ls --recursive {coop}/ | grep -v '/archived/' | wc -l",
        "find data/staging_source-coop -type f | wc -l",
    ]


EMITTERS = {"inference": cmds_inference, "mask-annual": cmds_mask_annual,
            "mask-quarterly": cmds_mask_quarterly, "publish": cmds_publish}


# --------------------------------------------------------------------------
# executed stages
# --------------------------------------------------------------------------

def run(cmd: List[str], dry: bool) -> int:
    # flush before handing the terminal to a child: our stdout is buffered when
    # piped, the child's is not, so without this the stage headers and the "$ cmd"
    # lines land after the output they introduce, and a multi-stage log becomes
    # impossible to attribute.
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def stage_concat(periods, dry) -> int:
    out = BASE / "raw_detections"
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for tag in periods:
        s, e = dates(tag)
        dest = out / f"Amazon_ACA_{MODEL}_0.40_{s}_{e}.geojson"
        if dest.is_file():
            continue
        parts = sorted(glob.glob(str(BASE / f"Amazon_ACA_?_{MODEL}_0.40_{s}_{e}.geojson")))
        if not parts:
            print(f"    {tag}: no subregion parts")
            continue
        rc |= run([sys.executable, str(SCRIPTS / "concatenate.py"), *parts,
                   "--outpath", str(dest)], dry)
    return rc


def stage_filter(periods, dry) -> int:
    """Clip andes detections to their boundary, writing straight to their home.

    ``geo_filter.py`` requires ``--outpath``, so the destination is passed rather
    than reconstructed from a ``-filt`` sibling afterwards: the filtered file is
    written once, in the one place both ``persistence.py`` (via
    ``--supplemental_dir``) and ``stage_outputs.raw_andes`` look for it. The
    ``0.20`` -> ``0.2`` shift between the inference output and the filtered name is
    the archive's existing convention, and is now visible in the command instead of
    buried in a rename.
    """
    out = BASE / "raw_detections" / "andes_supplemental"
    out.mkdir(parents=True, exist_ok=True)
    boundary = REPO / "data/boundaries/andes_supplemental.geojson"
    rc = 0
    for tag in periods:
        s, e = dates(tag)
        dest = out / f"andes_supplemental_{MODEL}_0.2_{s}_{e}.geojson"
        if dest.is_file():
            continue
        src = BASE / f"andes_supplemental_{MODEL}_0.20_{s}_{e}.geojson"
        if not src.is_file():
            print(f"    {tag}: no andes inference output")
            continue
        rc |= run([sys.executable, str(SCRIPTS / "geo_filter.py"),
                   str(src), str(boundary), "--outpath", str(dest)], dry)
    return rc


def stage_postprocess(periods, dry) -> int:
    rc = 0
    for t_main, t_iso in (("0.43", "0.75"), ("0.55", "0.8")):
        target = BASE / f"postprocessed_t{t_main}_d5_3km_t-iso{t_iso}"
        target.mkdir(parents=True, exist_ok=True)
        for tag in periods:
            s, e = dates(tag)
            name = (f"Amazon_ACA_{MODEL}_0.40_{s}_{e}"
                    f"_t{t_main}_d5_3km_t-iso{t_iso}.geojson")
            if (target / name).is_file():
                continue
            src = BASE / "raw_detections" / f"Amazon_ACA_{MODEL}_0.40_{s}_{e}.geojson"
            if not src.is_file():
                print(f"    {tag}: no raw detections")
                continue
            rc |= run([sys.executable, str(CORE / "postprocess.py"), str(src),
                       "--t-main", t_main, "--t-iso", t_iso], dry)
            produced = src.with_name(name)
            if produced.is_file() and not dry:
                produced.rename(target / name)
    return rc


def stage_persist_detections(periods, dry) -> int:
    years = [t for t in periods if Period.parse(t).is_annual]
    quarters = [t for t in periods if not Period.parse(t).is_annual]
    # --base explicitly, though persistence.py has a default: that default is
    # "../data/outputs/<model>", relative to the caller's cwd, so it only
    # resolves when run from core/. BASE is absolute and works from anywhere.
    return run([sys.executable, str(CORE / "persistence.py"),
                "--base", str(BASE),
                "--years", *years, "--quarters", *quarters, "--dissolve"], dry)


def stage_cog(periods, dry) -> int:
    rc = 0
    for d in sorted(SAM2.iterdir()):
        if not d.is_dir() or d.name.endswith("-logits-unsmoothed"):
            continue
        if not glob.glob(str(d / "*-msk.tif")):
            continue
        if glob.glob(str(d / "cog_outputs" / "*utm*.tif")):
            continue
        rc |= run([sys.executable, str(CORE / "sam2_build_cog.py"), str(d)], dry)
    return rc


#: Band group in a per-band mask COG name, e.g. utm21_lat_-16_-8.
GROUP_RE = re.compile(
    r"^mining_mask_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_(utm\S+?)_epsg4326\.tif$")


def stage_persist_masks(periods, dry) -> int:
    quarters = [t for t in periods if not Period.parse(t).is_annual]
    # mining_mask_<start>_<end>_<group>_epsg4326.tif. Matched explicitly rather
    # than by maxsplit on "_": the group itself contains underscores, so a
    # maxsplit off by one silently yields "2018-12-31_utm17_lat_-8_0" and every
    # downstream glob then matches nothing while the run still reports success.
    names = [p.name for p in SAM2.glob("*/cog_outputs/mining_mask_*utm*.tif")]
    groups = sorted({m.group(1) for m in (GROUP_RE.match(n) for n in names) if m})
    unparsed = [n for n in names if not GROUP_RE.match(n)]
    if unparsed or not groups:
        raise SystemExit(
            f"could not parse band group from {len(unparsed)} filename(s), "
            f"e.g. {unparsed[:2]}; got {len(groups)} groups")
    rc = 0
    for g in groups:
        rc |= run([sys.executable, str(CORE / "sam2_persistence.py"),
                   "--group", g, "--quarters", *quarters,
                   "--outdir", str(SAM2 / "persistence_masks")], dry)
    rc |= run([sys.executable, str(CORE / "sam2_persistence.py"), "--mosaic",
               str(SAM2 / "persistence_masks"
                   / "amazon_basin_mining_scar_masks_first_year.tif"),
               "--outdir", str(SAM2 / "persistence_masks")], dry)
    return rc


def stage_stage(periods, dry) -> int:
    return run([sys.executable, str(SCRIPTS / "stage_outputs.py"),
                "--periods", *periods], dry)


def stage_manifest(periods, dry) -> int:
    print("    MANIFEST.yaml is hand-maintained; update `updated`, `periods`,")
    print("    and any path_map changes. See data/outputs/MANIFEST.yaml.")
    return 0


RUNNERS = {"concat": stage_concat, "filter": stage_filter,
           "postprocess": stage_postprocess,
           "persist-detections": stage_persist_detections,
           "cog": stage_cog, "persist-masks": stage_persist_masks,
           "stage": stage_stage, "manifest": stage_manifest}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help=f"one or more of: {', '.join(ORDER)}")
    ap.add_argument("--periods", nargs="+", default=DEFAULT_PERIODS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="show the stage order")
    args = ap.parse_args()

    if args.list or not args.stages:
        print("stages, in dependency order:\n")
        for s in ORDER:
            who = "HUMAN (prints commands)" if s in HUMAN else "pipeline"
            print(f"  {s:<20} {who}")
        print("\nmask-quarterly depends on persist-detections: its prompts are "
              "patch_diffs/.")
        return

    # Validate every name first: a bad stage late in the list would otherwise
    # abort after earlier stages had already run for real.
    unknown = [s for s in args.stages if s not in ORDER]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; see --list")

    for s in args.stages:
        print(f"\n=== {s}", flush=True)
        if s in HUMAN:
            # These are long VM jobs, or outward-facing. Printing is the only
            # thing this driver does with them, so it needs no flag to ask for it.
            for line in EMITTERS[s](args.periods):
                print(line, flush=True)
            continue
        rc = RUNNERS[s](args.periods, args.dry_run)
        if rc:
            raise SystemExit(f"{s} failed (rc={rc}); not continuing")


if __name__ == "__main__":
    main()
