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

from periods import Period
from pipeline_config import (ALL_CURRENT_PERIODS, ANDES_RAW_TAG, ANDES_TAG,
                             ANDES_THRESHOLD, BASE, CORE, ISOLATION_KM, LOOSE,
                             MODEL, NEIGHBOURS, RAW_TAG, RAW_THRESHOLD, SAM2,
                             SCRIPTS, STRINGENT, SUBREGIONS, postprocess_tag)

LOOSE_TAG = postprocess_tag(*LOOSE)
STRINGENT_TAG = postprocess_tag(*STRINGENT)

HUMAN = {"review-config", "inference", "mask-annual", "mask-quarterly", "publish"}

#: These recompute from the whole history rather than from the periods named on
#: the command line, so they read ALL_CURRENT_PERIODS. Passing them one period
#: would compute onset with nothing to corroborate against.
WHOLE_HISTORY = {"persist-detections", "persist-masks", "stage", "manifest"}

ORDER = ["review-config", "inference", "concat", "filter", "postprocess", "persist-detections",
         "mask-annual", "mask-quarterly", "cog", "persist-masks", "stage",
         "manifest", "publish"]


def dates(tag: str):
    return Period.parse(tag).date_span.split("_")


def cache_dir(tag: str) -> str:
    """Image cache naming from the operator's runbook, one directory per period.

    The 552-12 suffix is DataConfig's tilesize and pad, which fix the tile
    geometry a cached tile was cut to; a cache is only reusable by a run that
    agrees on both.
    """
    return f"/mnt/tempdisk/amw_image_cache{Period.parse(tag).tag}_552-12/"


# --------------------------------------------------------------------------
# emitted commands (human-run)
# --------------------------------------------------------------------------

def cmds_review_config(periods: Sequence[str]) -> List[str]:
    """Stage 0: what a human sets, and where."""
    annual = [p for p in ALL_CURRENT_PERIODS if Period.parse(p).is_annual]
    quarters = [p for p in ALL_CURRENT_PERIODS if not Period.parse(p).is_annual]
    return [
        "# 0. Review core/pipeline_config.py before running anything.",
        "#",
        f"#    MODEL      {MODEL}",
        f"#    PERIODS    {len(annual)} annual   {' '.join(annual)}",
        f"#               {len(quarters)} quarterly {' '.join(quarters)}",
        f"#    SUBREGIONS {SUBREGIONS}",
        f"#    thresholds raw {RAW_THRESHOLD:g} (basin) / {ANDES_THRESHOLD:g} (andes)",
        f"#               loose     {LOOSE_TAG}",
        f"#               stringent {STRINGENT_TAG}",
        "#",
        "#    Add the period you are about to run, then pass it as --periods.",
        "#    Every --periods value must be a member of the list: "
        "persist-detections,",
        "#    persist-masks and stage read it rather than --periods, because they",
        "#    recompute from the whole history. A missing period is invisible to",
        "#    them and will not reach the product. --all uses the whole list.",
        "#",
        "#    Those values appear in filenames, so every stage must agree on them.",
        "#    Changing one on an emitted command line does not reconfigure the",
        "#    pipeline -- the next stage looks for a file that was never written.",
        "#",
        "#    Parameters that change a computation but not a path -- pad, tilesize,",
        "#    clear_threshold in core/gee.py::DataConfig; prior_sigma,",
        "#    smoothing_sigma in MaskConfig -- are edited in the dataclass. The",
        "#    outputs will be named identically to any produced before the change,",
        "#    and nothing in the published provenance records it, so note it.",
        "#",
        "#    To explore a parameter rather than change the product, call the",
        "#    underlying script directly with --outdir somewhere separate.",
    ]


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
        pp = (f"../data/outputs/{MODEL}/postprocessed_{LOOSE_TAG}/"
              f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}_{LOOSE_TAG}.geojson")
        andes = (f"../data/outputs/{MODEL}/raw_detections/andes_supplemental/"
                 f"andes_supplemental_{MODEL}_{ANDES_TAG}_{s}_{e}.geojson")
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


def cmds_publish(periods: Sequence[str]) -> List[str]:
    """Emitted, never executed: outward-facing, and the rasters want review.

    Checks the staging trees before emitting anything. Whatever is in them gets
    published, and they are assembled fresh each run, so a file stage() did not
    put there is a file that should not go out -- a quarto preview once left
    README.html and twelve bootstrap assets in the public tree.
    """
    # amw-published is the store of record and is versioned; amw-dev/published is
    # its backup, synced bucket-to-bucket so nothing round-trips through a laptop
    # -- every silent transfer failure we have hit was a local<->bucket sync.
    record = "gs://amw-published"
    backup = "gs://amw-dev/published"
    coop = "s3://earthgenome/amazon-mining-watch"
    warn: List[str] = []
    try:
        from stage_outputs import check_trees
        report = check_trees(periods)
    except Exception as exc:                      # never block the emit on this
        warn = [f"# NOTE: could not check the staging trees ({exc})", ""]
        report = {}

    def _lines(kind: str, files: List[str], tree: str, note: str) -> List[str]:
        out = [f"# WARNING: {len(files)} {kind} file(s) in {tree}/. {note}"]
        out += [f"#     {f}" for f in files[:10]]
        if len(files) > 10:
            out.append(f"#     ... and {len(files) - 10} more")
        return out + [""]

    for tree, r in report.items():
        if r["stray"]:
            warn += _lines("unexpected", r["stray"], tree.name,
                           "Staging did not put these here and they WILL be "
                           "published; remove them.")
        if r["missing"]:
            warn += _lines("MISSING", r["missing"], tree.name,
                           "Expected but absent -- an incomplete product would go "
                           "out. Re-run `pipeline.py stage`.")

    return warn + [
        "# review the rasters before running any of this",
        "",
        "# 1. store of record",
        "# add -c if objects were pre-populated by cp: rsync compares mtime, cp",
        "# does not set it, so those objects re-upload however identical they are",
        f"gsutil -m rsync -r data/staging_gs/ {record}/",
        "# Verify by name, not by count. Two different tools have silently",
        "# dropped files on this project: gsutil cp -I reported success having",
        "# copied 2 of 15,752, and aws s3 sync dropped 2 of 48. A count tells",
        "# you something is missing; this tells you which. Empty output = clean.",
        f"diff <(cd data/staging_gs && find . -type f | sed 's|^\./||' | sort) \\",
        f"     <(gsutil ls '{record}/**' | sed 's|^{record}/||' | sort)",
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
        "# Same verification. archived/ is on the bucket and not in staging, so",
        "# it is excluded -- otherwise the comparison can never come out clean",
        "# and the check gets ignored. Empty output = clean; re-run the sync for",
        "# anything listed, which is what the 2-of-48 drop needed.",
        f"diff <(cd data/staging_source-coop && find . -type f | sed 's|^\./||' | sort) \\",
        f"     <(aws s3 ls --recursive {coop}/ | awk '{{print $4}}' \\",
        f"        | sed 's|^amazon-mining-watch/||' | grep -v '^archived/' | sort)",
    ]


EMITTERS = {"review-config": cmds_review_config,
            "inference": cmds_inference, "mask-annual": cmds_mask_annual,
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
    missing: List[str] = []
    for tag in periods:
        s, e = dates(tag)
        dest = out / f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}.geojson"
        if dest.is_file():
            continue
        pattern = f"Amazon_ACA_?_{MODEL}_{RAW_TAG}_{s}_{e}.geojson"
        parts = sorted(glob.glob(str(BASE / pattern)))
        if len(parts) not in (0, len(SUBREGIONS)):
            # A partial concatenation is named exactly like a complete basin and
            # every later stage accepts it, so this stops rather than warns.
            raise SystemExit(
                f"{tag}: {len(parts)} of {len(SUBREGIONS)} subregion parts.\n"
                f"  Concatenating these would write a basin file missing whole "
                f"regions,\n  under a name nothing downstream can distinguish "
                f"from a complete one.\n"
                f"  Found: {[Path(x).name for x in parts]}")
        if not parts:
            print(f"    {tag}: no subregion parts matching {pattern}")
            missing.append(tag)
            continue
        rc |= run([sys.executable, str(SCRIPTS / "concatenate.py"), *parts,
                   "--outpath", str(dest)], dry)
    if missing:
        print(f"    {len(missing)} period(s) had no inference output: "
              f"{', '.join(missing)}")
        rc |= 1
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
        dest = out / f"andes_supplemental_{MODEL}_{ANDES_TAG}_{s}_{e}.geojson"
        if dest.is_file():
            continue
        src = BASE / f"andes_supplemental_{MODEL}_{ANDES_RAW_TAG}_{s}_{e}.geojson"
        if not src.is_file():
            print(f"    {tag}: no andes inference output at {src}")
            continue
        rc |= run([sys.executable, str(SCRIPTS / "geo_filter.py"),
                   str(src), str(boundary), "--outpath", str(dest)], dry)
    return rc


def stage_postprocess(periods, dry) -> int:
    rc = 0
    for t_main, t_iso in (LOOSE, STRINGENT):
        tag_ = postprocess_tag(t_main, t_iso)
        target = BASE / f"postprocessed_{tag_}"
        target.mkdir(parents=True, exist_ok=True)
        for tag in periods:
            s, e = dates(tag)
            name = f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}_{tag_}.geojson"
            if (target / name).is_file():
                continue
            src = BASE / "raw_detections" / f"Amazon_ACA_{MODEL}_{RAW_TAG}_{s}_{e}.geojson"
            if not src.is_file():
                print(f"    {tag}: no raw detections at {src}")
                continue
            rc |= run([sys.executable, str(CORE / "postprocess.py"), str(src),
                       "--t-main", f"{t_main:g}", "--t-iso", f"{t_iso:g}",
                       "--k", str(NEIGHBOURS), "--D", f"{ISOLATION_KM:g}"], dry)
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


def run_dirs(tag: str) -> List[Path]:
    """SAM2 run directories for one period.

    An annual period is segmented twice, basin and andes supplemental; a quarter
    once, from patch_diffs. sam2_mask.py names each directory after the detections
    file that prompted it, so the names are a function of the period.
    """
    period = Period.parse(tag)
    if period.is_annual:
        span = period.date_span
        return [SAM2 / f"Amazon_ACA_{MODEL}_{RAW_TAG}_{span}_{LOOSE_TAG}",
                SAM2 / f"andes_supplemental_{MODEL}_{ANDES_TAG}_{span}"]
    return [SAM2 / f"Amazon_ACA_{MODEL}_growth_{tag}"]


def stage_cog(periods, dry) -> int:
    """COG the run directories for the given periods.

    Named rather than discovered: data/outputs/sam2 also holds prior vintages and
    test output, and a stage that walks the directory has to recognise those to
    leave them alone.
    """
    rc = 0
    for tag in periods:
        for d in run_dirs(tag):
            if not d.is_dir():
                print(f"    {tag}: no run directory {d.name}")
                continue
            if glob.glob(str(d / "cog_outputs" / "*utm*.tif")):
                continue
            if not glob.glob(str(d / "*-msk.tif")):
                print(f"    {tag}: {d.name} holds no mask tiles")
                continue
            rc |= run([sys.executable, str(CORE / "sam2_build_cog.py"), str(d)], dry)
    return rc


#: Band group in a per-band mask COG name, e.g. utm21_lat_-16_-8.
GROUP_RE = re.compile(
    r"^mining_mask_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_(utm\S+?)_epsg4326\.tif$")


def stage_persist_masks(periods, dry) -> int:
    quarters = [t for t in periods if not Period.parse(t).is_annual]
    # mining_mask_<start>_<end>_<group>_epsg4326.tif. Matched by pattern rather
    # than by splitting on "_": the group name contains underscores itself, and a
    # wrong split yields a plausible string that matches nothing downstream.
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
                   / "amazon_basin_mining_scar_masks.tif"),
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
    ap.add_argument("--periods", nargs="+", default=None,
                    help=("Periods this run works on, e.g. --periods Q326, or "
                          "--periods 2026 Q127. Each must be a member of "
                          "ALL_CURRENT_PERIODS in core/pipeline_config.py. No default: "
                          "a stale list is the one silent failure here."))
    ap.add_argument("--all", action="store_true", dest="use_all",
                    help="Work on every period in ALL_CURRENT_PERIODS (full rebuild)")
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

    # review-config and the whole-history stages read ALL_CURRENT_PERIODS
    # directly, so only the remaining stages need --periods.
    consumers = [s for s in args.stages
                 if s not in WHOLE_HISTORY and s != "review-config"]

    if args.use_all:
        working = list(ALL_CURRENT_PERIODS)
    elif not consumers:
        working = list(ALL_CURRENT_PERIODS)      # not consumed by these stages
    elif args.periods:
        # Membership is required, not advisory: the whole-history stages read
        # ALL_CURRENT_PERIODS, so a period outside it silently never reaches the
        # product. Better to refuse than to half-run it.
        stray = [p for p in args.periods if p not in ALL_CURRENT_PERIODS]
        if stray:
            raise SystemExit(
                f"period(s) {stray} are not in ALL_CURRENT_PERIODS.\n"
                f"  Add them to core/pipeline_config.py first -- see "
                f"`pipeline.py review-config`.\n"
                f"  Without that, persist-detections / persist-masks / stage "
                f"cannot see them.")
        working = list(args.periods)
    else:
        raise SystemExit(
            f"--periods is required for {consumers} (or --all for every period).\n"
            f"  It must name periods listed as ALL_CURRENT_PERIODS in "
            f"core/pipeline_config.py,\n"
            f"  which currently holds {len(ALL_CURRENT_PERIODS)}: "
            f"{' '.join(ALL_CURRENT_PERIODS)}\n"
            f"  e.g. --periods {ALL_CURRENT_PERIODS[-1]:<10} one period\n"
            f"       {'--all':<20} a full rebuild\n"
            f"  `pipeline.py review-config` prints this list and what depends "
            f"on it.")

    for s in args.stages:
        print(f"\n=== {s}", flush=True)
        periods = ALL_CURRENT_PERIODS if s in WHOLE_HISTORY else working
        if s in HUMAN:
            # These are long VM jobs, or outward-facing. Printing is the only
            # thing this driver does with them, so it needs no flag to ask for it.
            for line in EMITTERS[s](periods):
                print(line, flush=True)
            continue
        rc = RUNNERS[s](periods, args.dry_run)
        if rc:
            raise SystemExit(f"{s} failed (rc={rc}); not continuing")


if __name__ == "__main__":
    main()
