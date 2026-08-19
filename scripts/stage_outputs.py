#!/usr/bin/env python3
"""
Lay out the published product trees, with clean names and provenance sidecars.

Two trees, neither ever checked in:

  data/staging_source-coop/   public: patches only
  data/staging_gs/            internal: the above plus cumulatives, dissolves, SAM2

Consumer filenames drop the model version and the isolation parameters --
``Amazon_ACA_48px_v4.10b-..._0.40_2018-01-01_2018-12-31_t0.43_d5_3km_t-iso0.75``
becomes ``amazon_basin_2018_t0.43_t-iso0.75`` -- so each directory gets a
``config.txt`` recording what produced it. That is not redundant with
MANIFEST.yaml: **the data leaves the repo and the manifest does not**, and a
directory copied out of context has to stay identifiable. Two vintage-ambiguity
incidents in one week argued for this; see docs/design/pipeline.md.

Thresholds in names are trimmed of trailing zeros -- ``t0.4``, ``t-iso0.8``, not
``t0.40``/``t-iso0.80`` -- consistently across raw, postprocessed and directory
names.

Copies rather than links: the staged trees are self-contained artifacts to push,
and a hardlink would make editing a staged file silently edit the source.

Idempotent -- a destination of the right size is left alone -- and it verifies
that every expected period is present rather than staging whatever it finds.

Usage:
    python scripts/stage_outputs.py --periods 2018 2019 ... Q226 [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "core"))

from persistence import Period
from postprocess import PostprocessConfig
# From sam2_logits, not inference_engine: the numbers are the same, and
# inference_engine imports tensorflow, which staging has no use for.
from sam2_logits import DEFAULT_LOGIT_CLAMP, DEFAULT_SMOOTHING_SIGMA

MODEL = "48px_v4.10b-18d-20g-21a-22bc-ensemble"
BASE = REPO / "data/outputs" / MODEL
SAM2 = REPO / "data/outputs/sam2"
SOURCE_COOP = REPO / "data/staging_source-coop"
GS = REPO / "data/staging_gs"

DEFAULT_PERIODS = [str(y) for y in range(2018, 2026)] + [
    "Q125", "Q225", "Q325", "Q425", "Q126", "Q226"]


#: Periods that appear as *published* cumulative layers. An annual period covered
#: by its own quarters is a witness, not a layer -- 2025 lets 2024 confirm but is
#: not published, since coverage past the last confirmed year arrives quarterly.
#: Mirrors persistence.published_periods; kept as data because staging must not
#: ask for a file the pipeline deliberately does not write.
def published_periods(periods):
    quartered = {Period.parse(t).year for t in periods
                 if not Period.parse(t).is_annual}
    return [t for t in periods
            if not (Period.parse(t).is_annual and Period.parse(t).year in quartered)]


@dataclass
class Product:
    """One directory of the staged output."""
    name: str
    dest: Path                       # relative to each tree it appears in
    trees: Sequence[Path]
    src_for: Callable[[str], Optional[Path]]   # period tag -> source file
    rename: Callable[[str], str]               # period tag -> destination name
    per_period: bool = True
    sidecar: Optional[str] = None
    #: Restrict to periods that are published as layers.
    published_only: bool = False


def span(tag: str) -> str:
    return Period.parse(tag).date_span


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def raw_amazon(tag: str) -> Optional[Path]:
    p = BASE / "raw_detections" / f"Amazon_ACA_{MODEL}_0.40_{span(tag)}.geojson"
    return p if p.is_file() else None


def raw_andes(tag: str) -> Optional[Path]:
    p = (BASE / "raw_detections" / "andes_supplemental"
         / f"andes_supplemental_{MODEL}_0.2_{span(tag)}.geojson")
    return p if p.is_file() else None


def postprocessed(tag: str, t_main: str = "0.43", t_iso: str = "0.75"
                  ) -> Optional[Path]:
    d = BASE / f"postprocessed_t{t_main}_d5_3km_t-iso{t_iso}"
    p = d / (f"Amazon_ACA_{MODEL}_0.40_{span(tag)}"
             f"_t{t_main}_d5_3km_t-iso{t_iso}.geojson")
    return p if p.is_file() else None


def postprocessed_strict(tag: str) -> Optional[Path]:
    return postprocessed(tag, "0.55", "0.8")


def cumulative(tag: str) -> Optional[Path]:
    p = BASE / "cumulative" / f"Amazon_ACA_{MODEL}_cumulative2018-{tag}.geojson"
    return p if p.is_file() else None


def cumulative_dissolved(tag: str) -> Optional[Path]:
    p = (BASE / "cumulative_dissolved"
         / f"Amazon_ACA_{MODEL}_cumulative2018-{tag}-dissolved.geojson")
    return p if p.is_file() else None


def dissolved_diff(tag: str) -> Optional[Path]:
    p = (BASE / "cumulative_dissolved" / "diffs"
         / f"Amazon_ACA_{MODEL}_growth_{tag}-dissolved.geojson")
    return p if p.is_file() else None


def patch_diff(tag: str) -> Optional[Path]:
    p = (BASE / "cumulative" / "patch_diffs"
         / f"Amazon_ACA_{MODEL}_growth_{tag}.geojson")
    return p if p.is_file() else None


PRODUCTS = [
    Product("raw detections (basin)", Path("raw_detections"),
            (SOURCE_COOP, GS), raw_amazon,
            lambda t: f"amazon_basin_{t}_t0.4.geojson",
            sidecar="raw"),
    Product("raw detections (andes)", Path("raw_detections"),
            (SOURCE_COOP, GS), raw_andes,
            lambda t: f"andes_supplemental_{t}_t0.2.geojson"),
    Product("postprocessed t0.43", Path("postprocessed"),
            (SOURCE_COOP, GS), postprocessed,
            lambda t: f"amazon_basin_{t}_t0.43_t-iso0.75.geojson",
            sidecar="pp043"),
    Product("postprocessed t0.55", Path("postprocessed_t0.55_d5_3km_t-iso0.8"),
            (GS,), postprocessed_strict,
            lambda t: f"amazon_basin_{t}_t0.55_t-iso0.8.geojson",
            sidecar="pp055"),
    Product("cumulative patches", Path("cumulative"), (GS,), cumulative,
            lambda t: f"amazon_basin_cumulative_2018-{t}.geojson",
            published_only=True),
    Product("patch diffs", Path("cumulative/patch_diffs"), (GS,), patch_diff,
            lambda t: f"amazon_basin_growth_{t}.geojson",
            published_only=True),
    Product("cumulative dissolved", Path("cumulative_dissolved"), (GS,),
            cumulative_dissolved,
            lambda t: f"amazon_basin_cumulative_2018-{t}-dissolved.geojson",
            published_only=True),
    Product("dissolved diffs", Path("cumulative_dissolved/diffs"), (GS,),
            dissolved_diff,
            lambda t: f"amazon_basin_growth_{t}-dissolved.geojson",
            published_only=True),
]

# The SAM2 identity is declared, not imported: MaskConfig leaves the checkpoint
# and weights as None and resolves them at run time from models/sam2/, so there
# is no default to read. These are the values every run in the archive used --
# confirmed by Ed, and matching the sam2_* fields in the per-run mask_config.txt
# written by the live path.
SAM2_PRIOR_SIGMA = 12.0   # mirrors MaskConfig.prior_sigma
SAM2_CHECKPOINT = "sam2.1_hiera_small.pt"
SAM2_FINETUNED_WEIGHTS = "SAM_model_96_px_final.pth"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

SIDECARS = {
    "raw": ("model = {model}\n"
            "region = Amazon_ACA (six subregions, concatenated and deduplicated)\n"
            "supplemental = andes_supplemental, clipped to its boundary\n"
            "pred_threshold = 0.40 (basin), 0.2 (andes supplemental)\n"
            "postprocessing = none\n"
            "coordinate_precision = {precision}\n"),
    "pp043": ("model = {model}\n"
              "source = raw detections at pred_threshold 0.40\n"
              "t_main = 0.43\nt_iso = 0.75\nk = {k}\n"
              "isolation_km = {iso}\n"
              "note = the loose per-period set; SAM2 is prompted from this\n"
              "coordinate_precision = {precision}\n"),
    "pp055": ("model = {model}\n"
              "source = raw detections at pred_threshold 0.40\n"
              "t_main = 0.55\nt_iso = 0.8\nk = {k}\n"
              "isolation_km = {iso}\n"
              "note = the stringent set; stands in for temporal evidence at the "
              "provisional edge\n"
              "coordinate_precision = {precision}\n"),
    "masks": ("detection_model = {model}\n"
              "segmentation = fine-tuned SAM2, prompted in the near field of "
              "view around each detection\n"
              "sam2_checkpoint = {ckpt}\n"
              "finetuned_weights = {weights}\n"
              "sam2_model_cfg = {cfg}\n"
              "prior_sigma = {prior_sigma}          # spatial prior, pixels\n"
              "smoothing_sigma = {smoothing_sigma}          # after upsampling, pixels\n"
              "logit_clamp = {logit_clamp}\n"
              "note = every run directory here used these settings\n"
              "logits_stored = varies; see the mask_config.txt inside each run "
              "directory\n"
              "  both clip(smooth(x)) and smooth(clip(x)) are present in the "
              "archive. Either\n"
              "  gives the mask as logits > 0; they disagree on roughly 1 tile "
              "in 7,000, by\n"
              "  1-3 px, because most of a tile sits at the clamp.\n"),
}


def write_sidecar(kind: str, directory: Path, dry_run: bool) -> None:
    text = SIDECARS[kind].format(
        model=MODEL, k=PostprocessConfig.k,
        iso=PostprocessConfig.isolation_km,
        precision=PostprocessConfig.coordinate_precision,
        ckpt=SAM2_CHECKPOINT, weights=SAM2_FINETUNED_WEIGHTS,
        cfg=SAM2_MODEL_CFG, prior_sigma=SAM2_PRIOR_SIGMA,
        smoothing_sigma=DEFAULT_SMOOTHING_SIGMA,
        logit_clamp=DEFAULT_LOGIT_CLAMP)
    if not dry_run:
        (directory / "config.txt").write_text(text)


def copy(src: Path, dst: Path, dry_run: bool) -> str:
    """Copy unless the destination is already the same size and no older.

    Size alone is not enough. A rebuilt raster can land at the same byte count
    as the one it replaces -- the master onset COG differed by 58 KB out of
    63 MB after a bug fix, which is well inside coincidence -- and a size-only
    check would then keep the stale copy silently. The mtime comparison costs
    nothing and turns that into a copy.
    """
    if (dst.exists() and dst.stat().st_size == src.stat().st_size
            and dst.stat().st_mtime >= src.stat().st_mtime):
        return "skip"
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "copy"


def expected_relpaths(periods: Sequence[str]) -> dict:
    """Per tree, the set of relative paths stage() is responsible for.

    Derived from the same PRODUCTS/singleton definitions stage() walks, so the
    two cannot drift. ``mining_scar_masks/`` is returned as a prefix rather than
    enumerated: it holds ~257,000 tile files, and listing them to compare would
    cost more than it tells us.

    The point is to catch what should *not* be there. Anything dropped into a
    staging tree gets published -- a quarto preview left README.html and twelve
    bootstrap assets under README_files/ in the public tree, which the next sync
    would have shipped to consumers alongside the data.
    """
    trees: dict = {GS: set(), SOURCE_COOP: set()}
    prefixes: dict = {GS: {"mining_scar_masks/"}, SOURCE_COOP: set()}

    for product in PRODUCTS:
        wanted = published_periods(periods) if product.published_only else periods
        for tree in product.trees:
            for tag in wanted:
                if product.src_for(tag) is not None:
                    trees[tree].add(str(product.dest / product.rename(tag)))
            if product.sidecar:
                trees[tree].add(str(product.dest / "config.txt"))

    for src, name, tgt in singleton_items():
        if not src.is_file():
            continue                 # stage() reports the missing source itself
        for tree in tgt:
            trees[tree].add(name)
            if src.suffix == ".tif":
                trees[tree].add(name + ".aux.xml")

    for _tpl, dest, _label in READMES:
        trees[dest].add("README.md")
    trees[GS].add("mining_scar_masks/config.txt")
    return {"files": trees, "prefixes": prefixes}


def check_trees(periods: Sequence[str]) -> dict:
    """Compare each staging tree against what stage() is responsible for.

    Both directions matter and they fail differently. Extra files get published
    -- that is the quarto case. Missing files mean an incomplete product goes out
    under a complete-looking name, which is worse, because nothing downstream
    would flag it.
    """
    spec = expected_relpaths(periods)
    out = {}
    for tree, allowed in spec["files"].items():
        if not tree.is_dir():
            continue
        prefixes = spec["prefixes"][tree]
        found = {str(f.relative_to(tree)) for f in tree.rglob("*") if f.is_file()}
        stray = sorted(f for f in found - allowed
                       if not any(f.startswith(pre) for pre in prefixes))
        missing = sorted(allowed - found)
        if stray or missing:
            out[tree] = {"stray": stray, "missing": missing}
    return out


def stage(periods: Sequence[str], dry_run: bool) -> int:
    missing_total = 0
    for product in PRODUCTS:
        for tree in product.trees:
            out = tree / product.dest
            if not dry_run:
                out.mkdir(parents=True, exist_ok=True)
            copied = skipped = 0
            missing: List[str] = []
            wanted = published_periods(periods) if product.published_only else periods
            for tag in wanted:
                src = product.src_for(tag)
                if src is None:
                    missing.append(tag)
                    continue
                action = copy(src, out / product.rename(tag), dry_run)
                copied += action == "copy"
                skipped += action == "skip"
            if product.sidecar:
                write_sidecar(product.sidecar, out, dry_run)
            missing_total += len(missing)
            label = f"{product.name} -> {tree.name}/{product.dest}"
            print(f"  {label:<62} {copied:>3} copied {skipped:>3} present"
                  + (f"   MISSING {missing}" if missing else ""))
    return missing_total


#: SAM2 run directories belonging to the current model. Everything else under
#: data/outputs/sam2 is a prior vintage or a test: the unsmoothed-logits stash,
#: the pre-persistence t0.55 cumulative masks, masks_July2026, masks2026-02 and
#: the persistence-tests folders. Matching explicitly rather than excluding by
#: name keeps a new stray directory out by default.
MASK_DIR_GLOBS = (
    f"Amazon_ACA_{MODEL}_0.40_*_t0.43_d5_3km_t-iso0.75",
    f"andes_supplemental_{MODEL}_0.2_*",
    f"Amazon_ACA_{MODEL}_growth_*",
    "persistence_masks",
)


# Each tree gets its own README: the public mirror carries a subset of the
# products and a different audience, so one shared file would have to hedge on
# every path. Same token set, so the substitution below is unchanged.
READMES = (
    (REPO / "scripts/templates/amw_published_README.md", GS, "staging_gs"),
    (REPO / "scripts/templates/source_coop_README.md", SOURCE_COOP,
     "staging_source-coop"),
)


def render_readme(periods: Sequence[str], dry_run: bool,
                  today: Optional[str] = None) -> int:
    """Render every bucket README into its tree."""
    return sum(_render_one(tpl, dest, label, periods, dry_run, today)
               for tpl, dest, label in READMES)


def _render_one(template: Path, dest: Path, label: str,
                periods: Sequence[str], dry_run: bool,
                today: Optional[str] = None) -> int:
    """Render one bucket README into its tree.

    Only the date and the coverage line change per period, so the prose lives in
    a version-controlled template and the pipeline fills those in -- a README
    edited by hand goes stale the first refresh nobody remembers to update it.

    Substitution is a plain replace rather than str.format: the template is
    markdown full of braces-adjacent syntax, and format() would be one stray
    brace away from failing.
    """
    if not template.is_file():
        print(f"  README: template missing at {template}")
        return 1

    years = sorted(t for t in periods if Period.parse(t).is_annual)
    quarters = sorted((t for t in periods if not Period.parse(t).is_annual),
                      key=lambda t: Period.parse(t).sort_key)
    if today is None:
        from datetime import date
        today = date.today().strftime("%-d %B %Y")

    text = template.read_text()
    # drop the template's own editing note; it is not for consumers
    text = "\n".join(l for l in text.splitlines()
                      if not l.startswith(("<!--", "     Edit this template")))
    for token, value in (("${updated}", today), ("${model}", MODEL),
                         ("${years}", f"{years[0]}\u2013{years[-1]}" if years else "none"),
                         ("${quarters}", f"{quarters[0]}\u2013{quarters[-1]}"
                          if quarters else "none")):
        text = text.replace(token, value)
    if "${" in text:
        leftover = [l for l in text.splitlines() if "${" in l]
        print(f"  README {label}: unsubstituted token(s): {leftover}")
        return 1

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(text.lstrip("\n") + "\n")
    print(f"  README -> {label}/README.md  ({today}, {len(years)} years, "
          f"{len(quarters)} quarters)")
    return 0


def stage_mask_dirs(dry_run: bool) -> int:
    """Copy the SAM2 run directories, under the published name.

    Destination is ``mining_scar_masks/``, not ``sam2/`` -- consumers get the
    product name, and it matches the prefix prior vintages already used. Run
    directory names keep their long form: they identify which detection set
    prompted the segmentation, which is the one thing a mask tile cannot
    otherwise tell you.
    """
    dest_root = GS / "mining_scar_masks"
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
    write_sidecar("masks", dest_root, dry_run)
    srcs = []
    for pattern in MASK_DIR_GLOBS:
        srcs += [Path(d) for d in glob.glob(str(SAM2 / pattern)) if Path(d).is_dir()]
    if not srcs:
        print("  mask dirs: none matched")
        return 1

    copied = skipped = 0
    for src in sorted(srcs):
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(SAM2)
            action = copy(f, dest_root / rel, dry_run)
            copied += action == "copy"
            skipped += action == "skip"
    total_gb = sum(f.stat().st_size for s in srcs for f in s.rglob("*")
                   if f.is_file()) / 1e9
    print(f"  mask dirs -> staging_gs/mining_scar_masks/  {len(srcs)} dirs, "
          f"{total_gb:.1f} GB: {copied} copied, {skipped} present")
    return 0


def ensure_statistics(raster: Path, dry_run: bool) -> bool:
    """Make sure ``raster`` has a current .aux.xml of full statistics.

    Computed once on the source and copied to each tree rather than scanned per
    tree: it is a full pass over 118 Gpx and takes minutes. Full, not
    ``-approx_stats`` -- onset pixels are 0.0985% of the raster, so approximate
    sampling finds none of them, reports STATISTICS_VALID_PERCENT=0, and GIS
    software then draws an empty layer. That is the failure this sidecar exists
    to prevent, so the cheap option is the wrong one.

    Skipped when the sidecar is already newer than the raster.
    """
    aux = raster.with_suffix(raster.suffix + ".aux.xml")
    if aux.is_file() and aux.stat().st_mtime >= raster.stat().st_mtime:
        return False
    if dry_run:
        print(f"  statistics: would compute for {raster.name}")
        return False
    aux.unlink(missing_ok=True)          # stale stats would otherwise be reused
    subprocess.run(["gdalinfo", "-stats", str(raster)],
                   check=True, capture_output=True, text=True)
    return True


def singleton_items():
    """(source, published name, trees) for the layers promoted to each tree top.

    A function rather than an inline list so expected_relpaths() and
    stage_singletons() cannot disagree about what belongs at the top of a tree.
    """
    return [
        (BASE / "cumulative" / f"Amazon_ACA_{MODEL}_detections_first_year.geojson",
         "amazon_basin_detections_first_year.geojson", (SOURCE_COOP, GS)),
        (SAM2 / "persistence_masks" / "amazon_basin_mining_scar_masks_first_year.tif",
         "amazon_basin_mining_scar_masks_first_year.tif", (SOURCE_COOP, GS)),
    ]


def stage_singletons(dry_run: bool) -> int:
    """The two first-year layers, promoted to the top of each tree."""
    missing = 0
    items = singleton_items()
    for src, name, trees in items:
        for tree in trees:
            if not src.is_file():
                print(f"  {name:<62} MISSING source {src}")
                missing += 1
                continue
            if not dry_run:
                tree.mkdir(parents=True, exist_ok=True)
            action = copy(src, tree / name, dry_run)
            print(f"  {name:<62} {action} -> {tree.name}/")

    # Statistics sidecars, after the rasters are in place. The README documents
    # these as part of the product, so the pipeline has to produce them -- left
    # to hand they go stale the first time a raster is rebuilt, which is exactly
    # what happened after the quarterly-edge fix.
    for src, name, trees in items:
        if src.suffix != ".tif" or not src.is_file():
            continue
        fresh = ensure_statistics(src, dry_run)
        aux = src.with_suffix(src.suffix + ".aux.xml")
        if not aux.is_file():
            continue
        for tree in trees:
            act = copy(aux, tree / (name + ".aux.xml"), dry_run)
            print(f"  {name + '.aux.xml':<62} {act} -> {tree.name}/"
                  + ("   (recomputed)" if fresh else ""))
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--periods", nargs="+", default=DEFAULT_PERIODS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"staging {len(args.periods)} periods"
          f"{'  (dry run)' if args.dry_run else ''}\n")
    missing = stage(args.periods, args.dry_run)
    print()
    missing += stage_singletons(args.dry_run)
    print()
    missing += stage_mask_dirs(args.dry_run)
    print()
    missing += render_readme(args.periods, args.dry_run)
    print(f"\n{'DRY RUN, nothing written' if args.dry_run else 'staged'}"
          f"  —  {missing} missing input(s)")
    if missing:
        print("Missing inputs are not fatal here, but a published tree should "
              "have none: check before publishing.")


if __name__ == "__main__":
    main()
