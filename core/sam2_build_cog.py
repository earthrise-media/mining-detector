""""Build mine scar segmentation COGs from tile GeoTiffs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import geopandas as gpd
import rasterio
from shapely.geometry import box
from tqdm import tqdm

# Required for gdalwarp to evaluate the Python pixel function in the mask
# union VRT (see build_mask_union_vrt). Warping that VRT is effectively
# single-threaded, so cap warp threads rather than leaving them unbounded.
os.environ.setdefault("GDAL_NUM_THREADS", "4")
os.environ.setdefault("GDAL_VRT_ENABLE_PYTHON", "YES")

MASK_SUFFIX = "-msk.tif"
LOGIT_SUFFIX = "-logits.tif"

MASK_NODATA = "2"
LOGIT_NODATA = "nan"

LAT_BAND_SIZE = 8  # degrees
UTM_ZONE_SIZE = 6  # degrees

# --- Fixed output grid -------------------------------------------------------
# Every published raster sits on one global lattice: pixel edges fall on integer
# multiples of GRID_RES degrees from (0, 0). This makes pixel (i, j) the same
# ground location in every period and every UTM/lat band, so temporal rules need
# no regridding and adjacent bands mosaic without resampling.
#
# GRID_RES is finer than every observed source resolution (masks span
# 9.0404e-05 .. 9.0952e-05 deg), so regridding never decimates. See
# docs/design/persistence-planning.md, "Fixing the raster grid".
GRID_RES = 0.00009  # degrees/pixel, ~10.02 m in latitude

# Tiles are assigned to a band by their center, so a tile may overhang the band
# boundary by up to half its width (~0.026 deg for a 576 px tile). This margin
# is the fixed allowance; overruns beyond it expand the extent and warn rather
# than clip.
GRID_MARGIN_DEG = 0.08

# Guard against float noise when snapping a value that is already on-lattice.
_SNAP_EPS = 1e-9

DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")


def snap_floor(value, resolution=GRID_RES):
    """Largest lattice coordinate <= value."""
    return math.floor(value / resolution + _SNAP_EPS) * resolution


def snap_ceil(value, resolution=GRID_RES):
    """Smallest lattice coordinate >= value."""
    return math.ceil(value / resolution - _SNAP_EPS) * resolution


def snap_extent(bounds, resolution=GRID_RES):
    """Snap (minx, miny, maxx, maxy) outward onto the lattice."""
    minx, miny, maxx, maxy = bounds
    return (
        snap_floor(minx, resolution),
        snap_floor(miny, resolution),
        snap_ceil(maxx, resolution),
        snap_ceil(maxy, resolution),
    )


def band_extent(utm_zone, lat_start, lat_end, resolution=GRID_RES,
                margin=GRID_MARGIN_DEG):
    """Fixed lattice-aligned extent for a UTM zone x lat band group.

    Derived from the band definition rather than from the tiles present, so the
    extent is identical for every period.
    """
    lon_start = -180 + UTM_ZONE_SIZE * (utm_zone - 1)
    return snap_extent(
        (lon_start - margin, lat_start - margin,
         lon_start + UTM_ZONE_SIZE + margin, lat_end + margin),
        resolution,
    )


def union_bounds(input_files):
    """Union of the bounds of input_files."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for path in input_files:
        with rasterio.open(path) as ds:
            b = ds.bounds
        minx, miny = min(minx, b.left), min(miny, b.bottom)
        maxx, maxy = max(maxx, b.right), max(maxy, b.top)
    return (minx, miny, maxx, maxy)


def resolve_extent(input_files, extent, resolution=GRID_RES, label=""):
    """Return a lattice-aligned extent that is guaranteed to contain the inputs.

    Expands (and warns) rather than clipping if the inputs overrun ``extent``:
    the result stays on the lattice either way, so temporal alignment holds even
    when the fixed extent has to grow.
    """
    src = union_bounds(input_files)
    if extent is None:
        return snap_extent(src, resolution)

    grown = (
        min(extent[0], snap_floor(src[0], resolution)),
        min(extent[1], snap_floor(src[1], resolution)),
        max(extent[2], snap_ceil(src[2], resolution)),
        max(extent[3], snap_ceil(src[3], resolution)),
    )
    if grown != tuple(extent):
        print(
            f"WARNING: {label or 'group'} tiles overrun the fixed extent "
            f"{tuple(round(v, 6) for v in extent)} -> "
            f"{tuple(round(v, 6) for v in grown)}. Data is preserved and the "
            f"grid stays aligned, but consider raising GRID_MARGIN_DEG."
        )
    return grown


def run(cmd):
    """Run a GDAL command, surfacing its stderr if it fails.

    GDAL is chatty on success and the noise is not worth keeping, but silencing
    stderr unconditionally makes a failure look like nothing happened at all --
    the traceback names the exit code and no reason. Capture, discard on
    success, re-raise with the message on failure.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd,
            output=proc.stdout,
            stderr=f"{proc.stderr.strip()}\n  command: {' '.join(map(str, cmd))}")

def _buildvrt_grid_args(extent, resolution, resampling, nodata):
    """gdalbuildvrt arguments pinning the output to the fixed lattice.

    Pinning the VRT (rather than only gdalwarp) means each source is placed on
    the final grid once, instead of being resampled twice.
    """
    # gdalwarp spells nearest "near"; gdalbuildvrt wants "nearest".
    resampling = "nearest" if resampling == "near" else resampling
    args = [
        "-resolution", "user",
        "-tr", repr(resolution), repr(resolution),
        "-te", *[repr(v) for v in extent],
        "-r", resampling,
    ]
    # Regions of the fixed extent not covered by a source must read as nodata,
    # not as 0 -- otherwise the mask OR reduces unobserved ground to
    # "observed, not mining".
    args += ["-srcnodata", str(nodata), "-vrtnodata", str(nodata)]
    return args


def build_mask_union_vrt(input_files, derived_vrt_path, nodata_val,
                         extent=None, resolution=GRID_RES, resampling="nearest"):
    """Build a VRT computing the pixel-wise union (logical OR) of N mask rasters.

    Plain gdalbuildvrt mosaicking is "last file wins" for overlapping pixels,
    which drops a 1 from an earlier tile if a later tile has 0 in the same
    spot. This instead stacks all inputs as separate bands and reduces them
    with a nodata-aware OR: output is 1 if any input is 1, nodata only if
    every input is nodata, and 0 otherwise.
    """
    vrt_dir = os.path.dirname(derived_vrt_path)
    stack_vrt = os.path.join(vrt_dir, "stack.vrt")

    grid_args = (
        _buildvrt_grid_args(extent, resolution, resampling, nodata_val)
        if extent is not None else []
    )
    run(["gdalbuildvrt", "-separate"] + grid_args + [stack_vrt] + input_files)

    tree = ET.parse(stack_vrt)
    root = tree.getroot()

    for band in root.findall("VRTRasterBand"):
        root.remove(band)

    derived = ET.SubElement(root, "VRTRasterBand", {
        "dataType": "Byte",
        "band": "1",
        "subClass": "VRTDerivedRasterBand",
    })
    ET.SubElement(derived, "NoDataValue").text = str(nodata_val)
    ET.SubElement(derived, "PixelFunctionType").text = "mask_or"
    ET.SubElement(derived, "PixelFunctionLanguage").text = "Python"
    code = ET.SubElement(derived, "PixelFunctionCode")
    code.text = f"""
import numpy as np
def mask_or(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    stack = np.stack(in_ar)
    is_nodata = stack == {nodata_val}
    out_ar[:] = np.where(
        is_nodata.all(axis=0),
        {nodata_val},
        np.where((stack == 1).any(axis=0), 1, 0),
    )
"""
    stack_basename = os.path.basename(stack_vrt)
    for band_index in range(1, len(input_files) + 1):
        src = ET.SubElement(derived, "SimpleSource")
        ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
        ET.SubElement(src, "SourceBand").text = str(band_index)

    tree.write(derived_vrt_path)

def build_logit_max_vrt(input_files, derived_vrt_path, extent=None,
                        resolution=GRID_RES, resampling="bilinear"):
    """Build a VRT computing the pixel-wise max of N logit rasters.

    Plain gdalbuildvrt mosaicking is "last file wins" for overlapping pixels,
    so a later tile's weaker logit would silently overwrite an earlier
    tile's stronger one. This instead stacks all inputs as separate bands
    and reduces them with a NaN-aware max: output is the largest non-NaN
    logit at each pixel, NaN only where every input is NaN. (Max is the
    logit-space equivalent of the mask OR: with a shared threshold,
    max(logits) > t iff any per-tile mask would be 1.)
    """
    vrt_dir = os.path.dirname(derived_vrt_path)
    stack_vrt = os.path.join(vrt_dir, "stack.vrt")

    grid_args = (
        _buildvrt_grid_args(extent, resolution, resampling, LOGIT_NODATA)
        if extent is not None else []
    )
    run(["gdalbuildvrt", "-separate"] + grid_args + [stack_vrt] + input_files)

    tree = ET.parse(stack_vrt)
    root = tree.getroot()

    for band in root.findall("VRTRasterBand"):
        root.remove(band)

    derived = ET.SubElement(root, "VRTRasterBand", {
        "dataType": "Float32",
        "band": "1",
        "subClass": "VRTDerivedRasterBand",
    })
    ET.SubElement(derived, "NoDataValue").text = "nan"
    ET.SubElement(derived, "PixelFunctionType").text = "logit_max"
    ET.SubElement(derived, "PixelFunctionLanguage").text = "Python"
    code = ET.SubElement(derived, "PixelFunctionCode")
    code.text = """
import numpy as np
def logit_max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    stack = np.stack(in_ar).astype(np.float32)
    filled = np.where(np.isnan(stack), -np.inf, stack)
    reduced = filled.max(axis=0)
    out_ar[:] = np.where(np.isneginf(reduced), np.nan, reduced)
"""
    stack_basename = os.path.basename(stack_vrt)
    for band_index in range(1, len(input_files) + 1):
        src = ET.SubElement(derived, "SimpleSource")
        ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
        ET.SubElement(src, "SourceBand").text = str(band_index)

    tree.write(derived_vrt_path)

def utm_zone_from_lon(lon):
    return int((lon + 180) // 6) + 1

def lat_band_from_lat(lat):
    band_start = math.floor(lat / LAT_BAND_SIZE) * LAT_BAND_SIZE
    band_end = band_start + LAT_BAND_SIZE
    return band_start, band_end

def extract_date_range(filename):
    m = DATE_RANGE_RE.search(filename)
    if not m:
        return None, None
    start, end = m.groups()
    return start, end

def build_cog(
    input_files,
    output_path,
    raster_type="mask",
    nodata=None,
    resampling=None,
    blocksize=512,
    predictor=None,
    extent=None,
    resolution=GRID_RES,
):
    """
    Build a Cloud-Optimized GeoTIFF (COG) from input rasters.

    Args:
        input_files (list[str]): list of input file paths
        output_path (str): path to output COG
        raster_type (str): "mask" or "logits"
        nodata (str|float): nodata value to set
        resampling (str): resampling method for gdalwarp
        blocksize (int): tile size for COG
        predictor (int): predictor for compression
        extent (tuple|None): fixed (minx, miny, maxx, maxy). Snapped onto the
            lattice and widened if the inputs overrun it. If None, the snapped
            union of the inputs is used -- still lattice-aligned, but the extent
            then varies with the tiles present.
        resolution (float): degrees/pixel; the lattice step. Defaults to
            GRID_RES and should not normally be overridden.
    """
    # Set defaults based on raster type
    if raster_type == "mask":
        resampling = resampling or "near"
        predictor = predictor or 2
        nodata = nodata or MASK_NODATA
    elif raster_type == "logits":
        resampling = resampling or "bilinear"
        predictor = predictor or 3
        nodata = nodata or LOGIT_NODATA
    else:
        raise ValueError(f"Unknown raster_type: {raster_type}")

    extent = resolve_extent(
        input_files, extent, resolution, label=os.path.basename(output_path))

    with tempfile.TemporaryDirectory(prefix="sam2_build_cog_") as tmpdir:
        vrt_path = os.path.join(tmpdir, "tmp.vrt")
        tmp_tif = os.path.join(tmpdir, "tmp.tif")

        if raster_type == "mask":
            build_mask_union_vrt(
                input_files, vrt_path, nodata_val=int(nodata),
                extent=extent, resolution=resolution, resampling=resampling)
        else:
            build_logit_max_vrt(
                input_files, vrt_path,
                extent=extent, resolution=resolution, resampling=resampling)

        # The VRT is already on the fixed lattice, so this warp is a no-op
        # grid-wise; -tr/-te are passed so the output grid is guaranteed rather
        # than inferred by GDALSuggestedWarpOutput.
        run([
            "gdalwarp",
            vrt_path,
            tmp_tif,
            "-r", resampling,
            "-tr", repr(resolution), repr(resolution),
            "-te", *[repr(v) for v in extent],
            "-srcnodata", str(nodata),
            "-dstnodata", str(nodata),
            "-co", "TILED=YES",
            "-co", "COMPRESS=ZSTD",
            "-co", "BIGTIFF=IF_SAFER",
            "-co", "NUM_THREADS=ALL_CPUS"
        ])

        run([
            "gdal_translate",
            tmp_tif,
            output_path,
            "-of", "COG",
            "-co", "COMPRESS=ZSTD",
            "-co", f"BLOCKSIZE={blocksize}",
            "-co", f"PREDICTOR={predictor}",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-a_nodata", str(nodata)
        ])

def main(input_dir, output_dir, index_out, stac_out, max_workers,
         extent_mode="union", raster_types=("mask",)):

    os.makedirs(output_dir, exist_ok=True)

    groups = {}
    tile_index_rows = []

    all_tiles = list(Path(input_dir).glob("*.tif"))
    if not all_tiles:
        raise ValueError('No geotiffs found.')

    for tif in tqdm(all_tiles, desc="Scanning tiles"):

        is_mask = tif.name.endswith(MASK_SUFFIX)
        is_logits = tif.name.endswith(LOGIT_SUFFIX)

        if not (is_mask or is_logits):
            continue

        raster_type = "mask" if is_mask else "logits"
        if raster_type not in raster_types:
            continue

        with rasterio.open(tif) as ds:
            bounds = ds.bounds
            minx, miny, maxx, maxy = bounds

            center_lon = (minx + maxx) / 2
            center_lat = (miny + maxy) / 2

            utm_zone = utm_zone_from_lon(center_lon)
            lat_start, lat_end = lat_band_from_lat(center_lat)

            start_date, end_date = extract_date_range(tif.name)

            key = (utm_zone, lat_start, lat_end, raster_type,
                   start_date, end_date)
            groups.setdefault(key, []).append(str(tif))

            data = ds.read(1)

            mine_pixels, mine_fraction = None, None
            if is_mask:
                mine_pixels = int((data == 1).sum())
                mine_fraction = mine_pixels / data.size

            tile_index_rows.append({
                "filename": tif.name,
                "utm_zone": utm_zone,
                "lat_start": lat_start,
                "lat_end": lat_end,
                "raster_type": raster_type,
                "start_date": start_date,
                "end_date": end_date,
                "mine_pixels": mine_pixels,
                "mine_fraction": mine_fraction,
                "geometry": box(minx, miny, maxx, maxy)
            })

    gdf = gpd.GeoDataFrame(tile_index_rows, crs="EPSG:4326")
    gdf.to_parquet(index_out)

    def build_group(group_key):
        utm_zone, lat_start, lat_end, raster_type, start_date, end_date = group_key
        date_tag = f"{start_date}_{end_date}" if start_date and end_date else "nodate"
        tag = f"{date_tag}_utm{utm_zone}_lat_{lat_start}_{lat_end}_epsg4326"
        cog_path = os.path.join(output_dir, f"mining_{raster_type}_{tag}.tif")
        build_cog(
            input_files=groups[group_key],
            output_path=cog_path,
            raster_type=raster_type,
            extent=(band_extent(utm_zone, lat_start, lat_end)
                    if extent_mode == "band" else None),
        )
        return cog_path, raster_type, utm_zone, lat_start, lat_end, start_date, end_date


    results = []

    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(build_group, key) for key in groups.keys()]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Building mosaics"):
            results.append(future.result())


    # Build one large mask mosaic across all mask COGs
    mask_cogs = [r[0] for r in results if r[1] == "mask"]

    if mask_cogs:
        # Extract unique date range (assuming all identical)
        start_dates = {r[5] for r in results if r[1] == "mask"}
        end_dates = {r[6] for r in results if r[1] == "mask"}

        if len(start_dates) == 1 and len(end_dates) == 1:
            start_date = start_dates.pop()
            end_date = end_dates.pop()
        else:
            raise ValueError("Mask groups have inconsistent date ranges.")

        big_vrt = os.path.join(output_dir, "big_mask.vrt")
        tmp_tif = os.path.join(output_dir, "big_mask_tmp.tif")
        big_mask_path = os.path.join(
            output_dir,
            f"mining_mask_{start_date}_{end_date}_epsg4326.tif"
        )
        build_cog(
            input_files=mask_cogs,
            output_path=big_mask_path,
            raster_type="mask"
        )
        # utm_zone/lat_start/lat_end are None: this mosaic spans all of them.
        results.append(
            (big_mask_path, "mask", None, None, None, start_date, end_date))

    # Build one large logits mosaic across all logits COGs
    logit_cogs = [r[0] for r in results if r[1] == "logits"]

    if logit_cogs:
        # Extract unique date range (assuming all identical)
        start_dates = {r[5] for r in results if r[1] == "logits"}
        end_dates = {r[6] for r in results if r[1] == "logits"}

        if len(start_dates) == 1 and len(end_dates) == 1:
            start_date = start_dates.pop()
            end_date = end_dates.pop()
        else:
            raise ValueError("Logits groups have inconsistent date ranges.")

        big_logits_path = os.path.join(
            output_dir,
            f"mining_logits_{start_date}_{end_date}_epsg4326.tif"
        )
        build_cog(
            input_files=logit_cogs,
            output_path=big_logits_path,
            raster_type="logits"
        )
        # utm_zone/lat_start/lat_end are None: this mosaic spans all of them.
        results.append(
            (big_logits_path, "logits", None, None, None, start_date, end_date))

    stac_items = []

    for cog_path, raster_type, utm_zone, lat_start, lat_end, start_date, end_date in results:
        with rasterio.open(cog_path) as ds:
            bounds = ds.bounds

        properties = {
            "utm_zone": utm_zone,
            "lat_start": lat_start,
            "lat_end": lat_end,
            "raster_type": raster_type,
            "start_date": start_date,
            "end_date": end_date
        }

        if start_date and end_date:
            properties["start_datetime"] = f"{start_date}T00:00:00Z"
            properties["end_datetime"] = f"{end_date}T23:59:59Z"

        stac_items.append({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": os.path.basename(cog_path),
            "properties": properties,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bounds.left, bounds.bottom],
                    [bounds.right, bounds.bottom],
                    [bounds.right, bounds.top],
                    [bounds.left, bounds.top],
                    [bounds.left, bounds.bottom]
                ]]
            },
            "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "assets": {
                raster_type: {
                    "href": cog_path,
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized"
                }
            }
        })

    stac_catalog = {
        "type": "FeatureCollection",
        "stac_version": "1.0.0",
        "description": "Amazon Mining Watch Scar Masks & Logits",
        "features": stac_items
    }

    with open(stac_out, "w") as f:
        json.dump(stac_catalog, f, indent=2)

    print("All processing complete.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build mine scar segmentation COGs from tile GeoTiffs.")
    parser.add_argument("input_dir", help="Directory containing EPSG:4326 DLTile GeoTIFFs")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/cog_outputs)")
    parser.add_argument("--index_out", default=None, help="Tile index GeoParquet path")
    parser.add_argument("--stac_out", default=None, help="STAC catalog output JSON")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 4, help="Parallel worker count")
    parser.add_argument(
        "--raster_types", nargs="+", choices=("mask", "logits"),
        default=["mask"],
        help=(
            "Which mosaics to build. Logit mosaics are for inspection only: "
            "re-deriving a mask at a different threshold must replay the "
            "smoothing per tile, because max-reduce across overlapping tiles "
            "does not commute with it. The durable logit artifact is the "
            "per-tile *-logits.tif, not the mosaic. Passing 'mask' alone skips "
            "the float32 half of the work, which dominates the run. Default is "
            "mask only; pass 'mask logits' if you want a logit mosaic to "
            "look at."
        ))
    parser.add_argument(
        "--extent_mode", choices=("band", "union"), default="union",
        help=(
            "Output extent for per-band mosaics. 'union' (default) uses the "
            "snapped union of the tiles present -- on the global lattice, so "
            "temporal rules stay valid with an integral pixel offset between "
            "periods. 'band' uses the fixed UTM-zone x lat-band box, giving "
            "byte-identical grids across periods, but writes 6.2 Gpx per band "
            "regardless of content: only ~1.1x the union on the dense core "
            "bands, but 16x on utm20 lat[-24,-16] and 79,000x on utm19 "
            "lat[-24,-16], whose real extent is 285x275 pixels. That has "
            "exhausted memory in gdalwarp on a loaded machine. Prefer 'band' "
            "only when identical extents are worth the cost and the run has "
            "the machine to itself."
        ))

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "cog_outputs")
    index_out = args.index_out or os.path.join(output_dir, "tile_index.parquet")
    stac_out = args.stac_out or os.path.join(output_dir, "stac_catalog.json")

    main(input_dir, output_dir, index_out, stac_out, args.max_workers,
         extent_mode=args.extent_mode, raster_types=tuple(args.raster_types))
