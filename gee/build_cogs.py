""""Build mine scar segmentation COGs from tile GeoTiffs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import math
from pathlib import Path
import re
import subprocess
import uuid

import geopandas as gpd
import rasterio
from shapely.geometry import box
from tqdm import tqdm

MASK_SUFFIX = "-msk.tif"
LOGIT_SUFFIX = "-logits.tif"

MASK_NODATA = "2"
LOGIT_NODATA = "nan"

LAT_BAND_SIZE = 8  # degrees

DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")

def run(cmd):
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    tmp_dir=None,
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
        tmp_dir (str|None): directory for temporary files
    """
    tmp_dir = tmp_dir or os.path.dirname(output_path)
    uid = uuid.uuid4().hex
    vrt_path = os.path.join(tmp_dir, f"tmp_{uid}.vrt")
    tmp_tif = os.path.join(tmp_dir, f"tmp_{uid}.tif")

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

    run(["gdalbuildvrt", vrt_path] + input_files)

    # Warp to align tiles and snap to a consistent grid
    run([
        "gdalwarp",
        vrt_path,
        tmp_tif,
        "-r", resampling,
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

    for f in (vrt_path, tmp_tif):
        if os.path.exists(f):
            os.remove(f)

def main(input_dir, output_dir, index_out, stac_out, max_workers):

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
            raster_type=raster_type
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

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "cog_outputs")
    index_out = args.index_out or os.path.join(output_dir, "tile_index.parquet")
    stac_out = args.stac_out or os.path.join(output_dir, "stac_catalog.json")

    main(input_dir, output_dir, index_out, stac_out, args.max_workers)
