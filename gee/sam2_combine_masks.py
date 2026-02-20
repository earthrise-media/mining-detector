#!/usr/bin/env python3
"""
Combine two segmentation mask COGs with OR logic.

- Output value 1 where either input has 1 (mask).
- Output value 0 where the logical OR is 0 (both 0, or 0 vs nodata).
- Output nodata (2) only where both inputs are nodata (2).
- Areas covered by only one raster pass through that raster's value.

Uses a VRT-derived band so processing is streaming (no full-raster load).
base_mask.tif defines output resolution and grid; update.tif is warped to match.

"""

import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Set before importing GDAL so VRT Python pixel functions and threading are enabled
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("GDAL_VRT_ENABLE_PYTHON", "YES")

from osgeo import gdal


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main(base_mask_path: str, update_path: str, output_path: str) -> None:
    nodata_val = 2

    # --- Resolution and grid from base_mask.tif ---
    ds = gdal.Open(base_mask_path)
    gt = ds.GetGeoTransform()
    xres = abs(gt[1])
    yres = abs(gt[5])
    ds = None
    print(f"Output resolution (from base_mask): {xres}, {yres}")

    with tempfile.TemporaryDirectory(prefix="combine_masks_") as tmpdir:
        stack_vrt = os.path.join(tmpdir, "stack.vrt")
        derived_vrt = os.path.join(tmpdir, "stack_calc.vrt")

        # --- Build aligned stack: both rasters warped to same grid ---
        run([
            "gdalbuildvrt",
            "-resolution", "user",
            "-tr", str(xres), str(yres),
            "-tap",
            "-separate",
            stack_vrt,
            base_mask_path,
            update_path,
        ])

        # --- Replace bands with a single derived band (OR + nodata logic) ---
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
        code.text = """
import numpy as np
def mask_or(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    A = in_ar[0]
    B = in_ar[1]
    out_ar[:] = np.where(
        (A == 2) & (B == 2),
        2,
        np.where((A == 1) | (B == 1), 1, 0),
    )
"""
        # Reference the two bands of the stack VRT (filename relative to derived VRT)
        stack_basename = os.path.basename(stack_vrt)
        for band_index in (1, 2):
            src = ET.SubElement(derived, "SimpleSource")
            ET.SubElement(src, "SourceFilename", {"relativeToVRT": "1"}).text = stack_basename
            ET.SubElement(src, "SourceBand").text = str(band_index)

        tree.write(derived_vrt)

        # --- Warp derived VRT to final COG ---
        run([
            "gdalwarp",
            derived_vrt,
            output_path,
            "-of", "COG",
            "-r", "near",
            "-multi",
            "-dstnodata", str(nodata_val),
            "-co", "COMPRESS=ZSTD",
            "-co", "BLOCKSIZE=512",
            "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=YES",
        ])

    print("Done.", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine two segmentation mask COGs with OR logic (base_mask.tif + update.tif -> output)."
    )
    parser.add_argument(
        "base_mask_path",
        type=str,
        help="Path to base mask COG (base_mask.tif)",
    )
    parser.add_argument(
        "update_path",
        type=str,
        help="Path to update mask COG (update.tif)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to output combined COG",
    )
    args = parser.parse_args()
    main(*vars(args).values())
