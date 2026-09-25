# Converts the geojson outputs to pmtiles using tippecanoe, for use on the website

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/convert_geojsons_to_pmtiles.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
# ]
# ///

import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from constants import COMBINED_MINING_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]

GEOJSONS_TO_PMTILES = [
    COMBINED_MINING_FILE,
    "data/boundaries/national_admin/out/national_admin_impacts_unfiltered.geojson",
    "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered.geojson",
]

# Max zoom per file. Only the mining polygons need the extra detail; the boundary layers are fine at 11.
DEFAULT_MAX_ZOOM = 11
MAX_ZOOM = {
    Path(COMBINED_MINING_FILE): 14,
}


def convert(filepath: str) -> tuple[str, bool]:
    input_path = REPO_ROOT / filepath
    output_path = input_path.with_suffix(".pmtiles")
    max_zoom = MAX_ZOOM.get(Path(filepath), DEFAULT_MAX_ZOOM)
    result = subprocess.run(
        ["tippecanoe", f"-z{max_zoom}", "-Z3", "-o", output_path, "-b5", "-r1", "-pk", "-pf", "-f", "-l", input_path.stem, input_path]
    )
    return filepath, result.returncode == 0


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        for filepath, success in executor.map(convert, GEOJSONS_TO_PMTILES):
            print(f"{'✓' if success else '✗'} {filepath}")
    print("All files processed!")
