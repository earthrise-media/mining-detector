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

REPO_ROOT = Path(__file__).resolve().parents[2]

FILES = [
    "data/boundaries/national_admin/out/national_admin_impacts_unfiltered.geojson",
    "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered.geojson",
]


def convert(filepath: str) -> tuple[str, bool]:
    input_path = REPO_ROOT / filepath
    output_path = input_path.with_suffix(".pmtiles")
    result = subprocess.run(
        ["tippecanoe", "-z14", "-Z2", "-o", output_path, "-b5", "-r1", "-pk", "-pf", "-f", "-l", input_path.stem, input_path]
    )
    return filepath, result.returncode == 0


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        for filepath, success in executor.map(convert, FILES):
            print(f"{'✓' if success else '✗'} {filepath}")
    print("All files processed!")
