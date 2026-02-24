"""
Standardizes illegality areas for analysis by the preprocess_mining_areas script.
The original source data had the following operations (in QGIS, because GeoPandas wasn't handling it):
- Fix geometry in both source files (Legality_v1.shp and LegalityBolivia_V2.shp)
- Needed to substitute Bolivia in v1 for v2, so:
    - Dissolved v2
    - Created whole area extents
    - Clipped extents by v2
    - Clipped v1 by it
Then saved as shapefiles and ran this script.
"""

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/standardize_illegality_areas.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
# ]
# ///
import geopandas as gpd
from shapely import set_precision

illegality_v1 = gpd.read_file("data/boundaries/illegality/source_data/illegality_v1_areas_fixed_clipped.shp")

categories = {"Low": 1, "Medium": 2, "High": 3, "Very_High": 4}
illegality_v1["illegality"] = illegality_v1.Category_1.map(categories)

illegality_v2_bolivia = gpd.read_file("data/boundaries/illegality/source_data/LegalityBolivia_V2_simplified_4326_fixed.shp")

categories = {"4.Low": 1, "3.Medium": 2, "2.High": 3, "1.Very_High": 4}
illegality_v2_bolivia["illegality"] = illegality_v2_bolivia.Category.map(categories)

# overlay Bolivia v2 to already clipped source data
cols = ["geometry", "illegality"]
combined = gpd.pd.concat([illegality_v1[cols], illegality_v2_bolivia[cols]], ignore_index=True)

combined_simplified = combined.copy()
combined_simplified["geometry"] = combined_simplified["geometry"].simplify(
    tolerance=0.0001, preserve_topology=True
)
combined_simplified["geometry"] = combined_simplified["geometry"].apply(
    lambda geom: set_precision(geom, grid_size=1e-6)
)
combined_simplified.to_file("data/boundaries/illegality/out/illegality_v2_areas_simplified.geojson", driver="GeoJSON", encoding="utf-8")
