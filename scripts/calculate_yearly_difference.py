# You can run this script with uv if you prefer, 
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/calculate_yearly_difference.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geopandas",
# ]
# ///
import geopandas as gpd
from pathlib import Path

base_folder = "data/outputs/48px_v3.2-3.7ensemble/cumulative"
output_folder = base_folder.replace("cumulative", "difference")

files = {
    2024: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2024cumulative.geojson",
    2023: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2023cumulative.geojson",
    2022: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2022cumulative.geojson",
    2021: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2021cumulative.geojson",
    2020: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2020cumulative.geojson",
    2019: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2019cumulative.geojson",
    2018: "amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2018cumulative.geojson",
}

def calculate_yearly_differences(save_individual_files):
    """Calculate geometric differences between consecutive years, using the
    cumulative GeoJSON files. Combines and saves all years into a single GeoJSON,
    tagged with a 'year' property."""
    years = sorted(files.keys())
    # start with first year
    first_year = years[0]
    first_gdf = gpd.read_file(f"{base_folder}/{files[first_year]}")
    first_gdf["year"] = first_year
    all_differences = [first_gdf]

    for i in range(1, len(years)):
        current_year = years[i]
        previous_year = years[i - 1]
        print(f"Calculating difference for {current_year} - {previous_year}")
        
        # load geodataframes
        current_gdf = gpd.read_file(f"{base_folder}/{files[current_year]}")
        previous_gdf = gpd.read_file(f"{base_folder}/{files[previous_year]}")
        
        # calculate difference (current - previous)
        difference_gdf = current_gdf.overlay(previous_gdf, how="difference")
        
        # convert to meters for accurate buffering
        difference_projected = difference_gdf.to_crs("EPSG:3857")  # web mercator
        # remove slivers with negative buffer then restore
        buffered_negative = difference_projected.buffer(-10, cap_style="square", join_style="mitre")
        buffered_restored = buffered_negative.buffer(10, cap_style="square", join_style="mitre")
        difference_gdf = gpd.GeoDataFrame(geometry=buffered_restored).to_crs("EPSG:4326")
        
        # create year property
        difference_gdf["year"] = current_year
        
        if save_individual_files:
            # generate output filename
            base_name = files[current_year].replace("cumulative.geojson", "")
            output_file = f"{output_folder}/{base_name}difference.geojson"
            
            # ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # save difference
            difference_gdf.to_file(output_file, driver="GeoJSON")
            print(f"Created: {output_file}")
        
        # append to list
        all_differences.append(difference_gdf)

    # combine frames
    combined_gdf = gpd.pd.concat(all_differences, ignore_index=True)
    output_combined_file = f"{output_folder}/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2024_all_differences.geojson"
    # ensure output directory exists
    output_path = Path(output_combined_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # save combined file
    combined_gdf.to_file(output_combined_file, driver="GeoJSON")
    print(f"Created: {output_combined_file}")

if __name__ == "__main__":
    save_individual_files = False
    calculate_yearly_differences(save_individual_files)