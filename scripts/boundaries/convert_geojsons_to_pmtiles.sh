#!/bin/bash

# Get the repository root (two levels up from scripts/boundaries/)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Array of input files (relative to repo root)
files=(
    "data/boundaries/national_admin/out/national_admin_impacts_unfiltered.geojson"
    "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered.geojson"
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered.geojson"
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered.geojson"
)

# Loop through each file
for filepath in "${files[@]}"; do
    # Construct full paths
    full_input="${REPO_ROOT}/${filepath}"
    
    # Extract the directory and filename without extension
    dir=$(dirname "$full_input")
    filename=$(basename "$filepath" .geojson)
    
    # Set output path (same directory, .pmtiles extension)
    output="${dir}/${filename}.pmtiles"
    
    echo "Processing: $full_input"
    echo "Output: $output"
    
    # Run tippecanoe
    tippecanoe -z14 -Z2 -o "$output" -b0 -r1 -pk -pf -f \
        -l "$filename" \
        "$full_input"
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Successfully created $output"
    else
        echo "✗ Failed to process $filepath"
    fi
    
    echo "---"
done

echo "All files processed!"