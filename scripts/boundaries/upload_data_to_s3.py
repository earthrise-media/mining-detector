# Uploads the processed data to S3, while renaming .geojson to .json
# to enable compression in Cloudfront.
# Environment variables (should be set in .env file):
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
# - AWS_BUCKET
# - CLOUDFRONT_DISTRIBUTION_ID

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/upload_data_to_s3.py`.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "dotenv",
# ]
# ///

import os
import sys
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv()

# File paths to upload
FILE_PATHS = [
    "data/boundaries/national_admin/out/national_admin_impacts.geojson",
    "data/boundaries/national_admin/out/national_admin_impacts_unfiltered.geojson",
    "data/boundaries/national_admin/out/national_admin_yearly.json",
    "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered.geojson",
    "data/boundaries/subnational_admin/out/admin_areas_display_yearly.json",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_yearly.json",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered.geojson",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_yearly.json",
    
    "data/outputs/website/mining_201800_simplified.geojson",
    "data/outputs/website/mining_201900_simplified.geojson",
    "data/outputs/website/mining_202000_simplified.geojson",
    "data/outputs/website/mining_202100_simplified.geojson",
    "data/outputs/website/mining_202200_simplified.geojson",
    "data/outputs/website/mining_202300_simplified.geojson",
    "data/outputs/website/mining_202400_simplified.geojson",
    "data/outputs/website/mining_202502_simplified.geojson",
    "data/outputs/website/mining_202503_simplified.geojson",
]

CONTENT_TYPES = {".json": "application/json", ".geojson": "application/geo+json"}

def main():
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_BUCKET"]
    if missing := [v for v in required_vars if not os.getenv(v)]:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    bucket = os.getenv("AWS_BUCKET")
    
    failed = []
    for path in FILE_PATHS:
        if not Path(path).exists():
            print(f"✗ Missing: {path}")
            failed.append(path)
            continue
        
        # Rename .geojson files to .json, because Cloudfront doesn't natively compress geojsons
        file_path = Path(path)
        if file_path.suffix.lower() == '.geojson':
            new_path = file_path.with_suffix('.json')
            print(f"Renaming {path} to {new_path}")
        else:
            new_path = file_path

        content_type = CONTENT_TYPES.get(new_path.suffix.lower(), "application/octet-stream")
        
        # Upload the file with the new path (if renamed)
        s3.upload_file(str(file_path), bucket, str(new_path), ExtraArgs={"ContentType": content_type})
        print(f"✓ Uploaded {new_path}")
    
    if cf_id := os.getenv("CLOUDFRONT_DISTRIBUTION_ID"):
        cf = boto3.client("cloudfront", region_name=os.getenv("AWS_REGION"))
        cf.create_invalidation(
            DistributionId=cf_id,
            InvalidationBatch={
                "Paths": {"Quantity": 1, "Items": ["/data/*"]},
                "CallerReference": str(int(os.times().elapsed * 1000)),
            }
        )
        print("\n✓ Invalidated CloudFront cache")

    print(f"\n{len(FILE_PATHS) - len(failed)}/{len(FILE_PATHS)} files uploaded")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()