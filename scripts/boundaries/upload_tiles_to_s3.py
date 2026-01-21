# Uploads tiles to S3.
# Environment variables (should be set in .env file):
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
# - AWS_BUCKET_TILES
# - CLOUDFRONT_DISTRIBUTION_ID_TILES

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/upload_tiles_to_s3.py`.

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
    "data/boundaries/national_admin/out/national_admin_impacts_unfiltered.pmtiles",
    "data/boundaries/subnational_admin/out/admin_areas_display_impacts_unfiltered.pmtiles",
    "data/boundaries/protected_areas_and_indigenous_territories/out/indigenous_territories_impacts_unfiltered.pmtiles",
    "data/boundaries/protected_areas_and_indigenous_territories/out/protected_areas_impacts_unfiltered.pmtiles",
]
BASE_FOLDER = "amw"


def main():
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_BUCKET_TILES",
    ]
    if missing := [v for v in required_vars if not os.getenv(v)]:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    bucket = os.getenv("AWS_BUCKET_TILES")

    failed = []
    for path in FILE_PATHS:
        if not Path(path).exists():
            print(f"✗ Missing: {path}")
            failed.append(path)
            continue

        file_path = Path(path)
        # Create S3 key using only the filename in the BASE_FOLDER
        s3_key = f"{BASE_FOLDER}/{file_path.name}"

        # Upload the file to BASE_FOLDER/filename
        s3.upload_file(str(file_path), bucket, s3_key)
        print(f"✓ Uploaded {file_path} → {s3_key}")

    if cf_id := os.getenv("CLOUDFRONT_DISTRIBUTION_ID_TILES"):
        cf = boto3.client("cloudfront", region_name=os.getenv("AWS_REGION"))
        cf.create_invalidation(
            DistributionId=cf_id,
            InvalidationBatch={
                "Paths": {"Quantity": 1, "Items": [f"/{BASE_FOLDER}/*"]},
                "CallerReference": str(int(os.times().elapsed * 1000)),
            },
        )
        print("\n✓ Invalidated CloudFront cache")

    print(f"\n{len(FILE_PATHS) - len(failed)}/{len(FILE_PATHS)} files uploaded")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
