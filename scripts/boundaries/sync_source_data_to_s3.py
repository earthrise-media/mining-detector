# Syncs local data directories to S3, replicating the behavior of:
#   aws s3 sync ./data/boundaries s3://amw-media/mining-detector-repo-backups/data/boundaries
#   aws s3 sync ./data/outputs/website s3://amw-media/mining-detector-repo-backups/data/outputs/website
# Excludes .DS_Store and .pmtiles files. Only uploads new or modified files (by size).
# Pass --download to also pull S3-only files to the local directory.
#
# Environment variables (should be set in .env file):
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
# - AWS_BUCKET_MEDIA

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/sync_data_to_s3.py` (upload only)
#     or: `uv run scripts/sync_data_to_s3.py --download` (bidirectional sync)

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "python-dotenv",
# ]
# ///

import argparse
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

S3_PREFIX = "mining-detector-repo-backups"

SYNC_PAIRS = [
    ("./data/boundaries", f"{S3_PREFIX}/data/boundaries"),
    ("./data/outputs/website", f"{S3_PREFIX}/data/outputs/website"),
]

EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_EXTENSIONS = {".pmtiles"}


def should_exclude(path: Path | str) -> bool:
    p = Path(path)
    return p.name in EXCLUDE_NAMES or p.suffix in EXCLUDE_EXTENSIONS


def get_s3_objects(s3, bucket: str, prefix: str) -> dict[str, int]:
    """Return a dict of {key: size} for all objects under the given prefix."""
    objects = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = obj["Size"]
    return objects


def sync_directory(
    s3, bucket: str, local_dir: str, s3_prefix: str, *, download: bool = False
) -> tuple[int, int, int]:
    """Sync a local directory to S3. Returns (uploaded, downloaded, skipped)."""
    local_path = Path(local_dir)
    if not local_path.is_dir():
        if download:
            local_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✗ Local directory not found: {local_dir}")
            return 0, 0, 0

    remote_objects = get_s3_objects(s3, bucket, s3_prefix)

    uploaded = 0
    downloaded = 0
    skipped = 0

    # Build a set of local relative keys for download comparison
    local_keys: dict[str, int] = {}
    for file_path in sorted(local_path.rglob("*")):
        if not file_path.is_file() or should_exclude(file_path):
            continue

        relative = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative}"
        local_size = file_path.stat().st_size
        local_keys[s3_key] = local_size

        if s3_key in remote_objects and remote_objects[s3_key] == local_size:
            skipped += 1
            continue

        print(f"  ↑ {relative}")
        s3.upload_file(str(file_path), bucket, s3_key)
        uploaded += 1

    # Download files that exist only on S3
    if download:
        for s3_key, remote_size in sorted(remote_objects.items()):
            if s3_key in local_keys or should_exclude(s3_key):
                continue

            relative = s3_key.removeprefix(f"{s3_prefix}/")
            dest = local_path / relative
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"  ↓ {relative}")
            s3.download_file(bucket, s3_key, str(dest))
            downloaded += 1

    return uploaded, downloaded, skipped


def main():
    parser = argparse.ArgumentParser(description="Sync local data directories to S3.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Also download S3-only files to the local directory",
    )
    args = parser.parse_args()

    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    ]
    if missing := [v for v in required_vars if not os.getenv(v)]:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    bucket = os.getenv("AWS_BUCKET")
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    total_uploaded = 0
    total_downloaded = 0
    total_skipped = 0

    for local_dir, s3_prefix in SYNC_PAIRS:
        print(f"\n⟳ Syncing {local_dir} → s3://{bucket}/{s3_prefix}")
        uploaded, downloaded, skipped = sync_directory(
            s3, bucket, local_dir, s3_prefix, download=args.download
        )
        total_uploaded += uploaded
        total_downloaded += downloaded
        total_skipped += skipped
        print(f"  ✓ {uploaded} uploaded, {downloaded} downloaded, {skipped} unchanged")

    print(
        f"\nDone — {total_uploaded} uploaded, {total_downloaded} downloaded, "
        f"{total_skipped} unchanged"
    )


if __name__ == "__main__":
    main()
