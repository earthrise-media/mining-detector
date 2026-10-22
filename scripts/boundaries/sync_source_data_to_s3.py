# Syncs local data directories to S3, replicating the behavior of:
#   aws s3 sync ./data/boundaries s3://amw-media/mining-detector-repo-backups/data/boundaries
#   aws s3 sync ./data/outputs/website s3://amw-media/mining-detector-repo-backups/data/outputs/website
# Excludes .DS_Store and .pmtiles files.
# A direction flag is required, so the script never guesses which way to sync:
#   --upload    pushes new or modified local files (by size) to S3
#   --download  pulls files missing locally and refreshes local files whose size
#               or ETag differs from the bucket copy. Never uploads anything.
# Add --dry-run to either to print what would be transferred without changing
# anything locally or on the bucket (it still lists the bucket, so credentials
# are required).
#
# Environment variables (should be set in .env file):
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
# - AWS_BUCKET

# You can run this script with uv if you prefer,
# see https://docs.astral.sh/uv/guides/scripts/.
# To run: `uv run scripts/boundaries/sync_source_data_to_s3.py --upload` (upload only)
# To run: `uv run scripts/boundaries/sync_source_data_to_s3.py --upload` --dry-run (preview upload only)
#     or: `uv run scripts/boundaries/sync_source_data_to_s3.py --download` (download only)
#     or: `uv run scripts/boundaries/sync_source_data_to_s3.py --download --dry-run` (preview download only)

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "python-dotenv",
# ]
# ///

import argparse
import hashlib
import math
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
    ("./data/outputs/rasters", f"{S3_PREFIX}/data/outputs/rasters"),
]

EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_EXTENSIONS = {".pmtiles"}

MIB = 1024 * 1024
# Default multipart chunk size used by boto3's upload_file and the AWS CLI.
DEFAULT_MULTIPART_CHUNK_SIZE = 8 * MIB


def should_exclude(path: Path | str) -> bool:
    p = Path(path)
    return p.name in EXCLUDE_NAMES or p.suffix in EXCLUDE_EXTENSIONS


def get_s3_objects(s3, bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
    """Return a dict of {key: (size, etag)} for all objects under the given prefix."""
    objects = {}
    paginator = s3.get_paginator("list_objects_v2")
    # Trailing slash so e.g. ".../boundaries" doesn't also match ".../boundaries_old/..."
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = (obj["Size"], obj["ETag"].strip('"'))
    return objects


def compute_etag(path: Path, chunk_size: int | None = None) -> str:
    """Compute an S3-style ETag for a local file.

    With no chunk_size, returns the plain MD5 (single-part upload ETag).
    With a chunk_size, returns the multipart ETag: MD5 of the concatenated
    per-part MD5 digests, suffixed with "-<number of parts>".
    """
    if chunk_size is None:
        h = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as f:
            while block := f.read(MIB):
                h.update(block)
        return h.hexdigest()

    part_digests = []
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            part_digests.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    combined = hashlib.md5(b"".join(part_digests), usedforsecurity=False).hexdigest()
    return f"{combined}-{len(part_digests)}"


def local_matches_etag(path: Path, size: int, remote_etag: str) -> bool:
    """Return True if the local file's content matches the remote ETag.

    Handles single-part (plain MD5) and multipart ETags. For multipart, tries the
    boto3/CLI default chunk size plus a chunk size inferred from the part count.
    Returns False when the ETag can't be reproduced (e.g. SSE-KMS objects), so
    the caller errs on the side of re-downloading.
    """
    if "-" not in remote_etag:
        return compute_etag(path) == remote_etag

    try:
        parts = int(remote_etag.rsplit("-", 1)[1])
    except ValueError:
        return False

    candidates = {DEFAULT_MULTIPART_CHUNK_SIZE}
    if parts > 0:
        inferred = math.ceil(size / parts)
        candidates.add(math.ceil(inferred / MIB) * MIB)

    for chunk_size in sorted(candidates):
        if math.ceil(size / chunk_size) != parts:
            continue
        if compute_etag(path, chunk_size) == remote_etag:
            return True
    return False


def upload_directory(
    s3, bucket: str, local_dir: str, s3_prefix: str, *, dry_run: bool = False
) -> tuple[int, int]:
    """Upload new or modified (by size) local files to S3. Returns (uploaded, skipped).

    With dry_run, only prints what would be uploaded; nothing is written to S3.
    """
    local_path = Path(local_dir)
    if not local_path.is_dir():
        print(f"✗ Local directory not found: {local_dir}")
        return 0, 0

    remote_objects = get_s3_objects(s3, bucket, s3_prefix)

    uploaded = 0
    skipped = 0

    for file_path in sorted(local_path.rglob("*")):
        if not file_path.is_file() or should_exclude(file_path):
            continue

        relative = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative.as_posix()}"
        local_size = file_path.stat().st_size

        remote = remote_objects.get(s3_key)
        if remote is not None and remote[0] == local_size:
            skipped += 1
            continue

        label = f"{relative} (overwriting bucket copy)" if remote is not None else relative
        print(f"  ↑ {label}")
        if not dry_run:
            s3.upload_file(str(file_path), bucket, s3_key)
        uploaded += 1

    return uploaded, skipped


def download_directory(
    s3, bucket: str, local_dir: str, s3_prefix: str, *, dry_run: bool = False
) -> tuple[int, int]:
    """Download new or changed S3 files to the local directory.

    A local file is refreshed when its size or ETag differs from the bucket copy.
    Local-only files are left untouched. Never writes to S3.
    With dry_run, only prints what would be downloaded; no local files or
    directories are created or changed.
    Returns (downloaded, skipped).
    """
    local_path = Path(local_dir)
    if not dry_run:
        local_path.mkdir(parents=True, exist_ok=True)

    remote_objects = get_s3_objects(s3, bucket, s3_prefix)

    downloaded = 0
    skipped = 0

    for s3_key, (remote_size, remote_etag) in sorted(remote_objects.items()):
        # Skip excluded files and "folder" placeholder keys created by the S3 console
        if should_exclude(s3_key) or s3_key.endswith("/"):
            continue

        relative = s3_key.removeprefix(f"{s3_prefix}/")
        dest = local_path / relative

        if dest.is_file():
            # Size check first so we only hash files that could plausibly match
            if dest.stat().st_size == remote_size and local_matches_etag(
                dest, remote_size, remote_etag
            ):
                skipped += 1
                continue
            label = f"{relative} (replacing stale local copy)"
        else:
            label = relative

        print(f"  ↓ {label}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # boto3 writes to a temp file and renames on completion, so an interrupted
            # download won't leave a truncated file in place of the old one.
            s3.download_file(bucket, s3_key, str(dest))
        downloaded += 1

    return downloaded, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Sync local data directories with S3. A direction flag is required."
    )
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument(
        "--upload",
        action="store_true",
        help="Upload new or modified (by size) local files to S3. Never downloads.",
    )
    direction.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download new or changed (by size/ETag) files from S3 to the local "
            "directory. Never uploads."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be transferred without changing anything locally or on S3.",
    )
    args = parser.parse_args()

    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_BUCKET",
    ]
    if missing := [v for v in required_vars if not os.getenv(v)]:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    bucket = os.getenv("AWS_BUCKET")
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    total_transferred = 0
    total_skipped = 0
    verb = "downloaded" if args.download else "uploaded"
    if args.dry_run:
        verb = f"would be {verb}"
        print("DRY RUN — nothing will be transferred")

    for local_dir, s3_prefix in SYNC_PAIRS:
        if args.download:
            print(f"\n⟳ Syncing s3://{bucket}/{s3_prefix} → {local_dir}")
            transferred, skipped = download_directory(
                s3, bucket, local_dir, s3_prefix, dry_run=args.dry_run
            )
        else:
            print(f"\n⟳ Syncing {local_dir} → s3://{bucket}/{s3_prefix}")
            transferred, skipped = upload_directory(
                s3, bucket, local_dir, s3_prefix, dry_run=args.dry_run
            )
        total_transferred += transferred
        total_skipped += skipped
        print(f"  ✓ {transferred} {verb}, {skipped} unchanged")

    print(f"\nDone — {total_transferred} {verb}, {total_skipped} unchanged")


if __name__ == "__main__":
    main()
