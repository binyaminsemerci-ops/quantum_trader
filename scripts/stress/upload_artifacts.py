#!/usr/bin/env python3
"""Zip and upload artifacts/stress to a cloud provider using SDK or CLI.

Supported providers: s3 (AWS), gcs (GCS), azure (Azure Blob).

Prefers SDK if available (boto3 for S3, google-cloud-storage for GCS, azure-storage-blob for Azure),
reading credentials from environment variables (AWS_ACCESS_KEY_ID, etc.).
Falls back to CLI if SDK not available or fails.

Examples:
  python scripts/stress/upload_artifacts.py --provider s3 --dest s3://my-bucket/path/stress.zip
  python scripts/stress/upload_artifacts.py --provider gcs --dest gs://my-bucket/path/stress.zip
  python scripts/stress/upload_artifacts.py --provider azure --dest mycontainer/stress.zip

Notes:
- For Azure the dest should be in the form <container>/<blobname>. The az CLI must be logged in and/or the storage account configured.
- SDK requires installing boto3, google-cloud-storage, azure-storage-blob.
"""
import argparse
import shutil
import subprocess
from pathlib import Path
import time
import sys
import os

# Try to import SDKs
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from azure.storage.blob import BlobServiceClient
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "artifacts" / "stress"


def zip_artifacts(outdir: Path) -> Path:
    if not outdir.exists():
        raise SystemExit(f"Artifacts dir not found: {outdir}")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    zip_base = outdir.parent / f"stress_artifacts_{ts}"
    archive = shutil.make_archive(str(zip_base), 'zip', root_dir=str(outdir))
    return Path(archive)


def rotate_zip_archives(artifacts_root: Path, keep: int = 5) -> None:
    """Keep only the newest `keep` stress_artifacts_*.zip files under artifacts_root.
    Older files beyond this count are deleted.
    """
    zips = sorted(
        artifacts_root.glob("stress_artifacts_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in zips[keep:]:
        try:
            old.unlink()
            print(f"Removed old archive: {old}")
        except Exception as e:
            print(f"Warning: could not remove {old}: {e}")


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
    return p.returncode


def upload_s3(zip_path: Path, dest: str, retries: int = 0, retry_delay: float = 1.0) -> int:
    if HAS_BOTO3:
        try:
            # Parse bucket and key from s3://bucket/key
            if not dest.startswith('s3://'):
                raise ValueError("S3 dest must start with s3://")
            bucket_key = dest[5:]
            if '/' not in bucket_key:
                raise ValueError("S3 dest must be s3://bucket/key")
            bucket, key = bucket_key.split('/', 1)
            s3 = boto3.client('s3')
            attempt = 0
            while True:
                try:
                    s3.upload_file(str(zip_path), bucket, key)
                    return 0
                except Exception as e:
                    attempt += 1
                    if attempt > retries:
                        raise
                    time.sleep(retry_delay)
        except Exception as e:
            print(f"SDK upload failed: {e}, falling back to CLI")
    # Fallback to CLI
    if not shutil.which('aws'):
        raise SystemExit('aws CLI not found on PATH and boto3 not available')
    attempt = 0
    while True:
        rc = run_cmd(['aws', 's3', 'cp', str(zip_path), dest])
        if rc == 0 or attempt >= retries:
            return rc
        attempt += 1
        time.sleep(retry_delay)


def upload_gcs(zip_path: Path, dest: str, retries: int = 0, retry_delay: float = 1.0) -> int:
    if HAS_GCS:
        try:
            if not dest.startswith('gs://'):
                raise ValueError("GCS dest must start with gs://")
            bucket_key = dest[5:]
            if '/' not in bucket_key:
                raise ValueError("GCS dest must be gs://bucket/key")
            bucket, key = bucket_key.split('/', 1)
            client = storage.Client()
            bucket_obj = client.bucket(bucket)
            blob = bucket_obj.blob(key)
            attempt = 0
            while True:
                try:
                    blob.upload_from_filename(str(zip_path))
                    return 0
                except Exception as e:
                    attempt += 1
                    if attempt > retries:
                        raise
                    time.sleep(retry_delay)
        except Exception as e:
            print(f"SDK upload failed: {e}, falling back to CLI")
    # Fallback to CLI
    if not shutil.which('gsutil'):
        raise SystemExit('gsutil not found on PATH and google-cloud-storage not available')
    attempt = 0
    while True:
        rc = run_cmd(['gsutil', 'cp', str(zip_path), dest])
        if rc == 0 or attempt >= retries:
            return rc
        attempt += 1
        time.sleep(retry_delay)


def upload_azure(zip_path: Path, dest: str, retries: int = 0, retry_delay: float = 1.0) -> int:
    if HAS_AZURE:
        try:
            # dest expected container/blob
            if '/' not in dest:
                raise ValueError('Azure dest must be container/blob')
            container, blob = dest.split('/', 1)
            account_url = os.environ.get('AZURE_STORAGE_ACCOUNT_URL')
            if not account_url:
                raise ValueError("AZURE_STORAGE_ACCOUNT_URL not set")
            credential = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY') or os.environ.get('AZURE_STORAGE_SAS_TOKEN')
            if not credential:
                raise ValueError("AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_SAS_TOKEN not set")
            blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob)
            attempt = 0
            while True:
                try:
                    with open(zip_path, 'rb') as data:
                        blob_client.upload_blob(data, overwrite=True)
                    return 0
                except Exception as e:
                    attempt += 1
                    if attempt > retries:
                        raise
                    time.sleep(retry_delay)
        except Exception as e:
            print(f"SDK upload failed: {e}, falling back to CLI")
    # Fallback to CLI
    if not shutil.which('az'):
        raise SystemExit('az CLI not found on PATH and azure-storage-blob not available')
    # dest expected container/blob
    if '/' not in dest:
        raise SystemExit('Azure dest must be container/blob')
    container, blob = dest.split('/', 1)
    attempt = 0
    while True:
        rc = run_cmd(['az', 'storage', 'blob', 'upload', '--container-name', container, '--file', str(zip_path), '--name', blob])
        if rc == 0 or attempt >= retries:
            return rc
        attempt += 1
        time.sleep(retry_delay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--provider', choices=['s3', 'gcs', 'azure'], required=False)
    p.add_argument('--dest', required=False, help='Destination path (s3://..., gs://..., or container/blob for azure)')
    p.add_argument('--zip-only', action='store_true', help='Only create the zip archive and exit')
    p.add_argument('--retries', type=int, default=2, help='Retry count for upload on transient failures')
    p.add_argument('--retry-delay', type=float, default=2.0, help='Delay between retries (seconds)')
    p.add_argument('--retain', type=int, default=0, help='After zipping, retain only the latest N zip archives (delete older). 0 disables.')
    args = p.parse_args()

    zip_path = zip_artifacts(OUTDIR)
    print('Created archive:', zip_path)
    if args.retain and args.retain > 0:
        rotate_zip_archives(OUTDIR.parent, keep=args.retain)
    if args.zip_only:
        print('Zip-only mode; skipping upload')
        return 0

    if not args.provider or not args.dest:
        print('Error: --provider and --dest are required unless --zip-only is used', file=sys.stderr)
        return 2

    if args.provider == 's3':
        rc = upload_s3(zip_path, args.dest, retries=args.retries, retry_delay=args.retry_delay)
    elif args.provider == 'gcs':
        rc = upload_gcs(zip_path, args.dest, retries=args.retries, retry_delay=args.retry_delay)
    else:
        rc = upload_azure(zip_path, args.dest, retries=args.retries, retry_delay=args.retry_delay)

    if rc == 0:
        print('Upload succeeded')
    else:
        print('Upload failed with code', rc)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
