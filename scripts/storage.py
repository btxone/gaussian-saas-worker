import os
from pathlib import Path

import boto3
import httpx


def download_file(url: str, destination: Path) -> Path:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    return destination


def upload_outputs(outputs: dict[str, Path], output_prefix: str) -> dict[str, str]:
    bucket = os.environ["S3_BUCKET"]
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT") or None,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY") or None,
        aws_secret_access_key=os.getenv("S3_SECRET_KEY") or None,
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )
    uploaded: dict[str, str] = {}
    for name, path in outputs.items():
        key = f"{output_prefix}/{path.name}"
        client.upload_file(str(path), bucket, key)
        uploaded[name] = key
    return uploaded

