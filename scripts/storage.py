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


def upload_outputs(outputs: dict[str, Path], output_prefix: str, output_bucket: str | None = None) -> dict[str, str]:
    bucket = output_bucket or _env_first("S3_BUCKET", "S3_BUCKET_NAME", "AWS_S3_BUCKET_NAME", "AWS_BUCKET_NAME")
    if not bucket or bucket == "local":
        raise RuntimeError("S3 bucket is required. Set S3_BUCKET or pass output_bucket in the RunPod payload.")

    client_kwargs = {
        "endpoint_url": _env_first("S3_ENDPOINT", "AWS_S3_ENDPOINT") or None,
        "region_name": _env_first("S3_REGION", "AWS_REGION") or "us-east-1",
    }
    access_key = _env_first("S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
    secret_key = _env_first("S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key

    client = boto3.client("s3", **client_kwargs)
    uploaded: dict[str, str] = {}
    for name, path in outputs.items():
        key = f"{output_prefix}/{path.name}"
        client.upload_file(str(path), bucket, key)
        uploaded[name] = key
    return uploaded


def _env_first(*names: str) -> str | None:
    for name in names:
        value = _clean_env_value(os.getenv(name))
        if value:
            return value
    return None


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    clean_value = value.strip().strip('"').strip("'").strip()
    if clean_value.lower() in {"", "none", "null", "undefined"}:
        return None
    return clean_value
