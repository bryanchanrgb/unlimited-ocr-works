"""Run AWS Textract's forms and tables analysis on an image or PDF."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ASYNC_SUFFIXES = {".pdf", ".tif", ".tiff"}
FEATURE_TYPES = ["FORMS", "TABLES"]
MAX_SYNC_BYTES = 10 * 1024 * 1024
MAX_ASYNC_PDF_BYTES = 500 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a Textract request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a PNG, JPEG, TIFF, or PDF")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: data/processed/<input-name>/textract/result.json)",
    )
    parser.add_argument(
        "--bucket",
        help="S3 bucket for PDF/TIFF asynchronous analysis (same region as Textract)",
    )
    parser.add_argument("--region", help="AWS region; defaults to AWS configuration")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between async job status checks (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Maximum async wait time in seconds (default: 1800)",
    )
    return parser.parse_args()


def validate_input(input_path: Path, bucket: str | None) -> bool:
    """Validate the input path and return whether it needs async Textract."""
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | ASYNC_SUFFIXES:
        raise ValueError("Input must be a PNG, JPEG, TIFF, or PDF file")

    if suffix in ASYNC_SUFFIXES:
        if not bucket:
            raise ValueError("PDF and TIFF analysis requires --bucket for S3 upload")
        if input_path.stat().st_size > MAX_ASYNC_PDF_BYTES:
            raise ValueError(
                f"Input exceeds the Textract asynchronous size limit of "
                f"{MAX_ASYNC_PDF_BYTES:,} bytes"
            )
        return True

    if input_path.stat().st_size > MAX_SYNC_BYTES:
        raise ValueError("Synchronous image analysis supports files up to 10 MiB")
    return False


def collect_async_result(
    textract: Any,
    job_id: str,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    """Poll an asynchronous Textract job and collect every paginated block."""
    deadline = time.monotonic() + timeout
    response = textract.get_document_analysis(JobId=job_id)
    while response.get("JobStatus") == "IN_PROGRESS":
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Textract job {job_id} did not finish within {timeout:g} seconds"
            )
        time.sleep(poll_interval)
        response = textract.get_document_analysis(JobId=job_id)

    terminal_response = response
    status = terminal_response.get("JobStatus")
    blocks = list(response.get("Blocks", []))
    warnings = list(response.get("Warnings", []))
    next_token = response.get("NextToken")
    while next_token:
        response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        blocks.extend(response.get("Blocks", []))
        warnings.extend(response.get("Warnings", []))
        next_token = response.get("NextToken")

    return {
        "job_status": status,
        "status_message": terminal_response.get("StatusMessage"),
        "model_version": terminal_response.get("AnalyzeDocumentModelVersion"),
        "document_metadata": terminal_response.get("DocumentMetadata", {}),
        "warnings": warnings,
        "blocks": blocks,
    }


def analyze_async(
    input_path: Path,
    bucket: str,
    textract: Any,
    s3: Any,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    """Upload a PDF/TIFF, run async forms-and-tables analysis, and remove its S3 copy."""
    key = f"textract-inputs/{uuid.uuid4().hex}/{input_path.name}"
    job_id = None
    terminal_status = False
    try:
        s3.upload_file(str(input_path), bucket, key)
        job_id = textract.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
            FeatureTypes=FEATURE_TYPES,
        )["JobId"]
        result = collect_async_result(textract, job_id, poll_interval, timeout)
        terminal_status = True
        if result["job_status"] == "FAILED":
            raise RuntimeError(
                result.get("status_message") or f"Textract job {job_id} failed"
            )
        if result["job_status"] not in {"SUCCEEDED", "PARTIAL_SUCCESS"}:
            raise RuntimeError(
                f"Unexpected Textract job status: {result['job_status']!r}"
            )
        return result
    finally:
        if job_id is None or terminal_status:
            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except (BotoCoreError, ClientError) as error:
                print(
                    f"Warning: could not delete temporary S3 object {key}: {error}",
                    file=sys.stderr,
                )
        else:
            print(
                f"Warning: async job {job_id} may still be running; temporary input remains at s3://{bucket}/{key}",
                file=sys.stderr,
            )


def make_output(
    input_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Wrap Textract blocks with source and feature metadata for the JSON file."""
    return {
        "source": str(input_path),
        "feature_types": FEATURE_TYPES,
        **result,
    }


def main() -> int:
    """Run Textract, save its complete block response as JSON, and report the path."""
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    try:
        use_async = validate_input(input_path, args.bucket)
        if args.poll_interval <= 0 or args.timeout <= 0:
            raise ValueError("--poll-interval and --timeout must be greater than zero")

        session = boto3.Session(region_name=args.region)
        region = session.region_name
        if not region:
            raise ValueError(
                "No AWS region configured; set AWS_DEFAULT_REGION or pass --region"
            )

        textract = session.client("textract", region_name=region)
        if use_async:
            s3 = session.client("s3", region_name=region)
            result = analyze_async(
                input_path,
                args.bucket,
                textract,
                s3,
                args.poll_interval,
                args.timeout,
            )
        else:
            response = textract.analyze_document(
                Document={"Bytes": input_path.read_bytes()},
                FeatureTypes=FEATURE_TYPES,
            )
            result = {
                "model_version": response.get("AnalyzeDocumentModelVersion"),
                "document_metadata": response.get("DocumentMetadata", {}),
                "warnings": [],
                "blocks": response.get("Blocks", []),
            }

        payload = make_output(input_path, result)
        output_path = (
            args.output or PROCESSED_DIR / input_path.stem / "textract" / "result.json"
        )
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved Textract JSON to {output_path}")
        return 0
    except (
        BotoCoreError,
        ClientError,
        OSError,
        TimeoutError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"Textract failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
