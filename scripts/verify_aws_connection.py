"""Verify that configured AWS credentials can authenticate with AWS STS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the AWS connectivity check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", help="AWS region; defaults to AWS configuration")
    return parser.parse_args()


def main() -> int:
    """Authenticate with STS and print the identity associated with the credentials."""
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    try:
        session = boto3.Session(region_name=args.region)
        credentials = session.get_credentials()
        if credentials is None:
            raise NoCredentialsError()

        identity = session.client("sts").get_caller_identity()
        print(f"AWS connection successful for {identity['Arn']}")
        print(f"Account: {identity['Account']}")
        return 0
    except (BotoCoreError, ClientError) as error:
        print(f"AWS connection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
