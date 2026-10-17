"""Verify AWS credentials by printing the authenticated IAM identity."""

from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from output_layout import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    """Parse the optional AWS region override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", help="AWS region; defaults to AWS configuration")
    return parser.parse_args()


def main() -> int:
    """Call STS GetCallerIdentity and print the authenticated IAM identity."""
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    try:
        session = boto3.Session(region_name=args.region)
        if not session.region_name:
            raise ValueError(
                "No AWS region configured; set AWS_DEFAULT_REGION or pass --region"
            )
        identity = session.client("sts").get_caller_identity()
        print(f"Account: {identity.get('Account')}")
        print(f"ARN: {identity.get('Arn')}")
        print(f"User ID: {identity.get('UserId')}")
        print(f"Region: {session.region_name}")
        return 0
    except (BotoCoreError, ClientError, ValueError) as error:
        print(f"AWS credential check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
