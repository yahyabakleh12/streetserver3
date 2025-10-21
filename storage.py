import os

import boto3

S3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT"),  # set for MinIO
    aws_access_key_id=os.getenv("S3_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET"),
)
BUCKET = os.getenv("S3_BUCKET", "parkonic-snapshots")


def put_snapshot(key: str, body: bytes, content_type: str = "image/jpeg"):
    S3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=content_type)
    return f"s3://{BUCKET}/{key}"
