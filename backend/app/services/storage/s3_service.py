import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("s3_service")

class S3Service:
    def __init__(self):
        self.bucket_name = settings.aws_s3_bucket
        self.region = settings.aws_region
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=self.region,
        )

    def upload_file(self, local_path: Path, s3_key: str, content_type: str) -> str:
        """Uploads a local file to S3 and returns the S3 key."""
        try:
            self.s3_client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info("s3_upload_success", bucket=self.bucket_name, key=s3_key)
            return s3_key
        except ClientError as e:
            logger.error("s3_upload_failed", error=str(e), key=s3_key)
            raise

    def generate_presigned_url(self, s3_key: str, expiration_seconds: int = 3600) -> str:
        """Generates a secure presigned download/streaming URL valid for 1 hour."""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration_seconds,
            )
            return url
        except ClientError as e:
            logger.error("s3_presigned_url_failed", error=str(e), key=s3_key)
            return ""

s3_service = S3Service()
