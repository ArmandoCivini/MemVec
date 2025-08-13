import os
from dotenv import load_dotenv
load_dotenv()

REDIS_HOST=os.getenv("REDIS_HOST", "localhost")
REDIS_PORT=int(os.getenv("REDIS_PORT", 6379))

AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION=os.getenv("AWS_REGION", "us-east-1")

S3_BUCKET_NAME=os.getenv("S3_BUCKET_NAME", "my-vector-bucket")
