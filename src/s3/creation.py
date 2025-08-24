"""
S3 bucket creation functionality for MemVec.

This module provides functions to create S3 buckets for vector storage.
"""

import boto3
from typing import Dict, Any, Optional
from ..config.env import AWS_REGION


def create_s3_bucket(
    bucket_name: str,
    region: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> Dict[str, Any]:
    """
    Create an S3 bucket if it doesn't exist.
    
    Args:
        bucket_name: Name of the S3 bucket to create
        region: AWS region. Uses default if None
        s3_client: Optional S3 client. Creates new one if None
        
    Returns:
        Dictionary containing:
        - bucket_name: The bucket name
        - created: Boolean indicating if bucket was created (False if already existed)
        - success: Boolean indicating operation success
        - error: Error message if operation failed
    """
    if region is None:
        region = AWS_REGION
    
    try:
        # Create S3 client if not provided
        if s3_client is None:
            s3_client = boto3.client('s3', region_name=region)
        
        # Check if bucket already exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            return {
                "bucket_name": bucket_name,
                "created": False,
                "success": True,
                "message": "Bucket already exists"
            }
        except Exception as e:
            # If it's a 404 or NoSuchBucket, the bucket doesn't exist - we should create it
            # For other errors (like permission issues), we should fail
            error_code = getattr(e.response, 'Error', {}).get('Code', '') if hasattr(e, 'response') else ''
            if '404' in str(e) or 'NoSuchBucket' in str(e) or error_code == 'NoSuchBucket':
                # Bucket doesn't exist, proceed to create it
                pass
            else:
                # Other errors (like permission issues)
                return {
                    "bucket_name": bucket_name,
                    "created": False,
                    "success": False,
                    "error": f"Error checking bucket: {str(e)}"
                }
        
        # Create the bucket
        if region == 'us-east-1':
            # us-east-1 doesn't need LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            # Other regions need LocationConstraint
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        return {
            "bucket_name": bucket_name,
            "created": True,
            "success": True,
            "message": "Bucket created successfully"
        }
        
    except Exception as e:
        return {
            "bucket_name": bucket_name,
            "created": False,
            "success": False,
            "error": str(e)
        }