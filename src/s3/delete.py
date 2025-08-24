"""
Vector chunk deletion functionality for S3 storage.

This module provides functions to delete vector chunks from S3.
"""

import boto3
from typing import Dict, Any, Optional
from ..config.env import AWS_REGION
from .chunker import create_chunk_key


def delete_vector_chunk(
    chunk_id: str,
    bucket_name: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> Dict[str, Any]:
    """
    Delete a vector chunk from S3.
    
    Args:
        chunk_id: The chunk identifier
        bucket_name: S3 bucket name
        s3_client: Optional S3 client. Creates new one if None
        
    Returns:
        Dictionary containing:
        - chunk_id: The chunk identifier
        - s3_key: The S3 object key that was deleted
        - success: Boolean indicating deletion success
    """
    # Create S3 key using chunker
    vectors_key = create_chunk_key(chunk_id)
    
    try:
        # Create S3 client if not provided
        if s3_client is None:
            s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Delete the chunk
        s3_client.delete_object(Bucket=bucket_name, Key=vectors_key)
        
        return {
            "chunk_id": chunk_id,
            "s3_key": vectors_key,
            "success": True
        }
        
    except Exception as e:
        return {
            "chunk_id": chunk_id,
            "s3_key": vectors_key,
            "success": False,
            "error": str(e)
        }