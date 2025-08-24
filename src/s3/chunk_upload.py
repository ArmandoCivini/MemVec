"""
Vector chunk upload functionality for S3 storage.

This module provides functions to upload batches of vectors to S3 as chunks.
Metadata is handled by the internal storage system, not stored in S3.
"""

import uuid
import pickle
import numpy as np
import boto3
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..config.env import S3_BUCKET_NAME, AWS_REGION
#TODO: implement chunker for vectors

def upload_vector_chunk(
    vectors: List[np.ndarray], 
    chunk_id: Optional[str] = None,
    bucket_name: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> Dict[str, Any]:
    """
    Upload a batch of vectors to S3 as a single chunk.
    
    Args:
        vectors: List of numpy arrays
        chunk_id: Chunk ID for the upload
        bucket_name: S3 bucket name
        s3_client: Optional S3 client. Creates new one if None
        
    Returns:
        Dictionary containing:
        - chunk_id: The chunk identifier
        - number_of_vectors: Number of vectors in the chunk
        - s3_key: The S3 object key for the chunk
        - success: Boolean indicating upload success
    """
    # Convert to float32 and stack vectors into a single numpy array
    vectors_array = np.stack([vector.astype(np.float32) for vector in vectors])
    
    # Define S3 key
    vectors_key = f"chunks/{chunk_id}.pkl"
    
    try:
        # Create S3 client if not provided
        if s3_client is None:
            s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Serialize vectors using pickle for efficient storage
        vectors_data = pickle.dumps(vectors_array)
        
        # Upload vectors to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=vectors_key,
            Body=vectors_data,
            ContentType='application/octet-stream'
        )
        
        return {
            "chunk_id": chunk_id,
            "number_of_vectors": len(vectors),
            "s3_key": vectors_key,
            "success": True
        }
        
    except Exception as e:
        return {
            "chunk_id": chunk_id,
            "number_of_vectors": len(vectors),
            "s3_key": vectors_key,
            "success": False,
            "error": str(e)
        }


def download_vector_chunk(
    chunk_id: str,
    bucket_name: Optional[str] = None,
    s3_client: Optional[boto3.client] = None
) -> Dict[str, Any]:
    """
    Download a vector chunk from S3.
    
    Args:
        chunk_id: The chunk identifier
        bucket_name: S3 bucket name
        s3_client: Optional S3 client. Creates new one if None
        
    Returns:
        Dictionary containing:
        - vectors: numpy array of vectors
        - success: Boolean indicating download success
    """
    # Define S3 key
    vectors_key = f"chunks/{chunk_id}.pkl"
    
    try:
        # Create S3 client if not provided
        if s3_client is None:
            s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Download vectors
        vectors_response = s3_client.get_object(Bucket=bucket_name, Key=vectors_key)
        vectors_data = vectors_response['Body'].read()
        vectors_array = pickle.loads(vectors_data)
        
        return {
            "vectors": vectors_array,
            "success": True
        }
        
    except Exception as e:
        return {
            "vectors": None,
            "success": False,
            "error": str(e)
        }


class MockS3Client:
    """Mock S3 client for testing purposes when S3 is not accessible."""
    
    def __init__(self):
        self.objects = {}
    
    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str = None):
        """Mock put_object method."""
        full_key = f"{Bucket}/{Key}"
        self.objects[full_key] = {
            'Body': Body,
            'ContentType': ContentType
        }
    
    def get_object(self, Bucket: str, Key: str):
        """Mock get_object method."""
        full_key = f"{Bucket}/{Key}"
        if full_key not in self.objects:
            raise Exception(f"Object {full_key} not found")
        
        class MockBody:
            def __init__(self, data):
                self.data = data
            
            def read(self):
                return self.data
        
        return {
            'Body': MockBody(self.objects[full_key]['Body'])
        }
