"""
Vector chunk upload functionality for S3 storage.

This module provides functions to upload batches of vectors to S3 as chunks.
Metadata is handled by the internal storage system, not stored in S3.
"""

import pickle
import boto3
from typing import List, Dict, Any, Optional
from ..vectors.vectors import Vector
from .chunker import prepare_vectors_for_storage, get_chunk_info, create_chunk_key

def upload_vector_chunk(
    vectors: List[Vector], 
    s3_client: boto3.client,
    bucket_name: str,
    chunk_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload a batch of vectors to S3 as a single chunk.
    
    Args:
        vectors: List of Vector objects
        s3_client: S3 client instance
        bucket_name: S3 bucket name
        chunk_id: Chunk ID for the upload
        
    Returns:
        Dictionary containing:
        - chunk_id: The chunk identifier
        - number_of_vectors: Number of vectors in the chunk
        - s3_key: The S3 object key for the chunk
        - success: Boolean indicating upload success
    """
    # Prepare vectors for storage
    vectors_array = prepare_vectors_for_storage(vectors)
    
    # Get chunk info
    chunk_info = get_chunk_info(vectors, chunk_id)
    vectors_key = chunk_info["s3_key"]
    
    try:
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
    s3_client: boto3.client,
    bucket_name: str
) -> Dict[str, Any]:
    """
    Download a vector chunk from S3.
    
    Args:
        chunk_id: The chunk identifier
        s3_client: S3 client instance
        bucket_name: S3 bucket name
        
    Returns:
        Dictionary containing:
        - vectors: numpy array of vectors
        - success: Boolean indicating download success
    """
    # Create S3 key using chunker
    vectors_key = create_chunk_key(chunk_id)
    
    try:
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


def download_multiple_vector_chunks(
    chunk_ids: List[str],
    s3_client: boto3.client,
    bucket_name: str
) -> Dict[str, Any]:
    """
    Download multiple vector chunks from S3.
    
    TODO: Optimize this with ThreadPoolExecutor or async for better performance
    
    Args:
        chunk_ids: List of chunk identifiers to download
        s3_client: S3 client instance
        bucket_name: S3 bucket name
        
    Returns:
        Dictionary containing:
        - chunks: Dict mapping chunk_id -> numpy array of vectors
        - success: Boolean indicating if all downloads succeeded
        - errors: Dict mapping chunk_id -> error message for failed downloads
    """
    chunks = {}
    errors = {}
    overall_success = True
    
    # Download each chunk sequentially
    for chunk_id in chunk_ids:
        result = download_vector_chunk(
            chunk_id=chunk_id,
            s3_client=s3_client,
            bucket_name=bucket_name
        )
        
        if result["success"]:
            chunks[chunk_id] = result["vectors"]
        else:
            errors[chunk_id] = result.get("error", "Unknown error")
            overall_success = False
    
    return {
        "chunks": chunks,
        "success": overall_success,
        "errors": errors,
        "total_requested": len(chunk_ids),
        "total_successful": len(chunks),
        "total_failed": len(errors)
    }
