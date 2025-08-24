"""
Vector chunking functionality for S3 storage.

This module provides functions to process and prepare vectors for storage,
separating the chunking logic from upload operations.
"""

import numpy as np
from typing import List, Dict, Any


def prepare_vectors_for_storage(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Prepare a list of vectors for storage by converting to consistent format.
    
    Args:
        vectors: List of numpy arrays
        
    Returns:
        numpy array with all vectors stacked and converted to float32
    """
    # Convert to float32 and stack vectors into a single numpy array
    vectors_array = np.stack([vector.astype(np.float32) for vector in vectors])
    return vectors_array


def create_chunk_key(chunk_id: str) -> str:
    """
    Create S3 key for a vector chunk.
    
    Args:
        chunk_id: The chunk identifier
        
    Returns:
        S3 key string for the chunk
    """
    return f"chunks/{chunk_id}.pkl"


def get_chunk_info(vectors: List[np.ndarray], chunk_id: str) -> Dict[str, Any]:
    """
    Get basic information about a vector chunk.
    
    Args:
        vectors: List of numpy arrays
        chunk_id: The chunk identifier
        
    Returns:
        Dictionary with chunk information
    """
    return {
        "chunk_id": chunk_id,
        "number_of_vectors": len(vectors),
        "s3_key": create_chunk_key(chunk_id)
    }


def get_vector_from_chunk(chunk_data: np.ndarray, offset: int) -> np.ndarray:
    """
    Extract a specific vector from a downloaded chunk using offset.
    
    Args:
        chunk_data: numpy array containing the chunk vectors (from download_vector_chunk)
        offset: Index of the vector to retrieve (0-based)
        
    Returns:
        numpy array containing the specific vector
        
    Raises:
        IndexError: If offset is out of bounds
    """
    if offset >= len(chunk_data) or offset < 0:
        raise IndexError(f"Offset {offset} is out of bounds for chunk with {len(chunk_data)} vectors")
    
    return chunk_data[offset]
