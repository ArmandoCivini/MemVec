"""
Query functionality for the MemVec system.

This module provides functionality to search the index and retrieve
corresponding vectors and metadata from S3.
"""

from typing import List, Dict, Any, Tuple
import heapq
import boto3
from .index.index import HNSWIndex
from .s3.chunk_upload import download_multiple_vector_chunks
from .vectors.vectors import Vector
from .vectors.pointer import Pointer
from .processes.components import SentenceTransformerEmbedding
from .config.env import AWS_REGION


def query_system(
    query_text: str,
    index: HNSWIndex,
    bucket_name: str,
    embedding_generator,
    k: int = 5,
    threshold: float = None,
    s3_client=None
) -> Dict[str, Any]:
    """
    Query the MemVec system with text and retrieve matching vectors from S3.
    
    Args:
        query_text: Text query to search for
        index: HNSW index to search
        bucket_name: S3 bucket name where chunks are stored
        embedding_generator: Embedding generator to encode the query
        k: Number of results to return
        threshold: Optional maximum distance threshold for results
        s3_client: Optional S3 client
        
    Returns:
        Dictionary containing:
        - query_embedding: The encoded query embedding
        - search_results: List of matching vectors with metadata
        - distances: Similarity distances for each result
        - success: Boolean indicating success
    """
    try:
        # Generate embedding for query text
        query_embeddings = embedding_generator.generate([query_text])
        query_embedding = query_embeddings[0]
        
        # Search the index (now accepts raw embeddings)
        distances, vector_ids = index.search(query_embedding, k=k, threshold=threshold)
        
        # If no results found
        if len(vector_ids) == 0:
            return {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "search_results": [],
                "total_found": 0,
                "success": True
            }
        
        # Group vector IDs by chunk for efficient S3 retrieval
        chunk_groups = _group_vectors_by_chunk(vector_ids)
        
        # Use heap to maintain order by distance
        result_heap = []
        
        # Create S3 client if not provided
        if s3_client is None:
            s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Fetch vectors from S3 using batch download
        chunk_ids = list(chunk_groups.keys())
        download_results = download_multiple_vector_chunks(
            chunk_ids=chunk_ids,
            s3_client=s3_client,
            bucket_name=bucket_name
        )
        
        # Process successfully downloaded chunks
        for chunk_id, vector_indices in chunk_groups.items():
            if chunk_id in download_results["chunks"]:
                chunk_vectors = download_results["chunks"][chunk_id]  # This is a numpy array
                
                # Find the specific vectors we need using offset directly
                for vector_index in vector_indices:
                    # Decode to get offset within chunk
                    document, chunk, offset = Pointer.decode(vector_index)
                    
                    # Get vector directly by offset
                    if offset < len(chunk_vectors):
                        vector_values = chunk_vectors[offset]  # numpy array
                        
                        # Find the distance for this vector
                        original_position = vector_ids.index(vector_index)
                        distance = distances[original_position]
                        
                        # Create a simple result without full Vector object
                        result = {
                            "vector_values": vector_values.tolist(),  # Convert to list
                            "distance": distance,
                            "document_id": document,
                            "chunk_id": Pointer.generate_chunk_id(document, chunk),
                            "offset": offset,
                            "metadata": {"vector_index": vector_index}  # Basic metadata
                        }
                        heapq.heappush(result_heap, (distance, original_position, result))
        
        # Extract results from heap (already ordered by distance)
        ordered_results = [result for _, _, result in sorted(result_heap, key=lambda x: x[1])]
        
        return {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "search_results": ordered_results,
            "total_found": len(ordered_results),
            "success": True
        }
        
    except Exception as e:
        return {
            "query_text": query_text,
            "query_embedding": None,
            "search_results": [],
            "total_found": 0,
            "success": False,
            "error": str(e)
        }


def _group_vectors_by_chunk(vector_ids: List[int]) -> Dict[int, List[int]]:
    """
    Group vector IDs by their chunk ID for efficient S3 retrieval.
    
    Args:
        vector_ids: List of vector indices
        
    Returns:
        Dictionary mapping chunk_id -> list of vector indices
    """
    chunk_groups = {}
    
    for vector_id in vector_ids:
        # Decode the vector ID to get document and chunk
        document, chunk, offset = Pointer.decode(vector_id)
        
        # Generate chunk ID
        chunk_id = Pointer.generate_chunk_id(document, chunk)
        
        if chunk_id not in chunk_groups:
            chunk_groups[chunk_id] = []
        
        chunk_groups[chunk_id].append(vector_id)
    
    return chunk_groups


def batch_query_system(
    query_texts: List[str],
    index: HNSWIndex,
    bucket_name: str,
    embedding_generator,
    k: int = 5,
    threshold: float = None,
    s3_client=None
) -> List[Dict[str, Any]]:
    """
    Query the MemVec system with multiple text queries.
    
    Args:
        query_texts: List of text queries to search for
        index: HNSW index to search
        bucket_name: S3 bucket name where chunks are stored
        embedding_generator: Embedding generator to encode the queries
        k: Number of results to return per query
        threshold: Optional maximum distance threshold for results
        s3_client: Optional S3 client
        
    Returns:
        List of query results, one per input query
    """
    results = []
    
    for query_text in query_texts:
        result = query_system(
            query_text=query_text,
            index=index,
            bucket_name=bucket_name,
            embedding_generator=embedding_generator,
            k=k,
            threshold=threshold,
            s3_client=s3_client
        )
        results.append(result)
    
    return results
