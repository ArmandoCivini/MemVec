"""
FAISS index management for MemVec.

This module provides a simple class to manage a FAISS HNSW index for vector search.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
from ..vectors.vectors import Vector
from ..config.contants import DEFAULT_HNSW_M, DEFAULT_SEARCH_K


class HNSWIndex:
    """
    FAISS HNSW index manager for vector search.
    """
    
    def __init__(self, dimension: int, m: int = DEFAULT_HNSW_M):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimension
            m: Number of connections per node (default from config)
        """
        self.dimension = dimension
        self.m = m
        
        # Base index (stores vectors internally)
        base_index = faiss.IndexHNSWFlat(dimension, m)
        
        # Wrap with ID mapping (decouples storage from index)
        self.index = faiss.IndexIDMap2(base_index)
    
    def add_vectors(self, vectors: List[Vector]) -> None:
        """
        Add vectors to the index using their own index values.
        
        Args:
            vectors: List of Vector objects to add (all vectors must have an index)
        """
        if not vectors:
            return
        
        # Extract embeddings and indices in single iteration using zip and unpacking
        embeddings_and_ids = [(vector.values, vector.index) for vector in vectors]
        embeddings, vector_ids = zip(*embeddings_and_ids)
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings, dtype=np.float32)
        vector_ids = np.array(vector_ids, dtype=np.int64)
        
        # Add embeddings to FAISS with vector IDs
        self.index.add_with_ids(embeddings, vector_ids)
    
    def search(self, query_embedding: List[float], k: int = DEFAULT_SEARCH_K, threshold: Optional[float] = None) -> Tuple[List[float], List[int]]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embedding: List of floats representing the query embedding
            k: Number of nearest neighbors to return
            threshold: Optional maximum distance threshold for results
            
        Returns:
            Tuple of (distances, vector_ids)
        """
        query_array = np.array([query_embedding], dtype=np.float32)
        
        if threshold is not None:
            # Use FAISS range_search for efficient threshold-based search
            # Note: FAISS uses strict inequality (distance < threshold), not (distance <= threshold)
            # Also: FAISS uses squared L2 distance by default
            lims, distances, indices = self.index.range_search(query_array, threshold)
            
            # range_search returns all results within threshold, but we may want to limit to k
            start_idx = lims[0]
            end_idx = lims[1]
            
            result_distances = distances[start_idx:end_idx]
            result_indices = indices[start_idx:end_idx]
            
            # Limit to k results if more than k found
            if len(result_distances) > k:
                # Sort by distance and take top k
                sorted_pairs = sorted(zip(result_distances, result_indices))
                result_distances = [d for d, _ in sorted_pairs[:k]]
                result_indices = [i for _, i in sorted_pairs[:k]]
            
            return result_distances.tolist(), result_indices.tolist()
        else:
            # Use regular search when no threshold
            distances, indices = self.index.search(query_array, k)
            return distances[0].tolist(), indices[0].tolist()
    
    def multi_search(self, query_vectors: List[Vector], k: int = DEFAULT_SEARCH_K, threshold: Optional[float] = None) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Search the index for similar vectors using multiple query vectors.
        
        Args:
            query_vectors: List of Vector objects to search for
            k: Number of nearest neighbors to return for each query
            threshold: Optional maximum distance threshold for results
            
        Returns:
            Tuple of (distances, vector_ids) where each is a list of lists
        """
        if not query_vectors:
            return [], []
        
        # Convert all query vectors to numpy array
        query_embeddings = np.array([vector.values for vector in query_vectors], dtype=np.float32)
        
        if threshold is not None:
            # Use FAISS range_search for efficient threshold-based search
            # Note: FAISS uses strict inequality (distance < threshold), not (distance <= threshold)
            # Also: FAISS uses squared L2 distance by default
            lims, distances, indices = self.index.range_search(query_embeddings, threshold)
            
            # Process results for each query
            result_distances = []
            result_indices = []
            
            for i in range(len(query_vectors)):
                start_idx = lims[i]
                end_idx = lims[i + 1]
                
                query_distances = distances[start_idx:end_idx]
                query_indices = indices[start_idx:end_idx]
                
                # Limit to k results if more than k found
                if len(query_distances) > k:
                    # Sort by distance and take top k
                    sorted_pairs = sorted(zip(query_distances, query_indices))
                    query_distances = [d for d, _ in sorted_pairs[:k]]
                    query_indices = [i for _, i in sorted_pairs[:k]]
                
                result_distances.append(query_distances.tolist())
                result_indices.append(query_indices.tolist())
            
            return result_distances, result_indices
        else:
            # Use regular search when no threshold
            distances, indices = self.index.search(query_embeddings, k)
            return distances.tolist(), indices.tolist()
    
    def get_info(self) -> dict:
        """
        Get information about the index.
        
        Returns:
            Dictionary with index information
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "m": self.m,
            "is_trained": self.index.is_trained
        }
    
    def size(self) -> int:
        """
        Get the number of vectors in the index.
        
        Returns:
            Number of vectors in the index
        """
        return self.index.ntotal