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
            # Use regular search with a larger k, then filter by threshold
            # This is more efficient than range_search + manual k-limiting
            # because FAISS search is optimized for k-nearest neighbors
            search_k = max(k * 2, 50)  # Search more to account for filtering
            distances, indices = self.index.search(query_array, search_k)
            
            # Filter by threshold and limit to k
            filtered_distances = []
            filtered_indices = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist < threshold:  # FAISS uses squared L2, strict inequality
                    filtered_distances.append(dist)
                    filtered_indices.append(idx)
                    if len(filtered_distances) >= k:  # Stop once we have k results
                        break
            
            return filtered_distances, filtered_indices
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
            # Use regular search with larger k, then filter by threshold
            search_k = max(k * 2, 50)  # Search more to account for filtering
            distances, indices = self.index.search(query_embeddings, search_k)
            
            # Process results for each query
            result_distances = []
            result_indices = []
            
            for i in range(len(query_vectors)):
                query_filtered_distances = []
                query_filtered_indices = []
                
                for dist, idx in zip(distances[i], indices[i]):
                    if dist < threshold:  # FAISS uses squared L2, strict inequality
                        query_filtered_distances.append(dist)
                        query_filtered_indices.append(idx)
                        if len(query_filtered_distances) >= k:  # Stop once we have k results
                            break
                
                result_distances.append(query_filtered_distances)
                result_indices.append(query_filtered_indices)
            
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