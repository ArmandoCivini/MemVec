"""
FAISS index management for MemVec.

This module provides a simple class to manage a FAISS HNSW index for vector search.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
from ..vectors.vectors import Vector


class HNSWIndex:
    """
    FAISS HNSW index manager for vector search.
    """
    
    def __init__(self, dimension: int, m: int = 16):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimension
            m: Number of connections per node (default 16)
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
    
    def search(self, query_vector: Vector, k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Search the index for similar vectors.
        
        Args:
            query_vector: Vector to search for
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, vector_ids)
        """
        query_embedding = np.array([query_vector.values], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return distances[0].tolist(), indices[0].tolist()
    
    def multi_search(self, query_vectors: List[Vector], k: int = 5) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Search the index for similar vectors using multiple query vectors.
        
        Args:
            query_vectors: List of Vector objects to search for
            k: Number of nearest neighbors to return for each query
            
        Returns:
            Tuple of (distances, vector_ids) where each is a list of lists
        """
        if not query_vectors:
            return [], []
        
        # Convert all query vectors to numpy array
        query_embeddings = np.array([vector.values for vector in query_vectors], dtype=np.float32)
        distances, indices = self.index.search(query_embeddings, k)
        
        # Convert to lists of lists
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