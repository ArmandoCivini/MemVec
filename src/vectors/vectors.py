"""
Simple Vector class for embedding storage.

This module provides a lightweight Vector class that represents a single embedding
with minimal dependencies.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class Vector:
    """
    Simple vector class for storing embeddings with metadata.
    """
    
    def __init__(self, id: str, values: List[float], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Vector instance.
        
        Args:
            id: Unique identifier for the vector
            values: The embedding values as a list of floats
            metadata: Optional dictionary with extra info (default empty)
        """
        self.id = id
        self.values = values
        self.metadata = metadata or {}
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert values to numpy array.
        
        Returns:
            numpy array of the embedding values
        """
        return np.array(self.values, dtype=np.float32)
    
    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"Vector(id='{self.id}', dim={len(self.values)}, metadata_keys={list(self.metadata.keys())})"
