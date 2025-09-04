"""
Simple Vector class for embedding storage.

This module provides a lightweight Vector class that represents a single embedding
with minimal dependencies.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .pointer import Pointer


class Vector:
    """
    Vector class for storing embeddings with pointer-based indexing.
    
    All vectors MUST have a pointer (either via components or raw index).
    The pointer encodes document, chunk, and offset information for storage management.
    """
    
    def __init__(self, values: List[float], document: int = None, chunk: int = None, offset: int = None, 
                 index: int = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Vector instance.
        
        Args:
            values: The embedding values as a list of floats
            document: Document number for pointer encoding (if using components)
            chunk: Chunk number for pointer encoding (if using components)
            offset: Offset within chunk for pointer encoding (if using components)
            index: Raw integer index (alternative to components)
            metadata: Optional dictionary with extra info (default empty)
        """
        self.values = values
        self.metadata = metadata or {}
        
        if index is not None:
            # Create pointer from raw integer index
            self.pointer = Pointer.from_index(index)
        else:
            # Create pointer from components
            self.pointer = Pointer(document, chunk, offset)
    
    @property
    def index(self) -> int:
        """Get the encoded index from the pointer."""
        return self.pointer.index
    
    @property
    def document(self) -> int:
        """Get the document number from the pointer."""
        return self.pointer.document
    
    @property
    def chunk(self) -> int:
        """Get the chunk number from the pointer."""
        return self.pointer.chunk
    
    @property
    def offset(self) -> int:
        """Get the offset from the pointer."""
        return self.pointer.offset
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert values to numpy array.
        
        Returns:
            numpy array of the embedding values
        """
        return np.array(self.values, dtype=np.float32)
    
    def get_pointer_components(self) -> Tuple[int, int, int]:
        """
        Get the pointer components (document, chunk, offset).
        
        Returns:
            Tuple of (document, chunk, offset)
        """
        return self.pointer.document, self.pointer.chunk, self.pointer.offset
    
    def set_pointer_components(self, document: int, chunk: int, offset: int) -> None:
        """
        Set the pointer components by creating a new pointer.
        
        Args:
            document: Document number
            chunk: Chunk number
            offset: Offset within chunk
        """
        self.pointer = Pointer(document, chunk, offset)
    
    def get_chunk_id(self) -> int:
        """
        Get the chunk ID from the pointer.
        
        Returns:
            Chunk ID as integer
        """
        return self.pointer.get_chunk_id()
    
    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"Vector(dim={len(self.values)}, doc={self.document}, chunk={self.chunk}, offset={self.offset}, index={self.index})"
