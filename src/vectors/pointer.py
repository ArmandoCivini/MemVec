"""
Pointer class for vector index management using bit manipulation.

This module provides a Pointer class that encodes/decodes vector indices
from document number, chunk number, and offset using bit manipulation.
"""

import random
from typing import Tuple
from ..config.contants import OFFSET_BITS, CHUNK_BITS, DOCUMENT_BITS


def generate_document_id() -> int:
    """
    Generate a random document ID within valid range.
    
    Returns:
        Random document ID
    """
    max_document = (1 << DOCUMENT_BITS) - 1
    return random.randint(0, max_document)


class Pointer:
    """
    Pointer object that encodes/decodes vector indices using bit manipulation.
    
    Index format (64-bit):
    [document_bits][chunk_bits][offset_bits]
    """
    
    # Derived constants
    MAX_OFFSET = (1 << OFFSET_BITS) - 1
    MAX_CHUNK = (1 << CHUNK_BITS) - 1
    MAX_DOCUMENT = (1 << DOCUMENT_BITS) - 1
    
    # Bit masks and shifts
    OFFSET_MASK = MAX_OFFSET
    CHUNK_MASK = MAX_CHUNK << OFFSET_BITS
    DOCUMENT_MASK = MAX_DOCUMENT << (OFFSET_BITS + CHUNK_BITS)
    
    CHUNK_SHIFT = OFFSET_BITS
    DOCUMENT_SHIFT = OFFSET_BITS + CHUNK_BITS
    
    def __init__(self, document: int, chunk: int, offset: int):
        """
        Initialize a Pointer with document, chunk, and offset.
        
        Args:
            document: Document number (0 to 2^27-1)
            chunk: Chunk number (0 to 2^20-1)
            offset: Offset within chunk (0 to 2^16-1)
        """
        self.document = document
        self.chunk = chunk
        self.offset = offset
        self._index = self._encode()
    
    def _encode(self) -> int:
        """Encode the pointer components into a single index."""
        return (self.document << self.DOCUMENT_SHIFT) | (self.chunk << self.CHUNK_SHIFT) | self.offset
    
    @property
    def index(self) -> int:
        """Get the encoded index value."""
        return self._index
    
    def get_chunk_id(self) -> int:
        """Get the chunk ID by combining document and chunk numbers."""
        return (self.document << CHUNK_BITS) | self.chunk
    
    def __repr__(self) -> str:
        """String representation of the pointer."""
        return f"Pointer(doc={self.document}, chunk={self.chunk}, offset={self.offset}, index={self._index})"
    
    @classmethod
    def from_index(cls, index: int) -> 'Pointer':
        """
        Create a Pointer from an encoded index.
        
        Args:
            index: Encoded index
            
        Returns:
            Pointer instance
        """
        offset = index & cls.OFFSET_MASK
        chunk = (index >> cls.CHUNK_SHIFT) & cls.MAX_CHUNK
        document = (index >> cls.DOCUMENT_SHIFT) & cls.MAX_DOCUMENT
        return cls(document, chunk, offset)
    
    @classmethod
    def encode(cls, document: int, chunk: int, offset: int) -> int:
        """
        Static method to encode document, chunk, and offset into a single index.
        (Kept for backward compatibility)
        """
        return (document << cls.DOCUMENT_SHIFT) | (chunk << cls.CHUNK_SHIFT) | offset
    
    @classmethod
    def decode(cls, index: int) -> Tuple[int, int, int]:
        """
        Static method to decode index back into document, chunk, and offset.
        (Kept for backward compatibility)
        """
        offset = index & cls.OFFSET_MASK
        chunk = (index >> cls.CHUNK_SHIFT) & cls.MAX_CHUNK
        document = (index >> cls.DOCUMENT_SHIFT) & cls.MAX_DOCUMENT
        return document, chunk, offset
    
    @classmethod
    def generate_chunk_id(cls, document: int, chunk: int) -> int:
        """
        Generate a chunk ID by combining document and chunk numbers.
        (Kept for backward compatibility)
        """
        return (document << CHUNK_BITS) | chunk
    
    @classmethod
    def decode_chunk_id(cls, chunk_id: int) -> Tuple[int, int]:
        """
        Decode chunk ID back into document and chunk numbers.
        (Kept for backward compatibility)
        """
        chunk = chunk_id & cls.MAX_CHUNK
        document = (chunk_id >> CHUNK_BITS) & cls.MAX_DOCUMENT
        return document, chunk
    
    @classmethod
    def get_limits(cls) -> dict:
        """
        Get the maximum values for each component.
        
        Returns:
            Dictionary with maximum values
        """
        return {
            "max_document": cls.MAX_DOCUMENT,
            "max_chunk": cls.MAX_CHUNK,
            "max_offset": cls.MAX_OFFSET,
            "total_capacity": cls.MAX_DOCUMENT * cls.MAX_CHUNK * cls.MAX_OFFSET
        }
