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
    Encodes/decodes vector indices using bit manipulation.
    
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
    
    @classmethod
    def encode(cls, document: int, chunk: int, offset: int) -> int:
        """
        Encode document, chunk, and offset into a single index.
        
        Args:
            document: Document number (0 to 2^28-1)
            chunk: Chunk number (0 to 2^20-1)
            offset: Offset within chunk (0 to 2^16-1)
            
        Returns:
            Encoded index as integer
        """
        return (document << cls.DOCUMENT_SHIFT) | (chunk << cls.CHUNK_SHIFT) | offset
    
    @classmethod
    def decode(cls, index: int) -> Tuple[int, int, int]:
        """
        Decode index back into document, chunk, and offset.
        
        Args:
            index: Encoded index
            
        Returns:
            Tuple of (document, chunk, offset)
        """
        offset = index & cls.OFFSET_MASK
        chunk = (index >> cls.CHUNK_SHIFT) & cls.MAX_CHUNK
        document = (index >> cls.DOCUMENT_SHIFT) & cls.MAX_DOCUMENT
        
        return document, chunk, offset
    
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
