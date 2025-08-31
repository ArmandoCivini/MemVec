"""
Configuration constants for MemVec.
"""

# Pointer bit allocation constants (powers of 2)
OFFSET_BITS = 16    # 2^16 = 65,536 vectors per chunk
CHUNK_BITS = 20     # 2^20 = 1,048,576 chunks per document  
DOCUMENT_BITS = 28  # 2^28 = 268,435,456 documents