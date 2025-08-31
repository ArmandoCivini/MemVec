"""
Configuration constants for MemVec.
"""

# Pointer bit allocation constants (powers of 2)
OFFSET_BITS = 16    # 2^16 = 65,536 vectors per chunk
CHUNK_BITS = 20     # 2^20 = 1,048,576 chunks per document  
DOCUMENT_BITS = 28  # 2^28 = 268,435,456 documents

# Chunk configuration
MAX_VECTORS_PER_CHUNK = 100  # Maximum number of vectors in a single chunk

# Text processing configuration
DEFAULT_TEXT_CHUNK_SIZE = 300  # Default words per text chunk
DEFAULT_TEXT_OVERLAP = 50      # Default word overlap between chunks
METADATA_TEXT_PREVIEW_LENGTH = 200  # Max characters for text preview in metadata

# FAISS index configuration
DEFAULT_HNSW_M = 16  # Default number of connections per node in HNSW index

# Search configuration
DEFAULT_SEARCH_K = 5  # Default number of results to return in searches