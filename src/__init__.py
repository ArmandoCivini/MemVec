"""
MemVec - S3-backed Hot Vector Cache

A two-tier vector search system combining fast in-memory cache
with cost-effective S3 cold storage.
"""

__version__ = "0.1.0"

from .cache import VectorCache
from .index import ANNIndex
from .storage import VectorStorage
from .main import MemVecRouter

# S3 utilities
from .s3 import (
    S3VectorManager,
    create_bucket_simple,
    create_index_simple,
    upload_embedded_simple,
    upload_texts_simple,
    upload_json_simple
)

__all__ = [
    "VectorCache", 
    "ANNIndex", 
    "VectorStorage", 
    "MemVecRouter",
    "S3VectorManager",
    "create_bucket_simple",
    "create_index_simple",
    "upload_embedded_simple",
    "upload_texts_simple",
    "upload_json_simple"
]
