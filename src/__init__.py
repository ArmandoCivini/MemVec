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
    Vector,
    append_vectors_to_bucket
)

__all__ = [
    "VectorCache", 
    "ANNIndex", 
    "VectorStorage", 
    "MemVecRouter",
    "Vector",
    "append_vectors_to_bucket"
]
